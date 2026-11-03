#pragma once

#include "hydro_lstm_module.h"

#include <torch/torch.h>

#include <tuple>
#include <vector>

// Daily-resolution antecedent-history branch (improvement_loop_20261023
// reports 66/67/68). A small LSTM processes `historyDays` of coarse daily
// features; its final (h, c) is transferred, via learned linear layers,
// into the existing hourly HydroLSTM's initial state -- Gauch et al.
// (2021, HESS)'s multi-branch MTS-LSTM variant. This module is only ever
// constructed when a config opts in (multiscale_daily_history_days > 0);
// it does not exist in, and has no effect on, the disabled code path.
struct DailyHistoryBranchImpl : torch::nn::Module {
    DailyHistoryBranchImpl(int64_t dailyInputDim, int64_t dailyHiddenDim,
                           int64_t hourlyHiddenDim, int64_t hourlyNumLayers)
        : lstm(torch::nn::LSTMOptions(dailyInputDim, dailyHiddenDim).num_layers(1).batch_first(true)),
          fc_h(dailyHiddenDim, hourlyHiddenDim * hourlyNumLayers),
          fc_c(dailyHiddenDim, hourlyHiddenDim * hourlyNumLayers),
          hourlyHiddenDim_(hourlyHiddenDim),
          hourlyNumLayers_(hourlyNumLayers) {
        register_module("lstm", lstm);
        register_module("fc_h", fc_h);
        register_module("fc_c", fc_c);
    }

    // dailyInputs: [batch, historyDays, dailyInputDim].
    // Returns (h0, c0), each [hourlyNumLayers, batch, hourlyHiddenDim] --
    // the exact shape torch::nn::LSTM::forward's hx argument requires.
    //
    // CORRECTED (report 69, caught on code review): the first version
    // discarded the LSTM's own returned (h_n, c_n) state tuple entirely
    // and projected BOTH h0 and c0 from the output sequence's last
    // timestep -- i.e. c0 was a learned function of h_n, never of the
    // actual final cell state. This version takes h_n and c_n from
    // torch::nn::LSTM::forward's own returned state tuple directly, so
    // h0 and c0 are genuinely independent projections of the daily
    // branch's own final hidden and cell states, matching the design
    // this module is documented as implementing (MULTISCALE_DESIGN.md).
    std::tuple<torch::Tensor, torch::Tensor> forward(const torch::Tensor& dailyInputs) {
        const auto result = lstm->forward(dailyInputs);
        const auto state = std::get<1>(result);          // (h_n, c_n), each [1, batch, dailyHiddenDim]
        const auto hLast = std::get<0>(state).select(0, 0);  // single daily layer -> [batch, dailyHiddenDim]
        const auto cLast = std::get<1>(state).select(0, 0);
        const auto batch = hLast.size(0);
        auto h0 = fc_h->forward(hLast).reshape({batch, hourlyNumLayers_, hourlyHiddenDim_})
                       .transpose(0, 1).contiguous();
        auto c0 = fc_c->forward(cLast).reshape({batch, hourlyNumLayers_, hourlyHiddenDim_})
                       .transpose(0, 1).contiguous();
        return {h0, c0};
    }

    torch::nn::LSTM lstm{nullptr};
    torch::nn::Linear fc_h{nullptr};
    torch::nn::Linear fc_c{nullptr};
    int64_t hourlyHiddenDim_;
    int64_t hourlyNumLayers_;
};
TORCH_MODULE(DailyHistoryBranch);

// Wraps an existing HydroLSTM with the daily branch above. Only
// instantiated when the opt-in config field is positive -- the disabled
// path constructs a plain HydroLSTM directly (see shared_scale_inference.cpp),
// never this wrapper, so old checkpoints and today's exact forward() call
// are both untouched by this file's existence.
struct MultiscaleHydroLSTMImpl : torch::nn::Module {
    MultiscaleHydroLSTMImpl(int64_t hourlyInputDim, int64_t hourlyHiddenDim,
                            int64_t outputDim, int64_t hourlyNumLayers,
                            int64_t dailyInputDim, int64_t dailyHiddenDim)
        : hourly(hourlyInputDim, hourlyHiddenDim, outputDim, hourlyNumLayers),
          daily(dailyInputDim, dailyHiddenDim, hourlyHiddenDim, hourlyNumLayers) {
        register_module("hourly", hourly);
        register_module("daily", daily);
        // Registered at CONSTRUCTION time, not lazily inside fitDailyScaler
        // -- torch::nn::Module's default save()/load() matches buffers by
        // name against whatever the CURRENT instance already has registered;
        // a freshly-constructed model being load()ed into must already have
        // these slots, or there is nothing for load() to fill from the
        // archive even though the archive itself has the right entries.
        // This was a real bug, caught by actually running the round-trip
        // test, not by reasoning about it -- see report 71.
        dailyScalerOffset = register_buffer("daily_scaler_offset", torch::zeros({1, 1, dailyInputDim}));
        dailyScalerScale = register_buffer("daily_scaler_scale", torch::ones({1, 1, dailyInputDim}));
    }

    // CORRECTED (report 69, caught on code review): the first version also
    // had a forwardWithoutDailyHistory() fallback for ineligible windows,
    // which conflicted with the design's own eligibility-mask policy
    // (report 67 Sec 4: both arms are scored on the IDENTICAL eligible-window
    // set, not "multiscale arm scores everything, falling back when it
    // can't"). Eligibility is now handled once, at the data level
    // (daily_history::eligibleWindowIndices), applied identically before
    // training loss and before scoring for BOTH arms -- this model never
    // sees an ineligible window at all, so no fallback forward exists here.
    torch::Tensor forward(const torch::Tensor& hourlyInputs, const torch::Tensor& dailyInputs) {
        const auto state = daily->forward(scaleDailyInputs(dailyInputs));
        return hourly->forward(hourlyInputs, std::get<0>(state), std::get<1>(state));
    }

    // Daily-feature scaling (improvement_loop_20261023 report 71): the
    // hourly branch's inputs are standardized via TensorScaler before this
    // model ever sees them (shared_scale_inference.cpp / lstm_pinn_wrapper.cpp),
    // but the daily branch's mean-hourly-rate features (report 70) were
    // otherwise passed through completely unscaled -- a real scale
    // mismatch between the two branches' inputs, left unresolved in report
    // 70 rather than risking the shared scalers.csv artifact format (used,
    // unmodified, by every other approach's checkpoints).
    //
    // Resolved here instead, inside THIS model's own checkpoint, which
    // already has its own dedicated, non-shared format
    // ("torch-module-v1-multiscale") -- fitDailyScaler stores a per-feature
    // offset/scale (standardize: mean/std, computed from training-eligible
    // rows only, by the caller) as ordinary registered buffers, so they
    // save/load with the rest of the model through the exact same
    // save()/load() calls, with no change to scalers.csv, artifact_loader.cpp,
    // or any other approach's checkpoint.
    void fitDailyScaler(const torch::Tensor& trainingDailyFeatures) {
        if (!trainingDailyFeatures.defined() || trainingDailyFeatures.numel() == 0) {
            throw std::invalid_argument("fitDailyScaler requires non-empty training daily features.");
        }
        if (!trainingDailyFeatures.isfinite().all().item<bool>()) {
            throw std::invalid_argument("fitDailyScaler requires finite training daily features.");
        }
        const std::vector<int64_t> reduceDims{0, 1};
        auto offset = trainingDailyFeatures.mean(reduceDims, /*keepdim=*/true).detach();
        auto scale = trainingDailyFeatures.std(reduceDims, /*unbiased=*/false, /*keepdim=*/true).detach();
        scale = torch::where(torch::abs(scale) < 1.0e-8, torch::ones_like(scale), scale);
        // Update the ALREADY-registered buffers in place (constructor above)
        // -- do not re-register here, which would create a second, orphaned
        // slot invisible to an instance that was load()ed rather than fit.
        torch::NoGradGuard noGrad;
        dailyScalerOffset.copy_(offset);
        dailyScalerScale.copy_(scale);
        dailyScalerFitted_ = true;
    }

    torch::Tensor scaleDailyInputs(const torch::Tensor& dailyInputs) const {
        if (!dailyScalerFitted_) {
            throw std::runtime_error(
                "MultiscaleHydroLSTM: daily scaler was never fit (call fitDailyScaler before forward(), "
                "or load a checkpoint that already has daily_scaler_offset/scale registered).");
        }
        return (dailyInputs - dailyScalerOffset) / dailyScalerScale;
    }

    // save()/load() are torch::nn::Module's own, inherited unchanged --
    // registered buffers (dailyScalerOffset/Scale above) are serialized
    // automatically by the base Module::save/load, exactly like every
    // parameter, with no override needed. dailyScalerFitted_ itself is a
    // plain bool, not a tensor, so it must be reconstructed from whether
    // the buffer round-tripped -- done once, immediately after load(), by
    // the loader (confirmDailyScalerLoaded below), not silently assumed.
    void confirmDailyScalerLoaded() {
        if (!dailyScalerOffset.defined() || !dailyScalerScale.defined() || dailyScalerOffset.numel() == 0) {
            throw std::runtime_error("MultiscaleHydroLSTM: checkpoint has no daily scaler state -- "
                                      "was it saved by a version of this class before fitDailyScaler existed?");
        }
        dailyScalerFitted_ = true;
    }

    HydroLSTM hourly{nullptr};
    DailyHistoryBranch daily{nullptr};
    torch::Tensor dailyScalerOffset;
    torch::Tensor dailyScalerScale;
    bool dailyScalerFitted_ = false;
};
TORCH_MODULE(MultiscaleHydroLSTM);
