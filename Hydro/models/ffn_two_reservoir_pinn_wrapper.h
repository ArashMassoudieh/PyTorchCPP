#pragma once

#include "hydro_run_types.h"
#include "../dataset/chronological_split.h"
#include "../dataset/lagged_tensor_builder.h"
#include "../dataset/reservoir_physics_tensor_builder.h"
#include "../dataset/tensor_scaler.h"
#include "../evaluation/hydro_metrics.h"
#include "../evaluation/model_checkpoint.h"

#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

// Process-aware FFN+PINN hybrid: the feed-forward network learns a
// non-negative effective runoff-generation signal R_nn from (optionally
// lagged) meteorological features, and a differentiable fast/slow reservoir
// routes it to total runoff exactly as HydroTwoReservoirLSTMImpl does for the
// LSTM+PINN path:
//
//   dQ_f/dt = k_f (alpha R_nn - Q_f)
//   dQ_s/dt = k_s ((1-alpha) R_nn - Q_s),  Q = Q_f + Q_s
//
// A single linear reservoir cannot represent a catchment with both a fast
// quickflow response and a slow baseflow recession at once (the two rates
// measured directly off the Sligo Creek record differ by ~75x), so this gives
// FFN+PINN the same structural routing capacity LSTM+PINN already has.
// storage_coeff=k_f, lambda_decay=k_s, runoff_coeff=alpha, matching the
// two_reservoir_hybrid config convention used by LSTM+PINN.
namespace hydro_ffn_two_reservoir_detail {

inline std::vector<int> parseHiddenLayers(const std::string& csv) {
    std::vector<int> layers;
    std::stringstream ss(csv);
    std::string token;
    while (std::getline(ss, token, ',')) {
        try {
            const int value = std::stoi(token);
            if (value > 0) layers.push_back(value);
        } catch (...) {}
    }
    if (layers.empty()) layers = {32, 16};
    return layers;
}

inline torch::nn::Sequential makeRunoffHead(int64_t inputDim,
                                            const std::vector<int>& hidden,
                                            const std::string& activation) {
    torch::nn::Sequential model;
    int64_t in = inputDim;
    for (const int width : hidden) {
        model->push_back(torch::nn::Linear(in, width));
        if (activation == "relu") model->push_back(torch::nn::ReLU());
        else if (activation == "sigmoid") model->push_back(torch::nn::Sigmoid());
        else model->push_back(torch::nn::Tanh());
        in = width;
    }
    model->push_back(torch::nn::Linear(in, 1));
    return model;
}

// Same forcing-only feature contract as LSTM+PINN's predictorFeatures(): drop
// the derived I*=max(P-PET,0) column so the network sees only raw drivers.
// GIStoOHQ physics layout: [time, I*, P, PET, T, RH, wind, solar].
inline torch::Tensor predictorFeatures(const torch::Tensor& physicsX) {
    if (physicsX.dim() != 2 || physicsX.size(1) < 3) {
        throw std::runtime_error("FFN two-reservoir predictor requires [time, forcing, ...].");
    }
    if (physicsX.size(1) >= 8) {
        return torch::cat({
            physicsX.slice(1, 2, 3), // P
            physicsX.slice(1, 4, 5), // T
            physicsX.slice(1, 5, 6), // RH
            physicsX.slice(1, 6, 7), // wind
            physicsX.slice(1, 7, 8), // solar
            physicsX.slice(1, 3, 4)  // PET
        }, 1).contiguous();
    }
    if (physicsX.size(1) >= 4) return physicsX.slice(1, 2, 4).contiguous(); // [P, PET]
    return physicsX.slice(1, 1, physicsX.size(1)).contiguous();
}

// Mirrors HydroTwoReservoirLSTMImpl::routeSingleReservoir/routeRunoff exactly
// (same affine recurrence, same parallel-scan formulation) so the two hybrids
// stay numerically consistent; kept as a separate copy rather than a shared
// header to avoid coupling this FFN wrapper to an LSTM-named struct/header.
//
// Solves dQ/dt = k(fraction*r - Q) via explicit Euler using a parallel
// (Hillis-Steele) associative scan over the affine recurrence
// q[i] = (1-step)*q[i-1] + step*fraction*r[i], instead of a per-timestep
// loop: log2(N) vectorized rounds instead of N sequential single-element
// tensor ops (~130x faster at N~7000; verified forward+gradient identical to
// floating-point precision against the sequential form it replaces).
inline torch::Tensor routeSingleReservoir(const torch::Tensor& runoff, double step, double fraction) {
    const int64_t n = runoff.size(0);
    torch::Tensor a = torch::full_like(runoff, 1.0 - step);
    torch::Tensor b = (step * fraction) * runoff;
    for (int64_t offset = 1; offset < n; offset *= 2) {
        torch::Tensor aShift = torch::ones_like(a);
        torch::Tensor bShift = torch::zeros_like(b);
        aShift.slice(0, offset, n) = a.slice(0, 0, n - offset);
        bShift.slice(0, offset, n) = b.slice(0, 0, n - offset);
        b = a * bShift + b;
        a = a * aShift;
    }
    return b;
}

inline torch::Tensor routeTwoReservoir(const torch::Tensor& runoff,
                                       double dtHours, double fastK, double slowK, double fastFraction) {
    if (!runoff.defined() || runoff.dim() != 2 || runoff.size(1) != 1) {
        throw std::invalid_argument("Two-reservoir routing expects runoff with shape [samples,1].");
    }
    const torch::Tensor fast = routeSingleReservoir(runoff, dtHours * fastK, fastFraction);
    const torch::Tensor slow = routeSingleReservoir(runoff, dtHours * slowK, 1.0 - fastFraction);
    return fast + slow;
}

inline std::vector<double> tensorValues(const torch::Tensor& tensor) {
    auto values = tensor.detach().to(torch::kCPU).reshape({-1}).contiguous();
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(values.size(0)));
    for (int64_t i = 0; i < values.size(0); ++i) out.push_back(values[i].item<double>());
    return out;
}

inline void fillPlotVectors(HydroRunResult& result, const torch::Tensor& time,
                            const torch::Tensor& truth, const torch::Tensor& prediction) {
    auto tc = time.reshape({-1}).to(torch::kCPU).contiguous();
    auto yc = truth.reshape({-1}).to(torch::kCPU).contiguous();
    auto pc = prediction.reshape({-1}).to(torch::kCPU).contiguous();
    result.x.reserve(static_cast<std::size_t>(tc.size(0)));
    result.y_true.reserve(static_cast<std::size_t>(tc.size(0)));
    result.y_pred.reserve(static_cast<std::size_t>(tc.size(0)));
    for (int64_t i = 0; i < tc.size(0); ++i) {
        result.x.push_back(tc[i].item<double>());
        result.y_true.push_back(yc[i].item<double>());
        result.y_pred.push_back(pc[i].item<double>());
    }
}

} // namespace hydro_ffn_two_reservoir_detail

class FFNTwoReservoirPINNWrapper {
public:
    HydroRunResult train(const HydroRunConfig& config) {
        using namespace hydro_ffn_two_reservoir_detail;
        HydroRunResult result;
        torch::manual_seed(static_cast<uint64_t>(std::max(0, config.random_seed)));

        torch::Tensor physicsX, y, plotX;
        if (!loadReservoirPhysicsTensors(config, physicsX, y, plotX)) {
            throw std::runtime_error("Unable to construct process-aware FFN+PINN tensors.");
        }
        if (physicsX.dim() != 2 || physicsX.size(1) < 4) {
            throw std::runtime_error("FFN two-reservoir hybrid requires [time, I*, P, PET, ...] forcing layout.");
        }

        const torch::Tensor precipitation = physicsX.slice(1, 2, 3).contiguous();
        torch::Tensor modelX = predictorFeatures(physicsX);
        torch::Tensor time = plotX;
        torch::Tensor precip = precipitation;
        torch::Tensor target = y;
        if (config.use_time_lagged_ffn) {
            const auto lagged = buildHydroLaggedTensor(modelX, config.input_lags_csv);
            modelX = lagged.inputs;
            time = time.slice(0, lagged.leading_rows, time.size(0)).contiguous();
            precip = precip.slice(0, lagged.leading_rows, precip.size(0)).contiguous();
            target = target.slice(0, lagged.leading_rows, target.size(0)).contiguous();
        }
        if (modelX.size(0) < 10) throw std::runtime_error("Too few samples for FFN two-reservoir hybrid.");

        const ChronologicalSplit split = makeChronologicalSplit(modelX.size(0),
                                                                config.train_split_ratio,
                                                                config.validation_split_ratio);
        const int64_t nTrain = split.train_end;
        const int64_t nValidationEnd = split.validation_end;

        torch::Tensor xTrainPhysical = modelX.slice(0, 0, nTrain).contiguous();
        torch::Tensor yTrainPhysical = target.slice(0, 0, nTrain).contiguous();
        torch::Tensor xValidationPhysical = modelX.slice(0, nTrain, nValidationEnd).contiguous();
        torch::Tensor yValidationPhysical = target.slice(0, nTrain, nValidationEnd).contiguous();
        torch::Tensor yTestPhysical = target.slice(0, nValidationEnd, target.size(0)).contiguous();

        TensorScaler inputScaler;
        TensorScaler targetScaler;
        inputScaler.fit(xTrainPhysical, "standardize");
        targetScaler.fit(yTrainPhysical, "standardize");
        const torch::Tensor xFull = inputScaler.transform(modelX);
        const torch::Tensor xTrain = xFull.slice(0, 0, nTrain).contiguous();
        const torch::Tensor yTrain = targetScaler.transform(yTrainPhysical);
        const torch::Tensor yValidation = targetScaler.transform(yValidationPhysical);

        const double dt = regularPhysicalTimeStepFromTime(time);
        const double fastK = config.storage_coeff;
        const double slowK = config.lambda_decay;
        const double alpha = config.runoff_coeff;
        if (!(fastK > 0.0 && slowK > 0.0 && fastK > slowK && alpha > 0.0 && alpha < 1.0)) {
            throw std::runtime_error("FFN two-reservoir hybrid requires fast_k>slow_k>0 and 0<routing_alpha<1.");
        }
        if (dt * fastK > 1.0 || dt * slowK > 1.0) {
            throw std::runtime_error("FFN two-reservoir hybrid explicit routing requires dt*k <= 1 for both stores.");
        }

        torch::nn::Sequential model = makeRunoffHead(xFull.size(1), parseHiddenLayers(config.hidden_layers_csv), config.activation);
        torch::optim::Adam optimizer(model->parameters(),
            torch::optim::AdamOptions(config.learning_rate).weight_decay(config.weight_decay));

        const torch::Tensor pTrain = precip.slice(0, 0, nTrain);
        const double precipitationScale = std::max(1.0e-8, pTrain.pow(2).mean().item<double>());
        const int totalEpochs = std::max(1, config.epochs);

        // softplus(0) ~= 0.69, which is roughly 10-100x the typical observed
        // flow scale here, so an untrained head starts wildly over-predicting.
        // Briefly pretrain the head directly against the observed (non-negative
        // physical) target before the routed physics objective takes over, the
        // same role the LSTM+PINN "supervised parent" warm start plays there.
        const int pretrainEpochs = std::min(totalEpochs - 1, std::max(10, (2 * totalEpochs) / 5));
        for (int epoch = 0; epoch < pretrainEpochs; ++epoch) {
            model->train();
            optimizer.zero_grad();
            const torch::Tensor pretrainRunoff = torch::softplus(model->forward(xTrain));
            const torch::Tensor pretrainLoss = torch::mse_loss(pretrainRunoff, yTrainPhysical);
            pretrainLoss.backward();
            torch::nn::utils::clip_grad_norm_(model->parameters(), 5.0);
            optimizer.step();
        }

        const int fineTuneEpochs = std::max(1, totalEpochs - pretrainEpochs);
        std::vector<torch::Tensor> bestParameters;
        std::vector<double> losses;
        std::vector<double> validationLosses;
        double bestValidationMse = std::numeric_limits<double>::infinity();
        int bestEpoch = 0;

        // best_epoch indexes this fine-tuning history only; the supervised
        // pretraining above is intentionally not part of training_loss_history,
        // matching the LSTM+PINN two-reservoir convention.
        for (int epoch = 0; epoch < fineTuneEpochs; ++epoch) {
            model->train();
            optimizer.zero_grad();
            const torch::Tensor runoff = model->forward(xTrain);
            const torch::Tensor runoffNonnegative = torch::softplus(runoff);
            const torch::Tensor qPhysical = routeTwoReservoir(runoffNonnegative, dt, fastK, slowK, alpha);
            const torch::Tensor qScaled = targetScaler.transform(qPhysical);
            const torch::Tensor dataLoss = torch::mse_loss(qScaled, yTrain);
            const torch::Tensor excess = torch::relu(runoffNonnegative - pTrain);
            const torch::Tensor availabilityLoss = torch::mean(excess * excess) / precipitationScale;
            const double ramp = static_cast<double>(epoch + 1) / static_cast<double>(fineTuneEpochs);
            const double effectivePhysicsWeight = config.physics_weight * ramp * ramp;
            const torch::Tensor totalLoss = config.data_weight * dataLoss + effectivePhysicsWeight * availabilityLoss;
            totalLoss.backward();
            torch::nn::utils::clip_grad_norm_(model->parameters(), 5.0);
            optimizer.step();
            losses.push_back(totalLoss.item<double>());

            model->eval();
            double validationMsePhysical = 0.0;
            {
                torch::NoGradGuard noGrad;
                // Route from the start of the record so validation receives the
                // physically carried storage state from the training period.
                const torch::Tensor runoffThroughValidation =
                    torch::softplus(model->forward(xFull.slice(0, 0, nValidationEnd)));
                const torch::Tensor qThroughValidation = routeTwoReservoir(runoffThroughValidation, dt, fastK, slowK, alpha);
                const torch::Tensor qValidation = qThroughValidation.slice(0, nTrain, nValidationEnd);
                validationMsePhysical = torch::mse_loss(qValidation, yValidationPhysical).item<double>();
            }
            validationLosses.push_back(validationMsePhysical);
            if (validationMsePhysical < bestValidationMse) {
                bestValidationMse = validationMsePhysical;
                bestEpoch = epoch + 1;
                bestParameters.clear();
                for (const auto& p : model->parameters()) bestParameters.push_back(p.detach().clone());
            }
        }

        if (bestParameters.empty()) throw std::runtime_error("FFN two-reservoir hybrid did not produce a validation checkpoint.");
        {
            torch::NoGradGuard noGrad;
            auto parameters = model->parameters();
            for (std::size_t i = 0; i < parameters.size(); ++i) parameters[i].copy_(bestParameters[i]);
        }

        result.training_loss_history = losses;
        result.validation_loss_history = validationLosses;
        result.best_epoch = bestEpoch;
        result.final_loss = losses.at(static_cast<std::size_t>(bestEpoch - 1));
        result.validation_mse = bestValidationMse;
        result.input_scaler = inputScaler.exportState();
        result.target_scaler = targetScaler.exportState();

        {
            const auto checkpoint = temporaryHydroCheckpointPath("hydro_ffn_pinn_two_reservoir");
            torch::serialize::OutputArchive archive;
            model->save(archive);
            archive.save_to(checkpoint.string());
            result.model_checkpoint = readHydroCheckpoint(checkpoint);
            result.model_checkpoint_format = "torch-sequential-v1";
            std::filesystem::remove(checkpoint);
        }

        model->eval();
        torch::NoGradGuard noGrad;
        const torch::Tensor runoffFull = torch::softplus(model->forward(xFull));
        const torch::Tensor predFullPhysical = routeTwoReservoir(runoffFull, dt, fastK, slowK, alpha);
        const torch::Tensor predTestPhysical = predFullPhysical.slice(0, nValidationEnd, predFullPhysical.size(0));
        if (!predFullPhysical.isfinite().all().item<bool>()) {
            throw std::runtime_error("FFN two-reservoir hybrid produced non-finite runoff.");
        }
        if (config.evaluate_metrics) {
            populateHydroMetrics(result, tensorValues(yTestPhysical), tensorValues(predTestPhysical));
            if (!hydroMetricsAreFinite(result)) throw std::runtime_error("FFN two-reservoir hybrid evaluation produced invalid core hydrology metrics.");
        }

        fillPlotVectors(result, time, target, predFullPhysical);
        result.split.resize(result.x.size(), "test");
        for (std::size_t i = 0; i < result.split.size(); ++i) {
            if (static_cast<int64_t>(i) < split.train_end) result.split[i] = "train";
            else if (static_cast<int64_t>(i) < split.validation_end) result.split[i] = "validation";
        }
        populateHydroPeakMetrics(result);

        const torch::Tensor availabilityResidual = torch::relu(runoffFull - precip);
        result.physics_loss = torch::mean(availabilityResidual * availabilityResidual).item<double>();
        result.physics_residual = tensorValues(availabilityResidual);
        populateHydroPhysicsResidualMetrics(result);

        result.success = true;
        result.message = "FFN+PINN completed with learned effective runoff generation and differentiable fast/slow reservoir routing.";
        return result;
    }
};
