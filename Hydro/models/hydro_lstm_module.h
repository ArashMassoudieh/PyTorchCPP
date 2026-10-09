#pragma once

#include "reservoir_routing_tensor.h"

#include <torch/torch.h>

#include <algorithm>
#include <stdexcept>
#include <tuple>
#include <vector>

struct HydroLSTMImpl : torch::nn::Module {
    HydroLSTMImpl(int64_t inputDim, int64_t hiddenDim, int64_t outputDim, int64_t numLayers, double dropout = 0.0)
        : lstm(torch::nn::LSTMOptions(inputDim, hiddenDim).num_layers(numLayers).batch_first(true).dropout(dropout)),
          fc(hiddenDim, outputDim) {
        register_module("lstm", lstm);
        register_module("fc", fc);
    }

    torch::Tensor forward(const torch::Tensor& inputs) {
        const auto output = std::get<0>(lstm->forward(inputs));
        return fc->forward(output.select(1, output.size(1) - 1));
    }

    torch::nn::LSTM lstm{nullptr};
    torch::nn::Linear fc{nullptr};
};
TORCH_MODULE(HydroLSTM);

// Process-aware recurrent hybrid used by the real-catchment LSTM+PINN path.
// The recurrent backbone learns a non-negative effective runoff-generation
// signal R_nn from meteorological sequences.  A differentiable two-reservoir
// routing layer then converts R_nn to total runoff:
//
//   dQ_f/dt = k_f (alpha R_nn - Q_f)
//   dQ_s/dt = k_s ((1-alpha) R_nn - Q_s)
//   Q = Q_f + Q_s
//
// k_f, k_s and alpha are validation-selected physical-routing hyperparameters.
// They are intentionally not trainable here so the held-out test set cannot
// influence their calibration.
struct HydroTwoReservoirLSTMImpl : torch::nn::Module {
    HydroTwoReservoirLSTMImpl(int64_t inputDim,
                              int64_t hiddenDim,
                              int64_t numLayers,
                              double dtHours,
                              double fastK,
                              double slowK,
                              double fastFraction)
        : lstm(torch::nn::LSTMOptions(inputDim, hiddenDim).num_layers(numLayers).batch_first(true)),
          runoff_head(hiddenDim, 1),
          dt_hours(dtHours),
          fast_k(fastK),
          slow_k(slowK),
          fast_fraction(std::max(0.0, std::min(1.0, fastFraction))) {
        register_module("lstm", lstm);
        register_module("runoff_head", runoff_head);
    }

    torch::Tensor runoffGeneration(const torch::Tensor& inputs) {
        const auto output = std::get<0>(lstm->forward(inputs));
        const auto last = output.select(1, output.size(1) - 1);
        return torch::softplus(runoff_head->forward(last));
    }

    static torch::Tensor routeSingleReservoir(const torch::Tensor& runoff, double step, double fraction) {
        return routeReservoirTensor(runoff, step, fraction);
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    routeRunoff(const torch::Tensor& runoff) const {
        if (!runoff.defined() || runoff.dim() != 2 || runoff.size(1) != 1) {
            throw std::invalid_argument("Two-reservoir routing expects runoff with shape [samples,1].");
        }
        if (runoff.size(0) == 0) {
            auto empty = torch::empty_like(runoff);
            return std::make_tuple(empty, empty, empty);
        }
        const torch::Tensor fast = routeReservoirTensor(runoff, dt_hours * fast_k, fast_fraction, exponential_routing);
        const torch::Tensor slow = routeReservoirTensor(runoff, dt_hours * slow_k, 1.0 - fast_fraction, exponential_routing);
        return std::make_tuple(fast + slow, fast, slow);
    }

    torch::Tensor forward(const torch::Tensor& inputs) {
        return std::get<0>(routeRunoff(runoffGeneration(inputs)));
    }

    // Old archives have no scheme marker and must retain their Euler behavior.
    void save(torch::serialize::OutputArchive& archive) const override {
        torch::nn::Module::save(archive);
        archive.write("routing_scheme", torch::tensor(exponential_routing ? 1 : 0));
    }
    void load(torch::serialize::InputArchive& archive) override {
        torch::Tensor scheme;
        exponential_routing = archive.try_read("routing_scheme", scheme) && scheme.item<int>() == 1;
        torch::nn::Module::load(archive);
    }
    bool exponential_routing = true;

    torch::nn::LSTM lstm{nullptr};
    torch::nn::Linear runoff_head{nullptr};
    double dt_hours = 1.0;
    double fast_k = 0.25;
    double slow_k = 0.01;
    double fast_fraction = 0.7;
};
TORCH_MODULE(HydroTwoReservoirLSTM);
