#pragma once

#include <torch/torch.h>

#include <algorithm>
#include <tuple>
#include <vector>

struct HydroLSTMImpl : torch::nn::Module {
    HydroLSTMImpl(int64_t inputDim, int64_t hiddenDim, int64_t outputDim, int64_t numLayers)
        : lstm(torch::nn::LSTMOptions(inputDim, hiddenDim).num_layers(numLayers).batch_first(true)),
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

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    routeRunoff(const torch::Tensor& runoff) const {
        if (!runoff.defined() || runoff.dim() != 2 || runoff.size(1) != 1) {
            throw std::invalid_argument("Two-reservoir routing expects runoff with shape [samples,1].");
        }
        if (runoff.size(0) == 0) {
            auto empty = torch::empty_like(runoff);
            return std::make_tuple(empty, empty, empty);
        }

        torch::Tensor qFast = torch::zeros({1, 1}, runoff.options());
        torch::Tensor qSlow = torch::zeros({1, 1}, runoff.options());
        std::vector<torch::Tensor> fastValues;
        std::vector<torch::Tensor> slowValues;
        std::vector<torch::Tensor> totalValues;
        fastValues.reserve(static_cast<std::size_t>(runoff.size(0)));
        slowValues.reserve(static_cast<std::size_t>(runoff.size(0)));
        totalValues.reserve(static_cast<std::size_t>(runoff.size(0)));

        for (int64_t i = 0; i < runoff.size(0); ++i) {
            const torch::Tensor r = runoff.slice(0, i, i + 1);
            qFast = qFast + dt_hours * fast_k * (fast_fraction * r - qFast);
            qSlow = qSlow + dt_hours * slow_k * ((1.0 - fast_fraction) * r - qSlow);
            fastValues.push_back(qFast);
            slowValues.push_back(qSlow);
            totalValues.push_back(qFast + qSlow);
        }
        return std::make_tuple(torch::cat(totalValues, 0),
                               torch::cat(fastValues, 0),
                               torch::cat(slowValues, 0));
    }

    torch::Tensor forward(const torch::Tensor& inputs) {
        return std::get<0>(routeRunoff(runoffGeneration(inputs)));
    }

    torch::nn::LSTM lstm{nullptr};
    torch::nn::Linear runoff_head{nullptr};
    double dt_hours = 1.0;
    double fast_k = 0.25;
    double slow_k = 0.01;
    double fast_fraction = 0.7;
};
TORCH_MODULE(HydroTwoReservoirLSTM);
