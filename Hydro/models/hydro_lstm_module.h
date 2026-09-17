#pragma once

#include <torch/torch.h>

#include <algorithm>
#include <stdexcept>
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

        // The recurrence is intentionally retained exactly: changing it to an
        // approximate parallel scan would alter the differentiable physics used
        // by the paper experiments.  Avoid, however, constructing/storing a third
        // per-timestep tensor for qFast+qSlow.  Concatenate the two state histories
        // once and form total runoff with one vectorized add after the recurrence.
        torch::Tensor qFast = torch::zeros({1, 1}, runoff.options());
        torch::Tensor qSlow = torch::zeros({1, 1}, runoff.options());
        std::vector<torch::Tensor> fastValues;
        std::vector<torch::Tensor> slowValues;
        fastValues.reserve(static_cast<std::size_t>(runoff.size(0)));
        slowValues.reserve(static_cast<std::size_t>(runoff.size(0)));

        const double fastDecay = dt_hours * fast_k;
        const double slowDecay = dt_hours * slow_k;
        const double fastInput = fastDecay * fast_fraction;
        const double slowInput = slowDecay * (1.0 - fast_fraction);
        const double fastCarry = 1.0 - fastDecay;
        const double slowCarry = 1.0 - slowDecay;

        for (int64_t i = 0; i < runoff.size(0); ++i) {
            const torch::Tensor r = runoff.slice(0, i, i + 1);
            qFast = fastCarry * qFast + fastInput * r;
            qSlow = slowCarry * qSlow + slowInput * r;
            fastValues.push_back(qFast);
            slowValues.push_back(qSlow);
        }
        const torch::Tensor fast = torch::cat(fastValues, 0);
        const torch::Tensor slow = torch::cat(slowValues, 0);
        return std::make_tuple(fast + slow, fast, slow);
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
