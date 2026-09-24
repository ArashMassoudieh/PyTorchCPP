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

    // Solves dQ/dt = k(fraction*r - Q) via explicit Euler using a parallel
    // (Hillis-Steele) associative scan over the affine recurrence
    // q[i] = (1-step)*q[i-1] + step*fraction*r[i], instead of a per-timestep
    // loop: log2(N) vectorized rounds instead of N sequential single-element
    // tensor ops. Verified against the original sequential recurrence to
    // floating-point precision, forward and gradient, at N up to 7000
    // (~130x faster there); see check_gui_physics_regressions.py.
    static torch::Tensor routeSingleReservoir(const torch::Tensor& runoff, double step, double fraction) {
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

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    routeRunoff(const torch::Tensor& runoff) const {
        if (!runoff.defined() || runoff.dim() != 2 || runoff.size(1) != 1) {
            throw std::invalid_argument("Two-reservoir routing expects runoff with shape [samples,1].");
        }
        if (runoff.size(0) == 0) {
            auto empty = torch::empty_like(runoff);
            return std::make_tuple(empty, empty, empty);
        }
        const torch::Tensor fast = routeSingleReservoir(runoff, dt_hours * fast_k, fast_fraction);
        const torch::Tensor slow = routeSingleReservoir(runoff, dt_hours * slow_k, 1.0 - fast_fraction);
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
