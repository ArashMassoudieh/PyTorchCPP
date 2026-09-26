#pragma once

#include "hydro_run_types.h"

#include <torch/torch.h>

#include <sstream>
#include <string>
#include <vector>

// Small helpers shared by the reservoir-physics FFN wrappers (plain
// single-reservoir profile in ffn_reservoir_pinn_wrapper.h and the
// residual-correction hybrid in residual_pinn_wrapper.h), factored out so
// both can include it without one wrapper header including the other.
namespace hydro_ffn_reservoir_detail {

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
    if (layers.empty()) layers = {16, 16};
    return layers;
}

inline torch::nn::Sequential makeNetwork(int64_t inputDim,
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

inline std::vector<double> tensorValues(const torch::Tensor& tensor) {
    auto values = tensor.detach().to(torch::kCPU).reshape({-1}).contiguous();
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(values.size(0)));
    for (int64_t i = 0; i < values.size(0); ++i) out.push_back(values[i].item<double>());
    return out;
}

inline void fillPlotVectors(HydroRunResult& result,
                            const torch::Tensor& time,
                            const torch::Tensor& truth,
                            const torch::Tensor& prediction) {
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

} // namespace hydro_ffn_reservoir_detail
