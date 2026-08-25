#pragma once

#include <torch/torch.h>

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
