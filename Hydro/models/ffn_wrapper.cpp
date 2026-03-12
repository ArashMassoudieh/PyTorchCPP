#include "ffn_wrapper.h"

#include "neuralnetworkwrapper.h"

#include <cmath>

bool FFNWrapper::train() {
    try {
        NeuralNetworkWrapper model;
        model.setHiddenLayers({16, 16});
        model.setLags({{1}});
        model.initializeNetwork(1, "tanh");

        torch::Tensor t = torch::linspace(0.0, 2.0, 128, torch::kFloat32).unsqueeze(1);
        torch::Tensor y = torch::exp(-0.8 * t);

        std::vector<double> losses = model.trainOnWindow(t, y, 120, 32, 0.01);
        return !losses.empty() && std::isfinite(losses.back());
    } catch (...) {
        return false;
    }
}
