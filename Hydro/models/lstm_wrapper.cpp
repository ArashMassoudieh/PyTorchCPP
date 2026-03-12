#include "lstm_wrapper.h"

#include "neuralnetworkwrapper.h"

#include <cmath>

bool LSTMWrapper::train() {
    try {
        NeuralNetworkWrapper model;
        model.setHiddenLayers({32, 16});
        model.setLags({{1}});
        model.initializeNetwork(1, "relu");

        torch::Tensor t = torch::linspace(0.0, 4.0, 160, torch::kFloat32).unsqueeze(1);
        torch::Tensor y = torch::sin(t) * torch::exp(-0.2 * t);

        std::vector<double> losses = model.trainOnWindow(t, y, 150, 32, 0.005);
        return !losses.empty() && std::isfinite(losses.back());
    } catch (...) {
        return false;
    }
}
