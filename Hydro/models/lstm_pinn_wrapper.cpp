#include "lstm_pinn_wrapper.h"

#include "neuralnetworkwrapper.h"

#include <cmath>

bool LSTMPINNWrapper::train() {
    try {
        NeuralNetworkWrapper model;
        model.setHiddenLayers({32, 16});
        model.setLags({{1}});
        model.initializeNetwork(1, "tanh");

        torch::Tensor t = torch::linspace(0.0, 3.0, 160, torch::kFloat32).unsqueeze(1);
        torch::Tensor y = torch::exp(-0.5 * t);

        model.setTensorData(DataType::Train, t, y);
        std::vector<double> losses = model.trainPINNExponentialDecay(150, 32, 0.005, 0.5, 1.0, 0.8);

        return !losses.empty() && std::isfinite(losses.back());
    } catch (...) {
        return false;
    }
}
