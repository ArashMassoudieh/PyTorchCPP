#include "lstm_pinn_wrapper.h"

#include "neuralnetworkwrapper.h"

#include <cmath>
#include <stdexcept>

bool LSTMPINNWrapper::train() {
    // Temporary backend: uses NeuralNetworkWrapper FFN core while LSTM-PINN is scaffolded.
    NeuralNetworkWrapper model;
    model.setHiddenLayers({32, 24});
    model.setLags({{1}});
    model.initializeNetwork(1, "tanh");

    constexpr double lambda = 0.5;
    torch::Tensor t = torch::linspace(0.0, 5.0, 240, torch::kFloat32).unsqueeze(1);
    torch::Tensor y = torch::exp(-lambda * t);

    const int64_t nTrain = 192;
    torch::Tensor tTrain = t.slice(0, 0, nTrain);
    torch::Tensor yTrain = y.slice(0, 0, nTrain);
    torch::Tensor tTest = t.slice(0, nTrain, t.size(0));
    torch::Tensor yTest = y.slice(0, nTrain, y.size(0));

    model.setTensorData(DataType::Train, tTrain, yTrain);
    model.setTensorData(DataType::Test, tTest, yTest);

    std::vector<double> losses = model.trainPINNExponentialDecay(220, 32, 0.0015, lambda, 1.0, 0.25);
    if (losses.empty() || !std::isfinite(losses.back())) {
        throw std::runtime_error("LSTM-PINN training produced empty/non-finite loss history.");
    }

    torch::Tensor pred = model.forward(DataType::Test);
    if (!pred.defined() || pred.size(0) != yTest.size(0) || !pred.isfinite().all().item<bool>()) {
        throw std::runtime_error("LSTM-PINN prediction failed or produced non-finite values.");
    }

    return true;
}
