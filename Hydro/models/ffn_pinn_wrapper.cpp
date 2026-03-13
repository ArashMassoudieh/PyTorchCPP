#include "ffn_pinn_wrapper.h"

#include "neuralnetworkwrapper.h"

#include <cmath>
#include <map>
#include <stdexcept>

bool FFNPINNWrapper::train() {
    NeuralNetworkWrapper model;
    model.setHiddenLayers({24, 24});
    model.setLags({{1}});
    model.initializeNetwork(1, "tanh");

    constexpr double lambda = 0.8;
    torch::Tensor t = torch::linspace(0.0, 3.5, 220, torch::kFloat32).unsqueeze(1);
    torch::Tensor y = torch::exp(-lambda * t);

    const int64_t nTrain = 176;
    torch::Tensor tTrain = t.slice(0, 0, nTrain);
    torch::Tensor yTrain = y.slice(0, 0, nTrain);
    torch::Tensor tTest = t.slice(0, nTrain, t.size(0));
    torch::Tensor yTest = y.slice(0, nTrain, y.size(0));

    model.setTensorData(DataType::Train, tTrain, yTrain);
    model.setTensorData(DataType::Test, tTest, yTest);

    std::vector<double> losses = model.trainPINNExponentialDecay(180, 32, 0.002, lambda, 1.0, 0.2);
    if (losses.empty() || !std::isfinite(losses.back())) {
        throw std::runtime_error("FFN-PINN training produced empty/non-finite loss history.");
    }

    torch::Tensor pred = model.forward(DataType::Test);
    if (!pred.defined() || pred.size(0) != yTest.size(0) || !pred.isfinite().all().item<bool>()) {
        throw std::runtime_error("FFN-PINN prediction on test set failed or produced non-finite values.");
    }

    std::map<std::string, double> metrics = model.evaluate();
    auto it = metrics.find("mse");
    if (it != metrics.end() && !std::isfinite(it->second)) {
        throw std::runtime_error("FFN-PINN evaluation produced non-finite MSE.");
    }

    return true;
}
