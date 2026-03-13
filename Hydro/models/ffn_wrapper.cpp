#include "ffn_wrapper.h"

#include "neuralnetworkwrapper.h"

#include <cmath>
#include <map>

HydroRunResult FFNWrapper::train(const HydroRunConfig& config) {
    HydroRunResult result;

    NeuralNetworkWrapper model;
    model.setHiddenLayers({24, 24});
    model.setLags({{1}});
    model.initializeNetwork(1, "tanh");

    torch::Tensor t = torch::linspace(0.0, 4.0, 200, torch::kFloat32).unsqueeze(1);
    torch::Tensor y = torch::sin(t) * torch::exp(-0.15 * t);

    const int64_t nTrain = 160;
    torch::Tensor tTrain = t.slice(0, 0, nTrain);
    torch::Tensor yTrain = y.slice(0, 0, nTrain);
    torch::Tensor tTest = t.slice(0, nTrain, t.size(0));
    torch::Tensor yTest = y.slice(0, nTrain, y.size(0));

    model.setTensorData(DataType::Train, tTrain, yTrain);
    model.setTensorData(DataType::Test, tTest, yTest);

    std::vector<double> losses = model.train(config.epochs, config.batch_size, config.learning_rate);
    if (losses.empty() || !std::isfinite(losses.back())) {
        throw std::runtime_error("FFN training produced empty/non-finite loss history.");
    }
    result.final_loss = losses.back();

    torch::Tensor pred = model.forward(DataType::Test);
    if (!pred.defined() || pred.size(0) != yTest.size(0) || !pred.isfinite().all().item<bool>()) {
        throw std::runtime_error("FFN prediction on test set failed or produced non-finite values.");
    }

    if (config.evaluate_metrics) {
        std::map<std::string, double> metrics = model.evaluate();
        auto it = metrics.find("mse");
        if (it != metrics.end()) {
            if (!std::isfinite(it->second)) {
                throw std::runtime_error("FFN evaluation produced non-finite MSE.");
            }
            result.mse = it->second;
        }
    }

    result.success = true;
    result.message = "FFN run completed.";
    return result;
}
