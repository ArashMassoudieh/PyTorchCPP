#include "lstm_wrapper.h"

#include "neuralnetworkwrapper.h"

#include <cmath>

HydroRunResult LSTMWrapper::train(const HydroRunConfig& config) {
    HydroRunResult result;

    // Temporary backend: uses NeuralNetworkWrapper FFN core while LSTM module is scaffolded.
    NeuralNetworkWrapper model;
    model.setHiddenLayers({32, 24});
    model.setLags({{1}});
    model.initializeNetwork(1, "relu");

    torch::Tensor t = torch::linspace(0.0, 6.0, 240, torch::kFloat32).unsqueeze(1);
    torch::Tensor y = 0.7 * torch::sin(1.5 * t) + 0.3 * torch::cos(0.5 * t);

    const int64_t nTrain = 192;
    torch::Tensor tTrain = t.slice(0, 0, nTrain);
    torch::Tensor yTrain = y.slice(0, 0, nTrain);
    torch::Tensor tTest = t.slice(0, nTrain, t.size(0));
    torch::Tensor yTest = y.slice(0, nTrain, y.size(0));

    model.setTensorData(DataType::Train, tTrain, yTrain);
    model.setTensorData(DataType::Test, tTest, yTest);

    std::vector<double> losses = model.train(config.epochs, config.batch_size, config.learning_rate);
    if (losses.empty() || !std::isfinite(losses.back())) {
        throw std::runtime_error("LSTM-baseline training produced empty/non-finite loss history.");
    }
    result.final_loss = losses.back();

    torch::Tensor pred = model.forward(DataType::Test);
    if (!pred.defined() || pred.size(0) != yTest.size(0) || !pred.isfinite().all().item<bool>()) {
        throw std::runtime_error("LSTM-baseline prediction failed or produced non-finite values.");
    }

    result.success = true;
    result.message = "LSTM-like run completed (temporary FFN backend).";
    return result;
}
