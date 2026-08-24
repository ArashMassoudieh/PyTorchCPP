#include "../evaluation/inference_runner.h"
#include "../evaluation/model_checkpoint.h"
#include "../dataset/lagged_tensor_builder.h"
#include "../models/hydro_lstm_module.h"
#include "../../neuralnetworkwrapper.h"

#include <cassert>
#include <filesystem>

namespace {
HydroScalerState identityScaler(const std::vector<int64_t>& shape, const std::size_t values) {
    return {"none", std::vector<double>(values, 0.0), std::vector<double>(values, 1.0), shape};
}
}

int main() {
    torch::manual_seed(7);
    HydroInferenceArtifacts artifacts;
    artifacts.experiment.config.hidden_layers_csv = "3";
    artifacts.experiment.config.activation = "tanh";
    artifacts.experiment.config.lstm_sequence_length = 2;
    const torch::Tensor feedForwardInputs = torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}});

    NeuralNetworkWrapper feedForward;
    feedForward.setHiddenLayers({3});
    feedForward.setLags({{1}, {1}});
    feedForward.initializeNetwork(1, "tanh");
    feedForward.setTensorData(DataType::Test, feedForwardInputs, torch::zeros({2, 1}));
    const torch::Tensor expectedFeedForward = feedForward.forward(DataType::Test).detach();
    const auto feedForwardPath = temporaryHydroCheckpointPath("hydro_inference_ffn_test");
    feedForward.saveModel(feedForwardPath.string());
    artifacts.models["ffn"] = {"models/ffn.pt", "neuralnetworkwrapper-v1", "", readHydroCheckpoint(feedForwardPath)};
    std::filesystem::remove(feedForwardPath);
    artifacts.scalers["ffn"] = {identityScaler({1, 2}, 2), identityScaler({1, 1}, 1)};
    const auto actualFeedForward = HydroInferenceRunner().predictFeedForward(artifacts, "ffn", feedForwardInputs);
    assert(torch::allclose(actualFeedForward, expectedFeedForward));
    HydroInferenceSession feedForwardSession(artifacts, "ffn");
    assert(torch::allclose(feedForwardSession.predict(feedForwardInputs), expectedFeedForward));
    const auto lagged = buildHydroLaggedTensor(
        torch::tensor({{1.0f, 10.0f}, {2.0f, 20.0f}, {3.0f, 30.0f}}), "1;2");
    assert(lagged.leading_rows == 2);
    assert(torch::allclose(lagged.inputs, torch::tensor({{3.0f, 2.0f, 30.0f, 10.0f}})));
    assert(torch::allclose(feedForwardSession.predict(feedForwardInputs), expectedFeedForward));

    const torch::Tensor recurrentInputs = torch::tensor(
        {{{1.0f, 2.0f}, {2.0f, 3.0f}}, {{3.0f, 4.0f}, {4.0f, 5.0f}}});
    HydroLSTM recurrent(2, 3, 1, 1);
    recurrent->eval();
    torch::NoGradGuard noGrad;
    const torch::Tensor expectedRecurrent = recurrent->forward(recurrentInputs).detach();
    const auto recurrentPath = temporaryHydroCheckpointPath("hydro_inference_lstm_test");
    torch::serialize::OutputArchive archive;
    recurrent->save(archive);
    archive.save_to(recurrentPath.string());
    artifacts.models["lstm"] = {"models/lstm.pt", "torch-module-v1", "", readHydroCheckpoint(recurrentPath)};
    std::filesystem::remove(recurrentPath);
    artifacts.scalers["lstm"] = {identityScaler({1, 1, 2}, 2), identityScaler({1, 1}, 1)};
    const auto actualRecurrent = HydroInferenceRunner().predictRecurrent(artifacts, "lstm", recurrentInputs);
    assert(torch::allclose(actualRecurrent, expectedRecurrent));
    HydroInferenceSession recurrentSession(artifacts, "lstm");
    assert(torch::allclose(recurrentSession.predict(recurrentInputs), expectedRecurrent));
    assert(torch::allclose(recurrentSession.predict(recurrentInputs), expectedRecurrent));
    const torch::Tensor recurrentSeries = torch::tensor(
        {{1.0f, 2.0f}, {2.0f, 3.0f}, {3.0f, 4.0f}, {4.0f, 5.0f}});
    const auto seriesPrediction = recurrentSession.predictSeries(recurrentSeries);
    assert(seriesPrediction.size(0) == 3);
    assert(torch::allclose(seriesPrediction.slice(0, 0, 1), expectedRecurrent.slice(0, 0, 1)));

    bool rejectedWidth = false;
    try { (void)HydroInferenceRunner().predictFeedForward(artifacts, "ffn", torch::zeros({2, 3})); }
    catch (const std::invalid_argument&) { rejectedWidth = true; }
    assert(rejectedWidth);

    const auto checkpointBytes = artifacts.models.at("ffn").bytes;
    artifacts.models.at("ffn").bytes.clear();
    bool rejectedEmptyCheckpoint = false;
    try { (void)HydroInferenceRunner().predictFeedForward(artifacts, "ffn", feedForwardInputs); }
    catch (const std::runtime_error&) { rejectedEmptyCheckpoint = true; }
    assert(rejectedEmptyCheckpoint);
    artifacts.models.at("ffn").bytes = checkpointBytes;
    return 0;
}
