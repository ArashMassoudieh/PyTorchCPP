#include "../evaluation/inference_runner.h"
#include "../evaluation/model_checkpoint.h"
#include "../evaluation/experiment_exporter.h"
#include "../evaluation/hydro_metrics.h"
#include "../dataset/lagged_tensor_builder.h"
#include "../dataset/csv_tensor_builder.h"
#include "../models/hydro_lstm_module.h"
#include "../../neuralnetworkwrapper.h"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>

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
    bool rejectedInvalidLag = false;
    try { (void)parseHydroLagSpecification("1,invalid;2", 2); }
    catch (const std::invalid_argument&) { rejectedInvalidLag = true; }
    assert(rejectedInvalidLag);
    bool rejectedExtraLagGroup = false;
    try { (void)parseHydroLagSpecification("1;2;3", 2); }
    catch (const std::invalid_argument&) { rejectedExtraLagGroup = true; }
    assert(rejectedExtraLagGroup);
    const std::filesystem::path csvPath = "/tmp/hydro_inference_input_test.csv";
    {
        std::ofstream csv(csvPath, std::ios::binary);
        csv << "time,target\r\n";
        for (int i = 0; i < 10; ++i) csv << '"' << i << "\",\"" << i * 2 << "\"\r\n";
    }
    HydroRunConfig csvConfig;
    csvConfig.csv_path = csvPath.string();
    csvConfig.csv_has_header = true;
    torch::Tensor csvInputs, csvTargets, csvPlot;
    loadHydroCsvTensors(csvConfig, csvInputs, csvTargets, csvPlot);
    assert(csvInputs.size(0) == 10 && csvInputs.size(1) == 1);
    assert(csvTargets[9].item<float>() == 18.0f);
    assert(parseHydroCsvRow("\"a,b\",\"c\"\"d\"") == std::vector<std::string>({"a,b", "c\"d"}));
    bool rejectedMalformedCsv = false;
    try { (void)parseHydroCsvRow("1\"2,3"); }
    catch (const std::runtime_error&) { rejectedMalformedCsv = true; }
    assert(rejectedMalformedCsv);
    std::filesystem::remove(csvPath);
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

    const std::filesystem::path integrationRoot = "/tmp/hydro_five_model_inference_test";
    std::filesystem::remove_all(integrationRoot);
    HydroRunResult feedForwardResult;
    feedForwardResult.success = true;
    feedForwardResult.x = {0.0, 1.0};
    feedForwardResult.y_true = {0.0, 0.0};
    feedForwardResult.y_pred = {0.0, 0.0};
    feedForwardResult.split = {"test", "test"};
    populateHydroMetrics(feedForwardResult, feedForwardResult.y_true, feedForwardResult.y_pred);
    populateHydroPeakMetrics(feedForwardResult);
    feedForwardResult.input_scaler = artifacts.scalers.at("ffn").input;
    feedForwardResult.target_scaler = artifacts.scalers.at("ffn").target;
    feedForwardResult.model_checkpoint_format = artifacts.models.at("ffn").format;
    feedForwardResult.model_checkpoint = artifacts.models.at("ffn").bytes;
    HydroRunResult recurrentResult = feedForwardResult;
    recurrentResult.input_scaler = artifacts.scalers.at("lstm").input;
    recurrentResult.target_scaler = artifacts.scalers.at("lstm").target;
    recurrentResult.model_checkpoint_format = artifacts.models.at("lstm").format;
    recurrentResult.model_checkpoint = artifacts.models.at("lstm").bytes;
    const std::map<std::string, HydroRunResult> fiveResults = {
        {"ffn", feedForwardResult}, {"ffn_pinn", feedForwardResult}, {"pinn", feedForwardResult},
        {"lstm", recurrentResult}, {"lstm_pinn", recurrentResult}};
    HydroExperimentExporter().exportRun(integrationRoot.string(), "run", artifacts.experiment.config, fiveResults);
    const auto reloaded = HydroArtifactLoader().loadForInference((integrationRoot / "run").string());
    assert(reloaded.models.size() == 5);
    for (const std::string approach : {"ffn", "ffn_pinn", "pinn"}) {
        assert(torch::allclose(HydroInferenceSession(reloaded, approach).predict(feedForwardInputs),
                               expectedFeedForward));
    }
    for (const std::string approach : {"lstm", "lstm_pinn"}) {
        assert(torch::allclose(HydroInferenceSession(reloaded, approach).predict(recurrentInputs),
                               expectedRecurrent));
    }
    std::filesystem::remove_all(integrationRoot);
    return 0;
}
