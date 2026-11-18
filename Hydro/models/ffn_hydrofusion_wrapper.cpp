#include "ffn_hydrofusion_wrapper.h"

#include "../experiments/gistohq_sligo/hydrofusion_synthetic_fixture.h"
#include "../experiments/gistohq_sligo/hydrofusion_dataset_assembly.h"
#include "../experiments/gistohq_sligo/hydrofusion_metadata_sidecar.h"
#include "../dataset/tensor_scaler.h"
#include "../evaluation/hydro_metrics.h"
#include "../evaluation/model_checkpoint.h"

#include "../../neuralnetworkwrapper.h"

#include <algorithm>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::vector<int> parseHiddenLayers(const std::string& csv) {
    std::vector<int> layers;
    std::stringstream ss(csv);
    std::string token;
    while (std::getline(ss, token, ',')) {
        std::size_t consumed = 0;
        const int value = std::stoi(token, &consumed);
        if (consumed != token.size() || value < 1) throw std::invalid_argument("Invalid HydroFusion hidden layer.");
        layers.push_back(value);
    }
    if (layers.empty()) throw std::invalid_argument("HydroFusion requires an explicit hidden layer.");
    return layers;
}
}  // namespace

HydroRunResult FFNHydroFusionWrapper::train(const HydroRunConfig& config) {
    using namespace hydrofusion;

    // This wrapper has no real-data loading path yet -- it always fits
    // the tiny synthetic fixture (hydrofusion_synthetic_fixture.h), never
    // config.hydro_package_path/csv_path. Silently substituting synthetic
    // data for a caller's requested real data source would be a dangerous
    // silent-failure mode (a job claiming to use a real package would
    // appear to succeed while training on nothing like the requested
    // data), so this is a hard, explicit, fail-closed guard, not a
    // best-effort warning. Configured real-data loading for this mode is
    // a separately-scoped, not-yet-implemented step -- see
    // CLAUDE_FIVE_METHOD_COMPATIBILITY_MATRIX_20261118.md's
    // implementation plan for why that step is deliberately deferred.
    if (config.use_hydro_package || config.use_csv_data) {
        throw std::runtime_error(
            "FFNHydroFusionWrapper: this mode is synthetic-fixture-only for now (HydroFusion Task 4 "
            "closure scope) -- it has no real-data loading path yet and must not silently substitute "
            "synthetic data for a configured hydro_package_path/csv_path. Set use_hydro_package=false "
            "and use_csv_data=false to run this mode's synthetic verification path explicitly.");
    }

    if (!config.hydrofusion_synthetic_verification) {
        throw std::runtime_error("FFNHydroFusionWrapper: explicit hydrofusion_synthetic_verification=true is required; no real-data path exists.");
    }
    if (config.normalization != "standardize" || config.optimizer != "adam") {
        throw std::runtime_error("FFNHydroFusionWrapper: prototype supports standardize normalization and adam only.");
    }
    TargetMode mode;
    if (config.hydrofusion_target_mode == "direct_log_q") {
        mode = TargetMode::DirectLogQ;
    } else if (config.hydrofusion_target_mode == "log_residual") {
        mode = TargetMode::LogResidual;
    } else {
        throw std::runtime_error(
            "FFNHydroFusionWrapper: hydrofusion_target_mode must be 'log_residual' or 'direct_log_q'.");
    }
    validateTargetMode(mode);

    // Reuses the split-cutoff fields every other mode already has
    // (train_end_x/validation_end_x/test_end_x). These must be set
    // explicitly in the job config rather than defaulted internally:
    // hydro_batch_runner.cpp's exportRun() call serializes the SAME
    // HydroRunConfig object this wrapper only sees as const&, so any
    // cutoff this wrapper chose internally without writing it back would
    // leave experiment_config.json showing -1/-1/-1 while the metadata
    // sidecar (sourced from result.hydrofusion_*, set below) reported the
    // real values -- a config/sidecar desync. Requiring the caller to set
    // all three up front means the exported config and the sidecar are
    // reading the same source of truth by construction.
    const bool allSet = config.train_end_x >= 0 && config.validation_end_x >= 0 && config.test_end_x >= 0;
    if (!allSet) {
        throw std::runtime_error(
            "FFNHydroFusionWrapper: train_end_x/validation_end_x/test_end_x must all be set explicitly "
            "in the job config (e.g. 150/220/280 for the synthetic fixture) -- this mode does not "
            "default them internally, since the exported experiment_config.json must agree with the "
            "HydroFusion metadata sidecar.");
    }
    const SplitCutoffs cutoffs{config.train_end_x, config.validation_end_x, config.test_end_x};

    const auto rows = buildSyntheticFixture();
    const PreparedData data = prepareHydroFusionData(rows, mode, cutoffs);

    TensorScaler inputScaler;
    inputScaler.fit(data.train.x, config.normalization);
    TensorScaler targetScaler;
    targetScaler.fit(data.train.y, config.normalization);

    torch::manual_seed(static_cast<uint64_t>(std::max(0, config.random_seed)));
    NeuralNetworkWrapper model;
    model.setHiddenLayers(parseHiddenLayers(config.hidden_layers_csv));
    model.setLags(std::vector<std::vector<int>>(featureOrderNames().size(), std::vector<int>{1}));
    model.initializeNetwork(1, config.activation);
    model.setTensorData(DataType::Train, inputScaler.transform(data.train.x), targetScaler.transform(data.train.y));
    model.setTensorData(DataType::Test, inputScaler.transform(data.validation.x),
                        targetScaler.transform(data.validation.y));
    std::vector<double> validationLosses;
    int bestEpoch = 0;
    const auto losses = model.train(config.epochs, config.batch_size, config.learning_rate,
                                   inputScaler.transform(data.validation.x),
                                   targetScaler.transform(data.validation.y),
                                   &validationLosses, &bestEpoch);
    if (losses.empty() || bestEpoch < 1 || static_cast<std::size_t>(bestEpoch) > losses.size()) {
        throw std::runtime_error("FFNHydroFusionWrapper: no validation-selected checkpoint.");
    }

    const torch::Tensor testInputsScaled = inputScaler.transform(data.test.x);
    model.setTensorData(DataType::Test, testInputsScaled, targetScaler.transform(data.test.y));
    const torch::Tensor predictedTargetSpace = targetScaler.inverseTransform(model.forward(DataType::Test)).detach();

    const auto checkpointPath = temporaryHydroCheckpointPath("hydrofusion_ffn_batch");
    model.saveModel(checkpointPath.string());
    const std::vector<std::uint8_t> checkpointBytes = readHydroCheckpoint(checkpointPath);
    std::filesystem::remove(checkpointPath);
    if (checkpointBytes.empty()) {
        throw std::runtime_error("FFNHydroFusionWrapper: saved checkpoint is empty.");
    }

    HydroRunResult result;
    result.success = true;
    result.best_epoch = bestEpoch;
    result.final_loss = losses.at(static_cast<std::size_t>(bestEpoch - 1));
    result.training_loss_history = losses;
    result.validation_loss_history = validationLosses;  // standardized transformed-target MSE, not discharge MSE

    result.model_checkpoint_format = "neuralnetworkwrapper-v1";
    result.model_checkpoint = checkpointBytes;
    result.input_scaler = inputScaler.exportState();
    result.target_scaler = targetScaler.exportState();
    // Real test-split TARGET-x identities (the time the observation/
    // prediction are AT), matching the already-verified integration test
    // -- see hydrofusion_metadata_sidecar.h's origin/target map for why
    // origin_x is ALSO preserved explicitly, just not as this column.
    result.x.assign(data.test.targetX.size(), 0.0);
    for (std::size_t i = 0; i < data.test.targetX.size(); ++i) {
        result.x[i] = static_cast<double>(data.test.targetX[i]);
    }
    // Public validation metric is physical discharge MSE; training histories
    // remain in the explicitly documented transformed/standardized target space.
    const auto validationTarget = targetScaler.inverseTransform(
        model.forwardTensor(inputScaler.transform(data.validation.x))).detach();
    const auto validationQ = reconstructHydroFusionColumn(mode, validationTarget, data.validation.originQ, data.qRef);
    double validationSse = 0.0;
    for (std::size_t i = 0; i < validationQ.size(); ++i) {
        const double error = validationQ[i] - data.validation.observedTargetQ[i];
        validationSse += error * error;
    }
    result.validation_mse = validationSse / validationQ.size();
    result.y_true = data.test.observedTargetQ;
    result.y_pred = reconstructHydroFusionColumn(mode, predictedTargetSpace, data.test.originQ, data.qRef);
    result.split.assign(data.test.targetX.size(), "test");
    populateHydroMetrics(result, result.y_true, result.y_pred);
    populateHydroPeakMetrics(result);

    // Threaded back through the unmodified runJob -> HydroRunResult ->
    // main() pipeline to hydro_batch_runner.cpp's post-exportRun sidecar
    // hook -- see hydro_run_types.h's doc comment on these fields.
    result.hydrofusion_target_mode = config.hydrofusion_target_mode;
    result.hydrofusion_q_ref = data.qRef;
    result.hydrofusion_train_end_x = data.cutoffs.train_end_x;
    result.hydrofusion_validation_end_x = data.cutoffs.validation_end_x;
    result.hydrofusion_test_end_x = data.cutoffs.test_end_x;
    result.hydrofusion_test_origin_x = data.test.originX;

    return result;
}
