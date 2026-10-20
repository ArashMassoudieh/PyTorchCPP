#include "lstm_regional_pretrain.h"
#include "hydro_lstm_module.h"
#include "../dataset/tensor_scaler.h"
#include "../dataset/multi_basin_sampling.h"
#include "../dataset/lstm_sequence_builder.h"
#include "../dataset/hydro_tensor_builder.h"
#include "../dataset/regional_window_filter.h"
#include "../evaluation/model_checkpoint.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace {

std::vector<int> parseHiddenLayers(const std::string& csv) {
    std::vector<int> layers;
    std::stringstream ss(csv);
    std::string token;
    while (std::getline(ss, token, ',')) {
        try {
            const int value = std::stoi(token);
            if (value > 0) layers.push_back(value);
        } catch (...) {}
    }
    if (layers.empty()) layers = {32};
    return layers;
}

std::vector<std::string> readBasinListFile(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Unable to open regional basin list file: " + path);
    std::vector<std::string> paths;
    std::string line;
    while (std::getline(input, line)) {
        const auto begin = line.find_first_not_of(" \t\r\n");
        if (begin == std::string::npos) continue;
        const auto end = line.find_last_not_of(" \t\r\n");
        paths.push_back(line.substr(begin, end - begin + 1));
    }
    if (paths.empty()) throw std::runtime_error("Regional basin list file has no entries: " + path);
    return paths;
}

std::string readManifestStudyStart(const std::string& packagePath) {
    const std::filesystem::path manifestPath = std::filesystem::path(packagePath) / "manifest.json";
    std::ifstream input(manifestPath);
    if (!input) throw std::runtime_error("Unable to open manifest.json for regional window filtering: " + manifestPath.string());
    const std::string json((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    static const std::regex pattern(R"json("study_start"\s*:\s*"([^"]+)")json");
    std::smatch match;
    if (!std::regex_search(json, match, pattern)) {
        throw std::runtime_error("manifest.json has no study_start field: " + manifestPath.string());
    }
    return match[1].str();
}

struct ScaledBasin {
    std::string staid;
    torch::Tensor xSeq;
    torch::Tensor ySeq;
};

// Builds ONE target scaler from the training pool only - basin-balanced
// (each training basin's own independently-fit scaler contributes equally
// to the average, regardless of record length, so a long-record basin
// cannot dominate the pooled statistics the same way balancedShuffledEpochIndices
// already prevents it from dominating the training draw). This single
// scaler is then applied to EVERY basin's target, training and validation
// alike.
//
// Why: fitting a validation basin's own target scaler from that basin's
// own complete target series (the prior, buggy behavior) let normalization
// statistics derived from the very data being evaluated leak into the
// validation-loss computation that selects the checkpoint - the
// model-selection metric was partly informed by the thing it was supposed
// to be blind to. Deriving every basin's target scale from the training
// pool alone removes that leakage path entirely: validation basins
// contribute nothing to this scaler's fit. Input (x) scaling is NOT
// changed by this fix - it remains per-basin (see scaleBasins below) since
// input features are not the evaluation target and so are not implicated
// by this leakage concern.
} // namespace

TensorScaler fitPooledTargetScalerFromTrainingPool(const std::vector<RegionalBasinInput>& trainingBasins,
                                                   const std::string& normalization) {
    std::vector<HydroScalerState> states;
    states.reserve(trainingBasins.size());
    for (const auto& basin : trainingBasins) {
        TensorScaler perBasin;
        perBasin.fit(basin.ySeq, normalization);
        states.push_back(perBasin.exportState());
    }
    for (const auto& state : states) {
        if (state.method != states.front().method || state.shape != states.front().shape) {
            throw std::runtime_error(
                "Training basins disagree on the target scaler's effective method or shape (one basin's "
                "target data likely fell outside log_standardize's domain) - cannot average incompatible "
                "per-basin scalers into one pooled target scaler.");
        }
    }
    const std::string method = states.front().method;
    const std::size_t basinCount = states.size();
    const std::size_t n = states.front().offset.size();

    HydroScalerState pooled;
    pooled.method = method;
    pooled.shape = states.front().shape;
    pooled.offset.assign(n, 0.0);
    pooled.scale.assign(n, 0.0);

    // Mean: always a simple equal-weighted average across basins - correct
    // for every method, since offset is a location parameter in all of
    // them (mean for the standardize family, min for minmax, 0 for none).
    for (const auto& state : states) {
        for (std::size_t i = 0; i < n; ++i) pooled.offset[i] += state.offset[i];
    }
    for (std::size_t i = 0; i < n; ++i) pooled.offset[i] /= static_cast<double>(basinCount);

    if (method == "standardize" || method == "log_standardize" || method == "asinh_standardize") {
        // Equally-weighted MIXTURE variance (law of total variance) -
        // NOT a naive average of each basin's own std dev, which was this
        // function's original (wrong) implementation. Averaging sigma_i
        // directly ignores the between-basin spread of means entirely,
        // understating the pooled scale whenever basin distributions
        // differ - confirmed independently before this fix: on a 3-basin
        // synthetic check with deliberately different means/stds, naive
        // sigma-averaging gave 0.740 against a true pooled sigma of 1.941
        // (verified two ways: this closed-form formula and a direct
        // concatenation of equal-size basins agree to full floating
        // precision) - a ~2.6x understatement, not a rounding difference.
        // For B equally-weighted basins, each with mean mu_i and variance
        // sigma_i^2 in the SAME transformed space TensorScaler::fit
        // already computes (log1p space for log_standardize):
        //   mu = (1/B) * sum(mu_i)
        //   sigma^2 = (1/B) * sum(sigma_i^2 + mu_i^2) - mu^2
        // See testPooledTargetScalerMatchesConcatenatedReference for the
        // independent verification this relies on.
        std::vector<double> sumOfSquarePlusMeanSquare(n, 0.0);
        for (const auto& state : states) {
            for (std::size_t i = 0; i < n; ++i) {
                sumOfSquarePlusMeanSquare[i] += state.scale[i] * state.scale[i] + state.offset[i] * state.offset[i];
            }
        }
        for (std::size_t i = 0; i < n; ++i) {
            const double meanOfSquares = sumOfSquarePlusMeanSquare[i] / static_cast<double>(basinCount);
            const double variance = meanOfSquares - pooled.offset[i] * pooled.offset[i];
            if (!(variance > 0.0)) {
                throw std::runtime_error(
                    "Pooled target variance computed as non-positive - training basins' target "
                    "distributions are degenerate or numerically inconsistent.");
            }
            pooled.scale[i] = std::sqrt(variance);
        }
    } else {
        // minmax/none: not exercised by this program's established target-
        // normalization convention (always log_standardize, falling back
        // to plain standardize only if the data's own domain forces it -
        // see TensorScaler::fit). Mean/variance mixture math does not
        // apply to these methods' offset/scale semantics (min/range, not
        // mean/std), so this falls back to a simple average - approximate,
        // not derived from a mixture-distribution identity like the
        // branch above, and not expected to matter for this target.
        for (const auto& state : states) {
            for (std::size_t i = 0; i < n; ++i) pooled.scale[i] += state.scale[i];
        }
        for (std::size_t i = 0; i < n; ++i) pooled.scale[i] /= static_cast<double>(basinCount);
    }

    TensorScaler pooledScaler;
    pooledScaler.importState(pooled);
    return pooledScaler;
}

namespace {

std::vector<ScaledBasin> scaleBasins(const std::vector<RegionalBasinInput>& basins, const std::string& normalization,
                                     const TensorScaler& pooledTargetScaler) {
    // log_standardize's transform (log1p) is domain-restricted to values
    // > -1 - TensorScaler::fit() checks this against the data it is FIT
    // from, but transform() applies log1p unconditionally regardless of
    // what is being transformed. A scaler fit purely from the training
    // pool's target domain does not know anything about a DIFFERENT
    // basin's target domain - if applied to a basin whose target dips to
    // or below -1, transform() silently returns NaN instead of throwing.
    // Real basin-normalized runoff depth is physically non-negative, so
    // this should not occur with genuine data, but the code must not
    // silently depend on that being true - check it explicitly and fail
    // loudly, caught directly by this module's own test construction (a
    // validation basin whose synthetic target dipped to -10 silently
    // produced NaN validation loss for every epoch before this guard was
    // added, aborting training entirely).
    const std::string targetMethod = pooledTargetScaler.exportState().method;
    for (const auto& basin : basins) {
        if (targetMethod == "log_standardize" && (basin.ySeq <= -1.0).any().item<bool>()) {
            throw std::runtime_error(
                "Basin " + basin.staid + "'s target data contains a value <= -1, outside the domain of the "
                "training pool's log_standardize target scaler - basin-normalized runoff depth should never "
                "be this negative; check this basin's data before trusting its validation loss.");
        }
    }
    std::vector<ScaledBasin> scaled;
    scaled.reserve(basins.size());
    for (const auto& basin : basins) {
        // Input (x): per-basin, fit from that basin's own data only - not
        // implicated by the target-leakage fix (see
        // fitPooledTargetScalerFromTrainingPool's doc comment).
        TensorScaler inputScaler;
        inputScaler.fit(basin.xSeq, normalization);
        // Target (y): the SAME training-pool-only pooled scaler for every
        // basin - never fit from a validation basin's own data.
        scaled.push_back({basin.staid, inputScaler.transform(basin.xSeq), pooledTargetScaler.transform(basin.ySeq)});
    }
    return scaled;
}

void validateSharedShape(const std::vector<RegionalBasinInput>& basins, int64_t inputDim, int64_t sequenceLength) {
    for (const auto& basin : basins) {
        if (!basin.xSeq.defined() || !basin.ySeq.defined() || basin.xSeq.dim() != 3 || basin.ySeq.dim() != 2 ||
            basin.xSeq.size(0) != basin.ySeq.size(0) || basin.xSeq.size(0) < 1) {
            throw std::invalid_argument("Basin " + basin.staid + " has malformed sequence tensors.");
        }
        if (basin.xSeq.size(1) != sequenceLength || basin.xSeq.size(2) != inputDim || basin.ySeq.size(1) != 1) {
            throw std::invalid_argument(
                "Basin " + basin.staid + " does not share the pool's sequence length/feature width - "
                "every basin must be ingested through the identical feature contract.");
        }
    }
}

} // namespace

RegionalBasinInput loadRegionalBasinInput(const std::string& packagePath, const HydroRunConfig& templateConfig) {
    HydroRunConfig basinConfig = templateConfig;
    basinConfig.use_hydro_package = true;
    basinConfig.hydro_package_path = packagePath;
    basinConfig.use_csv_data = false;
    basinConfig.use_hydro_forecast_feature = false;

    torch::Tensor x, y, plotX;
    if (!loadHydroPackageTensors(basinConfig, x, y, plotX)) {
        throw std::runtime_error("Unable to load regional basin package: " + packagePath);
    }

    // Exact calendar window enforcement (e.g. the pilot manifest's declared
    // 2015-01-01 through 2022-12-31) - loadHydroPackageTensors has no
    // date-range concept of its own and otherwise loads every row the
    // package contains, which for a CAMELSH-ingested package spans
    // 1980 through the 2023-10-15 seal, far more than any declared
    // pretraining window. See regional_window_filter.h and
    // HydroRunConfig::regional_window_start's doc comment for the gap this
    // closes - confirmed as a real violation, not theoretical, before this
    // fix (a first pilot attempt silently used the full available range).
    if (!templateConfig.regional_window_start.empty() || !templateConfig.regional_window_end.empty()) {
        if (templateConfig.regional_window_start.empty() || templateConfig.regional_window_end.empty()) {
            throw std::runtime_error("regional_window_start and regional_window_end must both be set, or both left empty.");
        }
        const std::string manifestStudyStart = readManifestStudyStart(packagePath);
        const auto window = elapsedHoursWindowFromStudyStart(
            manifestStudyStart, templateConfig.regional_window_start, templateConfig.regional_window_end);
        const auto flatPlotX = plotX.reshape({-1});
        const auto mask = (flatPlotX >= window.first).logical_and(flatPlotX < window.second);
        const auto indices = mask.nonzero().squeeze(1);
        if (indices.numel() < 4) {
            throw std::runtime_error(
                "Basin " + packagePath + " has too few rows (" + std::to_string(indices.numel()) +
                ") inside the requested regional window [" + templateConfig.regional_window_start + ", " +
                templateConfig.regional_window_end + ").");
        }
        x = x.index_select(0, indices);
        y = y.index_select(0, indices);
        plotX = plotX.index_select(0, indices);
    }

    const int sequenceLength = std::max(2, templateConfig.lstm_sequence_length);
    // enforceRegularTime=true: identical gap-aware windowing as single-basin
    // LSTM training (config.use_hydro_package controls this flag there too).
    const auto seq = makeLstmSequences(x, y, plotX, sequenceLength, true);

    RegionalBasinInput input;
    input.staid = packagePath;
    input.xSeq = seq.xSeq;
    input.ySeq = seq.ySeq;
    return input;
}

RegionalPretrainOutput trainLstmRegionalPool(
    const std::vector<RegionalBasinInput>& trainingBasins,
    const std::vector<RegionalBasinInput>& validationBasins,
    const RegionalPretrainConfig& config) {
    if (trainingBasins.empty()) throw std::invalid_argument("Regional pretraining requires at least one training basin.");
    if (validationBasins.empty()) throw std::invalid_argument("Regional pretraining requires at least one validation basin.");
    if (config.epochs < 1) throw std::invalid_argument("Regional pretraining requires epochs >= 1.");
    if (config.batch_size < 1) throw std::invalid_argument("Regional pretraining requires batch_size >= 1.");

    torch::manual_seed(static_cast<uint64_t>(std::max<int64_t>(0, config.random_seed)));

    const int64_t inputDim = trainingBasins.front().xSeq.size(2);
    const int64_t sequenceLength = trainingBasins.front().xSeq.size(1);
    validateSharedShape(trainingBasins, inputDim, sequenceLength);
    validateSharedShape(validationBasins, inputDim, sequenceLength);

    const TensorScaler pooledTargetScaler = fitPooledTargetScalerFromTrainingPool(trainingBasins, config.normalization);
    const std::vector<ScaledBasin> scaledTraining = scaleBasins(trainingBasins, config.normalization, pooledTargetScaler);
    const std::vector<ScaledBasin> scaledValidation = scaleBasins(validationBasins, config.normalization, pooledTargetScaler);

    std::vector<int64_t> trainingSizes;
    trainingSizes.reserve(scaledTraining.size());
    for (const auto& b : scaledTraining) trainingSizes.push_back(b.xSeq.size(0));
    const int64_t perBasinCount = config.per_basin_sequence_count > 0
        ? config.per_basin_sequence_count
        : *std::min_element(trainingSizes.begin(), trainingSizes.end());

    const std::vector<int> hiddenLayers = config.hidden_layers.empty() ? std::vector<int>{32} : config.hidden_layers;
    const int64_t hiddenDim = hiddenLayers.front();
    const int64_t numLayers = static_cast<int64_t>(std::max<std::size_t>(1, hiddenLayers.size()));
    HydroLSTM model(inputDim, hiddenDim, 1, numLayers, config.lstm_dropout);
    torch::optim::Adam optimizer(model->parameters(),
                                 torch::optim::AdamOptions(config.learning_rate).weight_decay(config.weight_decay));

    // Deterministic given random_seed - drives per-epoch basin-balanced
    // sampling only; the LSTM's own weight init is seeded above via
    // torch::manual_seed, matching the single-basin trainer's convention.
    std::mt19937_64 samplingRng(static_cast<uint64_t>(std::max<int64_t>(0, config.random_seed)));

    RegionalPretrainOutput out;
    double bestValidation = std::numeric_limits<double>::infinity();
    int bestEpoch = 0;
    std::vector<torch::Tensor> bestParameters;

    for (int epoch = 1; epoch <= config.epochs; ++epoch) {
        model->train();
        const auto epochIndices = balancedShuffledEpochIndices(trainingSizes, perBasinCount, samplingRng);
        double epochLossSum = 0.0;
        int64_t epochLossCount = 0;
        for (std::size_t start = 0; start < epochIndices.size(); start += static_cast<std::size_t>(config.batch_size)) {
            const std::size_t end = std::min(epochIndices.size(), start + static_cast<std::size_t>(config.batch_size));
            std::vector<torch::Tensor> xBatch, yBatch;
            xBatch.reserve(end - start);
            yBatch.reserve(end - start);
            for (std::size_t i = start; i < end; ++i) {
                const auto& pair = epochIndices[i];
                const auto& basin = scaledTraining[static_cast<std::size_t>(pair.first)];
                xBatch.push_back(basin.xSeq[pair.second].unsqueeze(0));
                yBatch.push_back(basin.ySeq[pair.second].unsqueeze(0));
            }
            const auto xb = torch::cat(xBatch, 0);
            const auto yb = torch::cat(yBatch, 0);
            optimizer.zero_grad();
            const auto pred = model->forward(xb);
            const auto loss = torch::mse_loss(pred, yb);
            loss.backward();
            optimizer.step();
            epochLossSum += loss.item<double>() * static_cast<double>(end - start);
            epochLossCount += static_cast<int64_t>(end - start);
        }
        const double trainLoss = epochLossCount > 0
            ? epochLossSum / static_cast<double>(epochLossCount)
            : std::numeric_limits<double>::quiet_NaN();
        out.training_loss_history.push_back(trainLoss);

        // Pooled validation loss: full available data from every held-out
        // basin (not basin-balanced sampling - that only governs the
        // training draw), each already normalized by its OWN independently
        // fit scaler, simple mean across basins (equal basin weight, same
        // spirit as the training draw's per-basin balance).
        model->eval();
        double validationLossSum = 0.0;
        int64_t validationBasinCount = 0;
        {
            torch::NoGradGuard noGrad;
            for (const auto& basin : scaledValidation) {
                const auto pred = model->forward(basin.xSeq);
                const auto loss = torch::mse_loss(pred, basin.ySeq);
                validationLossSum += loss.item<double>();
                ++validationBasinCount;
            }
        }
        const double validationLoss = validationBasinCount > 0
            ? validationLossSum / static_cast<double>(validationBasinCount)
            : std::numeric_limits<double>::quiet_NaN();
        out.validation_loss_history.push_back(validationLoss);

        if (validationLoss < bestValidation) {
            bestValidation = validationLoss;
            bestEpoch = epoch;
            bestParameters.clear();
            torch::NoGradGuard noGrad;
            for (const auto& p : model->parameters()) bestParameters.push_back(p.clone());
        }
    }

    if (bestParameters.empty()) throw std::runtime_error("Regional pretraining did not produce a validation-selected checkpoint.");
    {
        torch::NoGradGuard noGrad;
        auto parameters = model->parameters();
        if (parameters.size() != bestParameters.size()) {
            throw std::runtime_error("Regional pretraining checkpoint parameter count changed mid-run.");
        }
        for (std::size_t i = 0; i < parameters.size(); ++i) parameters[i].copy_(bestParameters[i]);
    }

    {
        const auto checkpoint = temporaryHydroCheckpointPath("hydro_lstm_regional_pretrain");
        torch::serialize::OutputArchive archive;
        model->save(archive);
        archive.save_to(checkpoint.string());
        out.model_checkpoint = readHydroCheckpoint(checkpoint);
        out.model_checkpoint_format = "torch-module-v1";
        std::filesystem::remove(checkpoint);
    }

    out.success = true;
    out.best_epoch = bestEpoch;
    out.best_validation_loss = bestValidation;
    return out;
}

HydroRunResult trainLstmRegionalPretrainFromConfig(const HydroRunConfig& config) {
    if (config.regional_training_basin_list_path.empty() || config.regional_validation_basin_list_path.empty()) {
        throw std::runtime_error(
            "lstm_regional_pretrain requires regional_training_basin_list_path and "
            "regional_validation_basin_list_path.");
    }
    if (config.optimizer != "adam") {
        throw std::runtime_error("lstm_regional_pretrain currently only supports optimizer=adam.");
    }

    const auto trainingPaths = readBasinListFile(config.regional_training_basin_list_path);
    const auto validationPaths = readBasinListFile(config.regional_validation_basin_list_path);

    std::vector<RegionalBasinInput> trainingBasins;
    trainingBasins.reserve(trainingPaths.size());
    for (const auto& path : trainingPaths) trainingBasins.push_back(loadRegionalBasinInput(path, config));

    std::vector<RegionalBasinInput> validationBasins;
    validationBasins.reserve(validationPaths.size());
    for (const auto& path : validationPaths) validationBasins.push_back(loadRegionalBasinInput(path, config));

    RegionalPretrainConfig poolConfig;
    poolConfig.epochs = config.epochs;
    poolConfig.batch_size = config.batch_size;
    poolConfig.learning_rate = config.learning_rate;
    poolConfig.weight_decay = config.weight_decay;
    poolConfig.normalization = config.normalization;
    poolConfig.hidden_layers = parseHiddenLayers(config.hidden_layers_csv);
    poolConfig.activation = config.activation;
    poolConfig.lstm_dropout = config.lstm_dropout;
    poolConfig.random_seed = config.random_seed;
    poolConfig.per_basin_sequence_count = config.regional_per_basin_sequence_count;

    const RegionalPretrainOutput result = trainLstmRegionalPool(trainingBasins, validationBasins, poolConfig);

    HydroRunResult hydroResult;
    hydroResult.success = result.success;
    hydroResult.message = result.message;
    hydroResult.model_checkpoint = result.model_checkpoint;
    hydroResult.model_checkpoint_format = result.model_checkpoint_format;
    hydroResult.best_epoch = result.best_epoch;
    hydroResult.validation_mse = result.best_validation_loss;
    hydroResult.training_loss_history = result.training_loss_history;
    hydroResult.validation_loss_history = result.validation_loss_history;
    hydroResult.final_loss = result.training_loss_history.empty()
        ? std::numeric_limits<double>::quiet_NaN()
        : result.training_loss_history.back();
    return hydroResult;
}
