#include "artifact_loader.h"

#include "../dataset/hydro_checksum.h"
#include "hydro_metrics.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace {
bool readCsvLine(std::istream& input, std::string& line) {
    if (!std::getline(input, line)) return false;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    return true;
}

std::vector<std::string> splitCsv(const std::string& line) {
    std::vector<std::string> values;
    std::string value;
    bool quoted = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char character = line[i];
        if (character == '"') {
            if (quoted && i + 1 < line.size() && line[i + 1] == '"') {
                value.push_back('"');
                ++i;
            } else {
                quoted = !quoted;
            }
        } else if (character == ',' && !quoted) {
            values.push_back(value);
            value.clear();
        } else {
            value.push_back(character);
        }
    }
    if (quoted) throw std::runtime_error("Unterminated quoted CSV field.");
    values.push_back(value);
    return values;
}

bool safeRelativePath(const std::string& value) {
    const std::filesystem::path path(value);
    if (value.empty() || path.is_absolute()) return false;
    for (const auto& component : path) if (component == "..") return false;
    return true;
}

std::vector<int64_t> parseShape(const std::string& value, const std::size_t row) {
    std::vector<int64_t> shape;
    std::stringstream stream(value);
    std::string dimension;
    while (std::getline(stream, dimension, ';')) {
        try {
            std::size_t consumed = 0;
            const auto parsed = std::stoll(dimension, &consumed);
            if (consumed != dimension.size() || parsed <= 0) throw std::invalid_argument("invalid dimension");
            shape.push_back(parsed);
        } catch (...) {
            throw std::runtime_error("Invalid scaler shape in row " + std::to_string(row) + ".");
        }
    }
    if (shape.empty()) throw std::runtime_error("Scaler shape is empty in row " + std::to_string(row) + ".");
    return shape;
}

double parseFiniteDouble(const std::string& value, const char* field, const std::size_t row) {
    try {
        std::size_t consumed = 0;
        const double parsed = std::stod(value, &consumed);
        if (consumed != value.size() || !std::isfinite(parsed)) throw std::invalid_argument("invalid number");
        return parsed;
    } catch (...) {
        throw std::runtime_error(std::string("Invalid scaler ") + field + " in row " + std::to_string(row) + ".");
    }
}

double parseMetricDouble(const std::string& value, const std::size_t row) {
    if (value == "nan" || value == "NaN") return std::numeric_limits<double>::quiet_NaN();
    return parseFiniteDouble(value, "metric", row);
}

bool equivalentMetric(const double left, const double right) {
    if (std::isnan(left) && std::isnan(right)) return true;
    if (!std::isfinite(left) || !std::isfinite(right)) return false;
    return std::abs(left - right) <= 1.0e-10 * std::max({1.0, std::abs(left), std::abs(right)});
}

std::string readArtifactText(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Experiment is missing " + path.filename().string() + ".");
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

std::string jsonString(const std::string& json, const std::string& key, const char* artifact) {
    std::smatch match;
    const std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*\\\"([^\\\"]*)\\\"");
    if (!std::regex_search(json, match, pattern)) {
        throw std::runtime_error(std::string(artifact) + " is missing string field: " + key);
    }
    return match[1].str();
}

std::int64_t jsonPositiveInteger(const std::string& json, const std::string& key, const char* artifact) {
    std::smatch match;
    const std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*([0-9]+)");
    if (!std::regex_search(json, match, pattern)) {
        throw std::runtime_error(std::string(artifact) + " is missing integer field: " + key);
    }
    try {
        const auto value = std::stoll(match[1].str());
        if (value <= 0) throw std::invalid_argument("non-positive");
        return value;
    } catch (...) {
        throw std::runtime_error(std::string(artifact) + " has an invalid integer field: " + key);
    }
}
}

std::map<std::string, HydroModelArtifact> HydroArtifactLoader::loadModels(
    const std::string& experimentDirectory) const {
    const std::filesystem::path root(experimentDirectory);
    const auto canonicalRoot = std::filesystem::weakly_canonical(root);
    std::ifstream manifest(root / "models.csv");
    if (!manifest) throw std::runtime_error("Experiment is missing models.csv.");
    std::string line;
    if (!readCsvLine(manifest, line) || line != "approach,file,format,size_bytes,sha256") {
        throw std::runtime_error("models.csv has an incompatible header.");
    }
    std::map<std::string, HydroModelArtifact> models;
    std::size_t row = 1;
    while (readCsvLine(manifest, line)) {
        ++row;
        if (line.empty()) continue;
        const auto fields = splitCsv(line);
        if (fields.size() != 5 || fields[0].empty()) throw std::runtime_error("Invalid models.csv row " + std::to_string(row) + ".");
        if (!safeRelativePath(fields[1])) throw std::runtime_error("Model artifact path escapes the experiment directory.");
        if (fields[2] != "neuralnetworkwrapper-v1" && fields[2] != "torch-module-v1") {
            throw std::runtime_error("Unsupported model checkpoint format: " + fields[2]);
        }
        std::size_t expectedSize = 0;
        try {
            std::size_t consumed = 0;
            expectedSize = std::stoull(fields[3], &consumed);
            if (consumed != fields[3].size()) throw std::invalid_argument("trailing size");
        } catch (...) { throw std::runtime_error("Invalid model size in row " + std::to_string(row) + "."); }
        const auto path = root / fields[1];
        const auto canonicalPath = std::filesystem::weakly_canonical(path);
        const auto relativeToRoot = canonicalPath.lexically_relative(canonicalRoot);
        if (relativeToRoot.empty() || *relativeToRoot.begin() == "..") {
            throw std::runtime_error("Model artifact resolves outside the experiment directory.");
        }
        if (!std::filesystem::is_regular_file(canonicalPath) || std::filesystem::file_size(canonicalPath) != expectedSize) {
            throw std::runtime_error("Model artifact size or existence check failed: " + fields[1]);
        }
        if (sha256File(canonicalPath.string()) != fields[4]) throw std::runtime_error("Model artifact SHA-256 mismatch: " + fields[1]);
        std::ifstream input(canonicalPath, std::ios::binary);
        HydroModelArtifact artifact;
        artifact.relative_path = fields[1];
        artifact.format = fields[2];
        artifact.sha256 = fields[4];
        artifact.bytes.assign(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
        if (!models.emplace(fields[0], std::move(artifact)).second) throw std::runtime_error("Duplicate approach in models.csv: " + fields[0]);
    }
    if (models.empty()) throw std::runtime_error("models.csv contains no model artifacts.");
    return models;
}

std::map<std::string, HydroArtifactScalers> HydroArtifactLoader::loadScalers(
    const std::string& experimentDirectory) const {
    std::ifstream manifest(std::filesystem::path(experimentDirectory) / "scalers.csv");
    if (!manifest) throw std::runtime_error("Experiment is missing scalers.csv.");
    std::string line;
    if (!readCsvLine(manifest, line) || line != "approach,kind,index,method,shape,offset,scale") {
        throw std::runtime_error("scalers.csv has an incompatible header.");
    }

    std::map<std::string, HydroArtifactScalers> scalers;
    std::map<std::string, std::map<std::string, std::size_t>> nextIndices;
    std::size_t row = 1;
    while (readCsvLine(manifest, line)) {
        ++row;
        if (line.empty()) continue;
        const auto fields = splitCsv(line);
        if (fields.size() != 7 || fields[0].empty() || (fields[1] != "input" && fields[1] != "target")) {
            throw std::runtime_error("Invalid scalers.csv row " + std::to_string(row) + ".");
        }
        if (fields[3] != "none" && fields[3] != "standardize" && fields[3] != "minmax") {
            throw std::runtime_error("Unsupported scaler method in row " + std::to_string(row) + ": " + fields[3]);
        }
        std::size_t index = 0;
        try {
            std::size_t consumed = 0;
            const auto parsed = std::stoull(fields[2], &consumed);
            if (consumed != fields[2].size() || parsed > std::numeric_limits<std::size_t>::max()) {
                throw std::invalid_argument("invalid index");
            }
            index = static_cast<std::size_t>(parsed);
        } catch (...) {
            throw std::runtime_error("Invalid scaler index in row " + std::to_string(row) + ".");
        }
        auto& expected = nextIndices[fields[0]][fields[1]];
        if (index != expected) throw std::runtime_error("Scaler indices are not contiguous in row " + std::to_string(row) + ".");

        auto& pair = scalers[fields[0]];
        auto& state = fields[1] == "input" ? pair.input : pair.target;
        const auto shape = parseShape(fields[4], row);
        if (!state.offset.empty() && (state.method != fields[3] || state.shape != shape)) {
            throw std::runtime_error("Inconsistent scaler metadata in row " + std::to_string(row) + ".");
        }
        state.method = fields[3];
        state.shape = shape;
        state.offset.push_back(parseFiniteDouble(fields[5], "offset", row));
        const double scale = parseFiniteDouble(fields[6], "scale", row);
        if (scale == 0.0) throw std::runtime_error("Scaler scale is zero in row " + std::to_string(row) + ".");
        state.scale.push_back(scale);
        ++expected;
    }
    if (scalers.empty()) throw std::runtime_error("scalers.csv contains no scaler states.");
    for (const auto& entry : scalers) {
        const auto validateState = [&](const HydroScalerState& state, const char* kind) {
            if (state.offset.empty()) {
                throw std::runtime_error("Missing " + std::string(kind) + " scaler for approach: " + entry.first);
            }
            std::size_t expectedValues = 1;
            for (const auto dimension : state.shape) {
                if (expectedValues > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(dimension)) {
                    throw std::runtime_error("Scaler shape is too large for approach: " + entry.first);
                }
                expectedValues *= static_cast<std::size_t>(dimension);
            }
            if (expectedValues != state.offset.size()) {
                throw std::runtime_error("Scaler shape does not match its values for approach: " + entry.first);
            }
        };
        validateState(entry.second.input, "input");
        validateState(entry.second.target, "target");
    }
    return scalers;
}

std::map<std::string, HydroRunResult> HydroArtifactLoader::loadPredictions(
    const std::string& experimentDirectory) const {
    std::ifstream input(std::filesystem::path(experimentDirectory) / "predictions.csv");
    if (!input) throw std::runtime_error("Experiment is missing predictions.csv.");
    std::string line;
    if (!readCsvLine(input, line) || line != "approach,index,split,x,observed,predicted,residual") {
        throw std::runtime_error("predictions.csv has an incompatible header.");
    }

    std::map<std::string, HydroRunResult> results;
    std::size_t row = 1;
    while (readCsvLine(input, line)) {
        ++row;
        if (line.empty()) continue;
        const auto fields = splitCsv(line);
        if (fields.size() != 7 || fields[0].empty() ||
            (fields[2] != "train" && fields[2] != "validation" && fields[2] != "test")) {
            throw std::runtime_error("Invalid predictions.csv row " + std::to_string(row) + ".");
        }
        auto& result = results[fields[0]];
        std::size_t index = 0;
        try {
            std::size_t consumed = 0;
            index = std::stoull(fields[1], &consumed);
            if (consumed != fields[1].size()) throw std::invalid_argument("invalid index");
        } catch (...) {
            throw std::runtime_error("Invalid prediction index in row " + std::to_string(row) + ".");
        }
        if (index != result.x.size()) {
            throw std::runtime_error("Prediction indices are not contiguous in row " + std::to_string(row) + ".");
        }
        const double x = parseFiniteDouble(fields[3], "x", row);
        const double observed = parseFiniteDouble(fields[4], "observed", row);
        const double predicted = parseFiniteDouble(fields[5], "predicted", row);
        const double residual = parseFiniteDouble(fields[6], "residual", row);
        const double expectedResidual = predicted - observed;
        const double tolerance = 1.0e-10 * std::max({1.0, std::abs(expectedResidual), std::abs(residual)});
        if (std::abs(residual - expectedResidual) > tolerance) {
            throw std::runtime_error("Prediction residual is inconsistent in row " + std::to_string(row) + ".");
        }
        result.x.push_back(x);
        result.y_true.push_back(observed);
        result.y_pred.push_back(predicted);
        result.split.push_back(fields[2]);
        result.success = true;
        result.message = "Reloaded exported predictions.";
    }
    if (results.empty()) throw std::runtime_error("predictions.csv contains no predictions.");
    for (auto& entry : results) {
        std::vector<double> observedTest;
        std::vector<double> predictedTest;
        for (std::size_t i = 0; i < entry.second.split.size(); ++i) {
            if (entry.second.split[i] != "test") continue;
            observedTest.push_back(entry.second.y_true[i]);
            predictedTest.push_back(entry.second.y_pred[i]);
        }
        if (observedTest.empty()) {
            throw std::runtime_error("Exported predictions contain no test partition for approach: " + entry.first);
        }
        populateHydroMetrics(entry.second, observedTest, predictedTest);
        populateHydroPeakMetrics(entry.second);
    }
    return results;
}

std::map<std::string, HydroRunResult> HydroArtifactLoader::loadMetrics(
    const std::string& experimentDirectory) const {
    std::ifstream input(std::filesystem::path(experimentDirectory) / "metrics.csv");
    if (!input) throw std::runtime_error("Experiment is missing metrics.csv.");
    std::string line;
    const std::string header = "approach,success,final_loss,validation_mse,test_mse,rmse,mae,nse,kge,correlation,pbias,volume_error_percent,peak_timing_error,peak_magnitude_error_percent,high_flow_rmse,low_flow_rmse,physics_residual_mean,physics_residual_rmse,cumulative_physics_residual,physics_loss";
    if (!readCsvLine(input, line) || line != header) throw std::runtime_error("metrics.csv has an incompatible header.");
    std::map<std::string, HydroRunResult> metrics;
    std::size_t row = 1;
    while (readCsvLine(input, line)) {
        ++row;
        if (line.empty()) continue;
        const auto fields = splitCsv(line);
        if (fields.size() != 20 || fields[0].empty() || (fields[1] != "0" && fields[1] != "1"))
            throw std::runtime_error("Invalid metrics.csv row " + std::to_string(row) + ".");
        HydroRunResult result;
        result.success = fields[1] == "1";
        double* values[] = {&result.final_loss, &result.validation_mse, &result.mse, &result.rmse, &result.mae,
                            &result.nse, &result.kge, &result.correlation, &result.pbias, &result.volume_error_percent,
                            &result.peak_timing_error, &result.peak_magnitude_error_percent, &result.high_flow_rmse,
                            &result.low_flow_rmse, &result.physics_residual_mean, &result.physics_residual_rmse,
                            &result.cumulative_physics_residual, &result.physics_loss};
        for (std::size_t i = 0; i < 18; ++i) *values[i] = parseMetricDouble(fields[i + 2], row);
        if (result.success) {
            if (!std::isfinite(result.mse) || !std::isfinite(result.rmse) || !std::isfinite(result.mae) ||
                result.mse < 0.0 || result.rmse < 0.0 || result.mae < 0.0) {
                throw std::runtime_error("Successful metrics row has invalid error metrics in row " + std::to_string(row) + ".");
            }
            const double expectedRmse = std::sqrt(result.mse);
            const double tolerance = 1.0e-10 * std::max({1.0, expectedRmse, result.rmse});
            if (std::abs(result.rmse - expectedRmse) > tolerance) {
                throw std::runtime_error("Metrics RMSE is inconsistent with MSE in row " + std::to_string(row) + ".");
            }
        }
        if (!metrics.emplace(fields[0], result).second) throw std::runtime_error("Duplicate approach in metrics.csv: " + fields[0]);
    }
    if (metrics.empty()) throw std::runtime_error("metrics.csv contains no approach rows.");
    return metrics;
}

std::map<std::string, HydroPhysicsResidualArtifact> HydroArtifactLoader::loadPhysicsResiduals(
    const std::string& experimentDirectory) const {
    std::ifstream input(std::filesystem::path(experimentDirectory) / "physics_residuals.csv");
    if (!input) throw std::runtime_error("Experiment is missing physics_residuals.csv.");
    std::string line;
    if (!readCsvLine(input, line) || line != "approach,index,split,x,physics_residual") {
        throw std::runtime_error("physics_residuals.csv has an incompatible header.");
    }
    std::map<std::string, HydroPhysicsResidualArtifact> residuals;
    std::size_t row = 1;
    while (readCsvLine(input, line)) {
        ++row;
        if (line.empty()) continue;
        const auto fields = splitCsv(line);
        if (fields.size() != 5 || fields[0].empty() ||
            (fields[2] != "train" && fields[2] != "validation" && fields[2] != "test")) {
            throw std::runtime_error("Invalid physics_residuals.csv row " + std::to_string(row) + ".");
        }
        auto& artifact = residuals[fields[0]];
        std::size_t index = 0;
        try {
            std::size_t consumed = 0;
            index = std::stoull(fields[1], &consumed);
            if (consumed != fields[1].size()) throw std::invalid_argument("invalid index");
        } catch (...) {
            throw std::runtime_error("Invalid physics residual index in row " + std::to_string(row) + ".");
        }
        if (index != artifact.values.size()) {
            throw std::runtime_error("Physics residual indices are not contiguous in row " + std::to_string(row) + ".");
        }
        artifact.split.push_back(fields[2]);
        artifact.x.push_back(parseFiniteDouble(fields[3], "x", row));
        if (fields[4] == "nan" || fields[4] == "NaN") artifact.values.push_back(std::numeric_limits<double>::quiet_NaN());
        else artifact.values.push_back(parseFiniteDouble(fields[4], "physics residual", row));
    }
    return residuals;
}

std::map<std::string, HydroTrainingHistoryArtifact> HydroArtifactLoader::loadTrainingHistory(
    const std::string& experimentDirectory) const {
    std::ifstream input(std::filesystem::path(experimentDirectory) / "training_history.csv");
    if (!input) throw std::runtime_error("Experiment is missing training_history.csv.");
    std::string line;
    if (!readCsvLine(input, line) ||
        line != "approach,epoch,training_loss,validation_loss,selected_checkpoint") {
        throw std::runtime_error("training_history.csv has an incompatible header.");
    }
    std::map<std::string, HydroTrainingHistoryArtifact> histories;
    std::size_t row = 1;
    while (readCsvLine(input, line)) {
        ++row;
        if (line.empty()) continue;
        const auto fields = splitCsv(line);
        if (fields.size() != 5 || fields[0].empty() || (fields[4] != "0" && fields[4] != "1")) {
            throw std::runtime_error("Invalid training_history.csv row " + std::to_string(row) + ".");
        }
        auto& history = histories[fields[0]];
        std::size_t epoch = 0;
        try {
            std::size_t consumed = 0;
            epoch = std::stoull(fields[1], &consumed);
            if (consumed != fields[1].size() || epoch != history.training_loss.size() + 1) {
                throw std::invalid_argument("non-contiguous epoch");
            }
        } catch (...) {
            throw std::runtime_error("Training history epochs are not contiguous in row " + std::to_string(row) + ".");
        }
        history.training_loss.push_back(parseFiniteDouble(fields[2], "training loss", row));
        history.validation_loss.push_back(parseMetricDouble(fields[3], row));
        if (fields[4] == "1") {
            if (history.best_epoch != -1) {
                throw std::runtime_error("Training history selects multiple checkpoints for approach: " + fields[0]);
            }
            history.best_epoch = static_cast<int>(epoch);
        }
    }
    return histories;
}

HydroEnvironmentArtifact HydroArtifactLoader::loadEnvironment(const std::string& experimentDirectory) const {
    const auto text = readArtifactText(std::filesystem::path(experimentDirectory) / "environment.json");
    HydroEnvironmentArtifact environment;
    environment.compiler = jsonString(text, "compiler", "environment.json");
    environment.cplusplus = jsonPositiveInteger(text, "cplusplus", "environment.json");
    environment.build_date = jsonString(text, "build_date", "environment.json");
    environment.build_time = jsonString(text, "build_time", "environment.json");
    if (environment.compiler.empty() || environment.build_date.empty() || environment.build_time.empty()) {
        throw std::runtime_error("environment.json contains empty build metadata.");
    }
    return environment;
}

HydroProvenanceArtifact HydroArtifactLoader::loadProvenance(const std::string& experimentDirectory) const {
    const std::filesystem::path root(experimentDirectory);
    const auto text = readArtifactText(root / "provenance.json");
    HydroProvenanceArtifact provenance;
    provenance.fingerprint_algorithm = jsonString(text, "fingerprint_algorithm", "provenance.json");
    provenance.dataset_manifest_sha256 = jsonString(text, "dataset_manifest_sha256", "provenance.json");
    if (provenance.fingerprint_algorithm != "sha256" || provenance.dataset_manifest_sha256.size() != 64 ||
        !std::all_of(provenance.dataset_manifest_sha256.begin(), provenance.dataset_manifest_sha256.end(),
                     [](const unsigned char value) { return std::isxdigit(value) && !std::isupper(value); })) {
        throw std::runtime_error("provenance.json contains unsupported or malformed fingerprint metadata.");
    }
    const auto manifest = root / "dataset_manifest.json";
    if (!std::filesystem::is_regular_file(manifest) ||
        sha256File(manifest.string()) != provenance.dataset_manifest_sha256) {
        throw std::runtime_error("Exported dataset manifest does not match provenance.json.");
    }
    return provenance;
}

HydroInferenceArtifacts HydroArtifactLoader::loadForInference(
    const std::string& experimentDirectory) const {
    const std::filesystem::path root(experimentDirectory);
    HydroInferenceArtifacts artifacts;
    artifacts.experiment = HydroExperimentLoader().loadConfig((root / "experiment_config.json").string());
    artifacts.models = loadModels(experimentDirectory);
    artifacts.scalers = loadScalers(experimentDirectory);
    artifacts.training_history = loadTrainingHistory(experimentDirectory);
    artifacts.environment = loadEnvironment(experimentDirectory);
    if (artifacts.experiment.config.use_hydro_package) artifacts.provenance = loadProvenance(experimentDirectory);
    auto predictions = loadPredictions(experimentDirectory);
    const auto metrics = loadMetrics(experimentDirectory);
    const auto residuals = loadPhysicsResiduals(experimentDirectory);

    for (const auto& entry : artifacts.models) {
        const auto scaler = artifacts.scalers.find(entry.first);
        if (scaler == artifacts.scalers.end()) {
            throw std::runtime_error("Model is missing scaler state for approach: " + entry.first);
        }
        const bool recurrent = entry.first == "lstm" || entry.first == "lstm_pinn";
        const bool feedForward = entry.first == "ffn" || entry.first == "ffn_pinn" || entry.first == "pinn";
        if (!recurrent && !feedForward) {
            throw std::runtime_error("Unsupported inference approach: " + entry.first);
        }
        const std::string expectedFormat = recurrent ? "torch-module-v1" : "neuralnetworkwrapper-v1";
        if (entry.second.format != expectedFormat) {
            throw std::runtime_error("Checkpoint format does not match approach " + entry.first + ".");
        }
    }
    for (const auto& entry : artifacts.scalers) {
        if (artifacts.models.find(entry.first) == artifacts.models.end()) {
            throw std::runtime_error("Scaler state has no matching model for approach: " + entry.first);
        }
    }
    if (predictions.size() != artifacts.models.size() || metrics.size() != artifacts.models.size()) {
        throw std::runtime_error("Inference models, predictions, and metrics have different approach sets.");
    }
    for (auto& entry : predictions) {
        const auto model = artifacts.models.find(entry.first);
        const auto metric = metrics.find(entry.first);
        if (model == artifacts.models.end() || metric == metrics.end()) {
            throw std::runtime_error("Inference summary artifacts have no matching model for approach: " + entry.first);
        }
        if (!metric->second.success) {
            throw std::runtime_error("Inference checkpoint has an unsuccessful metrics row for approach: " + entry.first);
        }
        if (!equivalentMetric(entry.second.mse, metric->second.mse) ||
            !equivalentMetric(entry.second.rmse, metric->second.rmse) ||
            !equivalentMetric(entry.second.mae, metric->second.mae) ||
            !equivalentMetric(entry.second.nse, metric->second.nse) ||
            !equivalentMetric(entry.second.kge, metric->second.kge) ||
            !equivalentMetric(entry.second.correlation, metric->second.correlation) ||
            !equivalentMetric(entry.second.pbias, metric->second.pbias) ||
            !equivalentMetric(entry.second.volume_error_percent, metric->second.volume_error_percent) ||
            !equivalentMetric(entry.second.peak_timing_error, metric->second.peak_timing_error) ||
            !equivalentMetric(entry.second.peak_magnitude_error_percent,
                              metric->second.peak_magnitude_error_percent) ||
            !equivalentMetric(entry.second.high_flow_rmse, metric->second.high_flow_rmse) ||
            !equivalentMetric(entry.second.low_flow_rmse, metric->second.low_flow_rmse)) {
            throw std::runtime_error("Inference metrics do not match predictions for approach: " + entry.first);
        }
        entry.second.final_loss = metric->second.final_loss;
        entry.second.validation_mse = metric->second.validation_mse;
        entry.second.physics_loss = metric->second.physics_loss;
    }
    for (const auto& entry : residuals) {
        const auto prediction = predictions.find(entry.first);
        const auto metric = metrics.find(entry.first);
        if (prediction == predictions.end() || metric == metrics.end() ||
            entry.second.x.size() != prediction->second.x.size() ||
            entry.second.split != prediction->second.split) {
            throw std::runtime_error("Inference physics residuals do not align for approach: " + entry.first);
        }
        for (std::size_t i = 0; i < entry.second.x.size(); ++i) {
            if (!equivalentMetric(entry.second.x[i], prediction->second.x[i])) {
                throw std::runtime_error("Inference residual timestamps do not match predictions for approach: " + entry.first);
            }
        }
        prediction->second.physics_residual = entry.second.values;
        populateHydroPhysicsResidualMetrics(prediction->second);
        if (!equivalentMetric(prediction->second.physics_residual_mean, metric->second.physics_residual_mean) ||
            !equivalentMetric(prediction->second.physics_residual_rmse, metric->second.physics_residual_rmse) ||
            !equivalentMetric(prediction->second.cumulative_physics_residual,
                              metric->second.cumulative_physics_residual)) {
            throw std::runtime_error("Inference residual summaries do not match residual samples for approach: " + entry.first);
        }
    }
    for (const auto& entry : artifacts.training_history) {
        const auto result = predictions.find(entry.first);
        if (result == predictions.end()) {
            throw std::runtime_error("Training history has no matching inference approach: " + entry.first);
        }
        if (!entry.second.training_loss.empty() && entry.second.best_epoch < 1) {
            throw std::runtime_error("Training history has no selected checkpoint for approach: " + entry.first);
        }
        result->second.training_loss_history = entry.second.training_loss;
        result->second.validation_loss_history = entry.second.validation_loss;
        result->second.best_epoch = entry.second.best_epoch;
    }
    artifacts.results = std::move(predictions);
    return artifacts;
}
