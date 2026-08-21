#include "artifact_loader.h"

#include "../dataset/hydro_checksum.h"

#include <algorithm>
#include <filesystem>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace {
std::vector<std::string> splitCsv(const std::string& line) {
    std::vector<std::string> values;
    std::stringstream stream(line);
    std::string value;
    while (std::getline(stream, value, ',')) values.push_back(value);
    return values;
}

bool safeRelativePath(const std::string& value) {
    const std::filesystem::path path(value);
    if (value.empty() || path.is_absolute()) return false;
    for (const auto& component : path) if (component == "..") return false;
    return true;
}

std::vector<int64_t> parseShape(std::string text) {
    if (text.size() >= 2 && text.front() == '"' && text.back() == '"') text = text.substr(1, text.size() - 2);
    std::vector<int64_t> shape;
    std::stringstream stream(text);
    std::string extent;
    while (std::getline(stream, extent, ';')) {
        try {
            std::size_t consumed = 0;
            const auto value = std::stoll(extent, &consumed);
            if (consumed != extent.size() || value <= 0) throw std::invalid_argument("invalid extent");
            shape.push_back(value);
        } catch (...) { throw std::runtime_error("Invalid scaler tensor shape: " + text); }
    }
    if (shape.empty()) throw std::runtime_error("Scaler tensor shape is empty.");
    return shape;
}
}

std::map<std::string, HydroModelArtifact> HydroArtifactLoader::loadModels(
    const std::string& experimentDirectory) const {
    const std::filesystem::path root(experimentDirectory);
    const auto canonicalRoot = std::filesystem::weakly_canonical(root);
    std::ifstream manifest(root / "models.csv");
    if (!manifest) throw std::runtime_error("Experiment is missing models.csv.");
    std::string line;
    if (!std::getline(manifest, line) || line != "approach,file,format,size_bytes,sha256") {
        throw std::runtime_error("models.csv has an incompatible header.");
    }
    std::map<std::string, HydroModelArtifact> models;
    std::size_t row = 1;
    while (std::getline(manifest, line)) {
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

std::map<std::string, HydroScalerArtifacts> HydroArtifactLoader::loadScalers(
    const std::string& experimentDirectory) const {
    std::ifstream manifest(std::filesystem::path(experimentDirectory) / "scalers.csv");
    if (!manifest) throw std::runtime_error("Experiment is missing scalers.csv.");
    std::string line;
    if (!std::getline(manifest, line) || line != "approach,kind,index,method,shape,offset,scale") {
        throw std::runtime_error("scalers.csv has an incompatible header.");
    }
    std::map<std::string, HydroScalerArtifacts> scalers;
    std::map<std::pair<std::string, std::string>, std::size_t> nextIndex;
    std::size_t row = 1;
    while (std::getline(manifest, line)) {
        ++row;
        if (line.empty()) continue;
        const auto fields = splitCsv(line);
        if (fields.size() != 7 || fields[0].empty() || (fields[1] != "input" && fields[1] != "target")) {
            throw std::runtime_error("Invalid scalers.csv row " + std::to_string(row) + ".");
        }
        std::size_t index = 0;
        double offset = 0.0;
        double scale = 0.0;
        try {
            std::size_t consumed = 0;
            index = std::stoull(fields[2], &consumed);
            if (consumed != fields[2].size()) throw std::invalid_argument("trailing index");
            consumed = 0;
            offset = std::stod(fields[5], &consumed);
            if (consumed != fields[5].size()) throw std::invalid_argument("trailing offset");
            consumed = 0;
            scale = std::stod(fields[6], &consumed);
            if (consumed != fields[6].size()) throw std::invalid_argument("trailing scale");
        } catch (...) { throw std::runtime_error("Invalid scaler numeric value at row " + std::to_string(row) + "."); }
        if (!std::isfinite(offset) || !std::isfinite(scale) || scale == 0.0) {
            throw std::runtime_error("Scaler row contains non-finite or zero scale.");
        }
        const auto key = std::make_pair(fields[0], fields[1]);
        if (index != nextIndex[key]++) throw std::runtime_error("Scaler indices must be contiguous and ordered.");
        HydroScalerState& state = fields[1] == "input" ? scalers[fields[0]].input : scalers[fields[0]].target;
        const auto shape = parseShape(fields[4]);
        if (!state.offset.empty() && (state.method != fields[3] || state.shape != shape)) {
            throw std::runtime_error("Scaler metadata changes within an approach/kind.");
        }
        state.method = fields[3];
        state.shape = shape;
        state.offset.push_back(offset);
        state.scale.push_back(scale);
    }
    for (const auto& entry : scalers) {
        if (entry.second.input.offset.empty() || entry.second.target.offset.empty()) {
            throw std::runtime_error("Approach is missing input or target scaler state: " + entry.first);
        }
    }
    return scalers;
}

std::map<std::string, HydroRunResult> HydroArtifactLoader::loadResults(
    const std::string& experimentDirectory) const {
    const std::filesystem::path root(experimentDirectory);
    std::ifstream metrics(root / "metrics.csv");
    if (!metrics) throw std::runtime_error("Experiment is missing metrics.csv.");
    std::string line;
    if (!std::getline(metrics, line) ||
        line != "approach,success,final_loss,validation_mse,test_mse,rmse,mae,nse,kge,correlation,pbias,volume_error_percent,physics_loss") {
        throw std::runtime_error("metrics.csv has an incompatible header.");
    }
    std::map<std::string, HydroRunResult> results;
    std::size_t row = 1;
    while (std::getline(metrics, line)) {
        ++row;
        if (line.empty()) continue;
        const auto fields = splitCsv(line);
        if (fields.size() != 13 || fields[0].empty()) throw std::runtime_error("Invalid metrics.csv row " + std::to_string(row) + ".");
        HydroRunResult result;
        try {
            if (fields[1] != "0" && fields[1] != "1") throw std::invalid_argument("invalid success flag");
            result.success = fields[1] == "1";
            result.final_loss = std::stod(fields[2]);
            result.validation_mse = std::stod(fields[3]);
            result.mse = std::stod(fields[4]);
            result.rmse = std::stod(fields[5]);
            result.mae = std::stod(fields[6]);
            result.nse = std::stod(fields[7]);
            result.kge = std::stod(fields[8]);
            result.correlation = std::stod(fields[9]);
            result.pbias = std::stod(fields[10]);
            result.volume_error_percent = std::stod(fields[11]);
            result.physics_loss = std::stod(fields[12]);
        } catch (...) { throw std::runtime_error("Invalid metric value at row " + std::to_string(row) + "."); }
        if (!results.emplace(fields[0], std::move(result)).second) throw std::runtime_error("Duplicate approach in metrics.csv: " + fields[0]);
    }
    if (results.empty()) throw std::runtime_error("metrics.csv contains no approaches.");

    std::ifstream predictions(root / "predictions.csv");
    if (!predictions) throw std::runtime_error("Experiment is missing predictions.csv.");
    if (!std::getline(predictions, line) || line != "approach,index,split,x,observed,predicted,residual") {
        throw std::runtime_error("predictions.csv has an incompatible header.");
    }
    std::map<std::string, std::size_t> nextIndex;
    row = 1;
    while (std::getline(predictions, line)) {
        ++row;
        if (line.empty()) continue;
        const auto fields = splitCsv(line);
        if (fields.size() != 7) throw std::runtime_error("Invalid predictions.csv row " + std::to_string(row) + ".");
        auto found = results.find(fields[0]);
        if (found == results.end()) throw std::runtime_error("Prediction references unknown approach: " + fields[0]);
        try {
            std::size_t consumed = 0;
            const std::size_t index = std::stoull(fields[1], &consumed);
            if (consumed != fields[1].size() || index != nextIndex[fields[0]]++) throw std::invalid_argument("unordered index");
            const double x = std::stod(fields[3]);
            const double observed = std::stod(fields[4]);
            const double predicted = std::stod(fields[5]);
            const double residual = std::stod(fields[6]);
            if (!std::isfinite(x) || !std::isfinite(observed) || !std::isfinite(predicted) ||
                !std::isfinite(residual) || std::abs((predicted - observed) - residual) > 1.0e-8 * std::max(1.0, std::abs(residual))) {
                throw std::invalid_argument("invalid prediction");
            }
            if (fields[2] != "train" && fields[2] != "validation" && fields[2] != "test") throw std::invalid_argument("invalid split");
            found->second.x.push_back(x);
            found->second.y_true.push_back(observed);
            found->second.y_pred.push_back(predicted);
            found->second.split.push_back(fields[2]);
        } catch (...) { throw std::runtime_error("Invalid prediction value at row " + std::to_string(row) + "."); }
    }
    for (const auto& entry : results) {
        if (entry.second.success && entry.second.x.empty()) {
            throw std::runtime_error("Successful approach has no exported predictions: " + entry.first);
        }
    }

    std::ifstream history(root / "training_history.csv");
    if (!history) throw std::runtime_error("Experiment is missing training_history.csv.");
    if (!std::getline(history, line) || line != "approach,epoch,training_loss,validation_loss,selected_checkpoint") {
        throw std::runtime_error("training_history.csv has an incompatible header.");
    }
    std::map<std::string, std::size_t> nextEpoch;
    row = 1;
    while (std::getline(history, line)) {
        ++row;
        if (line.empty()) continue;
        const auto fields = splitCsv(line);
        if (fields.size() != 5) throw std::runtime_error("Invalid training_history.csv row " + std::to_string(row) + ".");
        auto found = results.find(fields[0]);
        if (found == results.end()) throw std::runtime_error("Training history references unknown approach: " + fields[0]);
        try {
            const std::size_t epoch = std::stoull(fields[1]);
            if (epoch != ++nextEpoch[fields[0]]) throw std::invalid_argument("unordered epoch");
            const double trainingLoss = std::stod(fields[2]);
            const double validationLoss = std::stod(fields[3]);
            if (!std::isfinite(trainingLoss) || (!std::isfinite(validationLoss) && !std::isnan(validationLoss)) ||
                (fields[4] != "0" && fields[4] != "1")) {
                throw std::invalid_argument("invalid history value");
            }
            found->second.training_loss_history.push_back(trainingLoss);
            found->second.validation_loss_history.push_back(validationLoss);
            if (fields[4] == "1") {
                if (found->second.best_epoch != 0) throw std::invalid_argument("multiple selected checkpoints");
                found->second.best_epoch = static_cast<int>(epoch);
            }
        } catch (...) { throw std::runtime_error("Invalid training history value at row " + std::to_string(row) + "."); }
    }

    std::ifstream physics(root / "physics_residuals.csv");
    if (!physics) throw std::runtime_error("Experiment is missing physics_residuals.csv.");
    if (!std::getline(physics, line) || line != "approach,index,split,x,physics_residual") {
        throw std::runtime_error("physics_residuals.csv has an incompatible header.");
    }
    nextIndex.clear();
    row = 1;
    while (std::getline(physics, line)) {
        ++row;
        if (line.empty()) continue;
        const auto fields = splitCsv(line);
        if (fields.size() != 5) throw std::runtime_error("Invalid physics_residuals.csv row " + std::to_string(row) + ".");
        auto found = results.find(fields[0]);
        if (found == results.end()) throw std::runtime_error("Physics residual references unknown approach: " + fields[0]);
        try {
            const std::size_t index = std::stoull(fields[1]);
            if (index != nextIndex[fields[0]]++ || index >= found->second.x.size() ||
                fields[2] != found->second.split[index]) {
                throw std::invalid_argument("misaligned residual");
            }
            const double x = std::stod(fields[3]);
            const double residual = std::stod(fields[4]);
            if (!std::isfinite(x) || !std::isfinite(residual) ||
                std::abs(x - found->second.x[index]) > 1.0e-8 * std::max(1.0, std::abs(x))) {
                throw std::invalid_argument("invalid residual");
            }
            found->second.physics_residual.push_back(residual);
        } catch (...) { throw std::runtime_error("Invalid physics residual value at row " + std::to_string(row) + "."); }
    }
    return results;
}

void HydroArtifactLoader::validateCompatibility(
    const HydroRunConfig& config,
    const std::map<std::string, HydroModelArtifact>& models,
    const std::map<std::string, HydroScalerArtifacts>& scalers) const {
    if (models.empty()) throw std::runtime_error("Experiment contains no verified model checkpoints.");
    const auto elementCount = [](const HydroScalerState& state) {
        int64_t count = 1;
        for (const int64_t extent : state.shape) count *= extent;
        return count;
    };
    for (const auto& entry : models) {
        const bool recurrent = entry.first == "lstm" || entry.first == "lstm_pinn";
        const bool feedForward = entry.first == "ffn" || entry.first == "ffn_pinn" || entry.first == "pinn";
        if (!recurrent && !feedForward) throw std::runtime_error("Unknown Hydro approach in model artifacts: " + entry.first);
        const std::string expectedFormat = recurrent ? "torch-module-v1" : "neuralnetworkwrapper-v1";
        if (entry.second.format != expectedFormat) {
            throw std::runtime_error("Checkpoint format is incompatible with approach " + entry.first + ".");
        }
        const bool requiresScaler = entry.first == "ffn" || recurrent;
        const auto scaler = scalers.find(entry.first);
        if (requiresScaler && scaler == scalers.end()) {
            throw std::runtime_error("Checkpoint is missing fitted scaler artifacts for " + entry.first + ".");
        }
        if (scaler != scalers.end()) {
            if (scaler->second.input.offset.size() != scaler->second.input.scale.size() ||
                scaler->second.target.offset.size() != scaler->second.target.scale.size() ||
                elementCount(scaler->second.input) != static_cast<int64_t>(scaler->second.input.offset.size()) ||
                elementCount(scaler->second.target) != 1) {
                throw std::runtime_error("Scaler tensor dimensions are incompatible with approach " + entry.first + ".");
            }
            if ((entry.first == "ffn_pinn" || entry.first == "pinn" || entry.first == "lstm_pinn") &&
                scaler->second.input.method != "none") {
                throw std::runtime_error("PINN checkpoint cannot apply a non-physical input scaler.");
            }
        }
    }
    if (config.use_hydro_forecast_feature && config.hydro_forecast_variable.empty()) {
        throw std::runtime_error("Forecast-enabled checkpoint configuration has no forecast variable.");
    }
}
