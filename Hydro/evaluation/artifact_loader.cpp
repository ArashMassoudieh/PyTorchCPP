#include "artifact_loader.h"

#include "../dataset/hydro_checksum.h"

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
