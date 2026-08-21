#include "artifact_loader.h"

#include "../dataset/hydro_checksum.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace {
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

std::map<std::string, HydroArtifactScalers> HydroArtifactLoader::loadScalers(
    const std::string& experimentDirectory) const {
    std::ifstream manifest(std::filesystem::path(experimentDirectory) / "scalers.csv");
    if (!manifest) throw std::runtime_error("Experiment is missing scalers.csv.");
    std::string line;
    if (!std::getline(manifest, line) || line != "approach,kind,index,method,shape,offset,scale") {
        throw std::runtime_error("scalers.csv has an incompatible header.");
    }

    std::map<std::string, HydroArtifactScalers> scalers;
    std::map<std::string, std::map<std::string, std::size_t>> nextIndices;
    std::size_t row = 1;
    while (std::getline(manifest, line)) {
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
