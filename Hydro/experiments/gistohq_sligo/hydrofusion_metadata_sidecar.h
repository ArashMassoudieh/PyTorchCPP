#pragma once

// Persisted HydroFusion metadata sidecar: q_ref, TargetMode, split cutoffs
// ("bounds"), feature order+units, forecast horizon/latency, and a
// bundle-hash binding to the specific model+scaler artifacts it
// describes -- the trained-artifact state that is NOT re-derivable from a
// fresh raw-data load, unlike the input features or origin_q (both of
// which a real inference call always re-derives from fresh observations,
// same as production).
//
// Shared by the HydroFusion real-engine integration test AND the opt-in
// FFN+HydroFusion production dispatch path (Hydro/models/
// ffn_hydrofusion_wrapper.h, Hydro/batch/hydro_batch_runner.cpp) so both
// read/write the IDENTICAL sidecar format rather than risking two
// independently-fallible parsers drifting apart -- this project has
// already been burned once by exactly that failure mode (the retired
// hydrofusion_cli_smoke_test.cpp mock, which reproduced the same class of
// parsing bug across two independent-review rounds). Every lesson from
// that file's review findings is applied here: full-token-consumption
// numeric parsing, exact line/field structure, chronological-safe integer
// parsing (no arithmetic before domain bounding), and reuse of Task 2/3's
// own hardened validators (validateTargetMode, deserializeQRef/
// serializeQRef, validateSplitCutoffs) rather than re-deriving equivalent
// but independently-fallible logic.

#include "hydrofusion_feature_builder.h"
#include "hydrofusion_log_residual.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace hydrofusion {

// Exact feature-name order, matching hydrofusion_forecast_contract_schema.json's
// input_allowlist arrays -- persisted into the metadata sidecar below so a
// reload can independently verify it, not merely assume the caller's own
// column order is still correct.
inline const std::vector<std::string>& featureOrderNames() {
    static const std::vector<std::string> names = {
        "q_lag_0h",       "q_lag_1h",       "q_lag_3h",          "q_lag_6h",
        "q_lag_12h",      "q_lag_24h",      "q_trailing_mean_6h", "q_trailing_mean_24h",
        "q_trailing_mean_72h", "q_trailing_var_6h", "q_trailing_var_24h", "q_trailing_var_72h",
        "q_last_hourly_change", "q_past_slope_6h", "met_precip_mm_per_h", "met_temperature_c",
        "met_relative_humidity_pct", "met_wind_speed_m_per_s", "met_solar_mj_per_m2_per_h",
        "met_pet_mm_per_h"};
    return names;
}

// Units in the same order as featureOrderNames(), matching the contract's
// own unit conventions -- persisted so a reload can confirm the physical
// meaning of each column, not just its name/order.
inline const std::vector<std::string>& featureUnits() {
    static const std::vector<std::string> units = {
        "mm/h", "mm/h", "mm/h", "mm/h", "mm/h", "mm/h", "mm/h", "mm/h", "mm/h", "(mm/h)^2",
        "(mm/h)^2", "(mm/h)^2", "mm/h", "mm/h/h", "mm/h", "degC", "percent", "m/s",
        "MJ/m2/h", "mm/h"};
    return units;
}

inline const std::string kMetadataSchemaLine = "schema=hydrofusion_metadata_v1\n";

struct HydroFusionMetadata {
    TargetMode mode = TargetMode::LogResidual;
    double qRef = 0.0;
    SplitCutoffs cutoffs{0, 0, 0};
    // Binds this sidecar to the one model artifact it describes -- the
    // real exporter's own sha256 of the saved checkpoint file (read back
    // from models.csv / HydroModelArtifact::sha256), not a hash computed
    // independently by this file. A sidecar copied next to a different
    // model/run must be rejected on reload, not silently accepted.
    std::string modelSha256;
    // Model weights alone do not define inference: the SAME weights
    // combined with DIFFERENT fitted input/target scalers silently
    // produce different physical predictions, with no error from the
    // engine (scalers.csv has no schema reason to disagree with a given
    // model file). Binding separately to the bundle's scalers.csv (hashed
    // with the same sha256File() the engine itself trusts, since the
    // engine computes no per-scaler hash of its own) closes that gap --
    // a sidecar with the right model but swapped-in scalers must still
    // be rejected.
    std::string scalersSha256;
    // Forecast metadata mirroring hydrofusion_forecast_contract.md: fixed
    // at a 1-hour-ahead, idealized-availability forecast.
    int64_t horizonHours = 1;
    std::string latencyModel = "idealized";
};

inline void exportHydroFusionMetadata(const HydroFusionMetadata& metadata, const std::string& path) {
    validateTargetMode(metadata.mode);
    validateSplitCutoffs(metadata.cutoffs);
    const auto validDigest = [](const std::string& value) {
        return value.size() == 64 && std::all_of(value.begin(), value.end(), [](unsigned char c) { return std::isxdigit(c); });
    };
    validateQRef(metadata.qRef);
    if (!validDigest(metadata.modelSha256)) {
        throw std::invalid_argument("exportHydroFusionMetadata: modelSha256 must be a 64-character hex digest.");
    }
    if (!validDigest(metadata.scalersSha256)) {
        throw std::invalid_argument("exportHydroFusionMetadata: scalersSha256 must be a 64-character hex digest.");
    }
    if (metadata.horizonHours != 1) {
        throw std::invalid_argument("exportHydroFusionMetadata: horizonHours must be the contract's fixed 1h.");
    }
    if (metadata.latencyModel != "idealized") {
        throw std::invalid_argument("exportHydroFusionMetadata: latencyModel must be the contract's 'idealized'.");
    }
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("exportHydroFusionMetadata: could not open '" + path + "' for writing.");
    }
    out << kMetadataSchemaLine;
    out << "mode=" << (metadata.mode == TargetMode::DirectLogQ ? "direct_log_q" : "log_residual") << "\n";
    out << serializeQRef(metadata.qRef);
    out << "train_end_x=" << metadata.cutoffs.train_end_x << "\n";
    out << "validation_end_x=" << metadata.cutoffs.validation_end_x << "\n";
    out << "test_end_x=" << metadata.cutoffs.test_end_x << "\n";
    out << "model_sha256=" << metadata.modelSha256 << "\n";
    out << "scalers_sha256=" << metadata.scalersSha256 << "\n";
    out << "horizon_hours=" << metadata.horizonHours << "\n";
    out << "latency_model=" << metadata.latencyModel << "\n";
    out << "feature_order=";
    const auto& names = featureOrderNames();
    for (std::size_t i = 0; i < names.size(); ++i) {
        if (i > 0) out << ',';
        out << names[i];
    }
    out << "\n";
    out << "feature_units=";
    const auto& units = featureUnits();
    for (std::size_t i = 0; i < units.size(); ++i) {
        if (i > 0) out << ',';
        out << units[i];
    }
    out << "\n";
    if (!out.good()) {
        throw std::runtime_error("exportHydroFusionMetadata: write failed for '" + path + "'.");
    }
}

inline int64_t parseFullyConsumedInt64(const std::string& token, const char* fieldName) {
    std::size_t consumed = 0;
    int64_t value;
    try {
        value = std::stoll(token, &consumed);
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string("reloadHydroFusionMetadata: could not parse ") + fieldName + ".");
    }
    if (consumed != token.size()) {
        throw std::invalid_argument(std::string("reloadHydroFusionMetadata: trailing characters in ") + fieldName +
                                     ".");
    }
    return value;
}

inline std::string requireLineWithPrefix(std::ifstream& in, const std::string& prefix, const char* fieldName) {
    std::string line;
    if (!std::getline(in, line)) {
        throw std::invalid_argument(std::string("reloadHydroFusionMetadata: missing ") + fieldName + " line.");
    }
    if (line.rfind(prefix, 0) != 0) {
        throw std::invalid_argument(std::string("reloadHydroFusionMetadata: malformed ") + fieldName + " line.");
    }
    return line.substr(prefix.size());
}

inline HydroFusionMetadata reloadHydroFusionMetadata(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("reloadHydroFusionMetadata: could not open '" + path + "' for reading.");
    }

    std::string schemaLine;
    std::getline(in, schemaLine);
    if (schemaLine + "\n" != kMetadataSchemaLine) {
        throw std::invalid_argument("reloadHydroFusionMetadata: unexpected or missing schema line.");
    }

    const std::string modeToken = requireLineWithPrefix(in, "mode=", "mode");
    HydroFusionMetadata metadata;
    if (modeToken == "direct_log_q") metadata.mode = TargetMode::DirectLogQ;
    else if (modeToken == "log_residual") metadata.mode = TargetMode::LogResidual;
    else throw std::invalid_argument("reloadHydroFusionMetadata: unrecognized mode token '" + modeToken + "'.");
    validateTargetMode(metadata.mode);  // defense in depth; the branch above is already exhaustive

    std::string qRefLine;
    std::getline(in, qRefLine);
    metadata.qRef = deserializeQRef(qRefLine + "\n");  // reuses Task 3's own hardened parser

    const std::string trainEndStr = requireLineWithPrefix(in, "train_end_x=", "train_end_x");
    const std::string validationEndStr = requireLineWithPrefix(in, "validation_end_x=", "validation_end_x");
    const std::string testEndStr = requireLineWithPrefix(in, "test_end_x=", "test_end_x");
    metadata.cutoffs.train_end_x = parseFullyConsumedInt64(trainEndStr, "train_end_x");
    metadata.cutoffs.validation_end_x = parseFullyConsumedInt64(validationEndStr, "validation_end_x");
    metadata.cutoffs.test_end_x = parseFullyConsumedInt64(testEndStr, "test_end_x");
    validateSplitCutoffs(metadata.cutoffs);  // reuses Task 3's own hardened validator

    auto parseHexSha256 = [](const std::string& token, const char* fieldName) {
        if (token.size() != 64 ||
            !std::all_of(token.begin(), token.end(), [](unsigned char c) { return std::isxdigit(c); })) {
            throw std::invalid_argument(std::string("reloadHydroFusionMetadata: ") + fieldName +
                                         " is not a 64-character hex digest.");
        }
    };

    metadata.modelSha256 = requireLineWithPrefix(in, "model_sha256=", "model_sha256");
    parseHexSha256(metadata.modelSha256, "model_sha256");

    metadata.scalersSha256 = requireLineWithPrefix(in, "scalers_sha256=", "scalers_sha256");
    parseHexSha256(metadata.scalersSha256, "scalers_sha256");

    const std::string horizonStr = requireLineWithPrefix(in, "horizon_hours=", "horizon_hours");
    metadata.horizonHours = parseFullyConsumedInt64(horizonStr, "horizon_hours");
    if (metadata.horizonHours != 1) {
        throw std::invalid_argument("reloadHydroFusionMetadata: horizon_hours must be the contract's fixed 1h.");
    }

    metadata.latencyModel = requireLineWithPrefix(in, "latency_model=", "latency_model");
    if (metadata.latencyModel != "idealized") {
        throw std::invalid_argument("reloadHydroFusionMetadata: latency_model must be the contract's 'idealized'.");
    }

    const std::string featureOrderValue = requireLineWithPrefix(in, "feature_order=", "feature_order");
    std::vector<std::string> reloadedNames;
    std::istringstream featureStream(featureOrderValue);
    std::string name;
    while (std::getline(featureStream, name, ',')) reloadedNames.push_back(name);
    if (featureOrderValue.empty() || featureOrderValue.back() == ',' || reloadedNames != featureOrderNames()) {
        throw std::invalid_argument("reloadHydroFusionMetadata: feature_order does not match the expected order.");
    }

    const std::string featureUnitsValue = requireLineWithPrefix(in, "feature_units=", "feature_units");
    std::vector<std::string> reloadedUnits;
    std::istringstream unitsStream(featureUnitsValue);
    std::string unit;
    while (std::getline(unitsStream, unit, ',')) reloadedUnits.push_back(unit);
    if (featureUnitsValue.empty() || featureUnitsValue.back() == ',' || reloadedUnits != featureUnits()) {
        throw std::invalid_argument("reloadHydroFusionMetadata: feature_units does not match the expected units.");
    }

    if (in.peek() != std::char_traits<char>::eof()) {
        throw std::invalid_argument("reloadHydroFusionMetadata: unexpected extra record.");
    }
    return metadata;
}

// Confirms a reloaded sidecar actually describes the model AND scaler
// artifacts loaded alongside it, rather than being copied in from a
// different run. Both must be checked: model weights alone do not define
// inference -- the same weights combined with different fitted scalers
// silently produce different physical predictions, with no error from the
// engine itself. actualModelSha256 comes from HydroArtifactLoader's own
// already-verified HydroModelArtifact::sha256; actualScalersSha256 is a
// direct sha256File() of the bundle's scalers.csv (the engine computes no
// per-scaler hash of its own to read back).
inline void validateMetadataBoundToBundle(const HydroFusionMetadata& metadata, const std::string& actualModelSha256,
                                           const std::string& actualScalersSha256) {
    if (metadata.modelSha256 != actualModelSha256) {
        throw std::invalid_argument(
            "validateMetadataBoundToBundle: metadata sidecar's model_sha256 does not match the loaded model "
            "artifact -- sidecar/model mismatch.");
    }
    if (metadata.scalersSha256 != actualScalersSha256) {
        throw std::invalid_argument(
            "validateMetadataBoundToBundle: metadata sidecar's scalers_sha256 does not match the loaded bundle's "
            "scalers.csv -- sidecar/scalers mismatch.");
    }
}

// Reads back the real exporter's own models.csv to get the sha256 it
// already computed for this approach's checkpoint file -- the same value
// HydroArtifactLoader independently re-verifies at reload time -- rather
// than recomputing a hash ourselves and merely hoping it agrees.
inline std::string readModelSha256FromModelsCsv(const std::filesystem::path& root, const std::string& approach) {
    std::ifstream in(root / "models.csv");
    if (!in) {
        throw std::runtime_error("readModelSha256FromModelsCsv: could not open models.csv.");
    }
    std::string header;
    std::getline(in, header);
    if (header != "approach,file,format,size_bytes,sha256") {
        throw std::invalid_argument("readModelSha256FromModelsCsv: unexpected models.csv header.");
    }
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::string approachField, fileField, formatField, sizeField, sha256Field;
        std::getline(iss, approachField, ',');
        std::getline(iss, fileField, ',');
        std::getline(iss, formatField, ',');
        std::getline(iss, sizeField, ',');
        std::getline(iss, sha256Field, ',');
        if (approachField == approach) return sha256Field;
    }
    throw std::invalid_argument("readModelSha256FromModelsCsv: approach '" + approach + "' not found.");
}

// Explicit origin/target identity pairing. Exporting only target_x (the
// time the observation/prediction are AT -- the correct choice for
// scoring) would silently drop which origin hour's information the
// forecast was actually made FROM. This sidecar file keeps both explicit
// side by side, one row per test-split sample, in the same order as
// predictions.csv's own rows.
inline void validateOriginTargetPair(int64_t originX, int64_t targetX, int64_t horizonHours) {
    // Bound both identities before addition, including INT64_MAX cases.
    if (horizonHours != 1 || originX < 0 || originX >= kSealStartX ||
        targetX < 0 || targetX >= kSealStartX || targetX != originX + 1) {
        throw std::invalid_argument("HydroFusion origin/target pair violates fixed horizon or development seal.");
    }
}

inline void writeOriginTargetMapCsv(const std::filesystem::path& path, const std::vector<int64_t>& originX,
                                     const std::vector<int64_t>& targetX, int64_t horizonHours) {
    if (originX.size() != targetX.size()) {
        throw std::invalid_argument("writeOriginTargetMapCsv: size mismatch between originX and targetX.");
    }
    for (std::size_t i = 0; i < originX.size(); ++i) {
        validateOriginTargetPair(originX[i], targetX[i], horizonHours);
        if (i > 0 && targetX[i] <= targetX[i - 1]) {
            throw std::invalid_argument("writeOriginTargetMapCsv: unordered/duplicate target identity.");
        }
    }
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("writeOriginTargetMapCsv: could not open '" + path.string() + "' for writing.");
    }
    out << "origin_x,target_x\n";
    for (std::size_t i = 0; i < originX.size(); ++i) {
        out << originX[i] << ',' << targetX[i] << '\n';
    }
    if (!out.good()) {
        throw std::runtime_error("writeOriginTargetMapCsv: write failed for '" + path.string() + "'.");
    }
}

inline std::vector<std::pair<int64_t, int64_t>> readOriginTargetMapCsv(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("readOriginTargetMapCsv: could not open '" + path.string() + "'.");
    }
    std::string header;
    std::getline(in, header);
    if (header != "origin_x,target_x") {
        throw std::invalid_argument("readOriginTargetMapCsv: unexpected header.");
    }
    std::vector<std::pair<int64_t, int64_t>> rows;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const auto comma = line.find(',');
        if (comma == std::string::npos || line.find(',', comma + 1) != std::string::npos) {
            throw std::invalid_argument("readOriginTargetMapCsv: expected exactly two fields.");
        }
        const auto origin = parseFullyConsumedInt64(line.substr(0, comma), "origin_x");
        const auto target = parseFullyConsumedInt64(line.substr(comma + 1), "target_x");
        validateOriginTargetPair(origin, target, 1);
        if (!rows.empty() && target <= rows.back().second) {
            throw std::invalid_argument("readOriginTargetMapCsv: unordered/duplicate target identity.");
        }
        rows.emplace_back(origin, target);
    }
    if (rows.empty()) {
        throw std::invalid_argument("readOriginTargetMapCsv: no data rows.");
    }
    return rows;
}

// Independently re-derives the horizon relationship from the raw pairs --
// never trusts that a loaded file already satisfies the invariant it
// claims to, matching this file's existing fail-closed discipline.
inline void validateOriginTargetMap(const std::vector<std::pair<int64_t, int64_t>>& rows, int64_t horizonHours) {
    if (rows.empty()) throw std::invalid_argument("validateOriginTargetMap: no rows.");
    int64_t previous = -1;
    for (const auto& [originX, targetX] : rows) {
        validateOriginTargetPair(originX, targetX, horizonHours);
        if (targetX <= previous) throw std::invalid_argument("validateOriginTargetMap: unordered/duplicate identities.");
        previous = targetX;
    }
}

}  // namespace hydrofusion
