#include "gistohq_package_adapter.h"

#include "gistohq_temporal_csv.h"

#include <algorithm>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>

namespace {
std::string readFile(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("GIStoOHQ package is missing " + path.filename().string() + ".");
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

std::string jsonString(const std::string& json, const std::vector<std::string>& keys, const char* description) {
    for (const auto& key : keys) {
        const std::regex field("\\\"" + key + "\\\"\\s*:\\s*\\\"([^\\\"]+)\\\"");
        std::smatch match;
        if (std::regex_search(json, match, field)) return match[1].str();
    }
    throw std::runtime_error(std::string("GIStoOHQ manifest is missing required ") + description +
                             "; regenerate the producer export with HydroPINNExport schema 1.2 or newer.");
}

double jsonNumber(const std::string& json, const std::vector<std::string>& keys, const char* description) {
    for (const auto& key : keys) {
        const std::regex field("\\\"" + key + "\\\"\\s*:\\s*([-+0-9.eE]+)");
        std::smatch match;
        if (std::regex_search(json, match, field)) {
            const double value = std::stod(match[1].str());
            if (std::isfinite(value) && value > 0.0) return value;
        }
    }
    throw std::runtime_error(std::string("GIStoOHQ manifest is missing positive ") + description +
                             "; the adapter will not guess it from observations.");
}

std::int64_t utcEpoch(std::string value, const bool inclusiveEnd) {
    bool dateOnly = value.size() == 10;
    if (dateOnly) value += "T00:00:00Z";
    static const std::regex canonical(R"(^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$)");
    if (!std::regex_match(value, canonical)) {
        throw std::runtime_error("GIStoOHQ manifest has invalid UTC study bound: " + value);
    }
    std::tm parsed{};
    std::istringstream stream(value);
    stream >> std::get_time(&parsed, "%Y-%m-%dT%H:%M:%SZ");
    if (stream.fail()) throw std::runtime_error("GIStoOHQ manifest has invalid UTC study bound: " + value);
    const auto epoch = static_cast<std::int64_t>(timegm(&parsed));
    // Producer study_end identifies the final included day/hour, while the
    // harmonizer uses a half-open interval [start, end).
    return epoch + (inclusiveEnd ? (dateOnly ? 86400 : 3600) : 0);
}

bool supportedExportSchema(const std::string& version) {
    std::smatch match;
    if (!std::regex_match(version, match, std::regex(R"(^([0-9]+)\.([0-9]+)(?:\.[0-9]+)?$)"))) return false;
    return std::stoi(match[1].str()) == 1 && std::stoi(match[2].str()) >= 2;
}

std::map<std::string, std::string> variableUnits(const std::string& json) {
    std::map<std::string, std::string> units;
    const std::regex object(R"(\{[^{}]*\})");
    const std::regex name(R"json("(?:name|variable|variable_id)"\s*:\s*"([^"]+)")json");
    const std::regex unit(R"json("(?:unit|units|native_unit)"\s*:\s*"([^"]+)")json");
    for (auto it = std::sregex_iterator(json.begin(), json.end(), object); it != std::sregex_iterator(); ++it) {
        std::smatch nameMatch, unitMatch;
        const std::string value = it->str();
        if (std::regex_search(value, nameMatch, name) && std::regex_search(value, unitMatch, unit)) {
            units.emplace(nameMatch[1].str(), unitMatch[1].str());
        }
    }
    if (units.empty()) throw std::runtime_error("GIStoOHQ variables.json contains no variable/unit declarations.");
    return units;
}

bool isGisToOhqHydroPinnExportImpl(const std::string& package_directory) {
    const auto manifest = readFile(std::filesystem::path(package_directory) / "manifest.json");
    try {
        return jsonString(manifest, {"schema_name"}, "schema_name") == "HydroPINNExport";
    } catch (const std::runtime_error&) {
        return false;
    }
}

GisToOhqPreparedPackage prepareGisToOhqPackageFromManifest(
    const std::string& package_directory, const bool require_observed_discharge) {
    const auto manifest = readFile(std::filesystem::path(package_directory) / "manifest.json");
    if (jsonString(manifest, {"schema_name"}, "schema_name") != "HydroPINNExport") {
        throw std::runtime_error("Package is not a GIStoOHQ HydroPINNExport.");
    }
    const auto schemaVersion = jsonString(manifest, {"schema_version"}, "schema_version");
    if (!supportedExportSchema(schemaVersion)) {
        throw std::runtime_error("Unsupported GIStoOHQ HydroPINNExport schema version " + schemaVersion +
                                 "; schema 1.2 or newer is required for authoritative study bounds.");
    }
    const auto profile = jsonString(manifest, {"profile"}, "profile");
    if (profile != "water-balance-v1") {
        throw std::runtime_error("Unsupported GIStoOHQ HydroPINNExport profile: " + profile);
    }
    GisToOhqPackageConfig config;
    config.start_epoch_seconds = utcEpoch(
        jsonString(manifest, {"study_start", "requested_start", "start_time", "start_date", "start"}, "study start"), false);
    config.end_epoch_seconds = utcEpoch(
        jsonString(manifest, {"study_end", "requested_end", "end_time", "end_date", "end"}, "study end"), true);
    config.catchment_area_m2 = jsonNumber(
        manifest, {"catchment_area_m2", "area_m2", "drainage_area_m2"}, "catchment area in m2");
    config.require_observed_discharge = require_observed_discharge;
    return prepareGisToOhqPackage(package_directory, config);
}

bool acceptedUnit(const std::string& variable, const std::string& unit) {
    static const std::map<std::string, std::set<std::string>> accepted = {
        {"PRECTOTCORR", {"mm/day", "mm/d"}}, {"T2M", {"degC", "°C", "C"}},
        {"RH2M", {"%", "percent"}}, {"WS2M", {"m/s"}},
        {"ALLSKY_SFC_SW_DWN", {"MJ/m2/h", "MJ/m²/h", "MJ/hr"}},
        {"EVPTRNS", {"MJ/m2/day", "MJ/m²/day"}}, {"00060", {"ft3/s", "ft³/s", "cfs"}}
    };
    const auto found = accepted.find(variable);
    return found != accepted.end() && found->second.count(unit) != 0;
}
}

GisToOhqPreparedPackage prepareGisToOhqPackage(
    const std::string& package_directory, const GisToOhqPackageConfig& config) {
    const std::filesystem::path root(package_directory);
    if (!std::filesystem::is_directory(root)) {
        throw std::runtime_error("GIStoOHQ package directory does not exist: " + package_directory);
    }
    (void)readFile(root / "manifest.json");
    const auto units = variableUnits(readFile(root / "variables.json"));
    const std::vector<std::string> forcing = {
        "PRECTOTCORR", "T2M", "RH2M", "WS2M", "ALLSKY_SFC_SW_DWN", "EVPTRNS"};
    for (const auto& variable : forcing) {
        const auto found = units.find(variable);
        if (found == units.end()) throw std::runtime_error("GIStoOHQ variables.json is missing " + variable + ".");
        if (!acceptedUnit(variable, found->second)) {
            throw std::runtime_error("GIStoOHQ variable " + variable + " has unsupported unit: " + found->second);
        }
    }
    const bool hasDischarge = units.count("00060") != 0;
    if (hasDischarge && !acceptedUnit("00060", units.at("00060"))) {
        throw std::runtime_error("GIStoOHQ variable 00060 has unsupported unit: " + units.at("00060"));
    }
    if (config.require_observed_discharge && !hasDischarge) {
        throw std::runtime_error("GIStoOHQ weather-only package cannot be used for supervised training.");
    }
    std::vector<std::string> temporalFiles;
    const auto observations = root / "observations";
    if (!std::filesystem::is_directory(observations)) {
        throw std::runtime_error("GIStoOHQ package is missing observations directory.");
    }
    for (const auto& entry : std::filesystem::directory_iterator(observations)) {
        if (entry.is_regular_file() && entry.path().extension() == ".csv" &&
            entry.path().filename().string().rfind("temporal_", 0) == 0) {
            temporalFiles.push_back(entry.path().string());
        }
    }
    if (temporalFiles.empty()) throw std::runtime_error("GIStoOHQ package contains no temporal CSV assets.");
    std::sort(temporalFiles.begin(), temporalFiles.end());
    const auto inputs = loadGisToOhqTemporalCsvFiles(temporalFiles);
    if (hasDischarge != !inputs.discharge_ft3_per_second.empty()) {
        throw std::runtime_error("GIStoOHQ variables.json and temporal assets disagree about 00060 availability.");
    }
    GisToOhqHourlyConfig hourlyConfig;
    hourlyConfig.start_epoch_seconds = config.start_epoch_seconds;
    hourlyConfig.end_epoch_seconds = config.end_epoch_seconds;
    hourlyConfig.catchment_area_m2 = config.catchment_area_m2;
    hourlyConfig.discharge_sample_support_seconds = config.discharge_sample_support_seconds;
    hourlyConfig.minimum_discharge_coverage = config.minimum_discharge_coverage;
    hourlyConfig.latent_heat_mj_per_kg = config.latent_heat_mj_per_kg;
    auto hourly = harmonizeGisToOhqHourly(inputs, hourlyConfig);
    auto modelRows = buildGisToOhqModelRows(hourly, config.require_observed_discharge);
    return {std::move(hourly), std::move(modelRows), hasDischarge};
}

bool isGisToOhqHydroPinnExport(const std::string& package_directory) {
    return isGisToOhqHydroPinnExportImpl(package_directory);
}

GisToOhqPreparedPackage prepareGisToOhqPackage(
    const std::string& package_directory, const bool require_observed_discharge) {
    return prepareGisToOhqPackageFromManifest(package_directory, require_observed_discharge);
}
