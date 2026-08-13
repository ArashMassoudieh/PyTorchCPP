#include "ddrr_loader.h"

#include "hydro_units.h"

#include <ctime>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace {
std::vector<std::string> splitCsv(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream stream(line);
    std::string field;
    while (std::getline(stream, field, ',')) fields.push_back(field);
    if (!line.empty() && line.back() == ',') fields.emplace_back();
    return fields;
}

double parseUtcSeconds(const std::string& value) {
    std::tm parsed{};
    std::istringstream stream(value.substr(0, 19));
    stream >> std::get_time(&parsed, "%Y-%m-%dT%H:%M:%S");
    if (stream.fail()) throw std::runtime_error("Invalid canonical timestamp: " + value);
    double seconds = static_cast<double>(timegm(&parsed));
    const auto dot = value.find('.', 19);
    if (dot != std::string::npos) {
        const auto end = value.find('Z', dot);
        seconds += std::stod("0" + value.substr(dot, end - dot));
    }
    return seconds;
}

std::map<std::string, double> loadCatchmentAreas(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Unable to open catchment attributes: " + path.string());
    std::string line;
    if (!std::getline(input, line)) throw std::runtime_error("Catchment attributes file is empty.");
    const auto header = splitCsv(line);
    std::map<std::string, std::size_t> columns;
    for (std::size_t i = 0; i < header.size(); ++i) columns[header[i]] = i;
    if (columns.find("catchment_id") == columns.end() || columns.find("area_m2") == columns.end()) {
        throw std::runtime_error("catchment_attributes.csv requires catchment_id and area_m2 columns.");
    }
    std::map<std::string, double> areas;
    std::size_t lineNumber = 1;
    while (std::getline(input, line)) {
        ++lineNumber;
        if (line.empty()) continue;
        const auto fields = splitCsv(line);
        if (fields.size() != header.size()) {
            throw std::runtime_error("Catchment attributes row " + std::to_string(lineNumber) + " has inconsistent columns.");
        }
        const std::string id = fields.at(columns.at("catchment_id"));
        if (id.empty()) throw std::runtime_error("Catchment attributes contain an empty catchment_id.");
        double area = 0.0;
        try {
            std::size_t parsed = 0;
            const std::string& text = fields.at(columns.at("area_m2"));
            area = std::stod(text, &parsed);
            if (parsed != text.size()) throw std::invalid_argument("trailing characters");
        } catch (...) {
            throw std::runtime_error("Invalid area_m2 for catchment " + id + ".");
        }
        if (!std::isfinite(area) || area <= 0.0) throw std::runtime_error("area_m2 must be finite and positive for catchment " + id + ".");
        if (!areas.emplace(id, area).second) throw std::runtime_error("Duplicate catchment_id in attributes: " + id);
    }
    if (areas.empty()) throw std::runtime_error("Catchment attributes contain no data rows.");
    return areas;
}

std::string readText(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Unable to open file: " + path.string());
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

std::string jsonString(const std::string& json, const std::string& key, bool required = true) {
    const std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*\\\"([^\\\"]*)\\\"");
    std::smatch match;
    if (std::regex_search(json, match, pattern)) return match[1].str();
    if (required) throw std::runtime_error("Manifest is missing string field: " + key);
    return {};
}

int semanticMajor(const std::string& version) {
    const auto dot = version.find('.');
    const std::string major = version.substr(0, dot);
    if (major.empty()) throw std::runtime_error("Schema version has no major component.");
    try {
        std::size_t parsed = 0;
        const int value = std::stoi(major, &parsed);
        if (parsed != major.size() || value < 0) throw std::invalid_argument("invalid major");
        return value;
    } catch (...) {
        throw std::runtime_error("Invalid semantic schema version: " + version);
    }
}

bool safeRelativePath(const std::string& value) {
    if (value.empty()) return true;
    const std::filesystem::path path(value);
    if (path.is_absolute()) return false;
    for (const auto& component : path) {
        if (component == "..") return false;
    }
    return true;
}

void rejectPackageQcErrors(const std::filesystem::path& path) {
    if (path.empty()) return;
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Manifest references missing quality-control file: " + path.string());
    std::string line;
    if (!std::getline(input, line)) throw std::runtime_error("Quality-control file is empty.");
    const auto header = splitCsv(line);
    std::map<std::string, std::size_t> columns;
    for (std::size_t i = 0; i < header.size(); ++i) columns[header[i]] = i;
    if (columns.find("severity") == columns.end() || columns.find("rule_id") == columns.end()) {
        throw std::runtime_error("quality_control.csv requires rule_id and severity columns.");
    }
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        const auto fields = splitCsv(line);
        if (fields.size() != header.size()) throw std::runtime_error("Quality-control row has inconsistent columns.");
        const std::string severity = fields.at(columns.at("severity"));
        if (severity != "error" && severity != "warning" && severity != "information") {
            throw std::runtime_error("Unknown QC severity: " + severity);
        }
        if (severity == "error") {
            throw std::runtime_error("Package contains unresolved QC error: " + fields.at(columns.at("rule_id")));
        }
    }
}
}

bool DDRRLoader::load(const std::string& path) {
    return HydroDatasetValidator().validateCsv(path, HydroDatasetContract::rainfallRunoffV1()).valid;
}

HydroObservationDataset DDRRLoader::loadObservations(
    const std::string& path,
    const std::map<std::string, double>& catchmentAreasM2,
    const HydroDatasetContract& contract) const {
    const HydroDatasetValidation validation = HydroDatasetValidator().validateCsv(path, contract);
    if (!validation.valid) {
        std::string message = "Hydro observation validation failed";
        for (const auto& error : validation.errors) message += "\n- " + error;
        throw std::runtime_error(message);
    }
    std::ifstream input(path);
    std::string line;
    std::getline(input, line);
    const auto header = splitCsv(line);
    std::map<std::string, std::size_t> columns;
    for (std::size_t i = 0; i < header.size(); ++i) columns[header[i]] = i;

    HydroObservationDataset dataset;
    dataset.catchment_area_m2 = catchmentAreasM2;
    std::map<std::string, double> startByCatchment;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        const auto fields = splitCsv(line);
        HydroObservation observation;
        observation.timestamp = fields.at(columns.at("timestamp"));
        observation.catchment_id = fields.at(columns.at("catchment_id"));
        const auto area = catchmentAreasM2.find(observation.catchment_id);
        if (area == catchmentAreasM2.end() || area->second <= 0.0) {
            throw std::runtime_error("Missing positive catchment area for " + observation.catchment_id);
        }
        const double time = parseUtcSeconds(observation.timestamp);
        if (startByCatchment.find(observation.catchment_id) == startByCatchment.end()) {
            startByCatchment[observation.catchment_id] = time;
        }
        observation.elapsed_hours = (time - startByCatchment.at(observation.catchment_id)) / 3600.0;
        observation.precipitation_mm_per_hour = std::stod(fields.at(columns.at("precipitation")));
        observation.potential_et_mm_per_hour = std::stod(fields.at(columns.at("potential_et")));
        observation.observed_discharge_m3_per_second = std::stod(fields.at(columns.at("observed_discharge")));
        observation.observed_runoff_mm_per_hour = dischargeToDepthRate(
            observation.observed_discharge_m3_per_second, "m3/s", area->second);
        const auto storage = columns.find("storage");
        if (storage != columns.end() && !fields.at(storage->second).empty()) {
            observation.storage_mm = std::stod(fields.at(storage->second));
        }
        dataset.observations_by_catchment[observation.catchment_id].push_back(observation);
    }
    return dataset;
}

HydroObservationDataset DDRRLoader::loadPackageDirectory(
    const std::string& packageDirectory,
    const HydroDatasetContract& contract) const {
    const std::filesystem::path root(packageDirectory);
    if (!std::filesystem::is_directory(root)) {
        throw std::runtime_error("Hydro package directory does not exist: " + packageDirectory);
    }
    const auto manifestPath = root / "manifest.json";
    if (!std::filesystem::is_regular_file(manifestPath)) {
        throw std::runtime_error("Hydro package is missing manifest.json.");
    }
    const HydroPackageManifest manifest = loadManifest(manifestPath.string());
    if (manifest.schema_name != contract.schema_name) throw std::runtime_error("Package schema_name is incompatible with the selected contract.");
    if (semanticMajor(manifest.schema_version) != semanticMajor(contract.schema_version)) {
        throw std::runtime_error("Package schema major version is unsupported.");
    }
    if (manifest.profile != contract.profile) throw std::runtime_error("Package profile does not match the selected contract.");
    const auto observations = root / manifest.observations_file;
    const auto attributes = root / manifest.catchment_attributes_file;
    if (!std::filesystem::is_regular_file(observations)) {
        throw std::runtime_error("Hydro package is missing observations.csv.");
    }
    if (!std::filesystem::is_regular_file(attributes)) {
        throw std::runtime_error("Hydro package is missing catchment_attributes.csv.");
    }
    if (!manifest.quality_control_file.empty()) rejectPackageQcErrors(root / manifest.quality_control_file);
    HydroObservationDataset dataset = loadObservations(observations.string(), loadCatchmentAreas(attributes), contract);
    dataset.dataset_id = manifest.dataset_id;
    dataset.schema_version = manifest.schema_version;
    dataset.profile = manifest.profile;
    return dataset;
}

HydroPackageManifest DDRRLoader::loadManifest(const std::string& manifestPath) const {
    const std::string json = readText(manifestPath);
    HydroPackageManifest manifest;
    manifest.schema_name = jsonString(json, "schema_name");
    manifest.schema_version = jsonString(json, "schema_version");
    manifest.profile = jsonString(json, "profile");
    manifest.dataset_id = jsonString(json, "dataset_id");
    manifest.observations_file = jsonString(json, "observations_file");
    manifest.catchment_attributes_file = jsonString(json, "catchment_attributes_file");
    manifest.quality_control_file = jsonString(json, "quality_control_file", false);
    for (const auto& relative : {manifest.observations_file, manifest.catchment_attributes_file, manifest.quality_control_file}) {
        if (!safeRelativePath(relative)) {
            throw std::runtime_error("Manifest file paths must remain within the package directory.");
        }
    }
    return manifest;
}
