#include "ddrr_loader.h"

#include "hydro_units.h"

#include <ctime>
#include <fstream>
#include <iomanip>
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

std::time_t parseUtc(const std::string& value) {
    std::tm parsed{};
    std::istringstream stream(value.substr(0, 19));
    stream >> std::get_time(&parsed, "%Y-%m-%dT%H:%M:%S");
    if (stream.fail()) throw std::runtime_error("Invalid canonical timestamp: " + value);
    return timegm(&parsed);
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
    std::map<std::string, std::time_t> startByCatchment;
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
        const std::time_t time = parseUtc(observation.timestamp);
        if (startByCatchment.find(observation.catchment_id) == startByCatchment.end()) {
            startByCatchment[observation.catchment_id] = time;
        }
        observation.elapsed_hours = std::difftime(time, startByCatchment.at(observation.catchment_id)) / 3600.0;
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
