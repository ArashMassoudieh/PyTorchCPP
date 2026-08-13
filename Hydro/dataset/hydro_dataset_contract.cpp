#include "hydro_dataset_contract.h"

#include <cmath>
#include <fstream>
#include <regex>
#include <set>
#include <sstream>

namespace {
std::vector<std::string> splitCsv(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream stream(line);
    std::string field;
    while (std::getline(stream, field, ',')) fields.push_back(field);
    if (!line.empty() && line.back() == ',') fields.emplace_back();
    return fields;
}

bool finiteNumber(const std::string& text, double& value) {
    try {
        std::size_t parsed = 0;
        value = std::stod(text, &parsed);
        return parsed == text.size() && std::isfinite(value);
    } catch (...) {
        return false;
    }
}

bool canonicalUtcTimestamp(const std::string& text) {
    static const std::regex pattern(
        R"(^[0-9]{4}-(0[1-9]|1[0-2])-([0-2][0-9]|3[01])T([01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9](\.[0-9]+)?Z$)");
    return std::regex_match(text, pattern);
}
}

HydroDatasetContract HydroDatasetContract::observationsV1(bool requireStorage) {
    HydroDatasetContract contract;
    contract.variables = {
        {"timestamp", "UTC ISO-8601", "timestamp", true, false},
        {"catchment_id", "1", "identifier", true, false},
        {"precipitation", "mm/h", "forcing", true, true},
        {"potential_et", "mm/h", "forcing", true, true},
        {"observed_discharge", "m3/s", "target", true, true},
        {"storage", "mm", "state", requireStorage, false}
    };
    return contract;
}

HydroDatasetValidation HydroDatasetValidator::validateCsv(const std::string& path,
                                                           const HydroDatasetContract& contract,
                                                           bool hasHeader) const {
    HydroDatasetValidation result;
    if (!hasHeader) {
        result.errors.push_back("Canonical HydroPINN CSV files require a header row.");
        return result;
    }
    std::ifstream input(path);
    if (!input) {
        result.errors.push_back("Unable to open dataset: " + path);
        return result;
    }
    std::string line;
    if (!std::getline(input, line)) {
        result.errors.push_back("Dataset is empty.");
        return result;
    }
    const auto header = splitCsv(line);
    std::map<std::string, std::size_t> columns;
    for (std::size_t i = 0; i < header.size(); ++i) {
        if (!columns.emplace(header[i], i).second) result.errors.push_back("Duplicate column: " + header[i]);
    }
    for (const auto& variable : contract.variables) {
        if (variable.required && columns.find(variable.name) == columns.end()) {
            result.errors.push_back("Missing required column: " + variable.name);
        }
        if (variable.required && variable.unit.empty()) {
            result.errors.push_back("Required variable has no declared unit: " + variable.name);
        }
    }
    if (!result.errors.empty()) return result;

    std::map<std::string, std::string> previousTimestampByCatchment;
    std::map<std::string, std::size_t> rowCountByCatchment;
    std::set<std::pair<std::string, std::string>> observationKeys;
    std::size_t lineNumber = 1;
    while (std::getline(input, line)) {
        ++lineNumber;
        if (line.empty()) continue;
        const auto fields = splitCsv(line);
        if (fields.size() != header.size()) {
            result.errors.push_back("Row " + std::to_string(lineNumber) + " has an inconsistent column count.");
            continue;
        }
        const std::string& timestamp = fields[columns.at("timestamp")];
        const std::string& catchment = fields[columns.at("catchment_id")];
        if (catchment.empty()) {
            result.errors.push_back("Row " + std::to_string(lineNumber) + " has an empty catchment_id.");
        } else {
            ++rowCountByCatchment[catchment];
        }
        if (!canonicalUtcTimestamp(timestamp)) {
            result.errors.push_back("Row " + std::to_string(lineNumber) + " timestamp is not canonical UTC ISO-8601.");
        }
        if (!catchment.empty() && !timestamp.empty()) {
            if (!observationKeys.emplace(catchment, timestamp).second) {
                result.errors.push_back("Row " + std::to_string(lineNumber) + " duplicates catchment_id/timestamp key.");
            }
            const auto previous = previousTimestampByCatchment.find(catchment);
            if (previous != previousTimestampByCatchment.end() && timestamp <= previous->second) {
                result.errors.push_back("Row " + std::to_string(lineNumber) + " timestamp is non-increasing within catchment " + catchment + ".");
            }
            previousTimestampByCatchment[catchment] = timestamp;
        }
        for (const auto& variable : contract.variables) {
            if (variable.role == "timestamp" || variable.role == "identifier") continue;
            const auto it = columns.find(variable.name);
            if (it == columns.end()) continue;
            double value = 0.0;
            if (!finiteNumber(fields[it->second], value)) {
                result.errors.push_back("Row " + std::to_string(lineNumber) + " has a non-finite " + variable.name + ".");
            } else if (variable.nonnegative && value < 0.0) {
                result.errors.push_back("Row " + std::to_string(lineNumber) + " has negative " + variable.name + ".");
            }
        }
        ++result.row_count;
    }
    if (result.row_count < 3) result.errors.push_back("Dataset requires at least three data rows.");
    for (const auto& entry : rowCountByCatchment) {
        if (entry.second < 3) {
            result.errors.push_back("Catchment " + entry.first + " requires at least three data rows.");
        }
    }
    result.valid = result.errors.empty();
    return result;
}
