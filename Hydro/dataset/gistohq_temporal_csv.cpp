#include "gistohq_temporal_csv.h"

#include <algorithm>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace {
std::vector<std::string> parseCsvRow(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    bool quoted = false;
    bool closed = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (quoted) {
            if (c == '"' && i + 1 < line.size() && line[i + 1] == '"') { field.push_back('"'); ++i; }
            else if (c == '"') { quoted = false; closed = true; }
            else field.push_back(c);
        } else if (c == ',') {
            fields.push_back(field); field.clear(); closed = false;
        } else if (c == '"' && field.empty() && !closed) quoted = true;
        else if (c != '\r' || i + 1 != line.size()) {
            if (closed || c == '"') throw std::runtime_error("Temporal CSV contains malformed quoting.");
            field.push_back(c);
        }
    }
    if (quoted) throw std::runtime_error("Temporal CSV contains an unterminated quoted field.");
    fields.push_back(field);
    return fields;
}

std::int64_t parseTimestamp(const std::string& value) {
    static const std::regex canonical(
        R"(^([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2}):([0-9]{2}):([0-9]{2})(?:\.([0-9]+))?Z$)");
    std::smatch match;
    if (!std::regex_match(value, match, canonical)) {
        throw std::runtime_error("Temporal CSV timestamp is not canonical UTC: " + value);
    }
    if (match[7].matched && match[7].str().find_first_not_of('0') != std::string::npos) {
        throw std::runtime_error("Temporal CSV sub-second timestamps are unsupported by the hourly adapter: " + value);
    }
    std::tm parsed{};
    parsed.tm_year = std::stoi(match[1].str()) - 1900;
    parsed.tm_mon = std::stoi(match[2].str()) - 1;
    parsed.tm_mday = std::stoi(match[3].str());
    parsed.tm_hour = std::stoi(match[4].str());
    parsed.tm_min = std::stoi(match[5].str());
    parsed.tm_sec = std::stoi(match[6].str());
    const std::tm requested = parsed;
    const std::time_t epoch = timegm(&parsed);
    std::tm roundTrip{};
    gmtime_r(&epoch, &roundTrip);
    if (roundTrip.tm_year != requested.tm_year || roundTrip.tm_mon != requested.tm_mon ||
        roundTrip.tm_mday != requested.tm_mday || roundTrip.tm_hour != requested.tm_hour ||
        roundTrip.tm_min != requested.tm_min || roundTrip.tm_sec != requested.tm_sec) {
        throw std::runtime_error("Temporal CSV timestamp contains an invalid calendar time: " + value);
    }
    return static_cast<std::int64_t>(epoch);
}

double parseValue(const std::string& text, const std::string& variable, std::size_t line) {
    try {
        std::size_t consumed = 0;
        const double value = std::stod(text, &consumed);
        if (consumed != text.size() || !std::isfinite(value)) throw std::invalid_argument("invalid");
        return value;
    } catch (...) {
        throw std::runtime_error("Temporal CSV has invalid " + variable + " value on line " +
                                 std::to_string(line) + ".");
    }
}

std::vector<GisToOhqTimedValue>* series(GisToOhqHourlyInputs& inputs, const std::string& variable) {
    if (variable == "PRECTOTCORR") return &inputs.precipitation_mm_per_day;
    if (variable == "T2M") return &inputs.temperature_c;
    if (variable == "RH2M") return &inputs.relative_humidity_percent;
    if (variable == "WS2M") return &inputs.wind_speed_m_per_second;
    if (variable == "ALLSKY_SFC_SW_DWN") return &inputs.solar_energy_mj_per_m2;
    if (variable == "EVPTRNS") return &inputs.pet_energy_mj_per_m2_per_day;
    if (variable == "00060") return &inputs.discharge_ft3_per_second;
    return nullptr;
}

void sortAndRejectDuplicates(std::vector<GisToOhqTimedValue>& values, const char* variable) {
    std::sort(values.begin(), values.end(), [](const auto& left, const auto& right) {
        return left.epoch_seconds < right.epoch_seconds;
    });
    for (std::size_t i = 1; i < values.size(); ++i) {
        if (values[i].epoch_seconds == values[i - 1].epoch_seconds) {
            throw std::runtime_error(std::string("Duplicate ") + variable + " timestamp across temporal assets.");
        }
    }
}
}

GisToOhqHourlyInputs loadGisToOhqTemporalCsvFiles(const std::vector<std::string>& paths) {
    if (paths.empty()) throw std::invalid_argument("GIStoOHQ temporal CSV file list is empty.");
    GisToOhqHourlyInputs inputs;
    for (const auto& path : paths) {
        std::ifstream file(path);
        if (!file) throw std::runtime_error("Unable to open GIStoOHQ temporal CSV: " + path);
        std::string line;
        if (!std::getline(file, line)) throw std::runtime_error("GIStoOHQ temporal CSV is empty: " + path);
        const auto header = parseCsvRow(line);
        std::map<std::string, std::size_t> columns;
        for (std::size_t i = 0; i < header.size(); ++i) {
            if (!columns.emplace(header[i], i).second) throw std::runtime_error("Temporal CSV has a duplicate column: " + header[i]);
        }
        auto timestamp = columns.find("timestamp_utc");
        if (timestamp == columns.end()) timestamp = columns.find("timestamp");
        if (timestamp == columns.end()) {
            throw std::runtime_error("Temporal CSV requires a timestamp_utc or timestamp column.");
        }
        const bool longForm = columns.count("variable") && columns.count("value");
        std::vector<std::pair<std::string, std::size_t>> wideVariables;
        if (!longForm) {
            for (const auto& column : columns) if (series(inputs, column.first)) wideVariables.push_back(column);
            if (wideVariables.empty()) throw std::runtime_error("Temporal CSV contains no HydroPINN contract variables.");
        }
        std::size_t lineNumber = 1;
        while (std::getline(file, line)) {
            ++lineNumber;
            if (line.empty()) continue;
            const auto fields = parseCsvRow(line);
            if (fields.size() != header.size()) {
                throw std::runtime_error("Temporal CSV row " + std::to_string(lineNumber) + " has inconsistent columns.");
            }
            const auto epoch = parseTimestamp(fields[timestamp->second]);
            if (longForm) {
                const std::string& variable = fields[columns.at("variable")];
                auto* destination = series(inputs, variable);
                if (!destination) throw std::runtime_error("Temporal CSV contains unsupported variable: " + variable);
                const std::string& value = fields[columns.at("value")];
                if (!value.empty()) destination->push_back({epoch, parseValue(value, variable, lineNumber)});
            } else {
                for (const auto& variable : wideVariables) {
                    const std::string& value = fields[variable.second];
                    if (!value.empty()) series(inputs, variable.first)->push_back(
                        {epoch, parseValue(value, variable.first, lineNumber)});
                }
            }
        }
        if (file.bad()) throw std::runtime_error("Unable to read GIStoOHQ temporal CSV: " + path);
    }
    sortAndRejectDuplicates(inputs.precipitation_mm_per_day, "PRECTOTCORR");
    sortAndRejectDuplicates(inputs.temperature_c, "T2M");
    sortAndRejectDuplicates(inputs.relative_humidity_percent, "RH2M");
    sortAndRejectDuplicates(inputs.wind_speed_m_per_second, "WS2M");
    sortAndRejectDuplicates(inputs.solar_energy_mj_per_m2, "ALLSKY_SFC_SW_DWN");
    sortAndRejectDuplicates(inputs.pet_energy_mj_per_m2_per_day, "EVPTRNS");
    sortAndRejectDuplicates(inputs.discharge_ft3_per_second, "00060");
    return inputs;
}
