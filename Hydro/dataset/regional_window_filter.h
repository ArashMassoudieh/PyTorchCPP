#pragma once

#include <cstdint>
#include <ctime>
#include <regex>
#include <stdexcept>
#include <string>
#include <utility>

// Parses a UTC ISO8601 timestamp "YYYY-MM-DDTHH:MM:SSZ" into Unix epoch
// seconds. Shared, independently tested - do not reimplement this parsing
// inline elsewhere (this project has already been burned once this session
// by two independently-written pieces of similar logic silently disagreeing
// - see resolve_test_end_x.h's doc comment for that incident).
inline int64_t parseUtcIso8601ToEpochSeconds(const std::string& iso) {
    static const std::regex pattern(R"(^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})Z$)");
    std::smatch match;
    if (!std::regex_match(iso, match, pattern)) {
        throw std::invalid_argument("Expected UTC ISO8601 'YYYY-MM-DDTHH:MM:SSZ', got: " + iso);
    }
    std::tm tm{};
    tm.tm_year = std::stoi(match[1].str()) - 1900;
    tm.tm_mon = std::stoi(match[2].str()) - 1;
    tm.tm_mday = std::stoi(match[3].str());
    tm.tm_hour = std::stoi(match[4].str());
    tm.tm_min = std::stoi(match[5].str());
    tm.tm_sec = std::stoi(match[6].str());
    return static_cast<int64_t>(timegm(&tm));
}

// Converts a desired calendar window [windowStartIso, windowEndIso) into an
// elapsed-hours bound [startHours, endHours) in the SAME coordinate space
// loadHydroPackageTensors's plotX uses for a GIStoOHQ package - hours since
// that package's own first row (gistohq_model_rows.cpp:
// elapsed_hours = (row.epoch_seconds - hourly.front().epoch_seconds)/3600),
// given that package's manifest.json study_start.
//
// Correctness depends on hourly.front() actually corresponding to the
// manifest's declared study_start - true for every package
// ingest_camelsh_basin.py produces (it writes every hour from study_start
// onward with no leading gap, confirmed directly against 5 real extracted
// basins - see 01_camelsh_variable_mapping.md's "Status" section). A
// package whose first valid row starts later than its own declared
// study_start (e.g. missing data trimmed from the front) would need a
// different reference point - not assumed here, and not the case for any
// package this function has been used against so far.
inline std::pair<double, double> elapsedHoursWindowFromStudyStart(
    const std::string& manifestStudyStartIso, const std::string& windowStartIso, const std::string& windowEndIso) {
    const int64_t studyStart = parseUtcIso8601ToEpochSeconds(manifestStudyStartIso);
    const int64_t windowStart = parseUtcIso8601ToEpochSeconds(windowStartIso);
    const int64_t windowEnd = parseUtcIso8601ToEpochSeconds(windowEndIso);
    if (windowEnd <= windowStart) throw std::invalid_argument("Window end must be after window start.");
    const double startHours = static_cast<double>(windowStart - studyStart) / 3600.0;
    const double endHours = static_cast<double>(windowEnd - studyStart) / 3600.0;
    return {startHours, endHours};
}
