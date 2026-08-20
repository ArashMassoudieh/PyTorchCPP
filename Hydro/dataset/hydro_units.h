#pragma once

#include <cmath>
#include <ctime>
#include <iomanip>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

inline double dischargeToDepthRate(double discharge,
                                   const std::string& dischargeUnit,
                                   double catchmentAreaM2,
                                   const std::string& outputUnit = "mm/h") {
    if (catchmentAreaM2 <= 0.0) throw std::invalid_argument("Catchment area must be positive.");
    double cubicMetersPerSecond = discharge;
    if (dischargeUnit == "ft3/s") cubicMetersPerSecond *= 0.028316846592;
    else if (dischargeUnit != "m3/s") throw std::invalid_argument("Unsupported discharge unit: " + dischargeUnit);
    const double metersPerSecond = cubicMetersPerSecond / catchmentAreaM2;
    if (outputUnit == "mm/h") return metersPerSecond * 1000.0 * 3600.0;
    if (outputUnit == "mm/s") return metersPerSecond * 1000.0;
    throw std::invalid_argument("Unsupported depth-rate unit: " + outputUnit);
}

inline std::pair<std::time_t, long> parseCanonicalUtcTimestamp(const std::string& timestamp) {
    static const std::regex pattern(R"(^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d{1,9}))?Z$)");
    std::smatch match;
    if (!std::regex_match(timestamp, match, pattern)) {
        throw std::invalid_argument("Timestamp must be canonical UTC ISO-8601: " + timestamp);
    }
    std::tm parsed{};
    std::istringstream stream(match[1].str());
    stream >> std::get_time(&parsed, "%Y-%m-%dT%H:%M:%S");
    if (stream.fail()) throw std::invalid_argument("Invalid UTC timestamp: " + timestamp);
    const int year = parsed.tm_year;
    const int month = parsed.tm_mon;
    const int day = parsed.tm_mday;
    const int hour = parsed.tm_hour;
    const int minute = parsed.tm_min;
    const int second = parsed.tm_sec;
    const std::time_t epoch = timegm(&parsed);
    const std::tm* normalized = gmtime(&epoch);
    if (!normalized || normalized->tm_year != year || normalized->tm_mon != month ||
        normalized->tm_mday != day || normalized->tm_hour != hour ||
        normalized->tm_min != minute || normalized->tm_sec != second) {
        throw std::invalid_argument("Invalid UTC calendar timestamp: " + timestamp);
    }
    std::string fractional = match[2].str();
    fractional.append(9 - fractional.size(), '0');
    const long nanoseconds = fractional.empty() ? 0L : std::stol(fractional);
    return {epoch, nanoseconds};
}

inline bool forecastWasAvailable(const std::string& issueTime,
                                 const std::string& predictionTime) {
    return parseCanonicalUtcTimestamp(issueTime) <= parseCanonicalUtcTimestamp(predictionTime);
}

inline bool forecastTimingIsConsistent(const std::string& issueTime,
                                       const std::string& validTime,
                                       double leadHours,
                                       const std::string& predictionTime,
                                       double toleranceSeconds = 1.0) {
    if (!std::isfinite(leadHours) || leadHours < 0.0 || toleranceSeconds < 0.0) return false;
    const auto issue = parseCanonicalUtcTimestamp(issueTime);
    const auto valid = parseCanonicalUtcTimestamp(validTime);
    if (issue > valid || !forecastWasAvailable(issueTime, predictionTime)) return false;
    const double elapsedSeconds = std::difftime(valid.first, issue.first) +
        static_cast<double>(valid.second - issue.second) / 1.0e9;
    return std::abs(elapsedSeconds - leadHours * 3600.0) <= toleranceSeconds;
}
