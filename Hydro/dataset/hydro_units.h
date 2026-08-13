#pragma once

#include <stdexcept>
#include <string>

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

inline bool forecastWasAvailable(const std::string& issueTime,
                                 const std::string& predictionTime) {
    // Canonical UTC ISO-8601 timestamps sort chronologically as strings.
    return !issueTime.empty() && !predictionTime.empty() && issueTime <= predictionTime;
}
