#pragma once

#include "ddrr_loader.h"
#include "hydro_units.h"

#include <optional>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

struct AlignedForecastFeature {
    std::vector<double> values;
    std::string unit;
};

inline std::optional<HydroForecast> selectLatestAvailableForecast(
    const std::vector<HydroForecast>& forecasts,
    const std::string& catchmentId,
    const std::string& variable,
    const std::string& validTime,
    const std::string& predictionTime,
    const std::string& ensembleMember = {}) {
    const auto requestedValid = parseCanonicalUtcTimestamp(validTime);
    (void)parseCanonicalUtcTimestamp(predictionTime);
    std::optional<HydroForecast> selected;
    std::optional<std::string> selectedUnit;
    for (const auto& forecast : forecasts) {
        if (forecast.catchment_id != catchmentId || forecast.variable != variable ||
            (!ensembleMember.empty() && forecast.ensemble_member != ensembleMember) ||
            parseCanonicalUtcTimestamp(forecast.valid_time) != requestedValid ||
            !forecastWasAvailable(forecast.issue_time, predictionTime)) {
            continue;
        }
        if (selectedUnit && *selectedUnit != forecast.unit) {
            throw std::runtime_error("Available forecast candidates use inconsistent units for " + variable + ".");
        }
        selectedUnit = forecast.unit;
        if (!selected || parseCanonicalUtcTimestamp(selected->issue_time) < parseCanonicalUtcTimestamp(forecast.issue_time)) {
            selected = forecast;
        }
    }
    return selected;
}

inline AlignedForecastFeature buildAlignedForecastFeature(
    const std::vector<HydroForecast>& forecasts,
    const std::vector<std::string>& targetValidTimes,
    const std::string& catchmentId,
    const std::string& variable,
    double requiredLeadHours,
    const std::string& ensembleMember = {}) {
    if (!std::isfinite(requiredLeadHours) || requiredLeadHours < 0.0) {
        throw std::invalid_argument("Forecast feature lead must be finite and nonnegative.");
    }
    AlignedForecastFeature aligned;
    aligned.values.reserve(targetValidTimes.size());
    for (const auto& validTime : targetValidTimes) {
        const auto valid = parseCanonicalUtcTimestamp(validTime);
        const double cutoff = static_cast<double>(valid.first) + static_cast<double>(valid.second) / 1.0e9 -
                              requiredLeadHours * 3600.0;
        std::optional<HydroForecast> selected;
        for (const auto& forecast : forecasts) {
            if (forecast.catchment_id != catchmentId || forecast.variable != variable ||
                (!ensembleMember.empty() && forecast.ensemble_member != ensembleMember) ||
                parseCanonicalUtcTimestamp(forecast.valid_time) != valid) {
                continue;
            }
            const auto issue = parseCanonicalUtcTimestamp(forecast.issue_time);
            const double issueSeconds = static_cast<double>(issue.first) + static_cast<double>(issue.second) / 1.0e9;
            if (issueSeconds > cutoff) continue;
            if (!aligned.unit.empty() && aligned.unit != forecast.unit) {
                throw std::runtime_error("Aligned forecast feature changes unit for " + variable + ".");
            }
            if (!selected || parseCanonicalUtcTimestamp(selected->issue_time) < issue) selected = forecast;
        }
        if (!selected) {
            throw std::runtime_error("No leakage-safe forecast is available for " + variable + " at valid time " + validTime + ".");
        }
        aligned.unit = selected->unit;
        aligned.values.push_back(selected->value);
    }
    return aligned;
}
