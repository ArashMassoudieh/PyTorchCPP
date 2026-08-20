#pragma once

#include "ddrr_loader.h"
#include "hydro_units.h"

#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

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
