#include "gistohq_model_rows.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace {
constexpr std::int64_t kHourSeconds = 3600;
}

std::vector<GisToOhqModelRow> buildGisToOhqModelRows(
    const std::vector<GisToOhqHourlyRow>& hourly,
    const bool requireObservedTarget) {
    if (hourly.empty()) throw std::invalid_argument("GIStoOHQ hourly table is empty.");
    for (std::size_t i = 0; i < hourly.size(); ++i) {
        if (i > 0 && hourly[i].epoch_seconds - hourly[i - 1].epoch_seconds != kHourSeconds) {
            throw std::invalid_argument("GIStoOHQ hourly rows must form a strictly regular UTC grid.");
        }
        const bool forcingFlags = hourly[i].precipitation_valid && hourly[i].temperature_valid &&
            hourly[i].relative_humidity_valid && hourly[i].wind_valid && hourly[i].solar_valid && hourly[i].pet_valid;
        if (hourly[i].all_forcings_valid != forcingFlags) {
            throw std::invalid_argument("GIStoOHQ all_forcings_valid disagrees with its component masks.");
        }
    }

    std::vector<GisToOhqModelRow> result;
    std::size_t segment = 0;
    std::size_t indexInSegment = 0;
    bool previousAccepted = false;
    for (const auto& row : hourly) {
        const bool accepted = row.all_forcings_valid && (!requireObservedTarget || row.discharge_valid);
        if (!accepted) {
            previousAccepted = false;
            continue;
        }
        const std::array<double, 6> features = {
            row.precipitation_mm_per_hour,
            row.temperature_c,
            row.relative_humidity_percent,
            row.wind_speed_m_per_second,
            row.solar_energy_mj_per_m2_per_hour,
            row.pet_mm_per_hour};
        for (const double value : features) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument("GIStoOHQ valid forcing row contains a non-finite value.");
            }
        }
        if (row.discharge_valid && !std::isfinite(row.observed_runoff_mm_per_hour)) {
            throw std::invalid_argument("GIStoOHQ valid discharge row has a non-finite runoff target.");
        }
        if (!previousAccepted) {
            if (!result.empty()) ++segment;
            indexInSegment = 0;
        }
        GisToOhqModelRow selected;
        selected.epoch_seconds = row.epoch_seconds;
        selected.elapsed_hours = static_cast<double>(row.epoch_seconds - hourly.front().epoch_seconds) /
                                 static_cast<double>(kHourSeconds);
        selected.features = features;
        selected.target_runoff_mm_per_hour = row.discharge_valid
            ? row.observed_runoff_mm_per_hour : std::numeric_limits<double>::quiet_NaN();
        selected.target_valid = row.discharge_valid;
        selected.segment_id = segment;
        selected.index_in_segment = indexInSegment++;
        result.push_back(selected);
        previousAccepted = true;
    }
    if (result.empty()) {
        throw std::runtime_error(requireObservedTarget
            ? "GIStoOHQ table has no rows with valid forcings and observed discharge."
            : "GIStoOHQ table has no rows with valid forcings.");
    }
    return result;
}
