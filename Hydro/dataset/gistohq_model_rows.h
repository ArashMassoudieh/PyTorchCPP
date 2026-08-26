#pragma once

#include "gistohq_hourly_harmonizer.h"

#include <array>
#include <cstdint>
#include <vector>

struct GisToOhqModelRow {
    std::int64_t epoch_seconds = 0;
    double elapsed_hours = 0.0;
    // Stable feature order: P, T, RH, wind, solar, PET.
    std::array<double, 6> features{};
    double target_runoff_mm_per_hour = 0.0;
    bool target_valid = false;
    std::size_t segment_id = 0;
    std::size_t index_in_segment = 0;
};

/**
 * Selects forcing-valid hourly rows and records contiguous segment boundaries.
 * When requireObservedTarget is true, invalid discharge also breaks a segment.
 */
std::vector<GisToOhqModelRow> buildGisToOhqModelRows(
    const std::vector<GisToOhqHourlyRow>& hourly,
    bool requireObservedTarget);
