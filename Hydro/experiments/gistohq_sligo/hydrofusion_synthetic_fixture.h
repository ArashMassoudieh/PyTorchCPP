#pragma once

// Deterministic synthetic hourly fixture shared by the HydroFusion
// real-engine integration test and the opt-in FFN+HydroFusion production
// wrapper (Hydro/models/ffn_hydrofusion_wrapper.h), so both exercise the
// IDENTICAL data rather than two independently-written, independently-
// fallible copies. Not a real GisToOhq package -- see
// hydrofusion_ffn_engine_integration_test.cpp's own header comment for why
// a tiny synthetic fit is what HydroFusion Task 4 closure calls for.

#include "hydrofusion_feature_builder.h"

#include <cstdint>
#include <vector>

namespace hydrofusion {

inline HourlyRow makeSyntheticFixtureRow(int64_t hour, double q) {
    HourlyRow row;
    row.x = static_cast<double>(hour);
    row.observed_runoff_mm_per_hour = q;
    row.discharge_valid = true;
    row.precipitation_mm_per_hour = 0.1 + 0.01 * static_cast<double>(hour % 7);
    row.temperature_c = 12.0 + 0.1 * static_cast<double>(hour % 24);
    row.relative_humidity_percent = 55.0;
    row.wind_speed_m_per_second = 1.5;
    row.solar_energy_mj_per_m2_per_hour = 0.8;
    row.pet_mm_per_hour = 0.04;
    row.all_forcings_valid = true;
    return row;
}

inline std::vector<HourlyRow> buildSyntheticFixture() {
    std::vector<HourlyRow> rows;
    rows.reserve(300);
    for (int64_t h = 0; h < 300; ++h) {
        const double baseflow = 0.5 + 0.002 * static_cast<double>(h);
        const double stormPhase = static_cast<double>(h % 60);
        const double storm = stormPhase < 10 ? 2.0 * std::exp(-stormPhase / 4.0) : 0.0;
        rows.push_back(makeSyntheticFixtureRow(h, baseflow + storm));
    }
    return rows;
}

}  // namespace hydrofusion
