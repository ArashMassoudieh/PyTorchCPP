#pragma once

#include <cstdint>
#include <vector>

struct GisToOhqTimedValue {
    std::int64_t epoch_seconds = 0;
    double value = 0.0;
};

struct GisToOhqHourlyInputs {
    std::vector<GisToOhqTimedValue> precipitation_mm_per_day;
    std::vector<GisToOhqTimedValue> temperature_c;
    std::vector<GisToOhqTimedValue> relative_humidity_percent;
    std::vector<GisToOhqTimedValue> wind_speed_m_per_second;
    std::vector<GisToOhqTimedValue> solar_energy_mj_per_m2;
    std::vector<GisToOhqTimedValue> pet_energy_mj_per_m2_per_day;
    std::vector<GisToOhqTimedValue> discharge_ft3_per_second;
};

struct GisToOhqHourlyConfig {
    std::int64_t start_epoch_seconds = 0;
    std::int64_t end_epoch_seconds = 0; // exclusive
    double catchment_area_m2 = 0.0;
    double discharge_sample_support_seconds = 300.0;
    double minimum_discharge_coverage = 0.75;
    double latent_heat_mj_per_kg = 2.45;
};

struct GisToOhqHourlyRow {
    std::int64_t epoch_seconds = 0;
    double precipitation_mm_per_hour = 0.0;
    double temperature_c = 0.0;
    double relative_humidity_percent = 0.0;
    double wind_speed_m_per_second = 0.0;
    double solar_energy_mj_per_m2_per_hour = 0.0;
    double pet_mm_per_hour = 0.0;
    double observed_discharge_m3_per_second = 0.0;
    double observed_runoff_mm_per_hour = 0.0;
    double discharge_coverage = 0.0;
    bool precipitation_valid = false;
    bool temperature_valid = false;
    bool relative_humidity_valid = false;
    bool wind_valid = false;
    bool solar_valid = false;
    bool pet_valid = false;
    bool discharge_valid = false;
    bool all_forcings_valid = false;
};

/**
 * Harmonizes native GIStoOHQ forcing/PET/discharge series onto UTC hourly bins.
 * Invalid outputs remain NaN and are accompanied by explicit validity flags.
 */
std::vector<GisToOhqHourlyRow> harmonizeGisToOhqHourly(
    const GisToOhqHourlyInputs& inputs,
    const GisToOhqHourlyConfig& config);
