#pragma once

#include <cstdint>
#include <vector>

struct GisToOhqTimedValue {
    std::int64_t epoch_seconds = 0;
    double value = 0.0;
};

struct GisToOhqHourlyInputs {
    // Despite the field name (kept for compatibility), the unit of these
    // values is NOT assumed here - it is governed by
    // GisToOhqHourlyConfig::precipitation_is_daily_total, set by the caller
    // from each package's own declared PRECTOTCORR unit (see
    // gistohq_package_adapter.cpp). Never infer the unit from timestamp
    // cadence: a legacy package can carry a genuine daily total at
    // hourly-repeated timestamps, and an hourly-endpoint package reports
    // mm/hour at the same cadence - cadence alone cannot tell them apart.
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
    // Whether precipitation_mm_per_day values are a true daily total needing
    // /24 disaggregation to hourly, vs. already an hourly-cadence depth.
    // NASA POWER's hourly PRECTOTCORR endpoint reports mm/hour; only its
    // daily endpoint reports a true daily total (this package's legacy
    // default source). Must be derived from the package's own declared unit
    // (variables.json's native_unit for PRECTOTCORR), never from timestamp
    // spacing - a legacy daily total can still arrive at hourly-repeated
    // timestamps. Defaults to true (the legacy/original behavior) so any
    // caller that does not explicitly classify the unit keeps prior
    // behavior rather than silently changing under it.
    bool precipitation_is_daily_total = true;
    // USGS discharge telemetry occasionally drops out for a few hours for reasons
    // unrelated to the hydrologic process (radio/logger hiccups); linearly
    // interpolating across gaps at or below this bound is standard provisional-data
    // practice and avoids fracturing an otherwise-valid record into short unusable
    // segments. Gaps longer than this (e.g. a gauge knocked out during a flood) are
    // left untouched so a real storm hydrograph is never fabricated by interpolation.
    double max_interpolated_discharge_gap_seconds = 10800.0;
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
