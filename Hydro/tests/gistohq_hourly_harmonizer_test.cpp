#include "../dataset/gistohq_hourly_harmonizer.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>

namespace {
bool close(double left, double right, double tolerance = 1.0e-10) {
    return std::abs(left - right) <= tolerance * std::max({1.0, std::abs(left), std::abs(right)});
}
}

int main() {
    GisToOhqHourlyInputs inputs;
    for (std::int64_t hour = 0; hour < 3; ++hour) {
        const auto time = hour * 3600;
        inputs.precipitation_mm_per_day.push_back({time, 24.0 + hour});
        inputs.temperature_c.push_back({time, 10.0 + hour});
        inputs.relative_humidity_percent.push_back({time, 50.0});
        inputs.wind_speed_m_per_second.push_back({time, 2.0});
        inputs.solar_energy_mj_per_m2.push_back({time, 0.5});
    }
    inputs.pet_energy_mj_per_m2_per_day.push_back({0, 58.8}); // 1 mm/h at 2.45 MJ/kg
    for (int sample = 0; sample < 12; ++sample) {
        inputs.discharge_ft3_per_second.push_back({3600 + sample * 300, 100.0});
    }
    for (int sample = 0; sample < 6; ++sample) {
        inputs.discharge_ft3_per_second.push_back({7200 + sample * 300, 200.0});
    }
    GisToOhqHourlyConfig config;
    config.start_epoch_seconds = 0;
    config.end_epoch_seconds = 3 * 3600;
    config.catchment_area_m2 = 1.0e6;
    const auto rows = harmonizeGisToOhqHourly(inputs, config);
    assert(rows.size() == 3);
    assert(close(rows[0].precipitation_mm_per_hour, 1.0));
    assert(close(rows[0].pet_mm_per_hour, 1.0));
    assert(rows[0].all_forcings_valid);
    assert(!rows[0].discharge_valid && std::isnan(rows[0].observed_discharge_m3_per_second));
    assert(rows[1].discharge_valid && close(rows[1].discharge_coverage, 1.0));
    assert(close(rows[1].observed_discharge_m3_per_second, 2.8316846592));
    assert(close(rows[1].observed_runoff_mm_per_hour, 10.19406477312));
    assert(!rows[2].discharge_valid && close(rows[2].discharge_coverage, 0.5));
    assert(std::isnan(rows[2].observed_runoff_mm_per_hour));

    auto duplicate = inputs;
    duplicate.temperature_c.insert(duplicate.temperature_c.begin() + 1, duplicate.temperature_c.front());
    bool rejectedDuplicate = false;
    try { (void)harmonizeGisToOhqHourly(duplicate, config); }
    catch (const std::invalid_argument&) { rejectedDuplicate = true; }
    assert(rejectedDuplicate);

    auto invalidHumidity = inputs;
    invalidHumidity.relative_humidity_percent[0].value = 101.0;
    bool rejectedHumidity = false;
    try { (void)harmonizeGisToOhqHourly(invalidHumidity, config); }
    catch (const std::invalid_argument&) { rejectedHumidity = true; }
    assert(rejectedHumidity);

    // A short (<=3h) discharge dropout between two real samples should be
    // linearly interpolated and not fracture hourly coverage, while a longer
    // dropout (a gauge knocked out during a flood) must remain a real gap.
    GisToOhqHourlyInputs gapInputs;
    for (std::int64_t hour = 0; hour < 8; ++hour) {
        const auto time = hour * 3600;
        gapInputs.precipitation_mm_per_day.push_back({time, 24.0});
        gapInputs.temperature_c.push_back({time, 10.0});
        gapInputs.relative_humidity_percent.push_back({time, 50.0});
        gapInputs.wind_speed_m_per_second.push_back({time, 2.0});
        gapInputs.solar_energy_mj_per_m2.push_back({time, 0.5});
    }
    gapInputs.pet_energy_mj_per_m2_per_day.push_back({0, 58.8});
    // Hour 0: dense 5-min samples at 100 ft3/s.
    for (int sample = 0; sample < 12; ++sample) {
        gapInputs.discharge_ft3_per_second.push_back({sample * 300, 100.0});
    }
    // A 2-hour gap [3600, 10800) follows, then resumes at 300 ft3/s: within the
    // 3-hour interpolation bound, so hours 1-2 should still be fully covered.
    for (int sample = 0; sample < 12; ++sample) {
        gapInputs.discharge_ft3_per_second.push_back({10800 + sample * 300, 300.0});
    }
    // A 4-hour gap [14400, 28800) follows: beyond the interpolation bound, so
    // hours 4-7 must remain uncovered rather than fabricating a value.
    gapInputs.discharge_ft3_per_second.push_back({28800, 300.0});

    GisToOhqHourlyConfig gapConfig;
    gapConfig.start_epoch_seconds = 0;
    gapConfig.end_epoch_seconds = 8 * 3600;
    gapConfig.catchment_area_m2 = 1.0e6;
    const auto gapRows = harmonizeGisToOhqHourly(gapInputs, gapConfig);
    assert(gapRows.size() == 8);
    assert(gapRows[0].discharge_valid && close(gapRows[0].discharge_coverage, 1.0));
    assert(gapRows[1].discharge_valid && close(gapRows[1].discharge_coverage, 1.0));
    assert(gapRows[2].discharge_valid && close(gapRows[2].discharge_coverage, 1.0));
    // The interpolated ramp climbs from 100 to 300 ft3/s across the filled gap,
    // so later hours within it should read higher, not identical, values.
    assert(gapRows[2].observed_discharge_m3_per_second > gapRows[1].observed_discharge_m3_per_second);
    assert(gapRows[1].observed_discharge_m3_per_second > gapRows[0].observed_discharge_m3_per_second);
    assert(!gapRows[4].discharge_valid);
    assert(!gapRows[5].discharge_valid);
    assert(!gapRows[6].discharge_valid);
    return 0;
}
