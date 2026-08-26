#include "../dataset/gistohq_model_rows.h"

#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
GisToOhqHourlyRow validRow(std::int64_t time, double runoff) {
    GisToOhqHourlyRow row;
    row.epoch_seconds = time;
    row.precipitation_mm_per_hour = 1.0;
    row.temperature_c = 10.0;
    row.relative_humidity_percent = 50.0;
    row.wind_speed_m_per_second = 2.0;
    row.solar_energy_mj_per_m2_per_hour = 0.5;
    row.pet_mm_per_hour = 0.1;
    row.observed_runoff_mm_per_hour = runoff;
    row.precipitation_valid = row.temperature_valid = row.relative_humidity_valid =
        row.wind_valid = row.solar_valid = row.pet_valid = row.discharge_valid =
        row.all_forcings_valid = true;
    return row;
}
}

int main() {
    std::vector<GisToOhqHourlyRow> hourly;
    for (std::int64_t hour = 0; hour < 6; ++hour) hourly.push_back(validRow(hour * 3600, hour));
    hourly[0].discharge_valid = false; // forcing-only inference may retain this row
    hourly[0].observed_runoff_mm_per_hour = std::numeric_limits<double>::quiet_NaN();
    hourly[3].temperature_valid = false;
    hourly[3].all_forcings_valid = false;
    hourly[3].temperature_c = std::numeric_limits<double>::quiet_NaN();

    const auto supervised = buildGisToOhqModelRows(hourly, true);
    assert(supervised.size() == 4);
    assert(supervised[0].epoch_seconds == 3600 && supervised[0].segment_id == 0 && supervised[0].index_in_segment == 0);
    assert(supervised[1].index_in_segment == 1);
    assert(supervised[2].epoch_seconds == 4 * 3600 && supervised[2].segment_id == 1 && supervised[2].index_in_segment == 0);
    assert(supervised[3].index_in_segment == 1);
    assert(supervised[0].features[0] == 1.0 && supervised[0].features[5] == 0.1);

    const auto inference = buildGisToOhqModelRows(hourly, false);
    assert(inference.size() == 5);
    assert(!inference[0].target_valid && std::isnan(inference[0].target_runoff_mm_per_hour));
    assert(inference[0].segment_id == 0 && inference[2].index_in_segment == 2);
    assert(inference[3].segment_id == 1 && inference[3].index_in_segment == 0);

    auto irregular = hourly;
    irregular[2].epoch_seconds += 60;
    bool rejectedIrregular = false;
    try { (void)buildGisToOhqModelRows(irregular, true); }
    catch (const std::invalid_argument&) { rejectedIrregular = true; }
    assert(rejectedIrregular);

    auto inconsistentMask = hourly;
    inconsistentMask[1].all_forcings_valid = false;
    bool rejectedMask = false;
    try { (void)buildGisToOhqModelRows(inconsistentMask, true); }
    catch (const std::invalid_argument&) { rejectedMask = true; }
    assert(rejectedMask);
    return 0;
}
