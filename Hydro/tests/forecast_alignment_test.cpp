#include "../dataset/forecast_alignment.h"

#include <cassert>
#include <stdexcept>

int main() {
    HydroForecast early{"2024-01-01T00:00:00Z", "2024-01-01T06:00:00Z", 6.0,
                        "a", "precipitation", 1.0, "mm/h", "model", "00", "m01"};
    HydroForecast latest = early;
    latest.issue_time = "2024-01-01T01:00:00Z";
    latest.lead_hours = 5.0;
    latest.value = 2.0;
    HydroForecast future = latest;
    future.issue_time = "2024-01-01T02:00:00Z";
    future.lead_hours = 4.0;
    future.value = 99.0;

    const auto selected = selectLatestAvailableForecast(
        {future, early, latest}, "a", "precipitation", "2024-01-01T06:00:00Z",
        "2024-01-01T01:30:00Z", "m01");
    assert(selected.has_value());
    assert(selected->value == 2.0);

    assert(!selectLatestAvailableForecast(
        {future}, "a", "precipitation", "2024-01-01T06:00:00Z",
        "2024-01-01T01:30:00Z", "m01").has_value());

    HydroForecast wrongUnit = early;
    wrongUnit.unit = "in/h";
    bool rejected = false;
    try {
        (void)selectLatestAvailableForecast(
            {early, wrongUnit}, "a", "precipitation", "2024-01-01T06:00:00Z",
            "2024-01-01T01:30:00Z", "m01");
    } catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);

    HydroForecast next = latest;
    next.issue_time = "2024-01-01T07:00:00Z";
    next.valid_time = "2024-01-01T12:00:00Z";
    next.value = 3.0;
    const auto feature = buildAlignedForecastFeature(
        {future, early, latest, next}, {"2024-01-01T06:00:00Z", "2024-01-01T12:00:00Z"},
        "a", "precipitation", 5.0, "m01");
    assert(feature.unit == "mm/h");
    assert(feature.values.size() == 2);
    assert(feature.values[0] == 2.0 && feature.values[1] == 3.0);

    rejected = false;
    try {
        (void)buildAlignedForecastFeature(
            {future}, {"2024-01-01T06:00:00Z"}, "a", "precipitation", 5.0, "m01");
    } catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
    return 0;
}
