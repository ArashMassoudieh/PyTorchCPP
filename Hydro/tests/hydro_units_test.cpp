#include "../dataset/hydro_units.h"

#include <cassert>
#include <cmath>

int main() {
    const double area = 1.0e6;
    const double metric = dischargeToDepthRate(1.0, "m3/s", area);
    const double imperial = dischargeToDepthRate(35.3146667215, "ft3/s", area);
    assert(std::abs(metric - 3.6) < 1.0e-12);
    assert(std::abs(metric - imperial) < 1.0e-9);
    assert(forecastWasAvailable("2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z"));
    assert(!forecastWasAvailable("2024-01-01T02:00:00Z", "2024-01-01T01:00:00Z"));
    assert(!forecastWasAvailable("2024-01-01T00:00:00.500Z", "2024-01-01T00:00:00Z"));
    assert(forecastTimingIsConsistent("2024-01-01T00:00:00Z", "2024-01-01T06:00:00Z", 6.0,
                                      "2024-01-01T01:00:00Z"));
    assert(!forecastTimingIsConsistent("2024-01-01T02:00:00Z", "2024-01-01T06:00:00Z", 4.0,
                                       "2024-01-01T01:00:00Z"));
    assert(!forecastTimingIsConsistent("2024-01-01T00:00:00Z", "2024-01-01T06:00:00Z", 5.0,
                                       "2024-01-01T01:00:00Z"));
    bool rejected = false;
    try { (void)forecastWasAvailable("2024-02-30T00:00:00Z", "2024-03-01T00:00:00Z"); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
    return 0;
}
