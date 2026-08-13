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
    return 0;
}
