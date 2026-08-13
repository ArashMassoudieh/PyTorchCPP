#include "../dataset/chronological_split.h"
#include "../evaluation/hydro_metrics.h"

#include <cassert>
#include <cmath>
#include <stdexcept>
#include <vector>

int main() {
    const ChronologicalSplit split = makeChronologicalSplit(100, 0.7, 0.15);
    assert(split.train_end == 70);
    assert(split.validation_end == 85);
    assert(split.total - split.validation_end == 15);

    bool rejected = false;
    try {
        (void)makeChronologicalSplit(10, 0.9, 0.2);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    assert(rejected);

    HydroRunResult result;
    populateHydroMetrics(result, {1.0, 2.0, 3.0}, {1.0, 2.0, 3.0});
    assert(result.mse == 0.0);
    assert(result.rmse == 0.0);
    assert(result.mae == 0.0);
    assert(result.nse == 1.0);
    assert(result.pbias == 0.0);

    HydroRunResult constant;
    populateHydroMetrics(constant, {0.0, 0.0}, {1.0, 1.0});
    assert(std::isfinite(constant.mse));
    assert(std::isnan(constant.nse));
    assert(std::isnan(constant.pbias));
    assert(hydroMetricsAreFinite(constant));
    return 0;
}
