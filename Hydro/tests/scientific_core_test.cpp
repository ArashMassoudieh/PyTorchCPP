#include "../dataset/chronological_split.h"
#include "../evaluation/hydro_metrics.h"

#include <cassert>
#include <cmath>
#include <stdexcept>
#include <vector>

int main() {
    HydroRunConfig defaults;
    defaults.input_lags_csv = "24";
    assert(defaults.lstm_sequence_length == 6);

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
    assert(result.correlation == 1.0);
    assert(result.kge == 1.0);
    assert(result.volume_error_percent == 0.0);

    HydroRunResult biased;
    populateHydroMetrics(biased, {1.0, 2.0, 3.0}, {2.0, 3.0, 4.0});
    assert(std::abs(biased.volume_error_percent - 50.0) < 1.0e-12);
    assert(biased.kge < 1.0);

    HydroRunResult constant;
    populateHydroMetrics(constant, {0.0, 0.0}, {1.0, 1.0});
    assert(std::isfinite(constant.mse));
    assert(std::isnan(constant.nse));
    assert(std::isnan(constant.pbias));
    assert(hydroMetricsAreFinite(constant));
    HydroRunResult flows;
    flows.x = {0.0, 1.0, 2.0, 3.0};
    flows.y_true = {1.0, 2.0, 3.0, 10.0};
    flows.y_pred = {2.0, 2.0, 3.0, 8.0};
    flows.split = {"test", "test", "test", "test"};
    populateHydroPeakMetrics(flows);
    assert(flows.peak_timing_error == 0.0);
    assert(std::abs(flows.peak_magnitude_error_percent + 20.0) < 1.0e-12);
    assert(flows.low_flow_rmse == 1.0);
    assert(flows.high_flow_rmse == 2.0);
    HydroRunResult physics;
    physics.x = {0.0, 1.0, 3.0};
    physics.physics_residual = {std::numeric_limits<double>::quiet_NaN(), 2.0, -1.0};
    populateHydroPhysicsResidualMetrics(physics);
    assert(physics.physics_residual_mean == 0.5);
    assert(std::abs(physics.physics_residual_rmse - std::sqrt(2.5)) < 1.0e-12);
    assert(physics.cumulative_physics_residual == 0.0);
    return 0;
}
