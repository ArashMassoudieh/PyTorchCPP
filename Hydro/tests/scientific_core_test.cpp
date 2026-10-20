#include "../dataset/chronological_split.h"
#include "../dataset/resolve_test_end_x.h"
#include "../models/physics_batch.h"
#include "../evaluation/hydro_metrics.h"

#include <cassert>
#include <cmath>
#include <stdexcept>
#include <vector>

int main() {
    // Every observation and every adjacent transition is visited exactly once,
    // including singleton tails and batches larger than the whole series.
    for (int64_t total : {1, 2, 3, 17, 32, 33, 65}) {
        for (int64_t size : {2, 8, 16, 32, 100}) {
            std::vector<int> observations(total, 0), transitions(total, 0);
            for (int64_t start = 0; start < total; start += size) {
                const auto b = physicsBatch(start, total, size);
                assert(b.contextBegin + b.offset() == b.dataBegin);
                for (int64_t i = b.dataBegin; i < b.dataEnd; ++i) ++observations[i];
                for (int64_t i = b.contextBegin + 1; i < b.dataEnd; ++i) ++transitions[i];
            }
            assert(transitions[0] == 0);
            for (int64_t i = 0; i < total; ++i) {
                assert(observations[i] == 1);
                if (i > 0) assert(transitions[i] == 1);
            }
        }
    }
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

    // testEndRatio: default (-1.0, omitted) must reproduce the exact
    // pre-existing unbounded behavior - test_end == total.
    const ChronologicalSplit unbounded = makeChronologicalSplit(100, 0.7, 0.15);
    assert(unbounded.test_end == unbounded.total);

    // A valid bound clips test_end without touching train_end/validation_end.
    const ChronologicalSplit bounded = makeChronologicalSplit(100, 0.7, 0.15, 0.9);
    assert(bounded.train_end == 70);
    assert(bounded.validation_end == 85);
    assert(bounded.test_end == 90);
    assert(bounded.total == 100);

    // A bound at or before validation_end must leave no test partition and
    // is rejected (this is exactly the "named flood window too narrow to
    // follow validation" mistake a Phase 3 config generator could make -
    // see phase3_regional/00_phase3_design.md).
    bool boundRejected = false;
    try {
        (void)makeChronologicalSplit(100, 0.7, 0.15, 0.8);  // 0.8*100=80 <= validation_end=85
    } catch (const std::invalid_argument&) {
        boundRejected = true;
    }
    assert(boundRejected);

    // testEndX (exact row index) regression tests. These exist because
    // testEndRatio alone caused a real validation-row-contamination bug in
    // the Sligo Creek program: a ratio computed against an assumed total
    // row count lands at a different absolute row for callers whose
    // effective total differs (e.g. different input-sequence-length
    // warm-up consumption) - see chronological_split.h's testEndX doc
    // comment and results/nse08_recovery_20261015/phase3_regional/
    // 00_phase3_design.md's "Status update" section for the full account.

    // Default (-1, omitted) must reproduce the exact pre-existing unbounded
    // behavior, same as testEndRatio's default.
    const ChronologicalSplit noTestEndX = makeChronologicalSplit(100, 0.7, 0.15, -1.0);
    assert(noTestEndX.test_end == noTestEndX.total);

    // A valid exact bound clips test_end to precisely that row, without
    // touching train_end/validation_end - not an approximation via ratio.
    const ChronologicalSplit exactBound = makeChronologicalSplit(100, 0.7, 0.15, -1.0, 90);
    assert(exactBound.train_end == 70);
    assert(exactBound.validation_end == 85);
    assert(exactBound.test_end == 90);
    assert(exactBound.total == 100);

    // testEndX takes priority over testEndRatio when both are given -
    // ratio would bound to 95 (0.95*100), exact x says 90; 90 must win.
    const ChronologicalSplit xWinsOverRatio = makeChronologicalSplit(100, 0.7, 0.15, 0.95, 90);
    assert(xWinsOverRatio.test_end == 90);

    // An exact bound at or before validation_end is rejected, same safety
    // property as testEndRatio's equivalent check.
    bool exactBoundRejected = false;
    try {
        (void)makeChronologicalSplit(100, 0.7, 0.15, -1.0, 80);  // 80 <= validation_end=85
    } catch (const std::invalid_argument&) {
        exactBoundRejected = true;
    }
    assert(exactBoundRejected);

    // An exact bound beyond total is clamped to total, matching testEndRatio's
    // existing clamp behavior.
    const ChronologicalSplit exactBoundClamped = makeChronologicalSplit(100, 0.7, 0.15, -1.0, 500);
    assert(exactBoundClamped.test_end == 100);

    // THE actual regression: the same intended absolute cutoff (row 90),
    // requested via testEndX, must land at exactly row 90 regardless of a
    // caller-specific `total` difference (simulating two model families
    // with different input-sequence-length warm-up consumption on
    // otherwise-identical data) - while the SAME testEndRatio value does
    // NOT, because it is computed as a fraction of each caller's own
    // (different) total. This is the exact failure mode the real incident
    // exhibited (24h LSTM test start x=47502 vs. 120h LSTM's x=47838 for
    // the "same" Fold A bound).
    const int64_t totalA = 100;   // e.g. a 24h-sequence model's effective total
    const int64_t totalB = 106;   // e.g. a 120h-sequence model's smaller usable total shifted by warm-up
    const int64_t intendedCutoff = 90;
    const ChronologicalSplit exactA = makeChronologicalSplit(totalA, 0.5, 0.1, -1.0, intendedCutoff);
    const ChronologicalSplit exactB = makeChronologicalSplit(totalB, 0.5, 0.1, -1.0, intendedCutoff);
    assert(exactA.test_end == intendedCutoff);
    assert(exactB.test_end == intendedCutoff);  // exact x: identical cutoff regardless of total

    const double equivalentRatio = static_cast<double>(intendedCutoff) / static_cast<double>(totalA);
    const ChronologicalSplit ratioA = makeChronologicalSplit(totalA, 0.5, 0.1, equivalentRatio);
    const ChronologicalSplit ratioB = makeChronologicalSplit(totalB, 0.5, 0.1, equivalentRatio);
    assert(ratioA.test_end == intendedCutoff);     // ratio happens to match for the total it was tuned on
    assert(ratioB.test_end != intendedCutoff);     // but drifts for a different total - the bug testEndX fixes

    // resolveTestEndSequenceCountFromSorted: the gap-aware lookup
    // test_end_x actually needs. This reproduces the SECOND, deeper bug
    // found after the above fix shipped: a windowed sequence builder (or
    // any upstream row filter) skips positions across a gap, so a caller
    // cannot pass a desired exported-x cutoff straight through as if it
    // were already a sequence-index/array-position - sequence-index and
    // exported-x drift by the cumulative size of every gap before the
    // cutoff, not a constant offset. A first attempt at this feature did
    // exactly that (passed test_end_x straight through) and produced a
    // live test partition 951 x-units longer than intended; this lookup
    // is the fix. See resolve_test_end_x.h and
    // results/nse08_recovery_20261015/phase3_regional/00_phase3_design.md's
    // "Status update" section for the full account.
    {
        // Endpoints with two gaps: index 0..4 -> x 10,11,12,13,14 (contiguous),
        // then a gap (next kept endpoint is x=20, not 15), then 20,21,22
        // (contiguous), then another gap, then 30,31.
        const std::vector<double> endpoints = {10, 11, 12, 13, 14, 20, 21, 22, 30, 31};

        // Cutoff exactly on an endpoint before any gap: count includes it.
        assert(resolveTestEndSequenceCountFromSorted(endpoints, 12) == 3);

        // Cutoff that falls INSIDE a gap (no endpoint equals it) must count
        // only endpoints <= it, not assume contiguity across the gap. 17
        // falls between index 4 (x=14) and index 5 (x=20).
        assert(resolveTestEndSequenceCountFromSorted(endpoints, 17) == 5);

        // Cutoff exactly on an endpoint AFTER a gap: a naive constant-offset
        // formula (treating index as x-10) would predict x=20 at index 10,
        // not index 5 - the actual, correct answer, reflecting the 5
        // x-units (15-19) skipped by the first gap.
        assert(resolveTestEndSequenceCountFromSorted(endpoints, 20) == 6);

        // Cutoff after the second gap too: must reflect BOTH gaps' skipped
        // ranges cumulatively (15-19 and 23-29), not just the first.
        assert(resolveTestEndSequenceCountFromSorted(endpoints, 30) == 9);

        // Cutoff before the first endpoint: nothing qualifies.
        assert(resolveTestEndSequenceCountFromSorted(endpoints, 5) == 0);

        // Cutoff at or past the last endpoint: everything qualifies.
        assert(resolveTestEndSequenceCountFromSorted(endpoints, 31) == 10);
        assert(resolveTestEndSequenceCountFromSorted(endpoints, 1000) == 10);

        // Unset sentinel (-1): unbounded, matching testEndX's convention.
        assert(resolveTestEndSequenceCountFromSorted(endpoints, -1) == -1);
    }

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
    bool mismatchedMetricsRejected = false;
    try { populateHydroMetrics(constant, {1.0, 2.0}, {1.0}); }
    catch (const std::invalid_argument&) { mismatchedMetricsRejected = true; }
    assert(mismatchedMetricsRejected);
    bool nonFiniteMetricsRejected = false;
    try { populateHydroMetrics(constant, {1.0}, {std::numeric_limits<double>::infinity()}); }
    catch (const std::invalid_argument&) { nonFiniteMetricsRejected = true; }
    assert(nonFiniteMetricsRejected);
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
    bool misalignedPeakSeriesRejected = false;
    try {
        HydroRunResult misaligned = flows;
        misaligned.split.pop_back();
        populateHydroPeakMetrics(misaligned);
    } catch (const std::invalid_argument&) { misalignedPeakSeriesRejected = true; }
    assert(misalignedPeakSeriesRejected);
    bool missingTestPartitionRejected = false;
    try {
        HydroRunResult trainingOnly = flows;
        trainingOnly.split.assign(trainingOnly.split.size(), "train");
        populateHydroPeakMetrics(trainingOnly);
    } catch (const std::invalid_argument&) { missingTestPartitionRejected = true; }
    assert(missingTestPartitionRejected);
    HydroRunResult physics;
    physics.x = {0.0, 1.0, 3.0};
    physics.physics_residual = {std::numeric_limits<double>::quiet_NaN(), 2.0, -1.0};
    populateHydroPhysicsResidualMetrics(physics);
    assert(physics.physics_residual_mean == 0.5);
    assert(std::abs(physics.physics_residual_rmse - std::sqrt(2.5)) < 1.0e-12);
    assert(physics.cumulative_physics_residual == 0.0);
    bool misalignedResidualsRejected = false;
    try {
        HydroRunResult misalignedPhysics = physics;
        misalignedPhysics.x.pop_back();
        populateHydroPhysicsResidualMetrics(misalignedPhysics);
    } catch (const std::invalid_argument&) { misalignedResidualsRejected = true; }
    assert(misalignedResidualsRejected);
    bool unorderedResidualTimesRejected = false;
    try {
        HydroRunResult unorderedPhysics = physics;
        unorderedPhysics.x = {0.0, 2.0, 1.0};
        populateHydroPhysicsResidualMetrics(unorderedPhysics);
    } catch (const std::invalid_argument&) { unorderedResidualTimesRejected = true; }
    assert(unorderedResidualTimesRejected);
    return 0;
}
