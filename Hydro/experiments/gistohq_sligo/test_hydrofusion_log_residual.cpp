// Standalone, torch-free regression tests for hydrofusion_log_residual.h
// (Task 3 of CLAUDE_HYDROFUSION_IMPLEMENTATION_TASKS_20261115.md). No
// torch dependency -- builds in under a second, same convention as
// test_hydrofusion_feature_builder.cpp.
//
// Build: c++ -std=c++17 -Wall -Wextra -Werror test_hydrofusion_log_residual.cpp -o test_hydrofusion_log_residual
// Run:   ./test_hydrofusion_log_residual

#include "hydrofusion_log_residual.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

using hydrofusion::buildSample;
using hydrofusion::computeQRefFromTrainingSamples;
using hydrofusion::computeTrainingTarget;
using hydrofusion::deserializeQRef;
using hydrofusion::HourlyIndex;
using hydrofusion::HourlyRow;
using hydrofusion::ProvenanceRecord;
using hydrofusion::reconstructQ;
using hydrofusion::runPersistenceBenchmark;
using hydrofusion::serializeQRef;
using hydrofusion::SplitCutoffs;
using hydrofusion::SplitPartition;
using hydrofusion::TargetMode;
using hydrofusion::validateQRef;
using hydrofusion::zTransform;

namespace {

HourlyRow makeRow(int64_t hour, double q) {
    HourlyRow row;
    row.x = static_cast<double>(hour);
    row.observed_runoff_mm_per_hour = q;
    row.discharge_valid = true;
    row.precipitation_mm_per_hour = 0.1;
    row.temperature_c = 15.0;
    row.relative_humidity_percent = 60.0;
    row.wind_speed_m_per_second = 2.0;
    row.solar_energy_mj_per_m2_per_hour = 1.0;
    row.pet_mm_per_hour = 0.05;
    row.all_forcings_valid = true;
    return row;
}

std::vector<HourlyRow> contiguousFixture() {
    std::vector<HourlyRow> rows;
    rows.reserve(200);
    for (int64_t h = 0; h < 200; ++h) {
        rows.push_back(makeRow(h, 1.0 + 0.01 * static_cast<double>(h)));
    }
    return rows;
}

// Canonical cutoffs for the pure-unit tests below (well below kSealStartX).
// Train = x in [0,100], Validation = x in [101,150], Test = x in [151,180].
const SplitCutoffs kTestCutoffs{/*train_end_x=*/100, /*validation_end_x=*/150, /*test_end_x=*/180};

// originX must be a genuine chronological identity (target_x = originX+1),
// since computeQRefFromTrainingSamples now independently re-derives and
// verifies the partition from (originX, originX+1, kTestCutoffs) -- a
// fabricated/placeholder identity that disagrees with `claimedSplit` is
// exactly what the independent review's probe exploited, so these test
// fixtures must use real, consistent identities, not arbitrary ones.
ProvenanceRecord makeRecord(SplitPartition claimedSplit, bool included, int64_t originX, double targetQ) {
    ProvenanceRecord r;
    r.split = claimedSplit;
    r.split_assigned = true;
    r.included = included;
    r.predictor_availability_mask = included;
    r.target_native_mask = included;
    r.origin_x = originX;
    r.target_x = originX + 1;
    r.persistence_prediction = 999.0;  // origin reading is irrelevant to q_ref now
    r.observed_target_q = targetQ;
    return r;
}

void testZTransformDomainGuards() {
    bool threwForNegativeQ = false;
    try {
        zTransform(-1.0, 5.0);
    } catch (const std::invalid_argument&) {
        threwForNegativeQ = true;
    }
    assert(threwForNegativeQ);

    bool threwForNonfiniteQ = false;
    try {
        zTransform(std::numeric_limits<double>::infinity(), 5.0);
    } catch (const std::invalid_argument&) {
        threwForNonfiniteQ = true;
    }
    assert(threwForNonfiniteQ);

    bool threwForZeroQRef = false;
    try {
        zTransform(1.0, 0.0);
    } catch (const std::invalid_argument&) {
        threwForZeroQRef = true;
    }
    assert(threwForZeroQRef);

    bool threwForNegativeQRef = false;
    try {
        zTransform(1.0, -2.0);
    } catch (const std::invalid_argument&) {
        threwForNegativeQRef = true;
    }
    assert(threwForNegativeQRef);

    // Valid case: z(Q) = log1p(Q / q_ref), checked against the real formula.
    const double z = zTransform(3.0, 2.0);
    assert(std::abs(z - std::log1p(1.5)) < 1e-12);

    // Exact reproduction of the independent review's probe:
    // z_finite_ratio_overflow -- Q=DBL_MAX, q_ref=DBL_MIN are both finite,
    // but Q/q_ref overflows to +inf before log1p is even applied. Must
    // throw, not silently return +inf.
    bool threwForRatioOverflow = false;
    try {
        zTransform(std::numeric_limits<double>::max(), std::numeric_limits<double>::min());
    } catch (const std::overflow_error&) {
        threwForRatioOverflow = true;
    }
    assert(threwForRatioOverflow);
    std::cout << "testZTransformDomainGuards: PASS\n";
}

void testValidateQRefRejectsNegativeZeroAndNonfinite() {
    bool threwForNegative = false;
    try {
        validateQRef(-2.0);
    } catch (const std::invalid_argument&) {
        threwForNegative = true;
    }
    assert(threwForNegative);

    bool threwForZero = false;
    try {
        validateQRef(0.0);
    } catch (const std::invalid_argument&) {
        threwForZero = true;
    }
    assert(threwForZero);

    bool threwForNonfinite = false;
    try {
        validateQRef(std::numeric_limits<double>::quiet_NaN());
    } catch (const std::invalid_argument&) {
        threwForNonfinite = true;
    }
    assert(threwForNonfinite);

    validateQRef(1.0);  // must not throw
    std::cout << "testValidateQRefRejectsNegativeZeroAndNonfinite: PASS\n";
}

void testReconstructQRejectsNegativeQRefInEveryMode() {
    // Exact reproduction of the independent review's probe:
    // negative_qref_direct -- reconstructQ(DirectLogQ, 1, 1, -2) must
    // reject the negative q_ref, not silently clamp a negative
    // reconstruction to 0 and call that "accepted."
    bool threwForDirect = false;
    try {
        reconstructQ(TargetMode::DirectLogQ, 1.0, 1.0, -2.0);
    } catch (const std::invalid_argument&) {
        threwForDirect = true;
    }
    assert(threwForDirect);

    bool threwForResidual = false;
    try {
        reconstructQ(TargetMode::LogResidual, 1.0, 1.0, -2.0);
    } catch (const std::invalid_argument&) {
        threwForResidual = true;
    }
    assert(threwForResidual);
    std::cout << "testReconstructQRejectsNegativeQRefInEveryMode: PASS\n";
}

void testReconstructQRejectsUnknownTargetMode() {
    const auto bogusMode = static_cast<TargetMode>(99);
    bool threw = false;
    try {
        reconstructQ(bogusMode, 1.0, 1.0, 2.0);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
    std::cout << "testReconstructQRejectsUnknownTargetMode: PASS\n";
}

void testResidualMultiplierIsNotACallerAdjustableParameter() {
    // Exact reproduction of the independent review's probe: nonunit_alpha
    // -- an earlier version's reconstructQ accepted a 5th `alpha`
    // argument (e.g. 2.0), letting a caller tune the "fixed" multiplier.
    // The parameter no longer exists; this test exists to document and
    // lock that removal -- reconstructQ now only accepts (mode, originQ,
    // predicted, qRef), confirmed by this call compiling and matching
    // the fixed-multiplier formula exactly.
    const double qRef = 2.0;
    const double originQ = 1.0;
    const double predicted = 1.0;
    const double reconstructed = reconstructQ(TargetMode::LogResidual, originQ, predicted, qRef);
    const double zOrigin = zTransform(originQ, qRef);
    const double expected = std::max(0.0, qRef * std::expm1(zOrigin + 1.0 * predicted));
    assert(std::abs(reconstructed - expected) < 1e-12);
    std::cout << "testResidualMultiplierIsNotACallerAdjustableParameter: PASS\n";
}

void testComputeQRefMedianOfPositiveTargetValuesExactlyOnce() {
    // Independent-review correction: q_ref now pools each eligible
    // TRAINING TARGET observation exactly once (not origin+target
    // together, which double-counted interior hours). Distinct originX
    // values are used so each contributes a distinct target_x (the new
    // duplicate-target_x guard would otherwise fire).
    std::vector<ProvenanceRecord> records;
    records.push_back(makeRecord(SplitPartition::Train, true, 10, 3.0));
    records.push_back(makeRecord(SplitPartition::Train, true, 20, 7.0));
    records.push_back(makeRecord(SplitPartition::Train, true, 30, 5.0));
    // Train, but NOT included -- must be excluded entirely (origin_x/
    // target_x are never even inspected for a non-included record).
    records.push_back(makeRecord(SplitPartition::Train, false, 40, 1000.0));
    // Genuinely Validation under kTestCutoffs (originX=120 -> Validation) --
    // must be excluded (not Train), and its claimed split agrees with the
    // independently re-derived one, so this does not throw.
    records.push_back(makeRecord(SplitPartition::Validation, true, 120, 2000.0));
    // A genuine zero target reading: legitimately excluded from a
    // POSITIVE median, without being an error.
    records.push_back(makeRecord(SplitPartition::Train, true, 50, 0.0));

    // Target set used exactly once: {3, 7, 5} (zero excluded, not an
    // error) -> sorted {3, 5, 7} -> median = 5.
    const double qRef = computeQRefFromTrainingSamples(records, kTestCutoffs);
    assert(std::abs(qRef - 5.0) < 1e-12);
    std::cout << "testComputeQRefMedianOfPositiveTargetValuesExactlyOnce: PASS\n";
}

void testComputeQRefThrowsWhenUndefined() {
    std::vector<ProvenanceRecord> records;
    // Zero target reading: legitimately excluded (not an error), but
    // leaves nothing positive -> q_ref undefined.
    records.push_back(makeRecord(SplitPartition::Train, true, 10, 0.0));
    records.push_back(makeRecord(SplitPartition::Validation, true, 120, 5.0));  // wrong split

    bool threw = false;
    try {
        computeQRefFromTrainingSamples(records, kTestCutoffs);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);
    std::cout << "testComputeQRefThrowsWhenUndefined: PASS\n";
}

void testComputeQRefRejectsInvariantViolationRatherThanSilentlyDropping() {
    // Independent-review correction: an `included` record whose
    // observed_target_q is negative or non-finite violates
    // target_native_mask's own guarantee -- this is an upstream
    // invariant failure and must throw, not be silently treated as
    // "just not positive" the way a legitimate zero reading is. Uses a
    // genuine Train identity so the record passes the chronological/
    // split checks and reaches the numeric-domain check being tested.
    std::vector<ProvenanceRecord> records;
    records.push_back(makeRecord(SplitPartition::Train, true, 10, -4.0));  // invalid: included but negative

    bool threw = false;
    try {
        computeQRefFromTrainingSamples(records, kTestCutoffs);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);

    std::vector<ProvenanceRecord> nanRecords;
    nanRecords.push_back(
        makeRecord(SplitPartition::Train, true, 10, std::numeric_limits<double>::quiet_NaN()));
    bool threwForNaN = false;
    try {
        computeQRefFromTrainingSamples(nanRecords, kTestCutoffs);
    } catch (const std::invalid_argument&) {
        threwForNaN = true;
    }
    assert(threwForNaN);
    std::cout << "testComputeQRefRejectsInvariantViolationRatherThanSilentlyDropping: PASS\n";
}

void testComputeQRefOverflowSafeMedianOfExtremeValues() {
    // Exact reproduction of the independent review's probe:
    // qref_median_overflow -- two Train-included records whose target
    // readings are both DBL_MAX. The naive 0.5*(a+b) midpoint overflows
    // (DBL_MAX+DBL_MAX -> inf); the overflow-safe a+0.5*(b-a) form must
    // not. Distinct originX values avoid the (unrelated) duplicate-
    // target_x guard.
    std::vector<ProvenanceRecord> records;
    const double huge = std::numeric_limits<double>::max();
    records.push_back(makeRecord(SplitPartition::Train, true, 10, huge));
    records.push_back(makeRecord(SplitPartition::Train, true, 20, huge));
    const double qRef = computeQRefFromTrainingSamples(records, kTestCutoffs);
    assert(std::isfinite(qRef));
    assert(qRef > 0.0);
    std::cout << "testComputeQRefOverflowSafeMedianOfExtremeValues: PASS\n";
}

void testComputeTrainingTargetRejectsUnknownTargetMode() {
    // Exact reproduction of the independent review's probe:
    // unknown_target_mode -- computeTrainingTarget(static_cast<TargetMode>(42), 1, 2, 1)
    // previously fell through an implicit `else` and silently computed a
    // log-residual value (0.405465...) for an unrecognized mode, even
    // though reconstructQ had already been fixed to reject it. Both APIs
    // must use the same explicit, exhaustive mode check.
    const auto bogusMode = static_cast<TargetMode>(42);
    bool threw = false;
    try {
        computeTrainingTarget(bogusMode, 1.0, 2.0, 1.0);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
    std::cout << "testComputeTrainingTargetRejectsUnknownTargetMode: PASS\n";
}

void testComputeQRefRejectsRecordWhoseLabelDisagreesWithDerivedPartition() {
    // Exact reproduction of the independent review's probe:
    // fabricated_sealed_train_label -- a record claiming split==Train,
    // included, both masks true, with origin_x=77015/target_x=77016 (AT
    // the reserved-period seal) previously still contributed its target
    // reading to q_ref. The chronological/preseal check alone now catches
    // this exact case (target_x >= kSealStartX).
    ProvenanceRecord sealed;
    sealed.included = true;
    sealed.split_assigned = true;
    sealed.split = SplitPartition::Train;
    sealed.predictor_availability_mask = true;
    sealed.target_native_mask = true;
    sealed.origin_x = 77015;
    sealed.target_x = 77016;
    sealed.observed_target_q = 2.0;

    bool threwForSealed = false;
    try {
        computeQRefFromTrainingSamples({sealed}, kTestCutoffs);
    } catch (const std::invalid_argument&) {
        threwForSealed = true;
    }
    assert(threwForSealed);

    // A second, non-seal-related case: the record's own x values are
    // pre-seal and chronologically consistent (target_x = origin_x + 1),
    // but they actually fall in Validation under kTestCutoffs while the
    // record claims Train -- the independently re-derived partition must
    // be checked against the claimed label, not merely the raw domain.
    std::vector<ProvenanceRecord> records;
    records.push_back(makeRecord(SplitPartition::Train, true, 120, 5.0));  // 120/121 is Validation
    bool threwForMismatchedLabel = false;
    try {
        computeQRefFromTrainingSamples(records, kTestCutoffs);
    } catch (const std::invalid_argument&) {
        threwForMismatchedLabel = true;
    }
    assert(threwForMismatchedLabel);
    std::cout << "testComputeQRefRejectsRecordWhoseLabelDisagreesWithDerivedPartition: PASS\n";
}

void testDirectLogQModeRoundTrip() {
    const double qRef = 2.0;
    const double originQ = 1.0;
    const double targetQ = 3.0;
    const double target = computeTrainingTarget(TargetMode::DirectLogQ, originQ, targetQ, qRef);
    // A "perfect prediction" in direct-log-Q space is exactly the training target itself.
    const double reconstructed = reconstructQ(TargetMode::DirectLogQ, originQ, target, qRef);
    assert(std::abs(reconstructed - targetQ) < 1e-9);
    std::cout << "testDirectLogQModeRoundTrip: PASS\n";
}

void testLogResidualModeSyntheticPerfectReconstruction() {
    // "synthetic perfect residual reconstructs known Q" (Task 4 checklist item).
    const double qRef = 2.0;
    const double originQ = 1.0;
    const double targetQ = 3.0;
    const double rTarget = computeTrainingTarget(TargetMode::LogResidual, originQ, targetQ, qRef);
    const double reconstructed = reconstructQ(TargetMode::LogResidual, originQ, rTarget, qRef);
    assert(std::abs(reconstructed - targetQ) < 1e-9);
    std::cout << "testLogResidualModeSyntheticPerfectReconstruction: PASS\n";
}

void testRhatZeroRecoversPersistence() {
    // "rhat=0 recovers persistence" (Task 4 checklist item).
    const double qRef = 2.0;
    const double originQ = 4.5;
    const double reconstructed = reconstructQ(TargetMode::LogResidual, originQ, /*predicted=*/0.0, qRef);
    assert(std::abs(reconstructed - originQ) < 1e-9);
    std::cout << "testRhatZeroRecoversPersistence: PASS\n";
}

void testZeroClippingReachableAndNotOverTriggered() {
    const double qRef = 2.0;
    const double originQ = 0.1;  // small origin -> small, near-zero zOrigin
    // A strongly negative rhat pushes zHat below zero -> expm1 negative ->
    // qRef * expm1(zHat) negative -> must clip to exactly 0.0.
    const double clipped = reconstructQ(TargetMode::LogResidual, originQ, /*predicted=*/-5.0, qRef);
    assert(clipped == 0.0);

    // A mild negative rhat should NOT over-trigger the clip -- it must
    // still produce a small positive value less than originQ, proving
    // the clamp fires only when the math actually goes negative.
    const double mild = reconstructQ(TargetMode::LogResidual, originQ, /*predicted=*/-0.01, qRef);
    assert(mild > 0.0 && mild < originQ);
    std::cout << "testZeroClippingReachableAndNotOverTriggered: PASS\n";
}

void testExpm1OverflowProtected() {
    const double qRef = 2.0;
    const double originQ = 1.0;
    bool threw = false;
    try {
        // predicted=1000 is finite, but expm1(1000) overflows to +inf.
        reconstructQ(TargetMode::DirectLogQ, originQ, /*predicted=*/1000.0, qRef);
    } catch (const std::overflow_error&) {
        threw = true;
    }
    assert(threw);
    std::cout << "testExpm1OverflowProtected: PASS\n";
}

void testQRefExportReloadRoundTrip() {
    const double qRef = 0.123456789012345;
    const std::string serialized = serializeQRef(qRef);
    const double reloaded = deserializeQRef(serialized);
    assert(reloaded == qRef);  // 17 significant digits round-trips an IEEE double exactly

    bool threwForMalformed = false;
    try {
        deserializeQRef("not_a_valid_line\n");
    } catch (const std::invalid_argument&) {
        threwForMalformed = true;
    }
    assert(threwForMalformed);

    bool threwForNonpositive = false;
    try {
        deserializeQRef("q_ref=-1.0\n");
    } catch (const std::invalid_argument&) {
        threwForNonpositive = true;
    }
    assert(threwForNonpositive);

    // Exact reproduction of the independent review's probe:
    // malformed_serialized_scale -- "q_ref=2.0junk\n" must be rejected
    // (std::stod's prefix-parsing previously accepted "2.0" and silently
    // ignored the trailing "junk").
    bool threwForTrailingJunk = false;
    try {
        deserializeQRef("q_ref=2.0junk\n");
    } catch (const std::invalid_argument&) {
        threwForTrailingJunk = true;
    }
    assert(threwForTrailingJunk);

    // A second record/line appended must also be rejected, not silently
    // parsed from only the first line.
    bool threwForExtraRecord = false;
    try {
        deserializeQRef("q_ref=2.0\nq_ref=3.0\n");
    } catch (const std::invalid_argument&) {
        threwForExtraRecord = true;
    }
    assert(threwForExtraRecord);
    std::cout << "testQRefExportReloadRoundTrip: PASS\n";
}

void testIntegrationWithSharedFeatureBuilder() {
    // Demonstrates Task 3's explicit "same feature builder and
    // architecture" requirement concretely: q_ref and both modes' targets
    // are derived from the real hydrofusion_feature_builder.h sample
    // stream (Task 2), not a hand-constructed value disconnected from it.
    auto rows = contiguousFixture();
    HourlyIndex idx(rows);
    SplitCutoffs cutoffs{/*train_end_x=*/120, /*validation_end_x=*/160, /*test_end_x=*/190};
    std::vector<ProvenanceRecord> records = runPersistenceBenchmark(idx, rows, cutoffs);

    const double qRef = computeQRefFromTrainingSamples(records, cutoffs);
    assert(qRef > 0.0);

    // Pick one real included Train-split record and round-trip both modes
    // against its actual origin/target discharge values.
    bool checkedOne = false;
    for (const auto& r : records) {
        if (!r.included || r.split != SplitPartition::Train) continue;
        const double originQ = r.persistence_prediction;
        const double targetQ = r.observed_target_q;

        const double directTarget = computeTrainingTarget(TargetMode::DirectLogQ, originQ, targetQ, qRef);
        const double directReconstructed = reconstructQ(TargetMode::DirectLogQ, originQ, directTarget, qRef);
        assert(std::abs(directReconstructed - targetQ) < 1e-9);

        const double residualTarget = computeTrainingTarget(TargetMode::LogResidual, originQ, targetQ, qRef);
        const double residualReconstructed =
            reconstructQ(TargetMode::LogResidual, originQ, residualTarget, qRef);
        assert(std::abs(residualReconstructed - targetQ) < 1e-9);

        checkedOne = true;
        break;
    }
    assert(checkedOne);
    std::cout << "testIntegrationWithSharedFeatureBuilder: PASS\n";
}

void testQRefRejectsExtremeIdentityAndAllSplitContradictions() {
    auto reject = [](const ProvenanceRecord& r) {
        bool threw = false;
        try { (void)computeQRefFromTrainingSamples({r}, kTestCutoffs); }
        catch (const std::invalid_argument&) { threw = true; }
        assert(threw);
    };
    auto r = makeRecord(SplitPartition::Train, true, 10, 2.0);
    r.origin_x = std::numeric_limits<int64_t>::max();
    r.target_x = 0;
    reject(r);
    reject(makeRecord(SplitPartition::Validation, true, 155, 2.0));
    reject(makeRecord(SplitPartition::BeyondTestEnd, true, 185, 2.0));
    reject(makeRecord(SplitPartition::Validation, true, 100, 2.0));
    std::cout << "testQRefRejectsExtremeIdentityAndAllSplitContradictions: PASS\n";
}

}  // namespace

int main() {
    testQRefRejectsExtremeIdentityAndAllSplitContradictions();
    testZTransformDomainGuards();
    testValidateQRefRejectsNegativeZeroAndNonfinite();
    testReconstructQRejectsNegativeQRefInEveryMode();
    testReconstructQRejectsUnknownTargetMode();
    testResidualMultiplierIsNotACallerAdjustableParameter();
    testComputeQRefMedianOfPositiveTargetValuesExactlyOnce();
    testComputeQRefThrowsWhenUndefined();
    testComputeQRefRejectsInvariantViolationRatherThanSilentlyDropping();
    testComputeQRefOverflowSafeMedianOfExtremeValues();
    testComputeTrainingTargetRejectsUnknownTargetMode();
    testComputeQRefRejectsRecordWhoseLabelDisagreesWithDerivedPartition();
    testDirectLogQModeRoundTrip();
    testLogResidualModeSyntheticPerfectReconstruction();
    testRhatZeroRecoversPersistence();
    testZeroClippingReachableAndNotOverTriggered();
    testExpm1OverflowProtected();
    testQRefExportReloadRoundTrip();
    testIntegrationWithSharedFeatureBuilder();
    std::cout << "ALL HYDROFUSION LOG-RESIDUAL TESTS PASSED\n";
    return 0;
}
