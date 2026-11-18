// Standalone, torch-free regression tests for hydrofusion_feature_builder.h
// (Task 2 of CLAUDE_HYDROFUSION_IMPLEMENTATION_TASKS_20261115.md, feeding
// forward into Task 4's leakage/integration checks). No torch dependency --
// builds in under a second, matching the convention already used by
// test_verify_stageb_checkpoint_reproduction_parsing.cpp.
//
// Build: c++ -std=c++17 -Wall -Wextra -Werror test_hydrofusion_feature_builder.cpp -o test_hydrofusion_feature_builder
// Run:   ./test_hydrofusion_feature_builder

#include "hydrofusion_feature_builder.h"
#include "../../dataset/resolve_test_end_x.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

using hydrofusion::buildSample;
using hydrofusion::HourlyIndex;
using hydrofusion::HourlyRow;
using hydrofusion::ProvenanceRecord;
using hydrofusion::runPersistenceBenchmark;
using hydrofusion::SplitCutoffs;
using hydrofusion::SplitPartition;
using hydrofusion::assignSampleSplit;
using hydrofusion::partitionForX;
using hydrofusion::validateSealStartX;
using hydrofusion::validateSplitCutoffs;

namespace {

HourlyRow makeRow(int64_t hour, double q, bool dischargeValid = true, bool forcingsValid = true) {
    HourlyRow row;
    row.x = static_cast<double>(hour);
    row.observed_runoff_mm_per_hour = q;
    row.discharge_valid = dischargeValid;
    row.precipitation_mm_per_hour = 0.1;
    row.temperature_c = 15.0;
    row.relative_humidity_percent = 60.0;
    row.wind_speed_m_per_second = 2.0;
    row.solar_energy_mj_per_m2_per_hour = 1.0;
    row.pet_mm_per_hour = 0.05;
    row.all_forcings_valid = forcingsValid;
    return row;
}

// 200 contiguous hours (0..199) of synthetic discharge, no gaps, all valid.
std::vector<HourlyRow> contiguousFixture() {
    std::vector<HourlyRow> rows;
    rows.reserve(200);
    for (int64_t h = 0; h < 200; ++h) {
        rows.push_back(makeRow(h, 1.0 + 0.01 * static_cast<double>(h)));
    }
    return rows;
}

void testSealGuardBeforeAccess() {
    // Fixture spans well before the seal through just past it, with full
    // 72h-window support everywhere, so any rejection near the seal can
    // only be attributed to the seal check itself, not missing data.
    std::vector<HourlyRow> rows;
    for (int64_t h = hydrofusion::kSealStartX - 100; h <= hydrofusion::kSealStartX + 5; ++h) {
        rows.push_back(makeRow(h, 2.0));
    }
    HourlyIndex idx(rows);

    // Origin one hour before the seal: target (origin+1) is AT the seal,
    // so the sample must still be rejected (target checked too).
    auto rejectedAtTargetSeal = buildSample(idx, rows, hydrofusion::kSealStartX - 1);
    assert(!rejectedAtTargetSeal.has_value());

    // Origin at the seal itself must also be rejected.
    auto rejectedAtOriginSeal = buildSample(idx, rows, hydrofusion::kSealStartX);
    assert(!rejectedAtOriginSeal.has_value());

    // Origin safely two hours before the seal, with full data support:
    // must be accepted and fully eligible -- confirms the seal check
    // does not over-reject nearby-but-legal origins.
    auto acceptedBeforeSeal = buildSample(idx, rows, hydrofusion::kSealStartX - 2);
    assert(acceptedBeforeSeal.has_value());
    assert(acceptedBeforeSeal->predictor_availability_mask);
    assert(acceptedBeforeSeal->target_native_mask);
    std::cout << "testSealGuardBeforeAccess: PASS\n";
}

void testMissingOriginRowReturnsNullopt() {
    auto rows = contiguousFixture();
    // Remove row for hour 100 by rebuilding without it.
    std::vector<HourlyRow> withGap;
    for (const auto& r : rows) {
        if (static_cast<int64_t>(r.x) == 100) continue;
        withGap.push_back(r);
    }
    HourlyIndex idx(withGap);
    auto sample = buildSample(idx, withGap, 100);
    assert(!sample.has_value());
    std::cout << "testMissingOriginRowReturnsNullopt: PASS\n";
}

void testDuplicateHourRejected() {
    std::vector<HourlyRow> rows = {makeRow(10, 1.0), makeRow(10, 2.0)};
    bool threw = false;
    try {
        HourlyIndex idx(rows);
        (void)idx;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
    std::cout << "testDuplicateHourRejected: PASS\n";
}

void testNonMonotonicRejected() {
    std::vector<HourlyRow> rows = {makeRow(10, 1.0), makeRow(5, 2.0)};
    bool threw = false;
    try {
        HourlyIndex idx(rows);
        (void)idx;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
    std::cout << "testNonMonotonicRejected: PASS\n";
}

void testNonIntegerXRejected() {
    std::vector<HourlyRow> rows = {makeRow(10, 1.0)};
    rows[0].x = 10.5;
    bool threw = false;
    try {
        HourlyIndex idx(rows);
        (void)idx;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
    std::cout << "testNonIntegerXRejected: PASS\n";
}

void testSegmentGapVoidsCrossingLagWindow() {
    // Hours 0..79 contiguous, then a gap, then 90..199 contiguous.
    // An origin inside the second block whose 72h trailing window would
    // reach back across the gap must be ineligible on that account, even
    // though every individual hour it CAN reach exists.
    std::vector<HourlyRow> rows;
    for (int64_t h = 0; h <= 79; ++h) rows.push_back(makeRow(h, 1.0));
    for (int64_t h = 90; h <= 199; ++h) rows.push_back(makeRow(h, 1.0));
    HourlyIndex idx(rows);

    // Origin at hour 95: 72h trailing window needs hours 24..95, which
    // crosses the 80..89 gap -> must be ineligible (predictor mask false).
    auto sample = buildSample(idx, rows, 95);
    assert(sample.has_value());
    assert(!sample->predictor_availability_mask);

    // Origin at hour 170: fully within the second contiguous block, far
    // enough from its start to have a complete 72h window -> eligible.
    auto eligibleSample = buildSample(idx, rows, 170);
    assert(eligibleSample.has_value());
    assert(eligibleSample->predictor_availability_mask);
    std::cout << "testSegmentGapVoidsCrossingLagWindow: PASS\n";
}

void testIncompleteDischargeValidityVoidsSample() {
    auto rows = contiguousFixture();
    // Mark hour 150's discharge invalid; an origin at 150 needs q_lag_0h
    // from exactly that hour, so it must become ineligible.
    for (auto& r : rows) {
        if (static_cast<int64_t>(r.x) == 150) r.discharge_valid = false;
    }
    HourlyIndex idx(rows);
    auto sample = buildSample(idx, rows, 150);
    assert(sample.has_value());
    assert(!sample->predictor_availability_mask);
    std::cout << "testIncompleteDischargeValidityVoidsSample: PASS\n";
}

void testTargetNativeMaskIndependentOfPredictorMask() {
    auto rows = contiguousFixture();
    // Invalidate the met forcing at origin 150 (voids predictor mask) but
    // keep the target (hour 151) discharge valid -- target_native_mask
    // must still be true; the two masks are independent.
    for (auto& r : rows) {
        if (static_cast<int64_t>(r.x) == 150) r.all_forcings_valid = false;
    }
    HourlyIndex idx(rows);
    auto sample = buildSample(idx, rows, 150);
    assert(sample.has_value());
    assert(!sample->predictor_availability_mask);
    assert(sample->target_native_mask);
    std::cout << "testTargetNativeMaskIndependentOfPredictorMask: PASS\n";
}

void testPersistenceReconstructsQOrigin() {
    auto rows = contiguousFixture();
    HourlyIndex idx(rows);
    auto sample = buildSample(idx, rows, 150);
    assert(sample.has_value());
    assert(sample->predictor_availability_mask);
    assert(sample->target_native_mask);
    assert(std::abs(sample->persistence_prediction - sample->features.q_lag_0h) < 1e-12);
    assert(std::abs(sample->persistence_prediction - (1.0 + 0.01 * 150.0)) < 1e-9);
    std::cout << "testPersistenceReconstructsQOrigin: PASS\n";
}

void testSplitEdgeInclusiveSemantics() {
    SplitCutoffs cutoffs{/*train_end_x=*/100, /*validation_end_x=*/150, /*test_end_x=*/180};
    assert(partitionForX(100, cutoffs) == SplitPartition::Train);       // inclusive of train_end
    assert(partitionForX(101, cutoffs) == SplitPartition::Validation);  // first index of validation
    assert(partitionForX(150, cutoffs) == SplitPartition::Validation);  // inclusive of validation_end
    assert(partitionForX(151, cutoffs) == SplitPartition::Test);        // first index of test
    assert(partitionForX(180, cutoffs) == SplitPartition::Test);        // inclusive of test_end
    assert(partitionForX(181, cutoffs) == SplitPartition::BeyondTestEnd);
    std::cout << "testSplitEdgeInclusiveSemantics: PASS\n";
}

void testStraddlingSampleDiscarded() {
    SplitCutoffs cutoffs{/*train_end_x=*/100, /*validation_end_x=*/150, /*test_end_x=*/180};
    // origin=100 (train), target=101 (validation) -> straddles -> discard.
    auto straddling = assignSampleSplit(100, 101, cutoffs);
    assert(!straddling.has_value());
    // origin=99, target=100 -> both train -> assigned.
    auto clean = assignSampleSplit(99, 100, cutoffs);
    assert(clean.has_value());
    assert(*clean == SplitPartition::Train);
    std::cout << "testStraddlingSampleDiscarded: PASS\n";
}

void testHistoricalContextAcrossSplitAllowedButNotFutureLabel() {
    // A validation-origin sample's discharge-memory window is allowed to
    // read train-split hours (purely causal history). Confirm the feature
    // builder itself does not reject on split grounds at all -- split
    // assignment is a separate, later step from feature eligibility.
    auto rows = contiguousFixture();
    HourlyIndex idx(rows);
    SplitCutoffs cutoffs{/*train_end_x=*/140, /*validation_end_x=*/160, /*test_end_x=*/190};
    // Origin 145 is in validation; its 72h window reaches back to hour 73,
    // which is in train. The sample must still be eligible.
    auto sample = buildSample(idx, rows, 145);
    assert(sample.has_value());
    assert(sample->predictor_availability_mask);
    auto split = assignSampleSplit(sample->origin_x, sample->target_x, cutoffs);
    assert(split.has_value());
    assert(*split == SplitPartition::Validation);
    std::cout << "testHistoricalContextAcrossSplitAllowedButNotFutureLabel: PASS\n";
}

// --- Regression tests reproducing the independent review's exact probe
// (audits/hydrofusion_supervisor_20261117/synthetic_builder_probe.cpp) ---

void testNaNDischargeAtOriginVoidsSampleAndNeverProducesNonfinitePersistence() {
    // Exact reproduction of the independent review's probe: 100 rows,
    // x=0..99, all valid, q=1, except row 80's q set to NaN while
    // discharge_valid stays true. Before the fix: predictor_availability_
    // mask was true and persistence_prediction was non-finite. After the
    // fix: the discharge_valid flag alone must not be trusted numerically.
    std::vector<HourlyRow> rows;
    for (int64_t h = 0; h < 100; ++h) rows.push_back(makeRow(h, 1.0));
    rows[80].observed_runoff_mm_per_hour = std::numeric_limits<double>::quiet_NaN();
    assert(rows[80].discharge_valid);  // flag still claims valid -- that's the point

    HourlyIndex idx(rows);
    auto sample = buildSample(idx, rows, 80);
    assert(sample.has_value());
    assert(!sample->predictor_availability_mask);
    assert(std::isfinite(sample->persistence_prediction));
    std::cout << "testNaNDischargeAtOriginVoidsSampleAndNeverProducesNonfinitePersistence: PASS\n";
}

void testNaNDischargeAtTargetVoidsTargetNativeMask() {
    std::vector<HourlyRow> rows;
    for (int64_t h = 0; h < 100; ++h) rows.push_back(makeRow(h, 1.0));
    rows[81].observed_runoff_mm_per_hour = std::numeric_limits<double>::quiet_NaN();
    assert(rows[81].discharge_valid);

    HourlyIndex idx(rows);
    auto sample = buildSample(idx, rows, 80);
    assert(sample.has_value());
    assert(!sample->target_native_mask);
    std::cout << "testNaNDischargeAtTargetVoidsTargetNativeMask: PASS\n";
}

void testNegativeDischargeVoidsSample() {
    std::vector<HourlyRow> rows;
    for (int64_t h = 0; h < 100; ++h) rows.push_back(makeRow(h, 1.0));
    rows[75].observed_runoff_mm_per_hour = -0.5;  // flagged valid but physically impossible
    assert(rows[75].discharge_valid);

    HourlyIndex idx(rows);
    auto sample = buildSample(idx, rows, 80);  // 72h window reaches back to hour 8, includes 75
    assert(sample.has_value());
    assert(!sample->predictor_availability_mask);
    std::cout << "testNegativeDischargeVoidsSample: PASS\n";
}

void testInvalidMetValueVoidsPredictorMask() {
    std::vector<HourlyRow> rows;
    for (int64_t h = 0; h < 100; ++h) rows.push_back(makeRow(h, 1.0));
    rows[80].precipitation_mm_per_hour = -1.0;  // negative precipitation has no physical meaning
    assert(rows[80].all_forcings_valid);

    HourlyIndex idxNegative(rows);
    auto sampleNegative = buildSample(idxNegative, rows, 80);
    assert(sampleNegative.has_value());
    assert(!sampleNegative->predictor_availability_mask);

    rows[80].precipitation_mm_per_hour = 0.1;
    rows[80].temperature_c = std::numeric_limits<double>::quiet_NaN();
    HourlyIndex idxNaN(rows);
    auto sampleNaN = buildSample(idxNaN, rows, 80);
    assert(sampleNaN.has_value());
    assert(!sampleNaN->predictor_availability_mask);
    std::cout << "testInvalidMetValueVoidsPredictorMask: PASS\n";
}

void testNegativeXRejected() {
    std::vector<HourlyRow> rows = {makeRow(-5, 1.0), makeRow(-4, 1.0)};
    bool threw = false;
    try {
        HourlyIndex idx(rows);
        (void)idx;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
    std::cout << "testNegativeXRejected: PASS\n";
}

void testPersistenceBenchmarkExcludesStraddlingAndBeyondTestEnd() {
    // Exact reproduction of the independent review's probe: 200-row
    // fixture, cutoffs {train_end=80, validation_end=90, test_end=95}.
    // Before the fix: origin=80 (straddles train/validation) and
    // origin=96 (BeyondTestEnd) both had included=true. After the fix,
    // both must be false.
    auto rows = contiguousFixture();
    HourlyIndex idx(rows);
    SplitCutoffs cutoffs{/*train_end_x=*/80, /*validation_end_x=*/90, /*test_end_x=*/95};
    std::vector<ProvenanceRecord> records = runPersistenceBenchmark(idx, rows, cutoffs);

    bool foundOrigin80 = false;
    bool foundOrigin96 = false;
    for (const auto& r : records) {
        if (r.origin_x == 80) {
            foundOrigin80 = true;
            assert(!r.split_assigned);  // origin=80 (train) vs target=81 (validation): straddles
            assert(!r.included);
        }
        if (r.origin_x == 96) {
            foundOrigin96 = true;
            assert(r.split_assigned);
            assert(r.split == SplitPartition::BeyondTestEnd);
            assert(!r.included);
        }
    }
    assert(foundOrigin80);
    assert(foundOrigin96);
    std::cout << "testPersistenceBenchmarkExcludesStraddlingAndBeyondTestEnd: PASS\n";
}

void testPartitionForXParityWithProduction() {
    // Verifies partitionForX's inclusive-cutoff semantics agree with
    // resolveTestEndSequenceCountFromSorted -- the same pure, torch-free
    // core the production configuredChronologicalSplit path itself calls
    // -- across a GAPPED sequence, where row index and x diverge (exactly
    // the scenario resolve_test_end_x.h's own doc comment warns about).
    std::vector<double> endpointX;
    for (int64_t h = 0; h <= 50; ++h) endpointX.push_back(static_cast<double>(h));      // 0..50
    for (int64_t h = 60; h <= 120; ++h) endpointX.push_back(static_cast<double>(h));    // gap 51..59

    const int64_t trainEndX = 40;
    const int64_t validationEndX = 70;
    const int64_t testEndX = 100;
    SplitCutoffs cutoffs{trainEndX, validationEndX, testEndX};

    const int64_t trainCount = resolveTestEndSequenceCountFromSorted(endpointX, trainEndX);
    const int64_t validationCount = resolveTestEndSequenceCountFromSorted(endpointX, validationEndX);
    const int64_t testCount = resolveTestEndSequenceCountFromSorted(endpointX, testEndX);

    for (std::size_t i = 0; i < endpointX.size(); ++i) {
        const int64_t x = static_cast<int64_t>(endpointX[i]);
        SplitPartition expected;
        if (static_cast<int64_t>(i) < trainCount) expected = SplitPartition::Train;
        else if (static_cast<int64_t>(i) < validationCount) expected = SplitPartition::Validation;
        else if (static_cast<int64_t>(i) < testCount) expected = SplitPartition::Test;
        else expected = SplitPartition::BeyondTestEnd;

        const SplitPartition actual = partitionForX(x, cutoffs);
        assert(actual == expected);
    }

    // Explicit edge checks at the exact cutoff values and their neighbors,
    // spanning the gap (51..59 do not exist as rows, but the cutoff
    // comparison itself must not care).
    assert(partitionForX(trainEndX, cutoffs) == SplitPartition::Train);
    assert(partitionForX(trainEndX + 1, cutoffs) == SplitPartition::Validation);
    assert(partitionForX(validationEndX, cutoffs) == SplitPartition::Validation);
    assert(partitionForX(validationEndX + 1, cutoffs) == SplitPartition::Test);
    assert(partitionForX(testEndX, cutoffs) == SplitPartition::Test);
    assert(partitionForX(testEndX + 1, cutoffs) == SplitPartition::BeyondTestEnd);

    std::cout << "testPartitionForXParityWithProduction: PASS\n";
}

// --- Regression tests reproducing the independent review's second probe
// (audits/hydrofusion_supervisor_20261117/remaining_domain_probe.cpp) ---

void testFiniteInputOverflowVoidsSample() {
    // Exact reproduction: 100 rows, x=0..99, all valid, q=DBL_MAX. Every
    // single value is individually finite, but the 72h trailing mean/
    // variance arithmetic over 72 DBL_MAX values overflows. Before the
    // fix: predictor_availability_mask was true with non-finite derived
    // features. After the fix: the sample must be ineligible.
    std::vector<HourlyRow> rows;
    for (int64_t h = 0; h < 100; ++h) rows.push_back(makeRow(h, std::numeric_limits<double>::max()));
    HourlyIndex idx(rows);
    auto sample = buildSample(idx, rows, 80);
    assert(sample.has_value());
    assert(!sample->predictor_availability_mask);
    std::cout << "testFiniteInputOverflowVoidsSample: PASS\n";
}

void testInvalidSplitCutoffsRejected() {
    // Exact reproduction: cutoffs {train_end=90, validation_end=80,
    // test_end=95} -- train > validation, violating the required
    // ordering. Before the fix: runPersistenceBenchmark silently accepted
    // these and returned 100 records. After the fix: must throw before
    // any sample work.
    auto rows = contiguousFixture();
    HourlyIndex idx(rows);
    SplitCutoffs invalidCutoffs{/*train_end_x=*/90, /*validation_end_x=*/80, /*test_end_x=*/95};

    bool threwFromValidate = false;
    try {
        validateSplitCutoffs(invalidCutoffs);
    } catch (const std::invalid_argument&) {
        threwFromValidate = true;
    }
    assert(threwFromValidate);

    bool threwFromBenchmark = false;
    try {
        auto records = runPersistenceBenchmark(idx, rows, invalidCutoffs);
        (void)records;
    } catch (const std::invalid_argument&) {
        threwFromBenchmark = true;
    }
    assert(threwFromBenchmark);

    // A merely out-of-order-but-otherwise-plausible set is also rejected:
    // test_end_x == validation_end_x violates the strict '<'.
    SplitCutoffs equalBoundary{50, 100, 100};
    bool threwFromEqual = false;
    try {
        validateSplitCutoffs(equalBoundary);
    } catch (const std::invalid_argument&) {
        threwFromEqual = true;
    }
    assert(threwFromEqual);

    // And a valid set must pass without throwing.
    SplitCutoffs validCutoffs{80, 90, 95};
    validateSplitCutoffs(validCutoffs);  // must not throw
    std::cout << "testInvalidSplitCutoffsRejected: PASS\n";
}

void testHourlyIndexRejectsOutOfInt64RangeX() {
    std::vector<HourlyRow> rows = {makeRow(0, 1.0)};
    rows[0].x = 1.0e19;  // finite, integer-valued (at this magnitude), but beyond kMaxRepresentableX
    bool threw = false;
    try {
        HourlyIndex idx(rows);
        (void)idx;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
    std::cout << "testHourlyIndexRejectsOutOfInt64RangeX: PASS\n";
}

void testBuildSampleRejectsOutOfDomainOriginBeforeArithmetic() {
    auto rows = contiguousFixture();
    HourlyIndex idx(rows);
    // originHour passed directly, not sourced from a row -- buildSample's
    // own domain guard must reject it before computing origin+1, not rely
    // on the row lookup to fail gracefully afterward.
    auto negativeOrigin = buildSample(idx, rows, -1);
    assert(!negativeOrigin.has_value());
    // 9.1e18 is comfortably above kMaxRepresentableX (9.0e18) and still
    // safely within int64_t's own range (max ~9.223e18) -- a deliberately
    // out-of-domain origin, not an int64_t overflow in the test itself.
    auto hugeOrigin = buildSample(idx, rows, static_cast<int64_t>(9'100'000'000'000'000'000LL));
    assert(!hugeOrigin.has_value());
    std::cout << "testBuildSampleRejectsOutOfDomainOriginBeforeArithmetic: PASS\n";
}

// --- Regression tests reproducing the independent review's third probe
// (audits/hydrofusion_supervisor_20261117/fixed_seal_parameter_probe.cpp) --
// using only fabricated synthetic rows, never real reserved data.

void testSealStartXCannotBeWidenedPastFixedSeal() {
    bool threwForWideningValue = false;
    try {
        validateSealStartX(80000);  // > kSealStartX: an attempt to widen, must be rejected
    } catch (const std::invalid_argument&) {
        threwForWideningValue = true;
    }
    assert(threwForWideningValue);

    bool threwForZero = false;
    try {
        validateSealStartX(0);
    } catch (const std::invalid_argument&) {
        threwForZero = true;
    }
    assert(threwForZero);

    bool threwForNegative = false;
    try {
        validateSealStartX(-5);
    } catch (const std::invalid_argument&) {
        threwForNegative = true;
    }
    assert(threwForNegative);

    // Tightening (a smaller, positive value) must be permitted.
    validateSealStartX(1000);  // must not throw
    validateSealStartX(hydrofusion::kSealStartX);  // the fixed default itself must not throw

    // Exact reproduction of the independent review's probe: fabricated
    // synthetic rows at x=76940..77039 (never real reserved data), then
    // buildSample(idx, rows, 77016, 80000) -- origin AT the fixed seal,
    // with a wider caller-supplied sealStartX. Before the fix: this
    // returned a sample with target_native_mask=true. After the fix:
    // buildSample must throw before accepting the widened parameter.
    std::vector<HourlyRow> rows;
    for (int i = 0; i < 100; ++i) rows.push_back(makeRow(76940 + i, 1.0));
    HourlyIndex idx(rows);
    bool threwFromBuildSample = false;
    try {
        auto sample = buildSample(idx, rows, 77016, 80000);
        (void)sample;
    } catch (const std::invalid_argument&) {
        threwFromBuildSample = true;
    }
    assert(threwFromBuildSample);

    // Exact reproduction of the probe's second half: validateSplitCutoffs
    // with cutoffs past the fixed seal and a widened sealStartX. Before
    // the fix: accepted. After the fix: must throw.
    bool threwFromCutoffs = false;
    try {
        validateSplitCutoffs(SplitCutoffs{77000, 77010, 77020}, 80000);
    } catch (const std::invalid_argument&) {
        threwFromCutoffs = true;
    }
    assert(threwFromCutoffs);

    // The default call path (sealStartX left at its default) must remain
    // completely unaffected -- this was always correct and must stay so.
    auto defaultPathSample = buildSample(idx, rows, hydrofusion::kSealStartX - 50);
    assert(defaultPathSample.has_value());  // well before the seal, with full support
    std::cout << "testSealStartXCannotBeWidenedPastFixedSeal: PASS\n";
}

void testPersistenceBenchmarkProvenanceOnFixture() {
    auto rows = contiguousFixture();
    HourlyIndex idx(rows);
    SplitCutoffs cutoffs{/*train_end_x=*/120, /*validation_end_x=*/160, /*test_end_x=*/190};
    std::vector<ProvenanceRecord> records = runPersistenceBenchmark(idx, rows, cutoffs);
    // Every row from 0..198 is a candidate origin (199 is the last row, so
    // its target at 200 does not exist); origins before hour 71 cannot have
    // a complete 72h window, so they're excluded from `included`.
    bool sawIncluded = false;
    bool sawExcludedForWindow = false;
    for (const auto& r : records) {
        if (r.included) {
            sawIncluded = true;
            // contiguousFixture's q(h) = 1.0 + 0.01*h is deterministic, so
            // observed_target_q - persistence_prediction = q(o+1) - q(o)
            // = 0.01 exactly for every included record -- a real check,
            // not the previous vacuous `|| true`.
            assert(std::abs(r.observed_target_q - r.persistence_prediction - 0.01) < 1e-9);
        }
        if (r.origin_x < 71 && !r.predictor_availability_mask) sawExcludedForWindow = true;
    }
    assert(sawIncluded);
    assert(sawExcludedForWindow);
    std::cout << "testPersistenceBenchmarkProvenanceOnFixture: PASS (n=" << records.size() << ")\n";
}

}  // namespace

int main() {
    testSealGuardBeforeAccess();
    testMissingOriginRowReturnsNullopt();
    testDuplicateHourRejected();
    testNonMonotonicRejected();
    testNonIntegerXRejected();
    testSegmentGapVoidsCrossingLagWindow();
    testIncompleteDischargeValidityVoidsSample();
    testTargetNativeMaskIndependentOfPredictorMask();
    testPersistenceReconstructsQOrigin();
    testSplitEdgeInclusiveSemantics();
    testStraddlingSampleDiscarded();
    testHistoricalContextAcrossSplitAllowedButNotFutureLabel();
    testPersistenceBenchmarkProvenanceOnFixture();
    testNaNDischargeAtOriginVoidsSampleAndNeverProducesNonfinitePersistence();
    testNaNDischargeAtTargetVoidsTargetNativeMask();
    testNegativeDischargeVoidsSample();
    testInvalidMetValueVoidsPredictorMask();
    testNegativeXRejected();
    testPersistenceBenchmarkExcludesStraddlingAndBeyondTestEnd();
    testPartitionForXParityWithProduction();
    testFiniteInputOverflowVoidsSample();
    testInvalidSplitCutoffsRejected();
    testHourlyIndexRejectsOutOfInt64RangeX();
    testBuildSampleRejectsOutOfDomainOriginBeforeArithmetic();
    testSealStartXCannotBeWidenedPastFixedSeal();
    std::cout << "ALL HYDROFUSION FEATURE BUILDER TESTS PASSED\n";
    return 0;
}
