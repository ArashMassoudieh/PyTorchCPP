#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>

struct ChronologicalSplit {
    int64_t train_end = 0;
    int64_t validation_end = 0;
    int64_t total = 0;
    // End of the test partition (exclusive), i.e. test = [validation_end, test_end).
    // Equals `total` (the literal end of whatever series was loaded) unless a
    // bounded testEndRatio was requested - see makeChronologicalSplit's doc
    // comment below for why this field exists. Callers written before this
    // field was added slice test as [validation_end, total) directly and are
    // unaffected; new callers should prefer test_end over total so they
    // automatically respect a caller-requested bound.
    int64_t test_end = 0;
};

// trainRatio/validationRatio behave exactly as before. testEndRatio and
// testEndX are both optional (default -1 = unbounded, preserving every
// existing caller's behavior unchanged) and both bound the test partition's
// end instead of letting it run to the literal end of whatever series was
// loaded.
//
// Why a bound exists at all: without one, test is always
// [validation_end, total) - for a package whose `total` runs well past the
// period a caller actually cares about (e.g. a named flood window within a
// much longer series), the test partition silently sweeps in everything
// after it too. This produced nested, non-independent "folds" in the
// HydroPINN Sligo Creek program (same model family, same data, different
// training cutoffs, all scored on overlapping tails through the same
// endpoint) that were mistaken for independent evaluations until caught
// post hoc - see
// results/nse08_recovery_20261015/decision_gate_b/fold_c_diagnostic/02_fold_c_diagnostic_report.md
// and .../phase2_dataset/08_development_period_and_folds.md for the full
// account.
//
// Why testEndX exists, and takes priority over testEndRatio when both are
// given: testEndRatio bounds test_end as a *fraction of total row count*,
// but `total` itself - and therefore the same fraction's resulting absolute
// row index - varies by caller, because different input-sequence lengths
// consume different numbers of warm-up rows before the first usable index.
// Reusing one testEndRatio across model families (exactly what Phase 3's
// per-fold named-flood-window bound needed to do) silently lands at a
// different absolute row for each one - confirmed directly in the Sligo
// Creek program (within one fold, a 24h-sequence LSTM's genuine test start
// differed from a 120h-sequence LSTM's by 336 rows, same fold, same
// intended bound). testEndX sidesteps this by taking an exact row index
// directly, with no dependence on `total` or on any model-specific warm-up
// consumption - see .../phase3_regional/00_phase3_design.md's "Status
// update" section for the incident this fixes. testEndRatio is kept only
// for callers that cannot state an exact row index in advance.
//
// testEndX here means exactly what it says: an index into THIS function's
// own `total` (i.e. a sequence count/array position in whatever space the
// caller's `total` is measured in) - it is not aware of calendar time or
// of predictions.csv's exported "x" column. A caller whose `total` is a
// sequence count after windowing (e.g. an LSTM) must resolve a desired
// exported-x cutoff into that sequence-index space itself, by looking up
// its own actual per-sequence endpoint x values, BEFORE calling this
// function - see resolve_test_end_x.h's resolveTestEndSequenceCount() and
// HydroRunConfig::test_end_x's doc comment for why a formula-based
// conversion (e.g. subtracting a sequence length) is not safe: windowing
// skips sequences across gaps, so the offset between sequence-index and
// exported-x is not constant.
inline ChronologicalSplit makeChronologicalSplit(int64_t total,
                                                 double trainRatio,
                                                 double validationRatio,
                                                 double testEndRatio = -1.0,
                                                 int64_t testEndX = -1) {
    if (total < 3) {
        throw std::invalid_argument("A train/validation/test split requires at least three samples.");
    }
    const double train = std::clamp(trainRatio, 0.1, 0.9);
    const double validation = std::clamp(validationRatio, 0.01, 0.4);
    if (train + validation >= 0.99) {
        throw std::invalid_argument("Train and validation ratios must leave a non-empty test partition.");
    }
    int64_t trainEnd = std::clamp<int64_t>(static_cast<int64_t>(total * train), 1, total - 2);
    int64_t validationCount = std::max<int64_t>(1, static_cast<int64_t>(total * validation));
    int64_t validationEnd = std::min<int64_t>(total - 1, trainEnd + validationCount);

    int64_t testEnd = total;
    if (testEndX >= 0) {
        if (testEndX <= validationEnd) {
            throw std::invalid_argument(
                "testEndX must leave a non-empty bounded test partition after validation_end.");
        }
        testEnd = std::min(testEndX, total);
    } else if (testEndRatio >= 0.0) {
        const int64_t boundedEnd = static_cast<int64_t>(total * testEndRatio);
        if (boundedEnd <= validationEnd) {
            throw std::invalid_argument(
                "testEndRatio must leave a non-empty bounded test partition after validation_end.");
        }
        testEnd = std::min(boundedEnd, total);
    }
    return {trainEnd, validationEnd, total, testEnd};
}
