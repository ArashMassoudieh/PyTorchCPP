#pragma once

#include <cstdint>
#include <vector>

// Converts a `test_end_x` config value - expressed in the same exported-x
// coordinate space as predictions.csv's own "x" column (the canonical,
// gap-aware hour-index convention used throughout this program, see
// seal_guard.py) - into a sequence count for makeChronologicalSplit's
// testEndX parameter, by looking up the actual per-sequence/per-row
// endpoint x values a model's own data pipeline produced (e.g.
// LSTMNetworkWrapper's seq.plotSeq, LSTMPinnWrapper's seq.time, or
// pinn_wrapper.cpp's plotX directly).
//
// Why a lookup, not a formula: a windowed sequence builder silently skips
// a window wherever its span crosses a time-irregularity or GIStoOHQ
// segment gap, and even a non-windowed model's raw tensor can have
// upstream-filtered gaps - either way, array position and exported-x do
// not differ by a constant offset, they drift by the cumulative size of
// every gap before the cutoff. Confirmed directly in the Sligo Creek
// program on two separate model families: passing testEndX as a raw
// array/sequence position produced a test partition 951 x-units longer
// than intended for a 24h-sequence LSTM (traced to 928 x-units of
// cumulative drift from 8 time gaps earlier in the series), and a
// separate 744 x-units of drift from 8 gaps was confirmed present even in
// the non-windowed two-reservoir PINN path. Every model, windowed or not,
// must resolve test_end_x through this lookup - see
// results/nse08_recovery_20261015/phase3_regional/00_phase3_design.md's
// "Status update" section for the full account.
//
// Pure, torch-free core: endpointX must be sorted ascending (true by
// construction for every caller - sequence/row endpoints are built in
// time order). Returns -1 (unbounded, i.e. "testEndXCoordinate was never
// requested" or "not even the first endpoint is <= it") when
// testEndXCoordinate < 0 or no entry qualifies.
inline int64_t resolveTestEndSequenceCountFromSorted(const std::vector<double>& endpointX,
                                                      int64_t testEndXCoordinate) {
    if (testEndXCoordinate < 0) return -1;
    int64_t count = 0;
    for (std::size_t i = 0; i < endpointX.size(); ++i) {
        if (endpointX[i] <= static_cast<double>(testEndXCoordinate)) {
            count = static_cast<int64_t>(i) + 1;
        } else {
            break;  // endpointX is strictly increasing by construction
        }
    }
    return count;
}
