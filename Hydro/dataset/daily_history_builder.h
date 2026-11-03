#pragma once

#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <vector>

// Builds the daily-resolution antecedent-history input for the opt-in
// multiscale branch (improvement_loop_20261023 reports 66-69).
//
// CORRECTED (report 69, caught on code review): the first version of this
// file assumed row index == absolute hour, i.e. that 24 consecutive ROWS
// always form one calendar day. That assumption breaks the instant any row
// is missing (gistohq_model_rows.cpp drops, never pads, rows failing
// all_forcings_valid/discharge_valid -- confirmed in report 62's own
// audit) -- after a gap, row N no longer corresponds to hour N, so
// grouping by row position silently mixes hours from the wrong calendar
// days. This version uses each row's actual elapsed-hour value (`x`,
// project-wide convention: hours since 2015-01-01T00:00:00Z, itself
// midnight-aligned -- the same anchor `kSealStartX=77016` is defined
// against) to determine calendar-day membership, and requires every one
// of a day's 24 hours to have an existing row before that day counts as
// complete. A day with even one missing hour makes the whole window
// ineligible -- never partially aggregated from fewer than 24 hours.
//
// Segment starts are not assumed to fall on a midnight boundary either
// (a segment begins wherever the first accepted row after a gap happens
// to be) -- the segment-sufficiency check compares against the segment's
// actual first HOUR (from its first row's x value), not an assumed
// multiple of 24.
namespace daily_history {

constexpr int64_t kHoursPerDay = 24;

// x is expected to be an (effectively) integer number of hours -- the
// project-wide convention for every hourly package this builder is used
// against. Throws rather than silently rounding a genuinely irregular
// value, which would indicate a caller passed the wrong tensor.
//
// CORRECTED (report 70, caught on code review): NaN/Inf were not rejected
// explicitly. `llround(NaN)` is undefined behavior per the C++ standard
// (commonly returns an implementation-defined garbage value, not an
// error), and the old tolerance check `std::abs(x - rounded) > 1e-6` is
// ALWAYS false when x is NaN (every IEEE-754 comparison against NaN
// except != is false) -- so a NaN input silently produced a garbage hour
// index instead of throwing. `std::isfinite` is checked first, explicitly,
// before any arithmetic on x.
inline int64_t toHourIndex(double x) {
    if (!std::isfinite(x)) {
        throw std::invalid_argument("daily_history: x value is not finite (NaN or Inf).");
    }
    const int64_t rounded = static_cast<int64_t>(std::llround(x));
    if (std::abs(x - static_cast<double>(rounded)) > 1e-6) {
        throw std::invalid_argument("daily_history: x value is not an integer number of hours.");
    }
    return rounded;
}

// Pure index arithmetic on ABSOLUTE hour indices -- unchanged in spirit
// from the original (reversed, oldest-first, each block exactly 24 hours,
// every block ending strictly before windowStartHour), but the caller is
// now responsible for passing a real elapsed-hour value, not a row index.
inline std::vector<std::pair<int64_t, int64_t>> completedDayBlocks(
        int64_t windowStartHour, int64_t historyDays) {
    if (historyDays <= 0) return {};
    const int64_t lastCompleteDayEnd = (windowStartHour / kHoursPerDay) * kHoursPerDay - 1;
    std::vector<std::pair<int64_t, int64_t>> blocks;
    blocks.reserve(static_cast<std::size_t>(historyDays));
    int64_t end = lastCompleteDayEnd;
    for (int64_t i = 0; i < historyDays; ++i) {
        const int64_t start = end - kHoursPerDay + 1;
        blocks.push_back({start, end});
        end = start - 1;
    }
    std::reverse(blocks.begin(), blocks.end());
    return blocks;
}

inline bool hasSufficientSegmentHistory(
        const std::vector<std::pair<int64_t, int64_t>>& blocks,
        int64_t segmentFirstHour) {
    if (blocks.empty()) return true;
    return blocks.front().first >= segmentFirstHour;
}

struct DailyHistoryResult {
    torch::Tensor features;  // [numWindows, historyDays, 2] (daily MEAN hourly rate of P, of PET --
                              // i.e. the day's total divided by 24, same mm/h scale as the hourly
                              // branch's own P/PET inputs -- see report 70's normalization note below), float32
    torch::Tensor eligible;  // [numWindows] bool -- true iff every hour of every requested day block
                             // has an existing, same-segment row (no missing-row gaps, no cross-segment reach)
};

namespace detail {

struct HourIndex {
    std::unordered_map<int64_t, int64_t> hourToRow;
    std::vector<int64_t> hourOfRow;
    std::vector<int64_t> segmentFirstHour;  // per-row: this row's segment's actual first hour
};

// CORRECTED (report 70, caught on code review): three more gaps fixed
// here, all in the same function since every caller routes through it:
//   1. n==0 (empty input) previously reached `hourOfRow[0]` on a
//      zero-length vector -- undefined behavior, not a thrown error.
//      Rejected explicitly, first, before any indexing.
//   2. A duplicate hour (two rows with the same x value) previously
//      overwrote hourToRow[h] silently, keeping only the later row and
//      discarding the earlier one's contribution with no error at all.
//      Rejected explicitly via a presence check before insertion.
//   3. Unordered timestamps (row i+1's hour not strictly greater than row
//      i's) previously passed through silently, even though
//      segmentFirstHour's "track the running first hour as segments
//      change" logic -- and completedDayBlocks' day-boundary arithmetic
//      throughout this file -- both implicitly assume strictly increasing
//      hours within a segment. Rejected explicitly via a per-row
//      monotonicity check (hours may still have GAPS -- that is the whole
//      point of this file -- just never decrease or repeat).
inline HourIndex buildHourIndex(const torch::Tensor& xCpu, const torch::Tensor& segCpu, int64_t n) {
    if (n <= 0) {
        throw std::invalid_argument("daily_history: empty input (0 rows) -- nothing to index.");
    }
    HourIndex idx;
    idx.hourToRow.reserve(static_cast<std::size_t>(n) * 2);
    idx.hourOfRow.resize(static_cast<std::size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        const int64_t h = toHourIndex(xCpu[i].item<double>());
        if (i > 0 && h <= idx.hourOfRow[static_cast<std::size_t>(i - 1)]) {
            throw std::invalid_argument(
                "daily_history: timestamps are not strictly increasing at row " + std::to_string(i) +
                " (hour " + std::to_string(h) + " does not exceed the previous row's hour " +
                std::to_string(idx.hourOfRow[static_cast<std::size_t>(i - 1)]) + ") -- gaps are fine, "
                "duplicates/decreases/out-of-order rows are not.");
        }
        if (idx.hourToRow.find(h) != idx.hourToRow.end()) {
            throw std::invalid_argument("daily_history: duplicate hour " + std::to_string(h) +
                                         " at row " + std::to_string(i) + " -- refusing to silently overwrite.");
        }
        idx.hourOfRow[static_cast<std::size_t>(i)] = h;
        idx.hourToRow[h] = i;
    }
    idx.segmentFirstHour.resize(static_cast<std::size_t>(n));
    int64_t currentFirstHour = idx.hourOfRow[0];
    for (int64_t i = 0; i < n; ++i) {
        if (i > 0 && segCpu[i].item<int64_t>() != segCpu[i - 1].item<int64_t>()) {
            currentFirstHour = idx.hourOfRow[static_cast<std::size_t>(i)];
        }
        idx.segmentFirstHour[static_cast<std::size_t>(i)] = currentFirstHour;
    }
    return idx;
}

// Core logic, keyed directly on ABSOLUTE START HOURS (not row indices) --
// shared by both public entry points below so there is exactly one
// implementation of the alignment/completeness/segment rules.
inline DailyHistoryResult buildFromStartHours(
        const torch::Tensor& precipCpu, const torch::Tensor& petCpu, const torch::Tensor& segCpu,
        const HourIndex& idx, const std::vector<int64_t>& windowStartHours, int64_t historyDays) {
    const int64_t numWindows = static_cast<int64_t>(windowStartHours.size());
    auto features = torch::zeros({numWindows, historyDays, 2}, torch::kFloat32);
    auto eligible = torch::zeros({numWindows}, torch::kBool);
    auto featuresAcc = features.accessor<float, 3>();
    auto eligibleAcc = eligible.accessor<bool, 1>();

    for (int64_t w = 0; w < numWindows; ++w) {
        const int64_t windowStartHour = windowStartHours[static_cast<std::size_t>(w)];
        const auto startRowIt = idx.hourToRow.find(windowStartHour);
        if (startRowIt == idx.hourToRow.end()) {
            // The window's own start hour has no row at all -- cannot even
            // determine its segment; ineligible (this should not happen for
            // a window that was itself built from existing rows, but is
            // checked rather than assumed).
            eligibleAcc[w] = false;
            continue;
        }
        const int64_t windowSegment = segCpu[startRowIt->second].item<int64_t>();
        const int64_t segFirstHour = idx.segmentFirstHour[static_cast<std::size_t>(startRowIt->second)];
        const auto blocks = completedDayBlocks(windowStartHour, historyDays);

        if (!hasSufficientSegmentHistory(blocks, segFirstHour)) {
            eligibleAcc[w] = false;
            continue;
        }

        bool complete = true;
        std::vector<std::array<double, 2>> dailySums(static_cast<std::size_t>(historyDays), {0.0, 0.0});
        for (int64_t d = 0; d < historyDays && complete; ++d) {
            const auto& block = blocks[static_cast<std::size_t>(d)];
            for (int64_t h = block.first; h <= block.second; ++h) {
                const auto it = idx.hourToRow.find(h);
                if (it == idx.hourToRow.end() || segCpu[it->second].item<int64_t>() != windowSegment) {
                    complete = false;
                    break;
                }
                dailySums[static_cast<std::size_t>(d)][0] += precipCpu[it->second].item<float>();
                dailySums[static_cast<std::size_t>(d)][1] += petCpu[it->second].item<float>();
            }
        }
        eligibleAcc[w] = complete;
        if (complete) {
            // Mean hourly rate (sum / 24), not the raw daily total -- this
            // is the normalization report 70 asked for: it puts the daily
            // branch's inputs on the SAME physical scale (mm/h) as the
            // hourly branch's own precipitation/PET inputs, by construction,
            // with no separate fitted scaler to track, export, or load --
            // avoiding a much riskier change to the shared scalers.csv
            // artifact format (hardcoded to exactly "input"/"target" slots,
            // used by every other approach's checkpoints too). A fitted
            // scaler was considered and rejected for that reason.
            for (int64_t d = 0; d < historyDays; ++d) {
                featuresAcc[w][d][0] = static_cast<float>(dailySums[static_cast<std::size_t>(d)][0] / kHoursPerDay);
                featuresAcc[w][d][1] = static_cast<float>(dailySums[static_cast<std::size_t>(d)][1] / kHoursPerDay);
            }
        }
    }
    return {features, eligible};
}

}  // namespace detail

// x: [n] elapsed-hour values (project convention), float or double.
// precipitation, pet: [n] float, hourly, row-aligned with x.
// segmentId: [n] int64, row-aligned with x.
// windowStartRows: [numWindows] int64 ROW indices (not hours) -- the first
// row of each hourly window, matching this codebase's existing row-index
// convention for slicing hourly tensors; this function derives each
// window's absolute start HOUR from x[windowStartRows[w]] internally, so
// callers never need to compute hours themselves.
inline DailyHistoryResult buildDailyHistoryTensors(
        const torch::Tensor& x,
        const torch::Tensor& precipitation,
        const torch::Tensor& pet,
        const torch::Tensor& segmentId,
        const torch::Tensor& windowStartRows,
        int64_t historyDays) {
    if (historyDays <= 0) {
        throw std::invalid_argument("buildDailyHistoryTensors requires historyDays > 0.");
    }
    const auto xCpu = x.reshape({-1}).to(torch::kCPU).to(torch::kFloat64).contiguous();
    const auto precipCpu = precipitation.reshape({-1}).to(torch::kCPU).contiguous();
    const auto petCpu = pet.reshape({-1}).to(torch::kCPU).contiguous();
    const auto segCpu = segmentId.reshape({-1}).to(torch::kCPU).contiguous();
    const auto startsCpu = windowStartRows.reshape({-1}).to(torch::kCPU).contiguous();
    const int64_t n = xCpu.size(0);
    if (precipCpu.size(0) != n || petCpu.size(0) != n || segCpu.size(0) != n) {
        throw std::invalid_argument("buildDailyHistoryTensors requires aligned x/precipitation/pet/segmentId.");
    }
    const int64_t numWindows = startsCpu.size(0);
    const auto idx = detail::buildHourIndex(xCpu, segCpu, n);

    std::vector<int64_t> windowStartHours(static_cast<std::size_t>(numWindows));
    for (int64_t w = 0; w < numWindows; ++w) {
        const int64_t windowStartRow = startsCpu[w].item<int64_t>();
        if (windowStartRow < 0 || windowStartRow >= n) {
            throw std::invalid_argument("buildDailyHistoryTensors: windowStartRow out of range.");
        }
        windowStartHours[static_cast<std::size_t>(w)] = idx.hourOfRow[static_cast<std::size_t>(windowStartRow)];
    }
    return detail::buildFromStartHours(precipCpu, petCpu, segCpu, idx, windowStartHours, historyDays);
}

// Convenience entry point for callers that already have window END times
// (e.g. lstm_sequence_builder.h's LstmSequenceData::plotSeq) and the
// hourly sequence length, rather than a row index -- e.g. the multiscale
// training integration (lstmnetworkwrapper.cpp), which builds hourly
// windows via the existing, shared, gap-aware makeLstmSequences() and
// gets window END x-values, not start row indices, out of it. Converts
// windowEndX -> windowStartHour = windowEndHour - (sequenceLength - 1)
// (exact, because makeLstmSequences only keeps windows with a verified
// regular 1-hour grid throughout -- enforceRegularTime -- so this
// subtraction is not an approximation for any window that survived that
// check), then calls the same core logic as the row-index entry point.
inline DailyHistoryResult buildDailyHistoryTensorsFromWindowEndTimes(
        const torch::Tensor& x,
        const torch::Tensor& precipitation,
        const torch::Tensor& pet,
        const torch::Tensor& segmentId,
        const torch::Tensor& windowEndTimes,
        int64_t sequenceLength,
        int64_t historyDays) {
    if (historyDays <= 0) {
        throw std::invalid_argument("buildDailyHistoryTensorsFromWindowEndTimes requires historyDays > 0.");
    }
    const auto xCpu = x.reshape({-1}).to(torch::kCPU).to(torch::kFloat64).contiguous();
    const auto precipCpu = precipitation.reshape({-1}).to(torch::kCPU).contiguous();
    const auto petCpu = pet.reshape({-1}).to(torch::kCPU).contiguous();
    const auto segCpu = segmentId.reshape({-1}).to(torch::kCPU).contiguous();
    const auto endsCpu = windowEndTimes.reshape({-1}).to(torch::kCPU).to(torch::kFloat64).contiguous();
    const int64_t n = xCpu.size(0);
    if (precipCpu.size(0) != n || petCpu.size(0) != n || segCpu.size(0) != n) {
        throw std::invalid_argument("buildDailyHistoryTensorsFromWindowEndTimes requires aligned x/precipitation/pet/segmentId.");
    }
    const int64_t numWindows = endsCpu.size(0);
    const auto idx = detail::buildHourIndex(xCpu, segCpu, n);

    std::vector<int64_t> windowStartHours(static_cast<std::size_t>(numWindows));
    for (int64_t w = 0; w < numWindows; ++w) {
        const int64_t endHour = toHourIndex(endsCpu[w].item<double>());
        windowStartHours[static_cast<std::size_t>(w)] = endHour - (sequenceLength - 1);
    }
    return detail::buildFromStartHours(precipCpu, petCpu, segCpu, idx, windowStartHours, historyDays);
}

// One eligibility mask, meant to be applied identically to BOTH arms
// (control and multiscale) before training loss and before scoring --
// report 69's fix for the training/scoring inconsistency a per-model
// fallback method previously introduced. Returns the row indices (into
// the window-indexed tensors, e.g. sequence endpoints) that are eligible.
inline torch::Tensor eligibleWindowIndices(const torch::Tensor& eligible) {
    return eligible.nonzero().reshape({-1});
}

}  // namespace daily_history
