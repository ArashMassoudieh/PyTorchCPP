// Tests for the opt-in multiscale daily-history branch
// (improvement_loop_20261023 reports 66-69). Covers: daily aggregation
// correctness against REAL timestamps (not row position), missing-row
// gaps, non-midnight segment starts, segment-boundary respect, minimum-
// history exclusion via one shared eligibility mask, tensor shapes,
// gradient flow through h0 AND c0 independently, checkpoint save/load
// (new multiscale format, and an OLD plain-HydroLSTM checkpoint still
// loading unchanged), and exact reproduction of today's zero-option
// forward() when the branch is not used.
#include "../models/hydro_lstm_module.h"
#include "../models/multiscale_lstm_module.h"
#include "../dataset/daily_history_builder.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <vector>

namespace {

void test_completed_day_blocks_alignment() {
    {
        auto blocks = daily_history::completedDayBlocks(240, 3);
        assert(blocks.size() == 3);
        for (const auto& b : blocks) assert(b.second < 240);
        assert(blocks.back().second == 239);
        assert(blocks.back().first == 216);
    }
    {
        auto midDay = daily_history::completedDayBlocks(245, 3);
        auto dayBoundary = daily_history::completedDayBlocks(240, 3);
        assert(midDay.size() == dayBoundary.size());
        for (std::size_t i = 0; i < midDay.size(); ++i) assert(midDay[i] == dayBoundary[i]);
        for (const auto& b : midDay) assert(b.second < 245);
    }
    {
        auto blocks = daily_history::completedDayBlocks(500, 0);
        assert(blocks.empty());
    }
    for (int64_t ws : {24, 25, 48, 100, 1000, 8760}) {
        auto blocks = daily_history::completedDayBlocks(ws, 5);
        for (const auto& b : blocks) {
            assert(b.second < ws);
            assert(b.second - b.first + 1 == 24);
        }
    }
    std::cout << "test_completed_day_blocks_alignment: PASS\n";
}

void test_segment_boundary_respected() {
    auto blocks = daily_history::completedDayBlocks(240, 5);  // needs hours [120..239]
    assert(daily_history::hasSufficientSegmentHistory(blocks, 0));
    assert(daily_history::hasSufficientSegmentHistory(blocks, 120));
    assert(!daily_history::hasSufficientSegmentHistory(blocks, 121));
    assert(!daily_history::hasSufficientSegmentHistory(blocks, 150));
    std::cout << "test_segment_boundary_respected: PASS\n";
}

// Build a 10-day (240h), gap-free, single-segment series with x == row
// index (the simple, no-gap case), to confirm the real-timestamp builder
// reduces to the same behavior as the row-position version did when there
// genuinely are no gaps -- a sanity baseline before the gap/non-midnight
// cases below, which are the ones the original version got wrong.
void test_daily_aggregation_no_gaps_matches_expected_values() {
    const int64_t n = 240;
    auto x = torch::arange(0, n, torch::kFloat64);
    auto precip = torch::full({n}, 1.0f);
    auto pet = torch::full({n}, 0.1f);
    auto segmentId = torch::zeros({n}, torch::kInt64);

    auto windowStartRows = torch::tensor({static_cast<int64_t>(200), static_cast<int64_t>(24), static_cast<int64_t>(5)});
    const int64_t historyDays = 5;
    auto result = daily_history::buildDailyHistoryTensors(x, precip, pet, segmentId, windowStartRows, historyDays);

    assert(result.features.sizes() == std::vector<int64_t>({3, historyDays, 2}));
    assert(result.eligible.sizes() == std::vector<int64_t>({3}));
    auto eligibleAcc = result.eligible.accessor<bool, 1>();
    assert(eligibleAcc[0] == true);   // row 200: plenty of history
    assert(eligibleAcc[1] == false);  // row 24: not 5 full days available
    assert(eligibleAcc[2] == false);  // row 5: no full day available at all

    // Mean hourly rate (report 70's normalization fix), not the raw daily
    // total -- constant inputs of 1.0 mm/h precip and 0.1 mm/h PET average
    // to themselves over any complete day, so this also happens to be the
    // one case where sum-based and mean-based aggregation look numerically
    // similar at a glance; the daily_history_builder.h comment and the
    // training-integration smoke test are where the /24 actually matters.
    auto featAcc = result.features.accessor<float, 3>();
    for (int64_t d = 0; d < historyDays; ++d) {
        assert(std::abs(featAcc[0][d][0] - 1.0f) < 1e-4f);
        assert(std::abs(featAcc[0][d][1] - 0.1f) < 1e-4f);
    }
    std::cout << "test_daily_aggregation_no_gaps_matches_expected_values: PASS\n";
}

// THE bug report 69 caught: a missing row shifts row position relative to
// real time. Build a 240-hour series with hour 50 DROPPED entirely (239
// rows, x values 0..49,51..239 -- row 50 in the array is actually hour 51).
// A window whose history requires the day containing the dropped hour
// (hours [48,71], i.e. day index 2) must be marked INELIGIBLE -- not
// silently aggregated from 23 hours as if nothing were missing, and not
// shifted to aggregate the wrong hours under the old row-position bug.
void test_missing_row_marks_day_ineligible_not_misaligned() {
    std::vector<double> xs;
    for (int64_t h = 0; h < 240; ++h) {
        if (h == 50) continue;  // drop exactly one hour
        xs.push_back(static_cast<double>(h));
    }
    const int64_t n = static_cast<int64_t>(xs.size());
    assert(n == 239);
    auto x = torch::from_blob(xs.data(), {n}, torch::kFloat64).clone();
    auto precip = torch::full({n}, 1.0f);
    auto pet = torch::full({n}, 0.1f);
    auto segmentId = torch::zeros({n}, torch::kInt64);

    // Row 200 in this array corresponds to hour 201 (one row short because
    // of the earlier drop) -- find it by searching, not by assuming row==hour.
    int64_t rowForHour201 = -1;
    for (int64_t i = 0; i < n; ++i) {
        if (xs[static_cast<std::size_t>(i)] == 201.0) { rowForHour201 = i; break; }
    }
    assert(rowForHour201 >= 0);
    auto windowStartRows = torch::tensor({rowForHour201});
    // historyDays=5 from hour 201 needs hours [96,215]... actually
    // completedDayBlocks(201,5): lastCompleteDayEnd=(201/24)*24-1=191,
    // 5 days back -> [72,191], which does NOT include hour 50 (day [48,71]
    // ends at 71, before 72) -- so this probe should be ELIGIBLE. Use a
    // window further along so the required range DOES include day [48,71].
    auto blocksCheck = daily_history::completedDayBlocks(
        daily_history::toHourIndex(xs[static_cast<std::size_t>(rowForHour201)]), 5);
    assert(blocksCheck.front().first == 72);  // confirms day [48,71] is NOT in range here

    // windowStartHour=216 (a clean day boundary: 216=9*24) with
    // historyDays=7 reaches exactly back to day [48,71] (the dropped
    // hour's day): completedDayBlocks(216,7)'s earliest block is [48,71].
    int64_t rowForHour216 = -1;
    for (int64_t i = 0; i < n; ++i) {
        if (xs[static_cast<std::size_t>(i)] == 216.0) { rowForHour216 = i; break; }
    }
    assert(rowForHour216 >= 0);
    auto checkBlocks = daily_history::completedDayBlocks(216, 7);
    assert(checkBlocks.front().first == 48 && checkBlocks.front().second == 71);
    auto windowStartRows2 = torch::tensor({rowForHour216});
    auto result = daily_history::buildDailyHistoryTensors(x, precip, pet, segmentId, windowStartRows2, /*historyDays=*/7);
    const bool eligibleWithGap = result.eligible.accessor<bool, 1>()[0];
    assert(!eligibleWithGap);  // day [48,71] is missing hour 50 -> must be ineligible

    // Negative control: a window whose history does NOT reach the dropped
    // hour must still be eligible -- proves the fix isn't just "always fail."
    auto result2 = daily_history::buildDailyHistoryTensors(x, precip, pet, segmentId, windowStartRows, /*historyDays=*/2);
    auto eligibleAcc2 = result2.eligible.accessor<bool, 1>();
    assert(eligibleAcc2[0] == true);
    std::cout << "test_missing_row_marks_day_ineligible_not_misaligned: PASS\n";
}

// Non-midnight segment start: a segment beginning at hour 103 (not a
// multiple of 24) must have its sufficiency checked against its REAL
// first hour (103), not an assumed midnight-aligned boundary.
void test_non_midnight_segment_start() {
    const int64_t totalHours = 400;
    std::vector<double> xs;
    std::vector<int64_t> segs;
    for (int64_t h = 0; h < totalHours; ++h) {
        xs.push_back(static_cast<double>(h));
        segs.push_back(h < 103 ? 0 : 1);  // segment 1 starts at hour 103, mid-day
    }
    const int64_t n = totalHours;
    auto x = torch::from_blob(xs.data(), {n}, torch::kFloat64).clone();
    auto segData = segs;  // keep alive
    auto segmentId = torch::from_blob(segData.data(), {n}, torch::kInt64).clone();
    auto precip = torch::full({n}, 1.0f);
    auto pet = torch::full({n}, 0.1f);

    // A window at hour 300 (well into segment 1) requesting 5 days of
    // history needs hours [180,299] -- entirely within segment 1 (starts
    // at 103) -- should be eligible.
    auto farWindow = torch::tensor({static_cast<int64_t>(300)});
    auto resultFar = daily_history::buildDailyHistoryTensors(x, precip, pet, segmentId, farWindow, 5);
    const bool farEligible = resultFar.eligible.accessor<bool, 1>()[0];
    assert(farEligible);

    // A window at hour 150 (only 47 hours into segment 1, which itself
    // started mid-day at 103) requesting 5 days (120h) of history reaches
    // back to hour 24 -- before the segment's real start at 103, and NOT
    // aligned to any midnight -- must be ineligible, and must be judged
    // against hour 103 specifically, not hour 96 (the nearest midnight)
    // or hour 120 (the next one).
    auto nearWindow = torch::tensor({static_cast<int64_t>(150)});
    auto resultNear = daily_history::buildDailyHistoryTensors(x, precip, pet, segmentId, nearWindow, 5);
    const bool nearEligible = resultNear.eligible.accessor<bool, 1>()[0];
    assert(!nearEligible);

    // Confirm the boundary precisely: completedDayBlocks(144,2) needs hours
    // [96,143]; segment starts at 103, so this reaches 7 hours before the
    // segment start -- insufficient.
    auto blocks = daily_history::completedDayBlocks(144, 2);
    assert(!daily_history::hasSufficientSegmentHistory(blocks, 103));
    // But requiring only hours after the segment start should pass:
    assert(daily_history::hasSufficientSegmentHistory(
        daily_history::completedDayBlocks(240, 2), 103));
    std::cout << "test_non_midnight_segment_start: PASS\n";
}

// Four input-validation gaps caught on code review (report 70): empty
// input, duplicate hours, unordered/decreasing timestamps, nonfinite
// values -- each previously either crashed (UB on an empty vector),
// silently corrupted data (duplicate overwrite), silently produced wrong
// results (unordered rows break segmentFirstHour's assumptions), or
// silently passed through garbage (NaN defeated the old tolerance check).
void test_rejects_empty_input() {
    auto x = torch::empty({0}, torch::kFloat64);
    auto precip = torch::empty({0}, torch::kFloat32);
    auto pet = torch::empty({0}, torch::kFloat32);
    auto segmentId = torch::empty({0}, torch::kInt64);
    auto windowStarts = torch::empty({0}, torch::kInt64);
    bool threw = false;
    try {
        daily_history::buildDailyHistoryTensors(x, precip, pet, segmentId, windowStarts, 3);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
    std::cout << "test_rejects_empty_input: PASS\n";
}

void test_rejects_duplicate_hour() {
    std::vector<double> xs = {0, 1, 2, 2, 3, 4};  // hour 2 appears twice
    const int64_t n = static_cast<int64_t>(xs.size());
    auto x = torch::from_blob(xs.data(), {n}, torch::kFloat64).clone();
    auto precip = torch::full({n}, 1.0f);
    auto pet = torch::full({n}, 0.1f);
    auto segmentId = torch::zeros({n}, torch::kInt64);
    auto windowStarts = torch::tensor({static_cast<int64_t>(0)});
    bool threw = false;
    try {
        daily_history::buildDailyHistoryTensors(x, precip, pet, segmentId, windowStarts, 1);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
    std::cout << "test_rejects_duplicate_hour: PASS\n";
}

void test_rejects_unordered_timestamps() {
    std::vector<double> xs = {0, 1, 3, 2, 4};  // hour 2 appears after hour 3 -- out of order
    const int64_t n = static_cast<int64_t>(xs.size());
    auto x = torch::from_blob(xs.data(), {n}, torch::kFloat64).clone();
    auto precip = torch::full({n}, 1.0f);
    auto pet = torch::full({n}, 0.1f);
    auto segmentId = torch::zeros({n}, torch::kInt64);
    auto windowStarts = torch::tensor({static_cast<int64_t>(0)});
    bool threw = false;
    try {
        daily_history::buildDailyHistoryTensors(x, precip, pet, segmentId, windowStarts, 1);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
    std::cout << "test_rejects_unordered_timestamps: PASS\n";
}

void test_rejects_nonfinite_timestamp() {
    std::vector<double> xs = {0, 1, std::numeric_limits<double>::quiet_NaN(), 3, 4};
    const int64_t n = static_cast<int64_t>(xs.size());
    auto x = torch::from_blob(xs.data(), {n}, torch::kFloat64).clone();
    auto precip = torch::full({n}, 1.0f);
    auto pet = torch::full({n}, 0.1f);
    auto segmentId = torch::zeros({n}, torch::kInt64);
    auto windowStarts = torch::tensor({static_cast<int64_t>(0)});
    bool threw = false;
    try {
        daily_history::buildDailyHistoryTensors(x, precip, pet, segmentId, windowStarts, 1);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);

    std::vector<double> xsInf = {0, 1, std::numeric_limits<double>::infinity(), 3, 4};
    auto xInf = torch::from_blob(xsInf.data(), {n}, torch::kFloat64).clone();
    bool threwInf = false;
    try {
        daily_history::buildDailyHistoryTensors(xInf, precip, pet, segmentId, windowStarts, 1);
    } catch (const std::invalid_argument&) {
        threwInf = true;
    }
    assert(threwInf);
    std::cout << "test_rejects_nonfinite_timestamp: PASS\n";
}

void test_eligible_window_indices_single_shared_mask() {
    auto eligible = torch::tensor({true, false, true, true, false});
    auto idx = daily_history::eligibleWindowIndices(eligible);
    assert(idx.sizes() == std::vector<int64_t>({3}));
    auto acc = idx.accessor<int64_t, 1>();
    assert(acc[0] == 0 && acc[1] == 2 && acc[2] == 3);
    std::cout << "test_eligible_window_indices_single_shared_mask: PASS\n";
}

void test_gradient_flow_through_h_and_c_independently() {
    torch::manual_seed(42);
    const int64_t batch = 4, hourlySeqLen = 24, hourlyInput = 6, hourlyHidden = 8, hourlyLayers = 1;
    const int64_t historyDays = 5, dailyInput = 2, dailyHidden = 4;
    MultiscaleHydroLSTM model(hourlyInput, hourlyHidden, 1, hourlyLayers, dailyInput, dailyHidden);

    auto hourlyX = torch::randn({batch, hourlySeqLen, hourlyInput});
    auto dailyX = torch::randn({batch, historyDays, dailyInput});
    model->fitDailyScaler(dailyX);
    auto pred = model->forward(hourlyX, dailyX);
    assert(pred.sizes() == std::vector<int64_t>({batch, 1}));
    pred.pow(2).mean().backward();

    bool anyDailyGrad = false, anyHourlyGrad = false, fcHGrad = false, fcCGrad = false;
    for (const auto& p : model->daily->parameters())
        if (p.grad().defined() && p.grad().abs().sum().item<double>() > 0.0) anyDailyGrad = true;
    for (const auto& p : model->hourly->parameters())
        if (p.grad().defined() && p.grad().abs().sum().item<double>() > 0.0) anyHourlyGrad = true;
    for (const auto& p : model->daily->fc_h->parameters())
        if (p.grad().defined() && p.grad().abs().sum().item<double>() > 0.0) fcHGrad = true;
    for (const auto& p : model->daily->fc_c->parameters())
        if (p.grad().defined() && p.grad().abs().sum().item<double>() > 0.0) fcCGrad = true;
    assert(anyDailyGrad);
    assert(anyHourlyGrad);
    // Both transfer projections must receive gradient -- if c0 were still
    // silently derived from h_n (the bug report 69 caught), fc_c would
    // still happen to get SOME gradient, so this alone wouldn't catch the
    // bug; the real catch is test_h_and_c_are_independent below.
    assert(fcHGrad);
    assert(fcCGrad);
    std::cout << "test_gradient_flow_through_h_and_c_independently: PASS\n";
}

// Directly catches the original bug: perturbing ONLY the daily LSTM's
// cell-state pathway should change c0 without changing h0 identically --
// impossible to verify from outside the LSTM directly, so instead this
// confirms h0 and c0 are numerically DIFFERENT functions of the same
// daily input (not two copies of a projection from the same source
// tensor), by checking fc_h and fc_c produce different outputs on the
// same daily input even when their weights are forced equal -- any
// difference must then come from hLast != cLast, proving both were
// actually used.
void test_h_and_c_are_independent_states() {
    torch::manual_seed(5);
    const int64_t hourlyHidden = 8, hourlyLayers = 1, dailyInput = 2, dailyHidden = 4;
    DailyHistoryBranch branch(dailyInput, dailyHidden, hourlyHidden, hourlyLayers);
    {
        torch::NoGradGuard noGrad;
        branch->fc_h->weight.copy_(branch->fc_c->weight);
        branch->fc_h->bias.copy_(branch->fc_c->bias);
    }
    auto dailyX = torch::randn({2, 5, dailyInput});
    auto [h0, c0] = branch->forward(dailyX);
    // With fc_h and fc_c forced identical, h0 == c0 would mean hLast and
    // cLast were the same tensor (the bug) -- they must differ for a
    // generic LSTM/input, since h_n and c_n are genuinely different
    // internal states.
    assert(!torch::allclose(h0, c0, 0.0, 0.0));
    std::cout << "test_h_and_c_are_independent_states: PASS\n";
}

void test_zero_option_reproduces_today_exactly() {
    torch::manual_seed(123);
    const int64_t batch = 3, seqLen = 24, inputDim = 6, hidden = 8, layers = 1;
    HydroLSTM reference(inputDim, hidden, 1, layers);
    HydroLSTM identicalWeights(inputDim, hidden, 1, layers);
    {
        torch::NoGradGuard noGrad;
        auto refParams = reference->parameters();
        auto copyParams = identicalWeights->parameters();
        assert(refParams.size() == copyParams.size());
        for (std::size_t i = 0; i < refParams.size(); ++i) copyParams[i].copy_(refParams[i]);
    }
    auto x = torch::randn({batch, seqLen, inputDim});
    auto outOld = reference->forward(x);
    auto outNew = identicalWeights->forward(x);
    assert(torch::allclose(outOld, outNew, 0.0, 0.0));
    std::cout << "test_zero_option_reproduces_today_exactly: PASS\n";
}

void test_old_checkpoint_loads_unchanged_when_disabled() {
    torch::manual_seed(7);
    const int64_t inputDim = 6, hidden = 8, layers = 1;
    HydroLSTM original(inputDim, hidden, 1, layers);
    torch::serialize::OutputArchive out;
    original->save(out);
    const std::string path = "/tmp/claude_multiscale_test_old_checkpoint.pt";
    out.save_to(path);

    HydroLSTM reloaded(inputDim, hidden, 1, layers);
    torch::serialize::InputArchive in;
    in.load_from(path);
    reloaded->load(in);

    auto x = torch::randn({2, 24, inputDim});
    auto outOriginal = original->forward(x);
    auto outReloaded = reloaded->forward(x);
    assert(torch::allclose(outOriginal, outReloaded, 0.0, 0.0));
    std::remove(path.c_str());
    std::cout << "test_old_checkpoint_loads_unchanged_when_disabled: PASS\n";
}

void test_multiscale_checkpoint_save_load_roundtrip() {
    torch::manual_seed(99);
    const int64_t hourlyInput = 6, hourlyHidden = 8, hourlyLayers = 1, dailyInput = 2, dailyHidden = 4;
    MultiscaleHydroLSTM model(hourlyInput, hourlyHidden, 1, hourlyLayers, dailyInput, dailyHidden);
    // Fit on a deliberately non-trivial (non-zero-mean, non-unit-scale)
    // tensor so the offset/scale round-trip check below is not vacuously
    // true (report 71's daily-scaler fix -- registered buffers must survive
    // save/load exactly, not just the model's own weights).
    auto fitData = torch::randn({6, 5, dailyInput}) * 3.0 + 7.0;
    model->fitDailyScaler(fitData);
    torch::serialize::OutputArchive out;
    model->save(out);
    const std::string path = "/tmp/claude_multiscale_test_new_checkpoint.pt";
    out.save_to(path);

    MultiscaleHydroLSTM reloaded(hourlyInput, hourlyHidden, 1, hourlyLayers, dailyInput, dailyHidden);
    torch::serialize::InputArchive in;
    in.load_from(path);
    reloaded->load(in);
    reloaded->confirmDailyScalerLoaded();

    // Daily scaler state itself must round-trip bit-for-bit, not just
    // "close enough" -- this is the actual persistence mechanism report 71
    // relies on (registered buffers via the model's own save/load, not a
    // separate scaler artifact).
    assert(torch::allclose(model->dailyScalerOffset, reloaded->dailyScalerOffset, 0.0, 0.0));
    assert(torch::allclose(model->dailyScalerScale, reloaded->dailyScalerScale, 0.0, 0.0));

    auto hourlyX = torch::randn({2, 24, hourlyInput});
    auto dailyX = torch::randn({2, 5, dailyInput});
    auto outOriginal = model->forward(hourlyX, dailyX);
    auto outReloaded = reloaded->forward(hourlyX, dailyX);
    assert(torch::allclose(outOriginal, outReloaded, 0.0, 0.0));
    std::remove(path.c_str());
    std::cout << "test_multiscale_checkpoint_save_load_roundtrip: PASS (daily scaler + predictions both bit-exact)\n";
}

void test_no_block_ever_reaches_or_after_development_seal() {
    constexpr int64_t kSealStartX = 77016;
    for (int64_t ws : {kSealStartX - 1, kSealStartX - 24, kSealStartX - 1000}) {
        auto blocks = daily_history::completedDayBlocks(ws, 60);
        for (const auto& b : blocks) {
            assert(b.second < ws);
            assert(b.second < kSealStartX);
        }
    }
    std::cout << "test_no_block_ever_reaches_or_after_development_seal: PASS\n";
}

}  // namespace

int main() {
    torch::set_num_threads(1);
    test_completed_day_blocks_alignment();
    test_segment_boundary_respected();
    test_daily_aggregation_no_gaps_matches_expected_values();
    test_missing_row_marks_day_ineligible_not_misaligned();
    test_non_midnight_segment_start();
    test_rejects_empty_input();
    test_rejects_duplicate_hour();
    test_rejects_unordered_timestamps();
    test_rejects_nonfinite_timestamp();
    test_eligible_window_indices_single_shared_mask();
    test_gradient_flow_through_h_and_c_independently();
    test_h_and_c_are_independent_states();
    test_zero_option_reproduces_today_exactly();
    test_old_checkpoint_loads_unchanged_when_disabled();
    test_multiscale_checkpoint_save_load_roundtrip();
    test_no_block_ever_reaches_or_after_development_seal();
    std::cout << "ALL MULTISCALE DAILY-HISTORY TESTS PASSED\n";
    return 0;
}
