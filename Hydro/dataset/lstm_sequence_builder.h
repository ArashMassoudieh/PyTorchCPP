#pragma once

#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

// Extracted from lstmnetworkwrapper.cpp (2026-10-20) so the multi-basin
// regional-pretraining pooler (multi_basin_pool.h) uses the IDENTICAL
// gap-aware windowing logic as single-basin LSTM training, instead of a
// second, independently-maintained copy that could silently drift out of
// sync - exactly the class of bug the test_end_x fix (resolve_test_end_x.h)
// already found once this session from two copies of similar logic
// disagreeing. lstmnetworkwrapper.cpp includes this header and no longer
// defines its own copy.
struct LstmSequenceData {
    torch::Tensor xSeq;
    torch::Tensor ySeq;
    torch::Tensor plotSeq;
};

// enforceRegularTime: when true, a window is only kept if every consecutive
// pair of timestamps inside it has the same spacing as the series' own
// minimum positive interval - silently skipping (not erroring on) any
// window whose span crosses a time-irregularity. This is why sequence-index
// and exported-x (plotX) are not related by a constant offset once any gap
// exists - see resolve_test_end_x.h's doc comment for the bug this caused
// when assumed otherwise.
inline LstmSequenceData makeLstmSequences(const torch::Tensor& x,
                                          const torch::Tensor& y,
                                          const torch::Tensor& plotX,
                                          int sequenceLength,
                                          const bool enforceRegularTime) {
    if (!x.defined() || !y.defined() || !plotX.defined() || x.dim() != 2 || y.dim() != 2 ||
        plotX.numel() != x.size(0) || y.size(0) != x.size(0)) {
        throw std::runtime_error("LSTM sequence builder expects 2-D x/y tensors.");
    }
    sequenceLength = std::max(2, sequenceLength);
    const int64_t n = x.size(0);
    const int64_t inputDim = x.size(1);
    const int64_t m = n - sequenceLength + 1;
    if (m < 4) throw std::runtime_error("Too few samples for requested LSTM sequence length.");

    std::vector<torch::Tensor> windows, targets, endpoints;
    windows.reserve(static_cast<size_t>(m));
    double expectedDt = 0.0;
    if (enforceRegularTime) {
        const auto flatTime = plotX.reshape({-1});
        auto intervals = flatTime.slice(0, 1, n) - flatTime.slice(0, 0, n - 1);
        const auto positive = intervals.index({intervals > 0});
        if (positive.numel() == 0) throw std::runtime_error("LSTM package timestamps are not increasing.");
        expectedDt = positive.min().item<double>();
    }
    for (int64_t i = 0; i < m; ++i) {
        if (enforceRegularTime) {
            const auto time = plotX.slice(0, i, i + sequenceLength).reshape({-1});
            const auto intervals = time.slice(0, 1, sequenceLength) - time.slice(0, 0, sequenceLength - 1);
            const double tolerance = std::max(1.0e-12, std::abs(expectedDt) * 1.0e-6);
            if ((torch::abs(intervals - expectedDt) > tolerance).any().item<bool>()) continue;
        }
        windows.push_back(x.slice(0, i, i + sequenceLength).unsqueeze(0));
        targets.push_back(y.slice(0, i + sequenceLength - 1, i + sequenceLength));
        endpoints.push_back(plotX.slice(0, i + sequenceLength - 1, i + sequenceLength));
    }
    if (windows.size() < 4) throw std::runtime_error("Too few contiguous samples for requested LSTM sequence length.");

    LstmSequenceData seq;
    seq.xSeq = torch::cat(windows, 0).contiguous().view({static_cast<int64_t>(windows.size()), sequenceLength, inputDim});
    seq.ySeq = torch::cat(targets, 0).contiguous();
    seq.plotSeq = torch::cat(endpoints, 0).contiguous();
    return seq;
}
