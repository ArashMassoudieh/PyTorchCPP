#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

/**
 * @file precip_accumulation.h
 * @brief Trailing multi-window precipitation accumulation diagnostics
 * (NSE>=0.8 recovery program, Phase 2 Step 3: "6/24/72/168-hour
 * precipitation accumulations as diagnostics").
 *
 * These are reporting/analysis quantities (antecedent-wetness context for
 * flood-event tables, not yet a model input channel - wiring them into any
 * profile's feature contract is a separate, not-yet-made decision), so this
 * is a standalone utility rather than a change to predictorFeatures/
 * makeSequences in lstm_pinn_wrapper.cpp.
 */

// Trailing sum of `precipitation` over the preceding `windowHours` hours
// (inclusive of the current hour), computed separately within each
// contiguous block of `segmentId` - never summing across a block boundary.
// Hours near a block's own start that don't have `windowHours` of history
// yet within that same block get a partial sum over whatever history is
// actually available there (not padded with zero-as-if-dry, and not
// reaching into a different, possibly far-removed-in-time block).
inline torch::Tensor trailingPrecipAccumulation(const torch::Tensor& precipitation,
                                                const torch::Tensor& segmentId,
                                                int64_t windowHours) {
    if (windowHours < 1) throw std::invalid_argument("Accumulation window must be at least 1 hour.");
    const auto precip = precipitation.reshape({-1}).contiguous();
    const auto segments = segmentId.reshape({-1}).contiguous();
    if (precip.size(0) != segments.size(0)) {
        throw std::invalid_argument("Precipitation and segment id must have the same length.");
    }
    const int64_t n = precip.size(0);
    std::vector<float> precipVec(static_cast<std::size_t>(n));
    {
        const auto cpuPrecip = precip.to(torch::kFloat32).to(torch::kCPU).contiguous();
        std::copy(cpuPrecip.data_ptr<float>(), cpuPrecip.data_ptr<float>() + n, precipVec.begin());
    }
    std::vector<float> accumulated(static_cast<std::size_t>(n), 0.0f);

    int64_t begin = 0;
    while (begin < n) {
        const int64_t segment = segments[begin].item<int64_t>();
        int64_t end = begin + 1;
        while (end < n && segments[end].item<int64_t>() == segment) ++end;
        // Running-sum sliding window within [begin, end) only.
        double window = 0.0;
        for (int64_t i = begin; i < end; ++i) {
            window += precipVec[static_cast<std::size_t>(i)];
            const int64_t dropIndex = i - windowHours;
            if (dropIndex >= begin) window -= precipVec[static_cast<std::size_t>(dropIndex)];
            accumulated[static_cast<std::size_t>(i)] = static_cast<float>(window);
        }
        begin = end;
    }
    return torch::from_blob(accumulated.data(), {n}, torch::kFloat32).clone();
}

struct PrecipAccumulationDiagnostics {
    torch::Tensor accum_6h;
    torch::Tensor accum_24h;
    torch::Tensor accum_72h;
    torch::Tensor accum_168h;
};

inline PrecipAccumulationDiagnostics computeStandardPrecipAccumulations(
    const torch::Tensor& precipitation, const torch::Tensor& segmentId) {
    return {
        trailingPrecipAccumulation(precipitation, segmentId, 6),
        trailingPrecipAccumulation(precipitation, segmentId, 24),
        trailingPrecipAccumulation(precipitation, segmentId, 72),
        trailingPrecipAccumulation(precipitation, segmentId, 168),
    };
}
