#pragma once
#include "hydrofusion_feature_builder.h"

namespace hydrofusion {
// Causal LSTM input contract, independent of Torch/model dispatch. Each step
// contains the SAME twenty features as FFN at that step's origin. No padding,
// gap bridging, cross-partition context, or persistent hidden-state claim.
struct ForecastSequence {
    int64_t origin_x = 0, target_x = 0;
    SplitPartition split = SplitPartition::Train;
    std::vector<Features> steps;
    double observed_target_q = 0;
};
inline std::vector<ForecastSequence> buildForecastSequences(
    const std::vector<HourlyRow>& rows, const SplitCutoffs& cutoffs, int64_t length) {
    validateSplitCutoffs(cutoffs);
    if (length <= 0 || length > kSealStartX)
        throw std::invalid_argument("Invalid HydroFusion sequence length.");
    HourlyIndex index(rows);
    const auto records = runPersistenceBenchmark(index, rows, cutoffs);
    std::vector<ForecastSequence> output;
    for (const auto& r : records) {
        if (!r.included || r.origin_x < length-1) continue;
        ForecastSequence seq; seq.origin_x=r.origin_x; seq.target_x=r.target_x;
        seq.split=r.split; seq.observed_target_q=r.observed_target_q;
        bool eligible=true;
        for (int64_t h=r.origin_x-length+1; h<=r.origin_x; ++h) {
            const auto partition=assignSampleSplit(h,r.target_x,cutoffs);
            if (!partition || *partition!=r.split) { eligible=false; break; }
            auto step=buildSample(index,rows,h);
            if (!step || !step->predictor_availability_mask) { eligible=false; break; }
            seq.steps.push_back(step->features);
        }
        if (eligible) output.push_back(std::move(seq));
    }
    return output;
}
} // namespace hydrofusion
