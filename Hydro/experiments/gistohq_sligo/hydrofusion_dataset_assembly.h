#pragma once

// Shared HydroFusion dataset assembly: turns raw hourly rows into the
// 20-feature tensor / log-space target / origin+target identity triples
// used by both the real-engine integration test and the opt-in
// FFN+HydroFusion production wrapper. Promoted out of the test file so
// both consumers run the IDENTICAL assembly logic rather than risking two
// independently-fallible copies drifting apart -- this project has
// already been burned once by exactly that failure mode (the retired
// hydrofusion_cli_smoke_test.cpp mock).
//
// Deliberately parameterized on `rows` and `cutoffs` (not hardcoded to
// the synthetic fixture or to {150,220,280}): the test passes the
// synthetic fixture and its own fixed cutoffs; a future real-data caller
// would pass real rows and config-derived cutoffs through this same path
// unchanged.

#include "hydrofusion_feature_builder.h"
#include "hydrofusion_log_residual.h"

#include <torch/torch.h>

#include <cstdint>
#include <map>
#include <stdexcept>
#include <vector>

namespace hydrofusion {

inline torch::Tensor featuresToTensorRow(const Features& f) {
    auto values = torch::tensor({f.q_lag_0h, f.q_lag_1h, f.q_lag_3h, f.q_lag_6h, f.q_lag_12h, f.q_lag_24h,
                           f.q_trailing_mean_6h, f.q_trailing_mean_24h, f.q_trailing_mean_72h,
                           f.q_trailing_var_6h, f.q_trailing_var_24h, f.q_trailing_var_72h,
                           f.q_last_hourly_change, f.q_past_slope_6h, f.met_precip_mm_per_h,
                           f.met_temperature_c, f.met_relative_humidity_pct, f.met_wind_speed_m_per_s,
                           f.met_solar_mj_per_m2_per_h, f.met_pet_mm_per_h},
                          torch::kFloat64)
        .to(torch::kFloat32);
    if (!torch::isfinite(values).all().item<bool>()) {
        throw std::overflow_error("HydroFusion feature cannot be represented in model float32.");
    }
    return values;
}

struct DatasetSplit {
    torch::Tensor x;                  // [N,20], physical units
    torch::Tensor y;                  // [N,1], HydroFusion target space
    torch::Tensor originQ;            // [N,1], physical Q(origin) == x's own column 0
    std::vector<int64_t> originX;     // real sample identities, never fabricated
    std::vector<int64_t> targetX;     // originX[i] + 1 -- the time the observation/prediction are AT
    std::vector<double> observedTargetQ;
};

struct PreparedData {
    double qRef = 0.0;
    SplitCutoffs cutoffs{0, 0, 0};
    DatasetSplit train, validation, test;
};

inline PreparedData prepareHydroFusionData(const std::vector<HourlyRow>& rows, TargetMode mode,
                                            const SplitCutoffs& cutoffs) {
    validateSplitCutoffs(cutoffs);
    HourlyIndex idx(rows);
    std::vector<ProvenanceRecord> records = runPersistenceBenchmark(idx, rows, cutoffs);
    const double qRef = computeQRefFromTrainingSamples(records, cutoffs);

    std::map<SplitPartition, std::vector<torch::Tensor>> xBySplit;
    std::map<SplitPartition, std::vector<double>> yBySplit, originQBySplit, observedBySplit;
    std::map<SplitPartition, std::vector<int64_t>> originXBySplit, targetXBySplit;

    for (const auto& r : records) {
        if (!r.included) continue;
        auto sample = buildSample(idx, rows, r.origin_x);
        if (!sample.has_value() || !sample->predictor_availability_mask) continue;
        const double originQ = sample->features.q_lag_0h;
        const double targetQ = r.observed_target_q;
        const double target = computeTrainingTarget(mode, originQ, targetQ, qRef);
        // target_x's own seal/split consistency is checked explicitly,
        // not assumed from origin_x's -- assignSampleSplit already
        // requires both to agree for `r.included` to be true, but
        // re-asserting it here at the point target_x is actually used as
        // the exported identity makes the dependency direct rather than
        // inherited.
        if (r.target_x >= kSealStartX) {
            throw std::runtime_error("prepareHydroFusionData: target_x at/past the reserved-period seal.");
        }
        const auto targetPartition = assignSampleSplit(r.origin_x, r.target_x, cutoffs);
        if (!targetPartition.has_value() || *targetPartition != r.split) {
            throw std::runtime_error(
                "prepareHydroFusionData: target-based split re-derivation disagrees with record.");
        }
        xBySplit[r.split].push_back(featuresToTensorRow(sample->features));
        yBySplit[r.split].push_back(target);
        originQBySplit[r.split].push_back(originQ);
        observedBySplit[r.split].push_back(targetQ);
        originXBySplit[r.split].push_back(r.origin_x);
        targetXBySplit[r.split].push_back(r.target_x);
    }

    auto buildSplit = [&](SplitPartition split) -> DatasetSplit {
        auto it = xBySplit.find(split);
        if (it == xBySplit.end() || it->second.empty()) {
            throw std::runtime_error("prepareHydroFusionData: no eligible rows for a required split.");
        }
        DatasetSplit out;
        out.x = torch::stack(it->second);
        out.y = torch::tensor(yBySplit.at(split), torch::kFloat64).to(torch::kFloat32).reshape({-1, 1});
        out.originQ = torch::tensor(originQBySplit.at(split), torch::kFloat64).to(torch::kFloat32).reshape({-1, 1});
        out.originX = originXBySplit.at(split);
        out.targetX = targetXBySplit.at(split);
        out.observedTargetQ = observedBySplit.at(split);
        return out;
    };

    PreparedData data;
    data.qRef = qRef;
    data.cutoffs = cutoffs;
    data.train = buildSplit(SplitPartition::Train);
    data.validation = buildSplit(SplitPartition::Validation);
    data.test = buildSplit(SplitPartition::Test);
    return data;
}

inline std::vector<double> reconstructHydroFusionColumn(TargetMode mode, const torch::Tensor& predictedTargetSpace,
                                                         const torch::Tensor& originQ, double qRef) {
    const auto predictedAccessor = predictedTargetSpace.contiguous();
    const auto originAccessor = originQ.contiguous();
    std::vector<double> result;
    result.reserve(static_cast<std::size_t>(predictedAccessor.size(0)));
    for (int64_t i = 0; i < predictedAccessor.size(0); ++i) {
        const double predicted = predictedAccessor[i][0].item<double>();
        const double origin = originAccessor[i][0].item<double>();
        result.push_back(reconstructQ(mode, origin, predicted, qRef));
    }
    return result;
}

}  // namespace hydrofusion
