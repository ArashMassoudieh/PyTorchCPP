#pragma once
#include "hydrofusion_sequence_builder.h"
#include "hydrofusion_dataset_assembly.h"
#include <set>
#include <algorithm>

namespace hydrofusion {
// Additive LSTM assembly. Scales MUST be fitted on the returned Train tensors;
// q_ref is recomputed on final sequence-eligible training TARGET timestamps.
inline PreparedData prepareHydroFusionSequenceData(const std::vector<HourlyRow>& rows,
    TargetMode mode,const SplitCutoffs& cutoffs,int64_t length,bool developmentOnly=false,const std::set<int64_t>* targetMask=nullptr) {
    validateTargetMode(mode);
    validateSplitCutoffs(cutoffs);
    // Reject later identities BEFORE sequence construction can inspect any Q.
    if(developmentOnly){
        HourlyIndex identities(rows);(void)identities;
        for(const auto& row:rows)if(row.x>cutoffs.validation_end_x)throw std::invalid_argument("Development assembly refuses rows after validation before reading values.");
    }
    auto sequences=buildForecastSequences(rows,cutoffs,length);
    if(targetMask)sequences.erase(std::remove_if(sequences.begin(),sequences.end(),[&](const auto& sample){return targetMask->count(sample.target_x)==0;}),sequences.end());
    HourlyIndex index(rows);
    auto records=runPersistenceBenchmark(index,rows,cutoffs);
    std::set<int64_t> eligible;
    for(const auto& seq:sequences) eligible.insert(seq.target_x);
    for(auto& record:records) record.included=record.included && eligible.count(record.target_x)!=0;
    PreparedData data;data.cutoffs=cutoffs;
    data.qRef=computeQRefFromTrainingSamples(records,cutoffs);
    std::map<SplitPartition,std::vector<torch::Tensor>> inputs,targets,origins;
    std::map<SplitPartition,std::vector<int64_t>> originX,targetX;
    std::map<SplitPartition,std::vector<double>> observed;
    for(const auto& seq:sequences) {
        std::vector<torch::Tensor> steps;
        for(const auto& step:seq.steps) steps.push_back(featuresToTensorRow(step));
        inputs[seq.split].push_back(torch::stack(steps));
        const auto q=seq.steps.back().q_lag_0h;
        const double y=computeTrainingTarget(mode,q,seq.observed_target_q,data.qRef);
        auto value=torch::tensor({y},torch::kFloat64).to(torch::kFloat32);
        if(!torch::isfinite(value).all().item<bool>())throw std::overflow_error("Sequence target float32 overflow.");
        targets[seq.split].push_back(value);
        origins[seq.split].push_back(torch::tensor({q},torch::kFloat64));
        originX[seq.split].push_back(seq.origin_x);targetX[seq.split].push_back(seq.target_x);
        observed[seq.split].push_back(seq.observed_target_q);
    }
    auto split=[&](SplitPartition partition) {
        if(inputs[partition].empty())throw std::invalid_argument("Sequence assembly requires nonempty Train/Validation/Test.");
        DatasetSplit out;out.x=torch::stack(inputs[partition]);out.y=torch::stack(targets[partition]);
        out.originQ=torch::stack(origins[partition]);out.originX=originX[partition];out.targetX=targetX[partition];out.observedTargetQ=observed[partition];return out;
    };
    data.train=split(SplitPartition::Train);data.validation=split(SplitPartition::Validation);if(!developmentOnly)data.test=split(SplitPartition::Test);
    return data;
}
} // namespace hydrofusion
