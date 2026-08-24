#pragma once

#include "artifact_loader.h"

#include <torch/torch.h>

#include <string>

class HydroInferenceRunner {
public:
    torch::Tensor predictFeedForward(const HydroInferenceArtifacts& artifacts,
                                     const std::string& approach,
                                     const torch::Tensor& physicalInputs) const;
    torch::Tensor predictRecurrent(const HydroInferenceArtifacts& artifacts,
                                   const std::string& approach,
                                   const torch::Tensor& physicalSequences) const;
};
