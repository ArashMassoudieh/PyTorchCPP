#pragma once

#include "artifact_loader.h"

#include <torch/torch.h>

#include <memory>
#include <string>

class HydroInferenceSession {
public:
    HydroInferenceSession(const HydroInferenceArtifacts& artifacts, const std::string& approach);
    ~HydroInferenceSession();
    HydroInferenceSession(HydroInferenceSession&&) noexcept;
    HydroInferenceSession& operator=(HydroInferenceSession&&) noexcept;

    torch::Tensor predict(const torch::Tensor& physicalInputs) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class HydroInferenceRunner {
public:
    torch::Tensor predictFeedForward(const HydroInferenceArtifacts& artifacts,
                                     const std::string& approach,
                                     const torch::Tensor& physicalInputs) const;
    torch::Tensor predictRecurrent(const HydroInferenceArtifacts& artifacts,
                                   const std::string& approach,
                                   const torch::Tensor& physicalSequences) const;
};
