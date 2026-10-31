#pragma once

#include "artifact_loader.h"
#include "shared_scale_inference.h"

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
    torch::Tensor predictSeries(const torch::Tensor& physicalSeries) const;

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

    // The lstm_pinn profiles HydroInferenceSession explicitly rejects
    // (lstm_shared_scale_ablation_zero/_urban are the two this implements;
    // see shared_scale_inference.h) need raw access to the package named by
    // config.hydro_package_path -- a different shape of input than predict/
    // predictSeries take, since the baseline and segment-aware sequences
    // must be built from the raw series, not from an already-prepared
    // tensor. Callers (the UI) should check parseSharedScaleInferenceProfile
    // first and route here instead of constructing a HydroInferenceSession
    // when it returns true. holdoutMode defaults to false (development-only,
    // x>=77016 refused); callers must only ever pass true after verifying
    // their own explicit authorization, exactly as the CLI driver does.
    SharedScaleInferenceResult runSharedScaleAblation(const HydroRunConfig& config,
                                                      const std::string& frozenExperimentDir,
                                                      const std::string& approach,
                                                      bool holdoutMode = false) const;
};
