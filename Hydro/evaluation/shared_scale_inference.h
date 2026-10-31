#pragma once

// The single, shared, tested implementation of inference for
// lstm_shared_scale_ablation_zero and lstm_shared_scale_ablation_urban
// (Hydro/models/lstm_pinn_wrapper.cpp's trainSharedScaleAblation). Both the
// CLI driver (Hydro/experiments/gistohq_sligo/run_frozen_shared_scale_inference.cpp)
// and the Qt UI (Hydro/hydropinnwindow.cpp's runLoadedInferenceForMode) call
// this -- neither maintains its own copy of the baseline/clip/segment-aware
// sequence logic. Extracted per the explicit instruction not to keep
// divergent copies between CLI and UI (improvement_loop_20261023 Task 6).
//
// No optimizer is linked by anything this header declares. No training loop
// exists anywhere behind this interface. The entire computation runs under
// torch::NoGradGuard. Scalers are loaded via TensorScaler::importState only,
// never fit. The checkpoint is loaded once, hashed before and after the
// forward pass, and never written to.

#include "artifact_loader.h"
#include "../models/hydro_run_types.h"

#include <torch/torch.h>

#include <string>
#include <vector>

enum class SharedScaleInferenceProfile { Zero, UrbanManaged };

// Returns false (does not throw) if `profile` is not one of the two this
// implementation supports -- callers (HydroInferenceSession, the CLI driver)
// use this to decide whether to route here at all, keeping the "explicit
// rejection for unsupported profiles" behavior for everything else.
bool parseSharedScaleInferenceProfile(const std::string& profile, SharedScaleInferenceProfile& out);

struct SharedScaleInferenceResult {
    std::vector<double> x;
    std::vector<double> observed;
    std::vector<double> predicted;
    std::vector<std::string> split; // "train" | "validation" | "test" | "beyond_test_end"
    std::string checkpointSha256;   // hashed immediately before the forward pass
};

// Development-period seal (x>=77016, 2023-10-15T00:00:00Z) is enforced
// unless holdoutMode is true, in which case the bound relaxes to a generous
// but finite sanity ceiling (still refuses a grossly wrong/corrupted
// package). Callers are responsible for only ever passing holdoutMode=true
// after verifying their own explicit authorization token -- this function
// does not gate that itself, by design, so it has exactly one safety
// behavior to reason about regardless of caller.
SharedScaleInferenceResult runSharedScaleInference(
    const HydroRunConfig& config,
    SharedScaleInferenceProfile profile,
    const std::string& frozenExperimentDir,
    const std::string& approach,
    bool holdoutMode);
