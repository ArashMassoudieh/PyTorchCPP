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

// Additive counterpart (improvement_loop_20261023 reports 66-69) for a
// checkpoint trained by trainSharedScaleAblationMultiscale (approach
// "lstm_multiscale", Zero baseline only -- the only arm this program's
// multiscale screen manifest uses). Does not touch runSharedScaleInference
// above, which remains exactly as verified in reports 58-60.
//
// Result rows cover ELIGIBLE windows only (daily_history::eligibleWindowIndices)
// -- a multiscale checkpoint has no defined forward pass for a window
// lacking its required daily-history length, and there is no fallback (report
// 69). Callers comparing against a control-arm result must intersect on
// SharedScaleInferenceResult::x themselves; the two results are not
// guaranteed to have the same row count or the same length.
SharedScaleInferenceResult runMultiscaleSharedScaleInference(
    const HydroRunConfig& config,
    const std::string& frozenExperimentDir,
    const std::string& approach,
    bool holdoutMode);

// Additive counterpart (improvement_loop_20261023 report 72 / task #143's
// public-entry-point gap) for a checkpoint trained by
// trainSharedScaleAblationMatchedControl -- an ordinary plain HydroLSTM
// ("torch-module-v1", same approach/format as a non-matched-control zero
// run), so this reuses runSharedScaleInference's existing, unmodified Zero
// reload verbatim (never runMultiscaleSharedScaleInference's loader, which
// requires "torch-module-v1-multiscale" and would simply throw on a
// matched-control checkpoint). The one real difference: this filters the
// full, unrestricted reload result down to the SAME eligible-window set
// (daily_history::eligibleWindowIndices) the paired multiscale checkpoint's
// own reload is restricted to, recomputed independently from the same
// package/config rather than persisted anywhere, so a caller comparing this
// against runMultiscaleSharedScaleInference's result gets identical x/split
// coverage without needing to intersect on x itself.
SharedScaleInferenceResult runMatchedControlSharedScaleInference(
    const HydroRunConfig& config,
    const std::string& frozenExperimentDir,
    const std::string& approach,
    bool holdoutMode);
