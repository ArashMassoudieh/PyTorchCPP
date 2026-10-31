// Synthetic test proving HydroInferenceSession rejects the four known-
// unsupported lstm_pinn profiles BEFORE attempting to load a checkpoint,
// not after. Uses a deliberately-garbage, invalid model byte vector for
// the rejected profiles: if rejection happened after loading (the bug
// found and fixed in improvement_loop_20261023/50), this would instead
// throw a loader error about the garbage bytes, not the intended
// explicit-rejection message -- so this test fails loudly if the ordering
// regresses, not just if the rejection logic itself regresses.
//
// Only Hydro/evaluation/{inference_runner,artifact_loader,experiment_exporter,
// experiment_loader}.cpp and hydro_checksum.cpp are needed -- no real
// checkpoint, no real package, no reserved observation of any kind.

#include "../../evaluation/inference_runner.h"
#include "../../evaluation/artifact_loader.h"

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

HydroInferenceArtifacts makeArtifactsWithGarbageCheckpoint(const std::string& profile) {
    HydroInferenceArtifacts artifacts;
    artifacts.experiment.config.pinn_physics_profile = profile;
    artifacts.experiment.config.hidden_layers_csv = "4";
    artifacts.experiment.config.activation = "tanh";
    artifacts.experiment.config.lstm_sequence_length = 2;
    // Deliberately invalid: not a real torch checkpoint. If the rejection
    // check ever moves back to after the load attempt, this makes the
    // test fail with a loader error instead of the expected message,
    // rather than silently "passing" for the wrong reason.
    const std::vector<std::uint8_t> garbage = {0x00, 0x01, 0x02, 0x03, 0x04};
    artifacts.models["lstm_pinn"] = {"models/lstm_pinn.pt", "torch-module-v1", "", garbage};
    HydroScalerState identityScaler;
    identityScaler.method = "none";
    identityScaler.shape = {1, 1, 4};
    identityScaler.offset = {0.0, 0.0, 0.0, 0.0};
    identityScaler.scale = {1.0, 1.0, 1.0, 1.0};
    HydroScalerState identityTarget;
    identityTarget.method = "none";
    identityTarget.shape = {1};
    identityTarget.offset = {0.0};
    identityTarget.scale = {1.0};
    artifacts.scalers["lstm_pinn"] = {identityScaler, identityTarget};
    return artifacts;
}

bool rejectedWithExplicitMessage(const std::string& profile) {
    const auto artifacts = makeArtifactsWithGarbageCheckpoint(profile);
    try {
        HydroInferenceSession session(artifacts, "lstm_pinn");
        std::cerr << "FAIL (" << profile << "): no exception thrown at all\n";
        return false;
    } catch (const std::runtime_error& ex) {
        const std::string what = ex.what();
        const bool isExplicitRejection = what.find("does not support lstm_pinn profile") != std::string::npos;
        if (!isExplicitRejection) {
            std::cerr << "FAIL (" << profile << "): threw, but NOT the explicit-rejection message "
                         "(ordering regression -- loader ran first): " << what << "\n";
        }
        return isExplicitRejection;
    }
}

} // namespace

int main() {
    const std::vector<std::string> blocked = {
        "lstm_residual_pinn_hybrid",
        "lstm_shared_scale_ablation_zero",
        "lstm_shared_scale_ablation_urban",
        "lstm_shared_scale_ablation_physics",
    };
    bool allPassed = true;
    for (const auto& profile : blocked) {
        const bool ok = rejectedWithExplicitMessage(profile);
        std::cout << (ok ? "PASS" : "FAIL") << ": '" << profile << "' rejected before checkpoint load\n";
        allPassed = allPassed && ok;
    }

    // Negative control: "water_balance" (and anything else not on the
    // blocked list) must NOT be rejected by this check -- it should
    // proceed to (and fail at, given the garbage bytes) checkpoint
    // loading instead, proving the blacklist doesn't overreach.
    {
        const auto artifacts = makeArtifactsWithGarbageCheckpoint("water_balance");
        bool reachedLoader = false;
        try {
            HydroInferenceSession session(artifacts, "lstm_pinn");
        } catch (const std::exception& ex) {
            const std::string what = ex.what();
            reachedLoader = what.find("does not support lstm_pinn profile") == std::string::npos;
            if (!reachedLoader) std::cerr << "FAIL (water_balance): wrongly rejected: " << what << "\n";
        }
        std::cout << (reachedLoader ? "PASS" : "FAIL")
                  << ": 'water_balance' NOT rejected by the explicit check (reached loader instead)\n";
        allPassed = allPassed && reachedLoader;
    }

    if (!allPassed) {
        std::cerr << "SOME CHECKS FAILED\n";
        return 1;
    }
    std::cout << "ALL EXPLICIT-REJECTION CHECKS PASSED (no real checkpoint, package, or observation touched)\n";
    return 0;
}
