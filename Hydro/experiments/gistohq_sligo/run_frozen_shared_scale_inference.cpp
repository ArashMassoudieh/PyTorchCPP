// Thin CLI wrapper around the shared, common inference implementation in
// Hydro/evaluation/shared_scale_inference.{h,cpp}. This file no longer
// contains any of the baseline/clip/segment-aware-sequence logic itself --
// that now lives in exactly one place, shared with Hydro/hydropinnwindow.cpp's
// runLoadedInferenceForMode, per the explicit instruction not to maintain
// divergent copies between CLI and UI.
//
// No optimizer is linked into this program. No training loop exists in this
// file or in the shared implementation it calls. No checkpoint is ever
// written.

#include "evaluation/experiment_loader.h"
#include "evaluation/shared_scale_inference.h"

#include <torch/torch.h>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

// Must be passed verbatim as the 5th argument to lift the reserved-period
// seal. Deliberately long and specific so it cannot be typed or defaulted
// accidentally.
const std::string kHoldoutConfirmationToken = "HOLDOUT_MODE_CONFIRMED_RESERVED_PERIOD_READ_AUTHORIZED";

struct Args {
    std::string configPath;
    std::string experimentDir;
    std::string approach;
    std::string outputCsv;
    bool holdoutMode = false;
};

Args parseArgs(int argc, char** argv) {
    if (argc != 5 && argc != 6) {
        throw std::runtime_error(
            "Usage: run_frozen_shared_scale_inference <config.json> <frozen_experiment_dir> <approach> <output.csv> [HOLDOUT_TOKEN]\n"
            "  <frozen_experiment_dir> must contain models/<approach>.pt and scalers.csv from the already-trained job.\n"
            "  By default this refuses any x>=77016 (reserved-period seal). To lift that for an\n"
            "  explicitly authorized holdout run, pass the exact confirmation token as the 5th argument.");
    }
    Args args{argv[1], argv[2], argv[3], argv[4], false};
    if (argc == 6) {
        if (std::string(argv[5]) != kHoldoutConfirmationToken) {
            throw std::runtime_error("5th argument supplied but does not match the required holdout confirmation token. Refusing.");
        }
        args.holdoutMode = true;
    }
    return args;
}

} // namespace

int main(int argc, char** argv) {
    // Matches hydro_batch_runner.cpp's main() exactly -- single-threaded
    // execution is required for deterministic floating-point reduction
    // order in LibTorch's matmul/sum kernels (confirmed by direct comparison,
    // improvement_loop_20261023/47).
    torch::set_num_threads(1);
    torch::set_num_interop_threads(1);

    try {
        const Args args = parseArgs(argc, argv);
        std::cout << "[frozen-inference] config=" << args.configPath << "\n";
        std::cout << "[frozen-inference] frozen_experiment_dir=" << args.experimentDir << "\n";
        if (args.holdoutMode) {
            std::cout << "[frozen-inference] ################################################################\n"
                      << "[frozen-inference] # HOLDOUT MODE ENGAGED -- the x>=77016 reserved-period guard   #\n"
                      << "[frozen-inference] # has been explicitly lifted up to the shared sanity ceiling.  #\n"
                      << "[frozen-inference] # This run WILL read reserved-period rows.                     #\n"
                      << "[frozen-inference] ################################################################\n";
        }

        const auto loaded = HydroExperimentLoader().loadConfig(args.configPath);
        const HydroRunConfig& config = loaded.config;
        std::cout << "[frozen-inference] experiment_id=" << loaded.experiment_id
                  << " pinn_physics_profile=" << config.pinn_physics_profile << "\n";

        SharedScaleInferenceProfile profile;
        if (!parseSharedScaleInferenceProfile(config.pinn_physics_profile, profile)) {
            throw std::runtime_error(
                "Unsupported profile for this driver: '" + config.pinn_physics_profile +
                "'. Only lstm_shared_scale_ablation_zero and lstm_shared_scale_ablation_urban are implemented.");
        }

        const SharedScaleInferenceResult result =
            runSharedScaleInference(config, profile, args.experimentDir, args.approach, args.holdoutMode);
        std::cout << "[frozen-inference] checkpoint hash (verified unchanged before/after): "
                  << result.checkpointSha256 << "\n";

        std::ofstream out(args.outputCsv);
        if (!out) throw std::runtime_error("Cannot open output path: " + args.outputCsv);
        out << "approach,index,split,x,observed,predicted,residual\n";
        char buf[256];
        for (std::size_t i = 0; i < result.x.size(); ++i) {
            const double pred = result.predicted[i];
            const double obs = result.observed[i];
            std::snprintf(buf, sizeof(buf), "%s,%zu,%s,%.17g,%.17g,%.17g,%.17g\n",
                          args.approach.c_str(), i, result.split[i].c_str(), result.x[i], obs, pred, pred - obs);
            out << buf;
        }
        out.close();
        std::cout << "[frozen-inference] wrote " << result.x.size() << " rows to " << args.outputCsv << "\n";
        std::cout << "[frozen-inference] SUCCESS\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[frozen-inference] FAILED: " << ex.what() << "\n";
        return 1;
    }
}
