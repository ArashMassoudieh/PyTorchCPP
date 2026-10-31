// Reproducibility check for HydroInferenceRunner::runSharedScaleAblation --
// the UI-facing entry point added during the Task 6 shared-engine extraction
// (improvement_loop_20261023 reports 50-51) -- against one or more already-
// frozen shared-scale-ablation job directories' own predictions.csv.
//
// Does not hardcode any path: every job directory lives outside this repo
// (under the Dropbox results tree), so paths are passed as arguments. This
// mirrors Hydro/experiments/gistohq_sligo/run_frozen_shared_scale_inference.cpp's
// own CLI-argument convention rather than embedding environment-specific
// paths in committed source.
//
// Usage: verify_six_candidate_reproduction <approach> <job_dir> [<job_dir> ...]
//   Each <job_dir> must contain experiment_config.json, models/<approach>.pt,
//   scalers.csv, predictions.csv, and artifact_manifest.csv from an already-
//   completed, frozen run. <approach> is the same string for every job_dir in
//   one invocation (e.g. "lstm_pinn"); run the tool twice if candidates
//   differ in <approach>.
//
// Strengthened per explicit review feedback: the first version of this tool
// compared the "predicted" column only, which cannot detect an upstream
// data-loading bug that happened to still line up row-for-row with the old
// predictions (e.g. a shifted or mislabeled split). This version compares
// every column predictions.csv records (x, observed, split, predicted) and
// independently re-hashes the checkpoint and scaler files from disk, cross-
// checked against both the shared engine's own reported checkpoint hash and
// artifact_manifest.csv's independently-written record of both files --
// three independent sources, not one computed value trusted on its own.
//
// For each job_dir: loads its config, re-runs runSharedScaleAblation against
// the SAME job_dir (frozen scalers/checkpoint, holdoutMode=false), and
// compares every row's x/observed/split/predicted to predictions.csv exactly.
// Prints PASS/FAIL per directory and a final summary; exits nonzero if any
// directory fails.

#include "../../evaluation/inference_runner.h"
#include "../../evaluation/experiment_loader.h"
#include "../../dataset/hydro_checksum.h"

#include <torch/torch.h>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct ExistingRow {
    double x = 0.0;
    double observed = 0.0;
    std::string split;
    double predicted = 0.0;
};

std::vector<ExistingRow> readExistingPredictions(const std::string& predictionsCsvPath) {
    std::ifstream f(predictionsCsvPath);
    if (!f) throw std::runtime_error("Cannot open " + predictionsCsvPath);
    std::string line;
    std::getline(f, line);  // header: approach,index,split,x,observed,predicted,residual
    std::vector<ExistingRow> rows;
    while (std::getline(f, line)) {
        std::stringstream ss(line);
        std::string tok;
        std::getline(ss, tok, ',');  // approach (unused)
        std::getline(ss, tok, ',');  // index (unused)
        ExistingRow row;
        std::getline(ss, row.split, ',');
        std::getline(ss, tok, ','); row.x = std::stod(tok);
        std::getline(ss, tok, ','); row.observed = std::stod(tok);
        std::getline(ss, tok, ','); row.predicted = std::stod(tok);
        rows.push_back(row);
    }
    return rows;
}

// artifact_manifest.csv format (written independently by the exporter at job
// completion, not by anything in this verifier): a schema-version line, a
// header, then "file,size_bytes,sha256" rows. Returns sha256 for a given
// recorded file name, or empty string if not present.
std::map<std::string, std::string> readManifestHashes(const std::string& manifestPath) {
    std::map<std::string, std::string> hashes;
    std::ifstream f(manifestPath);
    if (!f) return hashes;  // tolerate absence; caller decides whether to require it
    std::string line;
    while (std::getline(f, line)) {
        std::stringstream ss(line);
        std::string file, sizeStr, sha;
        std::getline(ss, file, ',');
        std::getline(ss, sizeStr, ',');
        std::getline(ss, sha, ',');
        if (!file.empty() && !sha.empty() && sha.size() == 64) hashes[file] = sha;
    }
    return hashes;
}

bool verifyOne(const std::string& approach, const std::string& jobDir) {
    const std::string configPath = jobDir + "/experiment_config.json";
    const std::string predictionsPath = jobDir + "/predictions.csv";
    const std::string manifestPath = jobDir + "/artifact_manifest.csv";
    const std::string checkpointPath = jobDir + "/models/" + approach + ".pt";
    const std::string scalersPath = jobDir + "/scalers.csv";
    std::cout << "[verify] " << jobDir << "\n";

    // --- Integrity checks first: if these fail, don't even bother re-running
    // inference against files that may not be what they claim to be. ---
    const auto manifestHashes = readManifestHashes(manifestPath);
    bool integrityOk = true;
    const std::string checkpointRelName = "models/" + approach + ".pt";
    const std::string actualCheckpointHash = sha256File(checkpointPath);
    const std::string actualScalersHash = sha256File(scalersPath);
    if (auto it = manifestHashes.find(checkpointRelName); it != manifestHashes.end()) {
        if (it->second != actualCheckpointHash) {
            std::cout << "  FAIL: checkpoint hash mismatch vs artifact_manifest.csv (manifest="
                      << it->second << " actual=" << actualCheckpointHash << ")\n";
            integrityOk = false;
        }
    } else {
        std::cout << "  WARN: artifact_manifest.csv has no entry for " << checkpointRelName
                  << " -- cannot cross-check checkpoint integrity against it\n";
    }
    if (auto it = manifestHashes.find("scalers.csv"); it != manifestHashes.end()) {
        if (it->second != actualScalersHash) {
            std::cout << "  FAIL: scalers.csv hash mismatch vs artifact_manifest.csv (manifest="
                      << it->second << " actual=" << actualScalersHash << ")\n";
            integrityOk = false;
        }
    } else {
        std::cout << "  WARN: artifact_manifest.csv has no entry for scalers.csv\n";
    }
    if (!integrityOk) return false;

    const auto loaded = HydroExperimentLoader().loadConfig(configPath);
    const auto result = HydroInferenceRunner().runSharedScaleAblation(
        loaded.config, jobDir, approach, /*holdoutMode=*/false);

    // Cross-check the shared engine's own reported checkpoint hash against
    // the independent, direct-from-disk hash computed above -- two different
    // code paths (this file's sha256File() call vs. whatever
    // runSharedScaleAblation computed internally) agreeing is stronger
    // evidence than trusting either alone.
    if (result.checkpointSha256 != actualCheckpointHash) {
        std::cout << "  FAIL: engine-reported checkpoint hash (" << result.checkpointSha256
                  << ") disagrees with independently-computed hash (" << actualCheckpointHash << ")\n";
        return false;
    }

    const auto existing = readExistingPredictions(predictionsPath);
    if (existing.size() != result.predicted.size() || existing.size() != result.x.size() ||
        existing.size() != result.observed.size() || existing.size() != result.split.size()) {
        std::cout << "  FAIL: row count mismatch (existing=" << existing.size()
                  << " recomputed x/observed/split/predicted sizes="
                  << result.x.size() << "/" << result.observed.size() << "/"
                  << result.split.size() << "/" << result.predicted.size() << ")\n";
        return false;
    }

    std::size_t xMismatches = 0, obsMismatches = 0, splitMismatches = 0, predMismatches = 0;
    double maxPredDiff = 0.0, maxXDiff = 0.0, maxObsDiff = 0.0;
    for (std::size_t i = 0; i < existing.size(); ++i) {
        if (existing[i].x != result.x[i]) { ++xMismatches; maxXDiff = std::max(maxXDiff, std::abs(existing[i].x - result.x[i])); }
        if (existing[i].observed != result.observed[i]) { ++obsMismatches; maxObsDiff = std::max(maxObsDiff, std::abs(existing[i].observed - result.observed[i])); }
        if (existing[i].split != result.split[i]) ++splitMismatches;
        if (existing[i].predicted != result.predicted[i]) {
            ++predMismatches;
            maxPredDiff = std::max(maxPredDiff, std::abs(existing[i].predicted - result.predicted[i]));
        }
    }

    const bool pass = (xMismatches == 0 && obsMismatches == 0 && splitMismatches == 0 && predMismatches == 0);
    std::cout << "  n=" << existing.size()
              << " x_mismatches=" << xMismatches << " (max_diff=" << maxXDiff << ")"
              << " observed_mismatches=" << obsMismatches << " (max_diff=" << maxObsDiff << ")"
              << " split_mismatches=" << splitMismatches
              << " predicted_mismatches=" << predMismatches << " (max_diff=" << maxPredDiff << ")"
              << " checkpoint_sha256=" << actualCheckpointHash
              << " -> " << (pass ? "PASS" : "FAIL") << "\n";
    return pass;
}

}  // namespace

int main(int argc, char** argv) {
    torch::set_num_threads(1);
    torch::set_num_interop_threads(1);
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <approach> <job_dir> [<job_dir> ...]\n";
        return 2;
    }
    const std::string approach = argv[1];
    bool allPass = true;
    int count = 0;
    for (int i = 2; i < argc; ++i) {
        try {
            allPass = verifyOne(approach, argv[i]) && allPass;
        } catch (const std::exception& ex) {
            std::cout << "  FAIL: exception: " << ex.what() << "\n";
            allPass = false;
        }
        ++count;
    }
    std::cout << (allPass ? "ALL " : "SOME OF ") << count << " DIRECTORIES PASSED\n";
    return allPass ? 0 : 1;
}
