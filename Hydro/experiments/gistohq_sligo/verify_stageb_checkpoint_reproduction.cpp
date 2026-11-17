// Semantic checkpoint reload-reproduction verification for the Stage B
// multiscale-memory-screen jobs (improvement_loop_20261023), per
// CLAUDE_SUPERVISOR_TASKS.md's 2026-11-16 action checklist item 3: "Reload
// checkpoints through the real intended inference path ... and confirm
// reproduction of the exported predictions, preserving model/scaler
// identity and combined-output clipping. Structural ZIP checks are
// insufficient."
//
// Adapted directly from the existing, proven
// verify_six_candidate_reproduction.cpp (reports 50-51), which verified
// HydroInferenceRunner::runSharedScaleAblation reload-reproduction for the
// shared-scale-ablation candidates. The only real difference here: Stage B
// jobs dispatch on config.multiscale_matched_control /
// multiscale_daily_history_days, so this calls
// runMatchedControlSharedScaleInference (approach "lstm_pinn",
// matched_control=true) or runMultiscaleSharedScaleInference (approach
// "lstm_multiscale") instead of runSharedScaleAblation -- both are the
// real, unmodified entry points shared_scale_inference.h declares for
// exactly this purpose, already exercised by synthetic_multiscale_smoke_test.cpp
// against a synthetic package; this is their first use against the real,
// already-completed 18 Stage B job directories.
//
// For each job_dir: independently re-hashes the checkpoint/scalers from
// disk, cross-checks both against artifact_manifest.csv's own independently
// -written record, reloads via the real dispatch, cross-checks the engine's
// own reported checkpoint hash against the independent hash, then compares
// every row of x/observed/split/predicted against predictions.csv exactly.
// No optimizer is linked anywhere behind the called functions (see
// shared_scale_inference.h's own header comment); runs under
// torch::NoGradGuard; never writes to the checkpoint.
//
// Usage: verify_stageb_checkpoint_reproduction <job_dir> [<job_dir> ...]
//   Each job_dir's own experiment_config.json determines approach and
//   dispatch (matched-control vs multiscale) -- not a command-line flag,
//   so a job cannot be accidentally verified against the wrong path.
//
// Correction (2026-11-18, independent review): the first version of this
// tool loaded only experiment_config.json and left train_end_x/
// validation_end_x/test_end_x at HydroRunConfig's -1 ("unbounded")
// default, because this exporter (committed HEAD 592f684) never writes
// those fields into the exported config -- already documented in every
// job's own boundary_provenance.json sidecar, written specifically because
// of this exact gap, and already consulted correctly by this program's
// Python analysis path (compute_stageb_matched_support_metrics.py's
// _verify_sidecar). That first version produced an initially-reported
// large split-label mismatch with zero x/observed/predicted mismatch,
// which was misattributed to a possible hazard in
// chronological_split.h's index-based boundary computation. Independent
// review correctly identified the simpler, actual cause: a missing input,
// not an engine bug -- and separately confirmed that
// configuredChronologicalSplit's exact-bound path (resolveTestEndSequenceCount)
// already handles this correctly once given real bounds; it is the ratio
// fallback for ABSENT bounds that produced the original mismatch, exactly
// as this fix now avoids. Fixed below by resolving train_end_x/
// validation_end_x/test_end_x from the hash-verified staged input config
// (resolveBoundsFromSidecar) before the reload call, exactly as the Python
// path already does -- now also cross-checked against the independently
// frozen manifest hash, not merely the sidecar's own self-reported hash
// (same two-tier discipline as compute_stageb_matched_support_metrics.py's
// _verify_sidecar).
//
// Second correction (2026-11-18, same review): readExistingPredictions had
// no fail-closed validation at all (no finite/integer/nonnegative/seal/
// duplicate/header-schema checks) -- fixed in
// verify_stageb_checkpoint_reproduction_parsing.h, with guards applied to
// EVERY row before the approach filter. A missing artifact_manifest.csv
// entry for the checkpoint/scalers previously only WARNed and still
// returned success; this is now a hard failure, since an unverifiable
// integrity claim must not be silently treated as a passing one.

#include "verify_stageb_checkpoint_reproduction_parsing.h"

#include "../../evaluation/experiment_loader.h"
#include "../../evaluation/shared_scale_inference.h"
#include "../../dataset/hydro_checksum.h"

#include <torch/torch.h>

#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

const std::string kFrozenManifestHashesPath =
    "/mnt/3rd900/Projects/PyTorchCPP/build-hydrobatch/multiscale_stage_b_manifests_staged/"
    "STAGED_MANIFEST_HASHES.txt";

// Resolves the (train_end_x, validation_end_x, test_end_x) this job was
// actually trained/exported with, via boundary_provenance.json's staged
// input config -- NOT the exported experiment_config.json, which this
// checkpoint's own sidecar documents as never having carried these fields
// in the first place. The staged file's hash is checked against BOTH the
// sidecar's own declared hash AND the independently frozen manifest (two
// mutually-consistent-but-wrong values would otherwise pass if only one
// anchor were checked).
struct ResolvedBounds { int64_t trainEndX, validationEndX, testEndX; };

ResolvedBounds resolveBoundsFromSidecar(const std::string& jobDir, const std::string& label) {
    const std::string sidecarPath = jobDir + "/boundary_provenance.json";
    const std::string sidecarJson = readFile(sidecarPath);
    const std::string stagedPath = extractString(sidecarJson, "staged_input_config_path");
    const std::string declaredHash = extractString(sidecarJson, "staged_input_config_sha256");

    const std::string actualStagedHash = sha256File(stagedPath);
    if (actualStagedHash != declaredHash) {
        throw std::runtime_error(label + ": staged input " + stagedPath +
                                  "'s actual hash does not match the sidecar's declared hash -- "
                                  "refusing to trust its content");
    }

    const auto frozenHashes = parseFrozenManifestHashes(kFrozenManifestHashesPath);
    const std::string resolvedStagedPath = std::filesystem::weakly_canonical(stagedPath).string();
    const auto it = frozenHashes.find(resolvedStagedPath);
    if (it == frozenHashes.end()) {
        throw std::runtime_error(label + ": staged input " + stagedPath +
                                  " is not present in the frozen manifest hash file at all -- "
                                  "cannot independently verify it against anything");
    }
    if (it->second != actualStagedHash) {
        throw std::runtime_error(label + ": staged input " + stagedPath + "'s actual hash (" +
                                  actualStagedHash + ") does not match the independently frozen "
                                  "manifest's expected hash (" + it->second + ")");
    }

    const std::string stagedJson = readFile(stagedPath);
    return {
        extractInt64(stagedJson, "train_end_x"),
        extractInt64(stagedJson, "validation_end_x"),
        extractInt64(stagedJson, "test_end_x"),
    };
}

std::map<std::string, std::string> readManifestHashes(const std::string& manifestPath) {
    std::map<std::string, std::string> hashes;
    std::ifstream f(manifestPath);
    if (!f) return hashes;
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

bool verifyOne(const std::string& jobDir) {
    const std::string configPath = jobDir + "/experiment_config.json";
    const std::string predictionsPath = jobDir + "/predictions.csv";
    const std::string manifestPath = jobDir + "/artifact_manifest.csv";
    std::cout << "[verify] " << jobDir << "\n";

    auto loaded = HydroExperimentLoader().loadConfig(configPath);
    auto& config = loaded.config;
    const bool isMatchedControl = config.multiscale_matched_control;
    const bool isMultiscale = !isMatchedControl && config.multiscale_daily_history_days > 0;
    const std::string approach = isMultiscale ? "lstm_multiscale" : "lstm_pinn";
    const std::string checkpointPath = jobDir + "/models/" + approach + ".pt";
    const std::string scalersPath = jobDir + "/scalers.csv";

    std::cout << "  dispatch: " << (isMultiscale ? "runMultiscaleSharedScaleInference"
                                                  : isMatchedControl ? "runMatchedControlSharedScaleInference"
                                                                      : "NONE (neither multiscale nor matched-control)") << "\n";
    if (!isMultiscale && !isMatchedControl) {
        std::cout << "  FAIL: job is neither multiscale_daily_history_days>0 nor "
                     "multiscale_matched_control -- not a Stage B job this tool handles\n";
        return false;
    }

    // The exported experiment_config.json never carries train_end_x/
    // validation_end_x/test_end_x for these jobs (confirmed by independent
    // review, 2026-11-18) -- resolve them from the hash-verified staged
    // input config via the sidecar instead of leaving them at
    // HydroRunConfig's -1 ("unbounded") default, which would otherwise
    // silently mis-split the reload.
    const auto bounds = resolveBoundsFromSidecar(jobDir, "job");
    config.train_end_x = bounds.trainEndX;
    config.validation_end_x = bounds.validationEndX;
    config.test_end_x = bounds.testEndX;
    std::cout << "  resolved bounds from sidecar: train_end_x=" << config.train_end_x
              << " validation_end_x=" << config.validation_end_x
              << " test_end_x=" << config.test_end_x << "\n";

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
        // A missing manifest entry means the checkpoint's integrity cannot
        // be independently verified at all -- that is a failure, not a
        // warning to note and proceed past (2026-11-18 correction).
        std::cout << "  FAIL: artifact_manifest.csv has no entry for " << checkpointRelName
                  << " -- cannot verify checkpoint integrity\n";
        integrityOk = false;
    }
    if (auto it = manifestHashes.find("scalers.csv"); it != manifestHashes.end()) {
        if (it->second != actualScalersHash) {
            std::cout << "  FAIL: scalers.csv hash mismatch vs artifact_manifest.csv (manifest="
                      << it->second << " actual=" << actualScalersHash << ")\n";
            integrityOk = false;
        }
    } else {
        std::cout << "  FAIL: artifact_manifest.csv has no entry for scalers.csv -- cannot "
                     "verify scalers integrity\n";
        integrityOk = false;
    }
    if (!integrityOk) return false;

    SharedScaleInferenceResult result;
    try {
        if (isMultiscale) {
            result = runMultiscaleSharedScaleInference(config, jobDir, approach, /*holdoutMode=*/false);
        } else {
            result = runMatchedControlSharedScaleInference(config, jobDir, approach, /*holdoutMode=*/false);
        }
    } catch (const std::exception& ex) {
        std::cout << "  FAIL: exception during reload: " << ex.what() << "\n";
        return false;
    }

    if (result.checkpointSha256 != actualCheckpointHash) {
        std::cout << "  FAIL: engine-reported checkpoint hash (" << result.checkpointSha256
                  << ") disagrees with independently-computed hash (" << actualCheckpointHash << ")\n";
        return false;
    }

    const auto existing = readExistingPredictions(predictionsPath, approach);
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
    double sseExported = 0.0, sseReloaded = 0.0;
    for (std::size_t i = 0; i < existing.size(); ++i) {
        if (existing[i].x != result.x[i]) { ++xMismatches; maxXDiff = std::max(maxXDiff, std::abs(existing[i].x - result.x[i])); }
        if (existing[i].observed != result.observed[i]) { ++obsMismatches; maxObsDiff = std::max(maxObsDiff, std::abs(existing[i].observed - result.observed[i])); }
        if (existing[i].split != result.split[i]) ++splitMismatches;
        const double errExported = existing[i].observed - existing[i].predicted;
        const double errReloaded = existing[i].observed - result.predicted[i];
        sseExported += errExported * errExported;
        sseReloaded += errReloaded * errReloaded;
        if (existing[i].predicted != result.predicted[i]) {
            ++predMismatches;
            const double diff = std::abs(existing[i].predicted - result.predicted[i]);
            maxPredDiff = std::max(maxPredDiff, diff);
            // Per-row detail for every mismatched row, in the job's own
            // native unit (mm/h) -- no epsilon-based claim, the actual
            // values speak for themselves (2026-11-18 correction,
            // independent review: float32 epsilon is a dimensionless
            // relative spacing near 1.0, not a universal absolute-error
            // bound, so it does not by itself justify tolerance here).
            const double relDiff = existing[i].predicted != 0.0 ? diff / std::abs(existing[i].predicted) : std::numeric_limits<double>::infinity();
            std::cout << "  MISMATCH ROW: x=" << existing[i].x << " split=" << existing[i].split
                      << " observed=" << existing[i].observed
                      << " exported_predicted=" << existing[i].predicted
                      << " reloaded_predicted=" << result.predicted[i]
                      << " abs_diff_mm_h=" << diff
                      << " rel_diff=" << relDiff << "\n";
        }
    }
    const double rmseExported = std::sqrt(sseExported / existing.size());
    const double rmseReloaded = std::sqrt(sseReloaded / existing.size());
    if (predMismatches > 0) {
        std::cout << "  METRIC IMPACT: job RMSE from exported predictions=" << rmseExported
                  << " mm/h; RMSE from reloaded predictions=" << rmseReloaded
                  << " mm/h; abs RMSE difference=" << std::abs(rmseExported - rmseReloaded)
                  << " mm/h (over all " << existing.size() << " rows, from " << predMismatches
                  << " differing row(s))\n";
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
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <job_dir> [<job_dir> ...]\n";
        return 2;
    }
    bool allPass = true;
    int count = 0;
    for (int i = 1; i < argc; ++i) {
        try {
            allPass = verifyOne(argv[i]) && allPass;
        } catch (const std::exception& ex) {
            std::cout << "  FAIL: exception: " << ex.what() << "\n";
            allPass = false;
        }
        ++count;
    }
    std::cout << (allPass ? "ALL " : "SOME OF ") << count << " DIRECTORIES PASSED\n";
    return allPass ? 0 : 1;
}
