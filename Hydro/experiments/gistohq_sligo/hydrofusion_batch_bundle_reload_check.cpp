// Fresh-process verification for the REAL CLI-produced "ffn_hydrofusion"
// bundle: given only the directory a genuine `HydroBatch <batchfile>
// <outdir>` run just wrote (plus its two HydroFusion sidecars), this
// program -- itself a separate OS process from HydroBatch, invoked after
// it has already exited -- independently reloads the bundle through the
// real, unmodified HydroArtifactLoader/HydroInferenceSession, re-derives
// the same deterministic synthetic test split from the metadata sidecar's
// own recorded cutoffs (not hardcoded, not assumed), predicts, and
// compares against the real exporter's own recorded predictions.csv
// values for this approach within tolerance.
//
// This is the production-CLI counterpart to the hardened in-process
// fresh-process check already verified in
// hydrofusion_ffn_engine_integration_test.cpp -- there, the check is a
// forked/exec'd child of the SAME binary; here, no fork/exec is needed at
// all, because this verification is naturally a distinct process already
// (a separate `./hydrofusion_batch_bundle_reload_check` invocation, run
// after the real `HydroBatch` process has exited).
//
// Usage: hydrofusion_batch_bundle_reload_check <bundleDir> <metadataPath>
//   <originTargetMapPath> <approach>
// bundleDir is <output_root>/<experiment_id> (the real exporter's own
// managed directory); metadataPath/originTargetMapPath are the siblings
// hydro_batch_runner.cpp's exportHydroFusionSidecarForBatchJob wrote next
// to it.

#include "hydrofusion_synthetic_fixture.h"
#include "hydrofusion_dataset_assembly.h"
#include "hydrofusion_metadata_sidecar.h"
#include "../../dataset/hydro_checksum.h"
#include "../../evaluation/hydrofusion_inference.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: " << argv[0] << " <bundleDir> <metadataPath> <originTargetMapPath> <approach>\n";
        return 2;
    }
    const std::string bundleDir = argv[1];
    const std::string metadataPath = argv[2];
    const std::string originTargetMapPath = argv[3];
    const std::string approach = argv[4];

    try {
        using namespace hydrofusion;

        const auto reloaded = HydroArtifactLoader().loadForInference(bundleDir);
        if (reloaded.models.count(approach) != 1) {
            throw std::runtime_error("bundle is missing approach '" + approach + "'.");
        }

        const HydroFusionMetadata metadata = reloadHydroFusionMetadata(metadataPath);
        const std::string actualScalersSha256 =
            sha256File((std::filesystem::path(bundleDir) / "scalers.csv").string());
        validateMetadataBoundToBundle(metadata, reloaded.models.at(approach).sha256, actualScalersSha256);

        const auto originTargetRows = readOriginTargetMapCsv(originTargetMapPath);
        validateOriginTargetMap(originTargetRows, metadata.horizonHours);

        // Re-derive the SAME deterministic synthetic test split the real
        // training job used, from the metadata's own recorded cutoffs --
        // not hardcoded, not assumed equal to any value this process
        // hasn't itself re-checked.
        const auto rows = buildSyntheticFixture();
        const PreparedData data = prepareHydroFusionData(rows, metadata.mode, metadata.cutoffs);

        if (data.test.targetX.size() != originTargetRows.size()) {
            throw std::runtime_error("re-derived test split size disagrees with the origin/target map's row count.");
        }
        for (std::size_t i = 0; i < data.test.targetX.size(); ++i) {
            if (data.test.originX[i] != originTargetRows[i].first || data.test.targetX[i] != originTargetRows[i].second) {
                throw std::runtime_error("re-derived test split identities disagree with the origin/target map.");
            }
        }

        bool rejectedGeneric = false;
        try { HydroInferenceSession wrongSpace(reloaded, approach); }
        catch (const std::exception&) { rejectedGeneric = true; }
        if (!rejectedGeneric) throw std::runtime_error("Generic inference exposed transformed target as discharge.");
        if (metadata.qRef != data.qRef) throw std::runtime_error("Saved q_ref disagrees with training-only fit.");
        HydroFusionInferenceSession session(reloaded, bundleDir, metadataPath);
        bool rejectedNonfinite = false, rejectedWidth = false;
        auto bad = data.test.x.clone();
        bad[0][0] = std::numeric_limits<float>::quiet_NaN();
        try { session.predict(bad); } catch (const std::exception&) { rejectedNonfinite = true; }
        try { session.predict(data.test.x.slice(1, 0, 10)); } catch (const std::exception&) { rejectedWidth = true; }
        if (!rejectedNonfinite || !rejectedWidth) throw std::runtime_error("Physical inference input guards failed.");
        const auto physicalQ = session.predict(data.test.x);
        std::vector<double> reconstructed;
        for (int64_t i = 0; i < physicalQ.size(0); ++i) reconstructed.push_back(physicalQ[i][0].item<double>());
        const auto& exported = reloaded.results.at(approach);
        const auto& expected = exported.y_pred;
        if (exported.x.size() != data.test.targetX.size()) throw std::runtime_error("Exported target support size mismatch.");
        for (std::size_t i = 0; i < exported.x.size(); ++i) {
            if (exported.x[i] != data.test.targetX[i] || exported.split[i] != "test") {
                throw std::runtime_error("Exported target identity/split mismatch.");
            }
        }
        if (reconstructed.size() != expected.size()) {
            throw std::runtime_error("reconstructed/expected size mismatch.");
        }
        double maxAbsDiff = 0.0;
        for (std::size_t i = 0; i < reconstructed.size(); ++i) {
            if (!std::isfinite(reconstructed[i]) || !std::isfinite(expected[i])) throw std::runtime_error("Nonfinite prediction.");
            maxAbsDiff = std::max(maxAbsDiff, std::abs(reconstructed[i] - expected[i]));
        }
        if (!(maxAbsDiff < 1e-6)) {
            std::cerr << "FAIL: max_abs_diff=" << maxAbsDiff << " against the real exported predictions.csv\n";
            return 1;
        }
        std::cout << "PASS: n=" << reconstructed.size() << " max_abs_diff=" << maxAbsDiff
                  << " against the real HydroBatch-exported predictions.csv for approach '" << approach << "'\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "EXCEPTION: " << e.what() << "\n";
        return 1;
    }
}
