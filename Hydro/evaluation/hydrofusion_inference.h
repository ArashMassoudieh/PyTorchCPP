#pragma once

// Explicit physical-discharge adapter for the synthetic-only HydroFusion prototype.
// Generic inference rejects this approach so log targets cannot be shown as Q.
#include "inference_runner.h"
#include "../dataset/hydro_checksum.h"
#include "../experiments/gistohq_sligo/hydrofusion_dataset_assembly.h"
#include "../experiments/gistohq_sligo/hydrofusion_metadata_sidecar.h"

class HydroFusionInferenceSession {
public:
    HydroFusionInferenceSession(const HydroInferenceArtifacts& artifacts,
                               const std::string& bundleDirectory,
                               const std::string& metadataPath)
    {
        const auto managedMetadata = std::filesystem::path(bundleDirectory) / "hydrofusion_metadata.txt";
        if (std::filesystem::weakly_canonical(metadataPath) != std::filesystem::weakly_canonical(managedMetadata)) {
            throw std::invalid_argument("HydroFusion metadata must be inside its managed artifact bundle.");
        }
        HydroArtifactLoader().validateBundleManifest(bundleDirectory);
        metadata_ = hydrofusion::reloadHydroFusionMetadata(managedMetadata.string());
        const auto model = artifacts.models.find("ffn_hydrofusion");
        if (model == artifacts.models.end()) throw std::invalid_argument("Missing ffn_hydrofusion model.");
        hydrofusion::validateMetadataBoundToBundle(metadata_, model->second.sha256,
            sha256File((std::filesystem::path(bundleDirectory) / "scalers.csv").string()));
        const auto& config = artifacts.experiment.config;
        const std::string mode = metadata_.mode == hydrofusion::TargetMode::LogResidual ? "log_residual" : "direct_log_q";
        if (config.hydrofusion_target_mode != mode || config.train_end_x != metadata_.cutoffs.train_end_x ||
            config.validation_end_x != metadata_.cutoffs.validation_end_x || config.test_end_x != metadata_.cutoffs.test_end_x) {
            throw std::invalid_argument("HydroFusion config and metadata mode/bounds disagree.");
        }
        session_ = std::make_unique<HydroInferenceSession>(artifacts, "ffn_hydrofusion",
                                        HydroInferenceOutputSpace::TransformedHydroFusionTarget);
    }

    torch::Tensor predict(const torch::Tensor& physicalInputs) const {
        if (physicalInputs.dim() != 2 || physicalInputs.size(1) != 20 || physicalInputs.size(0) < 1 ||
            !torch::isfinite(physicalInputs).all().item<bool>()) {
            throw std::invalid_argument("HydroFusion expects a finite, nonempty [N,20] physical feature tensor.");
        }
        if ((physicalInputs.slice(1, 0, 12) < 0).any().item<bool>() ||
            (physicalInputs.select(1, 14) < 0).any().item<bool>() ||
            (physicalInputs.slice(1, 16, 20) < 0).any().item<bool>() ||
            (physicalInputs.select(1, 16) > 100).any().item<bool>()) {
            throw std::invalid_argument("HydroFusion physical feature domain violation.");
        }
        const auto predicted = session_->predict(physicalInputs);
        const auto origins = physicalInputs.select(1, 0).reshape({-1, 1});
        const auto q = hydrofusion::reconstructHydroFusionColumn(metadata_.mode, predicted, origins, metadata_.qRef);
        return torch::tensor(q, torch::kFloat64).reshape({-1, 1});  // raw discharge mm/h
    }
private:
    hydrofusion::HydroFusionMetadata metadata_;
    std::unique_ptr<HydroInferenceSession> session_;
};
