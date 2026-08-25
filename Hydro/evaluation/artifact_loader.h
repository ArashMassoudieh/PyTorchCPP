#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "../models/hydro_run_types.h"
#include "experiment_loader.h"

struct HydroModelArtifact {
    std::string relative_path;
    std::string format;
    std::string sha256;
    std::vector<std::uint8_t> bytes;
};

struct HydroArtifactScalers {
    HydroScalerState input;
    HydroScalerState target;
};

struct HydroPhysicsResidualArtifact {
    std::vector<double> x;
    std::vector<std::string> split;
    std::vector<double> values;
};

struct HydroEnvironmentArtifact {
    std::string compiler;
    std::int64_t cplusplus = 0;
    std::string build_date;
    std::string build_time;
};

struct HydroProvenanceArtifact {
    std::string fingerprint_algorithm;
    std::string dataset_manifest_sha256;
};

struct HydroTrainingHistoryArtifact {
    std::vector<double> training_loss;
    std::vector<double> validation_loss;
    int best_epoch = -1;
};

struct HydroInferenceArtifacts {
    LoadedHydroExperiment experiment;
    std::map<std::string, HydroModelArtifact> models;
    std::map<std::string, HydroArtifactScalers> scalers;
    std::map<std::string, HydroRunResult> results;
    std::map<std::string, HydroTrainingHistoryArtifact> training_history;
    HydroEnvironmentArtifact environment;
    HydroProvenanceArtifact provenance;
};

class HydroArtifactLoader {
public:
    std::map<std::string, HydroModelArtifact> loadModels(const std::string& experimentDirectory) const;
    std::map<std::string, HydroArtifactScalers> loadScalers(const std::string& experimentDirectory) const;
    std::map<std::string, HydroRunResult> loadPredictions(const std::string& experimentDirectory) const;
    std::map<std::string, HydroRunResult> loadMetrics(const std::string& experimentDirectory) const;
    std::map<std::string, HydroPhysicsResidualArtifact> loadPhysicsResiduals(const std::string& experimentDirectory) const;
    std::map<std::string, HydroTrainingHistoryArtifact> loadTrainingHistory(const std::string& experimentDirectory) const;
    HydroEnvironmentArtifact loadEnvironment(const std::string& experimentDirectory) const;
    HydroProvenanceArtifact loadProvenance(const std::string& experimentDirectory) const;
    HydroInferenceArtifacts loadForInference(const std::string& experimentDirectory) const;
};
