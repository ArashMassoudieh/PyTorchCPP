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

struct HydroInferenceArtifacts {
    LoadedHydroExperiment experiment;
    std::map<std::string, HydroModelArtifact> models;
    std::map<std::string, HydroArtifactScalers> scalers;
};

class HydroArtifactLoader {
public:
    std::map<std::string, HydroModelArtifact> loadModels(const std::string& experimentDirectory) const;
    std::map<std::string, HydroArtifactScalers> loadScalers(const std::string& experimentDirectory) const;
    std::map<std::string, HydroRunResult> loadPredictions(const std::string& experimentDirectory) const;
    std::map<std::string, std::vector<double>> loadPhysicsResiduals(const std::string& experimentDirectory) const;
    HydroInferenceArtifacts loadForInference(const std::string& experimentDirectory) const;
};
