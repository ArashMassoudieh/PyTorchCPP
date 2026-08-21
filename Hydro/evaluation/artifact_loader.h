#pragma once

#include "../models/hydro_run_types.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

struct HydroModelArtifact {
    std::string relative_path;
    std::string format;
    std::string sha256;
    std::vector<std::uint8_t> bytes;
};

struct HydroScalerArtifacts {
    HydroScalerState input;
    HydroScalerState target;
};

class HydroArtifactLoader {
public:
    std::map<std::string, HydroModelArtifact> loadModels(const std::string& experimentDirectory) const;
    std::map<std::string, HydroScalerArtifacts> loadScalers(const std::string& experimentDirectory) const;
    std::map<std::string, HydroRunResult> loadResults(const std::string& experimentDirectory) const;
    void validateCompatibility(const HydroRunConfig& config,
                               const std::map<std::string, HydroModelArtifact>& models,
                               const std::map<std::string, HydroScalerArtifacts>& scalers) const;
};
