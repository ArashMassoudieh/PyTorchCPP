#pragma once

#include "../models/hydro_run_types.h"

#include <string>

struct LoadedHydroExperiment {
    std::string experiment_id;
    HydroRunConfig config;
};

class HydroExperimentLoader {
public:
    LoadedHydroExperiment loadConfig(const std::string& configPath) const;
};
