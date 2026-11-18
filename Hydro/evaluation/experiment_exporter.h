#pragma once

#include "../models/hydro_run_types.h"

#include <cstddef>
#include <map>
#include <string>

class HydroExperimentExporter {
public:
    void exportRun(const std::string& outputDirectory,
                   const std::string& experimentId,
                   const HydroRunConfig& config,
                   const std::map<std::string, HydroRunResult>& results,
                   std::size_t callerConfigSize = sizeof(HydroRunConfig),
                   std::size_t callerResultSize = sizeof(HydroRunResult)) const;
};
