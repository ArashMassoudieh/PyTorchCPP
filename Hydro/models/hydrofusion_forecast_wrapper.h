#pragma once
#include "hydro_run_types.h"
#include <string>
class HydroFusionForecastWrapper {
public:
    HydroRunResult train(const HydroRunConfig& config,const std::string& approach);
};
