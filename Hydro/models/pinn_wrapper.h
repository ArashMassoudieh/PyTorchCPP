#pragma once

#include "hydro_run_types.h"

/** Explicit physics-only fifth approach. */
class PINNWrapper {
public:
    HydroRunResult train(const HydroRunConfig& config);
};
