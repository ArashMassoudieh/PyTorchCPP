#include "pinn_wrapper.h"

#include "ffn_pinn_wrapper.h"

#include <algorithm>

HydroRunResult PINNWrapper::train(const HydroRunConfig& config) {
    HydroRunConfig physicsOnly = config;
    physicsOnly.use_time_lagged_ffn = false;
    physicsOnly.data_weight = 0.0;
    physicsOnly.physics_weight = std::max(1.0, physicsOnly.physics_weight);
    FFNPINNWrapper backend;
    HydroRunResult result = backend.train(physicsOnly);
    if (result.success) result.message = "Physics-only PINN run completed (no supervised data loss).";
    return result;
}
