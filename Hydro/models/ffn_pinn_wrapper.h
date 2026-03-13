#pragma once

#include "hydro_run_types.h"

/**
 * @file ffn_pinn_wrapper.h
 * @brief PINN-enabled feed-forward wrapper interface.
 */

/**
 * @brief FFN + physics-informed training wrapper.
 */
class FFNPINNWrapper {
public:
    /**
     * @brief Run FFN training with physics residual terms.
     * @param config Runtime configuration.
     * @return Run result with status and summary metrics.
     */
    HydroRunResult train(const HydroRunConfig& config);
};
