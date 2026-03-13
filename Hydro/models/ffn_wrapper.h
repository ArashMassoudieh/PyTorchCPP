#pragma once

#include "hydro_run_types.h"

/**
 * @file ffn_wrapper.h
 * @brief Baseline feed-forward network wrapper interface.
 */

/**
 * @brief Lightweight FFN baseline wrapper for HydroPINN experiments.
 */
class FFNWrapper {
public:
    /**
     * @brief Run training/evaluation for the FFN baseline.
     * @param config Runtime configuration.
     * @return Run result with status and summary metrics.
     */
    HydroRunResult train(const HydroRunConfig& config);
};
