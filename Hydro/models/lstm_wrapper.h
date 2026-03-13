#pragma once

#include "hydro_run_types.h"

/**
 * @file lstm_wrapper.h
 * @brief Baseline LSTM wrapper interface.
 */

/**
 * @brief Lightweight LSTM baseline wrapper for HydroPINN experiments.
 */
class LSTMWrapper {
public:
    /**
     * @brief Run baseline LSTM-like training backend.
     * @param config Runtime configuration.
     * @return Run result with status and summary metrics.
     */
    HydroRunResult train(const HydroRunConfig& config);
};
