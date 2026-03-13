#pragma once

#include "hydro_run_types.h"

/**
 * @file lstm_pinn_wrapper.h
 * @brief PINN-enabled LSTM wrapper interface.
 */

/**
 * @brief LSTM + physics-informed training wrapper.
 */
class LSTMPINNWrapper {
public:
    /**
     * @brief Run LSTM-like backend with physics constraints.
     * @param config Runtime configuration.
     * @return Run result with status and summary metrics.
     */
    HydroRunResult train(const HydroRunConfig& config);
};
