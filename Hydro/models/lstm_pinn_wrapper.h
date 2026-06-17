#pragma once

#include "hydro_run_types.h"

/**
 * @file lstm_pinn_wrapper.h
 * @brief Physics-informed LibTorch LSTM wrapper interface.
 */

/**
 * @brief Lightweight true LSTM-PINN wrapper for HydroPINN experiments.
 *
 * The data term is computed on rolling LSTM windows. The physics term is
 * computed from finite-difference residuals of ordered LSTM predictions.
 */
class LSTMPINNWrapper {
public:
    /**
     * @brief Run physics-informed LSTM training backend.
     * @param config Runtime configuration.
     * @return Run result with status and summary metrics.
     */
    HydroRunResult train(const HydroRunConfig& config);
};
