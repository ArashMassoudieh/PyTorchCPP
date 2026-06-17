#pragma once

#include "hydro_run_types.h"

/**
 * @file lstmnetworkwrapper.h
 * @brief Shared true LibTorch LSTM backend used by LSTMWrapper and LSTMPINNWrapper.
 */
class LSTMNetworkWrapper {
public:
    /**
     * @brief Train/evaluate a true torch::nn::LSTM model.
     * @param config HydroPINN runtime configuration.
     * @param physicsInformed When true, adds the finite-difference physics residual loss.
     */
    HydroRunResult train(const HydroRunConfig& config, bool physicsInformed = false);
};
