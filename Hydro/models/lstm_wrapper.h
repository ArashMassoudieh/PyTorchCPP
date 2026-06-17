#pragma once

#include "hydro_run_types.h"

/**
 * @file lstm_wrapper.h
 * @brief Baseline LibTorch LSTM wrapper interface.
 */

/**
 * @brief Lightweight true LSTM baseline wrapper for HydroPINN experiments.
 *
 * This wrapper uses a torch::nn::LSTM backend, builds rolling input windows
 * from the provided series, and reports held-out test metrics plus full-series
 * aligned predictions for plotting.
 */
class LSTMWrapper {
public:
    /**
     * @brief Run baseline LSTM training backend.
     * @param config Runtime configuration.
     * @return Run result with status and summary metrics.
     */
    HydroRunResult train(const HydroRunConfig& config);
};
