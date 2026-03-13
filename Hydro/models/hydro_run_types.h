#pragma once

#include <string>

/**
 * @brief Runtime configuration used by Hydro mode wrappers.
 */
struct HydroRunConfig {
    int epochs = 150;
    int batch_size = 32;
    double learning_rate = 0.003;

    // PINN-specific options
    double lambda_decay = 0.8;
    double data_weight = 1.0;
    double physics_weight = 0.2;

    bool evaluate_metrics = true;
};

/**
 * @brief Basic run result reported by Hydro mode wrappers.
 */
struct HydroRunResult {
    bool success = false;
    double final_loss = 0.0;
    double mse = 0.0;
    std::string message;
};
