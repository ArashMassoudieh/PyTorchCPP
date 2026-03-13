#pragma once

#include <string>
#include <vector>

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

    // Data input options
    int sample_count = 220;
    double t_start = 0.0;
    double t_end = 5.0;
    std::string synthetic_profile = "exp_decay"; // exp_decay | damped_sine | mixed_wave

    // Network options
    std::string hidden_layers_csv = "24,24";
    std::string activation = "tanh"; // relu | tanh | sigmoid

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

    // Optional series for plotting
    std::vector<double> x;
    std::vector<double> y_true;
    std::vector<double> y_pred;
};
