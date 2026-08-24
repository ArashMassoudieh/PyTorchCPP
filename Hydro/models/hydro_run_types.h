#pragma once

#include <string>
#include <cstdint>
#include <limits>
#include <vector>

struct HydroScalerState {
    std::string method = "none";
    std::vector<double> offset;
    std::vector<double> scale;
    std::vector<int64_t> shape;
};

/**
 * @brief Runtime configuration used by Hydro mode wrappers.
 */
struct HydroRunConfig {
    int epochs = 150;
    int batch_size = 32;
    double learning_rate = 0.003;

    // PINN-specific options
    // Physics profiles:
    // - water_balance:       watershed mass-balance residual P/precip - ET - Q - dS/dt for watershed_balance and rainfall_runoff;
    // - linear_reservoir:    dy/dt + lambda*y - forcing_gain*u = 0
    // - cstr_first_order:    dy/dt + lambda*y - forcing_gain*u = 0 (same residual form, different interpretation)
    // - exp_decay:           dy/dt + lambda*y = 0
    //                        falls back to forcing-driven training for other profiles without explicit P/ET/S columns.
    double lambda_decay = 0.8;
    double data_weight = 1.0;
    double physics_weight = 0.2;
    std::string pinn_physics_profile = "water_balance"; // water_balance | linear_reservoir | cstr_first_order | exp_decay
    double forcing_gain = 1.0;
    double runoff_coeff = 0.7;
    double storage_coeff = 1.0;
    double physics_dt = 1.0;
    int pinn_collocation_points = 0; // 0 => use batch inputs only; >0 => sample extra Raissi-style collocation points per batch

    // Data input options
    bool use_csv_data = false;
    bool use_hydro_package = false;
    std::string hydro_package_path;
    std::string hydro_catchment_id;
    std::string hydro_package_profile = "rainfall-runoff";
    bool use_hydro_forecast_feature = false;
    std::string hydro_forecast_variable = "precipitation";
    double hydro_forecast_lead_hours = 0.0;
    std::string hydro_forecast_ensemble_member;
    std::string csv_path;
    int csv_x_column = 0;
    int csv_y_column = 1;
    bool csv_has_header = true;

    int sample_count = 220;
    double t_start = 0.0;
    double t_end = 5.0;
    std::string synthetic_profile = "watershed_balance"; // watershed_balance | rainfall_runoff | neuroforge_inputs_target | exp_decay | damped_sine | mixed_wave

    // Network options
    std::string hidden_layers_csv = "24,24";
    // Lag groups are separated by ';' and each group contains comma-separated positive integer lags.
    // Examples: "1" (all features use lag 1), "1,2;1;1,3" (feature-specific lag groups).
    std::string input_lags_csv = "1";
    bool use_time_lagged_ffn = false; // Applies only to FFN and FFN + PINN; LSTM keeps sequence memory internally.
    int lstm_sequence_length = 6; // Independent of FFN lag-search settings.
    std::string activation = "tanh"; // single backend activation used across hidden/output layers

    bool evaluate_metrics = true;

    // NeuroForge-style extra options (currently informational/plumbing for Hydro UI compatibility)
    double train_split_ratio = 0.8;
    double validation_split_ratio = 0.1; // chronological fraction reserved for model selection
    bool shuffle_training = true;
    int random_seed = 42;
    std::string optimizer = "adam";      // adam | sgd | rmsprop
    double weight_decay = 0.0;
    double momentum = 0.9;
    std::string normalization = "none";  // none | standardize | minmax

    // Incremental/rolling-window options (future backend compatibility)
    bool use_incremental_training = false;
    double window_size = 1.0;
    double window_step = 0.5;
    int epochs_per_window = 25;
    bool reset_optimizer_on_new_window = false;
};

/**
 * @brief Basic run result reported by Hydro mode wrappers.
 */
struct HydroRunResult {
    bool success = false;
    double final_loss = std::numeric_limits<double>::quiet_NaN();
    double mse = std::numeric_limits<double>::quiet_NaN();
    double validation_mse = std::numeric_limits<double>::quiet_NaN();
    double rmse = std::numeric_limits<double>::quiet_NaN();
    double mae = std::numeric_limits<double>::quiet_NaN();
    double nse = std::numeric_limits<double>::quiet_NaN();
    double pbias = std::numeric_limits<double>::quiet_NaN();
    double correlation = std::numeric_limits<double>::quiet_NaN();
    double kge = std::numeric_limits<double>::quiet_NaN();
    double volume_error_percent = std::numeric_limits<double>::quiet_NaN();
    double peak_timing_error = std::numeric_limits<double>::quiet_NaN();
    double peak_magnitude_error_percent = std::numeric_limits<double>::quiet_NaN();
    double high_flow_rmse = std::numeric_limits<double>::quiet_NaN();
    double low_flow_rmse = std::numeric_limits<double>::quiet_NaN();
    double physics_residual_mean = std::numeric_limits<double>::quiet_NaN();
    double physics_residual_rmse = std::numeric_limits<double>::quiet_NaN();
    double cumulative_physics_residual = std::numeric_limits<double>::quiet_NaN();
    double physics_loss = std::numeric_limits<double>::quiet_NaN();
    std::string message;

    // Optional series for plotting
    std::vector<double> x;
    std::vector<double> y_true;
    std::vector<double> y_pred;
    std::vector<std::string> split;
    std::vector<double> training_loss_history;
    std::vector<double> validation_loss_history;
    int best_epoch = 0;
    HydroScalerState input_scaler;
    HydroScalerState target_scaler;
    std::string model_checkpoint_format;
    std::vector<std::uint8_t> model_checkpoint;
    std::vector<double> physics_residual;
};
