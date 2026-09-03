#pragma once

#include <string>
#include <cstdint>
#include <limits>
#include <vector>

inline int& hydroLstmSequenceLengthRuntimeDefault() {
    static int value = 6;
    return value;
}

struct HydroScalerState {
    std::string method = "none";
    std::vector<double> offset = {0.0};
    std::vector<double> scale = {1.0};
    std::vector<int64_t> shape = {1};
};

struct HydroRunConfig {
    int epochs = 150;
    int batch_size = 32;
    double learning_rate = 0.003;

    // PINN-specific options.
    // Physics profiles:
    // - linear_reservoir: reduced runoff evolution dQ/dt = k(Peff-Q), shared by
    //   Synthetic, CSV, and Hydro-package physics modes. Peff=max(P-PET,0).
    // - water_balance: explicit known-state balance P-ET-Q-dS/dt=0; intended for
    //   controlled synthetic or packages that independently provide storage S.
    // - cstr_first_order: dy/dt + lambda*y - forcing_gain*u = 0.
    // - exp_decay: dy/dt + lambda*y = 0.
    double lambda_decay = 0.8;
    double data_weight = 1.0;
    double physics_weight = 0.2;
    std::string pinn_physics_profile = "water_balance";
    double forcing_gain = 1.0;
    double runoff_coeff = 0.7;
    double storage_coeff = 1.0;
    double physics_dt = 1.0;
    int pinn_collocation_points = 0;

    // Historical flag name retained for experiment compatibility. For GIStoOHQ
    // reduced-reservoir physics it now selects the contiguous forcing-only layout
    // [time, Peff, P, PET, T, RH, wind, solar]; no latent storage is generated.
    bool use_latent_storage_physics = false;
    double latent_storage_recession_per_hour = 0.08;

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
    std::string synthetic_profile = "watershed_balance";

    // Ground-truth reservoir coefficient used only to GENERATE the controlled
    // reduced_reservoir synthetic target. It is intentionally independent from
    // lambda_decay/storage_coeff, which represent the candidate model k during
    // PINN calibration/tuning. Every method and every candidate must see the same
    // synthetic truth hydrograph.
    double synthetic_reservoir_truth_k = 0.08;

    // Network options
    std::string hidden_layers_csv = "24,24";
    std::string input_lags_csv = "1";
    bool use_time_lagged_ffn = false;
    int lstm_sequence_length = hydroLstmSequenceLengthRuntimeDefault();
    std::string activation = "tanh";

    bool evaluate_metrics = true;

    double train_split_ratio = 0.8;
    double validation_split_ratio = 0.1;
    bool shuffle_training = true;
    int random_seed = 42;
    std::string optimizer = "adam";
    double weight_decay = 0.0;
    double momentum = 0.9;
    std::string normalization = "none";

    bool use_incremental_training = false;
    double window_size = 1.0;
    double window_step = 0.5;
    int epochs_per_window = 25;
    bool reset_optimizer_on_new_window = false;
};

struct HydroRunResult {
    bool success = false;
    double final_loss = std::numeric_limits<double>::quiet_NaN();
    double mse = std::numeric_limits<double>::quiet_NaN();
    double validation_mse = std::numeric_limits<double>::quiet_NaN();
    double rmse = std::numeric_limits<double>::quiet_NaN();
    double mae = std::numeric_limits<double>::quiet_NaN();
    double nse = std::numeric_limits<double>::quiet_NaN();
    double r2 = std::numeric_limits<double>::quiet_NaN();
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
