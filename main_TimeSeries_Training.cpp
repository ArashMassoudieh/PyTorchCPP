#include <QCoreApplication>
#include <iostream>
#include <vector>
#include <iomanip>
#include "neuralnetworkwrapper.h"
#include "TimeSeriesSet.h"
#include "TimeSeries.h"
#include "TestHyperParameters.h"
#include "hyperparameters.h"
#include "Normalization.h"

void createSyntheticData();

int main(int argc, char *argv[]) {
    QCoreApplication app(argc, argv);

    std::cout << "Running HyperParameters class tests...\n" << std::endl;

    try {
        testConstructorAndDefaults();
        testTimeSeriesSelection();
        testNetworkArchitecture();
        testLagConfiguration();
        testTrainingParameters();
        testValidation();
        //testErrorHandling();
        testStringRepresentation();
        testReset();

        std::cout << "\n🎉 All tests passed successfully!" << std::endl;

    } catch (const std::exception& e) {
        std::cout << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cout << "❌ Test failed with unknown exception" << std::endl;
        return 1;
    }

    try {
        std::cout << "=== Neural Network Wrapper with HyperParameters Example ===" << std::endl;
        std::cout << "\n1. Creating synthetic data..." << std::endl;
        //createSyntheticData();

        // Step 1: Load your data
        std::cout << "\n1. Loading data..." << std::endl;
        TimeSeriesSet<double> input_data;
        input_data.read("/mnt/3rd900/Projects/PyTorchCPP/Data/Inputs.txt", true);
        //input_data.read("input_data.csv", true); // CSV with multiple time series

        TimeSeries<double> target_series;
        target_series.readfile("/mnt/3rd900/Projects/PyTorchCPP/Data/Output.txt");
        //target_series.readfile("target_output.txt"); // Single target series

        std::cout << "Loaded " << input_data.size() << " input time series" << std::endl;
        std::cout << "Target series has " << target_series.size() << " points" << std::endl;

        // --------------------------------------Debug----------------------------------------------------
        std::cout << "\n[DEBUG] Input data size: " << input_data.size() << " series" << std::endl;
        for (size_t i = 0; i < input_data.size(); ++i) {
            std::cout << "  Series " << i
                      << " length = " << input_data[i].size()
                      << " (t range: " << input_data[i].front().t
                      << " → " << input_data[i].back().t << ")" << std::endl;

            // Print first 3 values as a sample
            std::cout << "    First 3 values: ";
            for (size_t j = 0; j < std::min<size_t>(3, input_data[i].size()); ++j) {
                std::cout << "(" << input_data[i][j].t << "," << input_data[i][j].c << ") ";
            }
            std::cout << std::endl;

            // Print last 3 values as a sample
            std::cout << "    Last 3 values: ";
            for (size_t j = input_data[i].size() > 3 ? input_data[i].size()-3 : 0;
                 j < input_data[i].size(); ++j) {
                std::cout << "(" << input_data[i][j].t << "," << input_data[i][j].c << ") ";
            }
            std::cout << std::endl;
        }

        std::cout << "[DEBUG] Target series length = " << target_series.size()
                  << " (t range: " << target_series.front().t
                  << " → " << target_series.back().t << ")" << std::endl;

        // Print sample of target values
        std::cout << "  First 5 target values: ";
        for (size_t j = 0; j < std::min<size_t>(5, target_series.size()); ++j) {
            std::cout << "(" << target_series[j].t << "," << target_series[j].c << ") ";
        }
        std::cout << std::endl;

        std::cout << "  Last 5 target values: ";
        for (size_t j = target_series.size() > 5 ? target_series.size()-5 : 0;
             j < target_series.size(); ++j) {
            std::cout << "(" << target_series[j].t << "," << target_series[j].c << ") ";
        }
        std::cout << std::endl;
        // -----------------------------------------------------------------------------------------------

        // Normalize input + target
        Normalizer<double> inputScaler(NormType::MinMax);
        inputScaler.fit(input_data);
        inputScaler.transform(input_data);

        Normalizer<double> targetScaler(NormType::MinMax);
        targetScaler.fit(target_series);
        targetScaler.transform(target_series);

        // --------------------------------------Debug----------------------------------------------------
        std::cout << "\n[DEBUG] Normalization applied." << std::endl;
        std::cout << "  First 5 values of series 0: ";
        for (size_t i = 0; i < std::min<size_t>(5, input_data[0].size()); ++i)
            std::cout << input_data[0][i].c << " ";
        std::cout << std::endl;

        std::cout << "  First 5 target values: ";
        for (size_t i = 0; i < std::min<size_t>(5, target_series.size()); ++i)
            std::cout << target_series[i].c << " ";
        std::cout << std::endl;
        // -----------------------------------------------------------------------------------------------

        // Step 2: Configure hyperparameters
        std::cout << "\n2. Configuring hyperparameters..." << std::endl;

        HyperParameters hyperparams;

        // --- Time series selection ---
        hyperparams.setSelectedSeriesFromBinary(511L, 9); // 9 of 9 inputs (All)

        /*
        | Bitmask | Binary | Selected series (0-based) |
        | ------- | ------ | ------------------------- |
        | `1L`    | 001    | Series 0                  |
        | `2L`    | 010    | Series 1                  |
        | `4L`    | 100    | Series 2                  |
        | `3L`    | 011    | Series 0 + 1              |
        | `5L`    | 101    | Series 0 + 2              |
        | `7L`    | 111    | Series 0 + 1 + 2          |


        | Bitmask (decimal) | Binary (9 bits) | Selected series  |
        | ----------------- | --------------- | ---------------- |
        | `1L`              | `000000001`     | Series 0         |
        | `2L`              | `000000010`     | Series 1         |
        | `4L`              | `000000100`     | Series 2         |
        | `8L`              | `000001000`     | Series 3         |
        | `16L`             | `000010000`     | Series 4         |
        | `32L`             | `000100000`     | Series 5         |
        | `64L`             | `001000000`     | Series 6         |
        | `128L`            | `010000000`     | Series 7         |
        | `256L`            | `100000000`     | Series 8         |
        | `511L`            | `111111111`     | All 9 (0–8)      |
        | `341L`            | `101010101`     | Series 0,2,4,6,8 |
        | `85L`             | `0001010101`    | Series 0,2,4,6   |

        */

        // --- Lag structure ---
        hyperparams.setMaxLags(10);
        hyperparams.setLagSelectionOdd(2);

        std::vector<std::vector<int>> lags = {
                {0,1,2,5,10},  // input 1
                {0,1,2,5,10},  // input 2
                {0,1,2,5,10},  // input 3
                {0,1,2,5,10},
                {0,1,2,5,10},
                {0,1,2,5,10},
                {0,1,2,5,10},
                {0,1,2,5,10},
                {0,1,2,5,10}
        };
        hyperparams.setLags(lags);

        // --- Lag multipliers ---
        std::vector<int> lag_multipliers = {1, 1, 1, 1, 1, 1, 1, 1, 1};
        hyperparams.setLagMultiplier(lag_multipliers);
        hyperparams.setMaxLagMultiplier(10);

        // --- Network architecture ---
        hyperparams.setMaxNumberOfHiddenNodes(128);
        hyperparams.setMaxNumberOfHiddenLayers(3);
        hyperparams.setHiddenLayers({64, 32, 16});

        // --- Activation functions ---
        hyperparams.setInputActivation("sigmoid");
        hyperparams.setHiddenActivation("relu");
        hyperparams.setOutputActivation("");

        // --- Training parameters ---
        hyperparams.setNumEpochs(100);
        hyperparams.setBatchSize(32);
        hyperparams.setLearningRate(0.0001);
        hyperparams.setTrainTestSplit(0.7);

        if (!hyperparams.isValid()) {
            throw std::runtime_error("Invalid hyperparameter configuration");
        }

        // --------------------------------------Debug----------------------------------------------------
        std::cout << "\n[DEBUG] Hyperparameters summary:" << std::endl;
        std::cout << hyperparams.toString() << std::endl;

        std::cout << "[DEBUG] Selected series indices: ";
        for (auto idx : hyperparams.getSelectedSeriesIds()) {
            std::cout << idx << " ";
        }
        std::cout << std::endl;
        // -----------------------------------------------------------------------------------------------

        // Step 3: Create and initialize neural network
        std::cout << "\n3. Initializing neural network..." << std::endl;
        NeuralNetworkWrapper net;
        net.initializeNetwork(&hyperparams, 1);

        std::cout << "Network initialized with " << net.getTotalParameters() << " parameters" << std::endl;

        // Step 4: Prepare training and test data
        std::cout << "\n4. Preparing data using hyperparameters..." << std::endl;
        if (input_data.size() == 0) throw std::runtime_error("No input series loaded!");
        if (target_series.size() == 0) throw std::runtime_error("No target series loaded!");

        double t_start = input_data[0].front().t;
        double t_end   = input_data[0].back().t;
        double dt      = input_data[0][1].t - input_data[0][0].t;

        double split_ratio = hyperparams.getTrainTestSplit();
        double split_time = t_start + split_ratio * (t_end - t_start);

        net.setInputDataFromHyperParams(DataType::Train, input_data, t_start, split_time, dt);

        // --------------------------------------Debug----------------------------------------------------
        // Debug: print first training sample
        {
            auto train_inputs = net.getInputData(DataType::Train);
            std::cout << "\n[DEBUG] First training sample features ("
                      << train_inputs.size(1) << " features):" << std::endl;

            auto first_row = train_inputs[0];  // first sample (1D tensor)
            for (int i = 0; i < first_row.size(0); ++i) {
                std::cout << "  Feature[" << i << "] = "
                          << first_row[i].item<double>() << std::endl;
            }
        }
        // --------------------------------------Debug----------------------------------------------------

        net.setTargetData(DataType::Train, target_series, t_start, split_time, dt);

        net.setInputDataFromHyperParams(DataType::Test, input_data, split_time, t_end, dt);
        net.setTargetData(DataType::Test, target_series, split_time, t_end, dt);

        std::cout << "Training data: " << net.getInputData(DataType::Train).size(0) << " samples" << std::endl;
        std::cout << "Test data: " << net.getInputData(DataType::Test).size(0) << " samples" << std::endl;

        // --------------------------------------Debug----------------------------------------------------
        std::cout << "\n[DEBUG] Network architecture initialized." << std::endl;
        std::cout << "  Total parameters = " << net.getTotalParameters() << std::endl;
        std::cout << "  Expected input dimension (features per sample) = "
                  << net.getInputData(DataType::Train).size(1) << std::endl;
        // --------------------------------------Debug----------------------------------------------------

        // Step 5: Train
        std::cout << "\n5. Training network..." << std::endl;
        std::vector<double> training_losses = net.train(
            hyperparams.getNumEpochs(),
            hyperparams.getBatchSize(),
            hyperparams.getLearningRate()
        );

        // Debug: show first 5 and last 5 losses
        std::cout << "Debug: Loss progression (first 5 epochs):" << std::endl;
        for (size_t i = 0; i < std::min<size_t>(5, training_losses.size()); ++i) {
            std::cout << "  Epoch " << (i+1) << ": " << training_losses[i] << std::endl;
        }
        std::cout << "Debug: Loss progression (last 5 epochs):" << std::endl;
        for (size_t i = training_losses.size() > 5 ? training_losses.size()-5 : 0;
             i < training_losses.size(); ++i) {
            std::cout << "  Epoch " << (i+1) << ": " << training_losses[i] << std::endl;
        }



        std::cout << "Training completed. Final loss: " << training_losses.back() << std::endl;
        double train_r2 = net.calculateR2(DataType::Train);
        std::cout << "Training R²: " << std::fixed << std::setprecision(4) << train_r2 << std::endl;

        // Step 6: Evaluate
        std::cout << "\n6. Evaluating performance..." << std::endl;
        TimeSeriesSet<double> test_predictions = net.predict(
            DataType::Test, split_time, t_end, dt, {"predicted_output"});

        // Inverse-transform predictions back to original scale
        targetScaler.inverseTransform(test_predictions);


        // Optionally also inverse-transform target series if you want
        // to save or visualize them on original scale
        // targetScaler.inverseTransform(target_series);

        // Save predictions in original units
        test_predictions.write("test_predictions_rescaled.csv");

        // --------------------------------------Debug----------------------------------------------------
                std::cout << "\n[DEBUG] Predictions generated: " << test_predictions.size() << " series" << std::endl;
        for (size_t i = 0; i < std::min<size_t>(5, test_predictions[0].size()); ++i) {
            std::cout << "  t=" << test_predictions[0][i].t
                      << " pred=" << test_predictions[0][i].c << std::endl;
        }
        // --------------------------------------Debug----------------------------------------------------

        // Normalized-space metrics
        auto metrics = net.evaluate();
        std::cout << "\n--- Normalized scale metrics ---" << std::endl;
        std::cout << "Test MSE: " << metrics["mse"] << std::endl;
        std::cout << "Test RMSE: " << metrics["rmse"] << std::endl;
        std::cout << "Test MAE: " << metrics["mae"] << std::endl;
        std::cout << "Test R²: " << metrics["r_squared"] << std::endl;

        // Original-scale metrics (manual calc)
        TimeSeries<double> target_test;
        for (const auto& point : target_series) {
            if (point.t >= split_time && point.t <= t_end) {
                target_test.addPoint(point.t, point.c);
            }
        }
        targetScaler.inverseTransform(target_test);

        auto pred_tensor = test_predictions.toTensor();
        auto true_tensor = target_test.toTensor();

        auto ss_res = torch::sum(torch::pow(true_tensor - pred_tensor, 2));
        auto ss_tot = torch::sum(torch::pow(true_tensor - torch::mean(true_tensor), 2));

        double mse_original  = torch::mean(torch::pow(true_tensor - pred_tensor, 2)).item<double>();
        double rmse_original = std::sqrt(mse_original);
        double mae_original  = torch::mean(torch::abs(true_tensor - pred_tensor)).item<double>();
        double r2_original   = 1.0 - (ss_res.item<double>() / ss_tot.item<double>());

        std::cout << "\n--- Original scale metrics ---" << std::endl;
        std::cout << "Test MSE: " << mse_original << std::endl;
        std::cout << "Test RMSE: " << rmse_original << std::endl;
        std::cout << "Test MAE: " << mae_original << std::endl;
        std::cout << "Test R²: " << r2_original << std::endl;

        // === Baseline check ===
        {
            torch::NoGradGuard no_grad;
            auto test_outputs = net.getTargetData(DataType::Test);

            double target_mean = torch::mean(test_outputs).item<double>();
            auto baseline_predictions = torch::full_like(test_outputs, target_mean);

            auto ss_res_base = torch::sum(torch::pow(test_outputs - baseline_predictions, 2));
            auto ss_tot2 = torch::sum(torch::pow(test_outputs - target_mean, 2));
            float baseline_r2 = 1.0f - (ss_res_base.item<float>() / ss_tot2.item<float>());

            std::cout << "Baseline R² (mean predictor): "
                      << std::fixed << std::setprecision(4)
                      << baseline_r2 << std::endl;
        }

        // === Continue with saving, optimization demo, etc. ===
        net.saveModel("trained_model.pt");
        test_predictions.write("test_predictions.csv");

        std::ofstream hyperparams_file("hyperparameters_config.txt");
        hyperparams_file << hyperparams.toString() << std::endl;
        hyperparams_file.close();

        std::cout << "Training loss history (last 10 epochs):" << std::endl;
        for (size_t i = std::max<size_t>(0, training_losses.size() - 10);
             i < training_losses.size(); ++i) {
            std::cout << "  Epoch " << (i+1) << ": " << training_losses[i] << std::endl;
        }

        std::cout << "\n=== HyperParameters example completed successfully! ===" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
}

// Updated synthetic data creation function
void createSyntheticData() {
    std::cout << "Creating synthetic data for testing..." << std::endl;

    // Create synthetic input data (3 time series)
    TimeSeriesSet<double> synthetic_input(3);

    // Extend the time range to include negative values for proper lag handling
    for (double t = -10.0; t <= 100.0; t += 0.1) { // Extended range for larger lags
        // Series 0: sine wave
        synthetic_input[0].addPoint(t, std::sin(0.1 * t) + 0.1 * std::sin(0.5 * t));

        // Series 1: cosine wave with trend
        synthetic_input[1].addPoint(t, std::cos(0.15 * t) + 0.01 * t);

        // Series 2: noisy exponential decay
        double noise = 0.1 * (static_cast<double>(rand()) / RAND_MAX - 0.5);
        synthetic_input[2].addPoint(t, std::exp(-0.02 * std::abs(t)) + noise);
    }

    synthetic_input.setSeriesName(0, "temperature");
    synthetic_input.setSeriesName(1, "pressure");
    synthetic_input.setSeriesName(2, "flow_rate");

    synthetic_input.write("input_data.csv");

    // Create synthetic target data using the hyperparameter-like structure
    TimeSeries<double> synthetic_target;
    for (double t = 0.0; t <= 100.0; t += 0.1) {
        // Simulate the lag structure that will be used:
        // Series 0 with lags [0,1,2,...] × multiplier 1
        // Series 1 with lags [1,2] × multiplier 5 = [5,10]
        // Series 2 with lags [1,3] × multiplier 2 = [2,6]
        double target =
            0.3 * synthetic_input[0].interpol(t - 0.0) +  // lag 0
            0.2 * synthetic_input[0].interpol(t - 0.1) +  // lag 1
            0.2 * synthetic_input[1].interpol(t - 0.5) +  // lag 5
            0.2 * synthetic_input[2].interpol(t - 0.2) +  // lag 2
            0.1 * (static_cast<double>(rand()) / RAND_MAX - 0.5); // noise

        synthetic_target.addPoint(t, target);
    }

    synthetic_target.writefile("target_output.txt");

    std::cout << "Synthetic data created successfully!" << std::endl;
}
