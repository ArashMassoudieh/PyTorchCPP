#include <QCoreApplication>
#include <iostream>
#include <vector>
#include "neuralnetworkwrapper.h"
#include "TimeSeriesSet.h"
#include "TimeSeries.h"
#include "TestHyperParameters.h"
#include "hyperparameters.h"

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
        testErrorHandling();
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
        createSyntheticData();

        // Step 1: Load your data
        std::cout << "\n1. Loading data..." << std::endl;
        TimeSeriesSet<double> input_data;
        input_data.read("input_data.csv", true);  // CSV with multiple time series

        TimeSeries<double> target_series;
        target_series.readfile("target_output.txt");  // Single target series

        std::cout << "Loaded " << input_data.size() << " input time series" << std::endl;
        std::cout << "Target series has " << target_series.size() << " points" << std::endl;

        // Step 2: Configure hyperparameters
        std::cout << "\n2. Configuring hyperparameters..." << std::endl;
        HyperParameters hyperparams;

        // Configure time series selection (select all 3 series: binary 111 = 7)
        hyperparams.setSelectedSeriesFromBinary(7L, 3);  // Selects series {0, 1, 2}

        // Configure lag structure for all potential series
        hyperparams.setMaxLags(10);           // Lags numbered 0-9
        hyperparams.setLagSelectionOdd(2);    // Base 2 for lag selection

        // Set lag codes for each series (3 codes for 3 potential series)
        std::vector<long int> lag_codes = {
            0L,   // Series 0: all lags selected (0%2=0 for all)
            6L,   // Series 1: binary 110 -> lags [1, 2] selected
            10L   // Series 2: binary 1010 -> lags [1, 3] selected
        };
        hyperparams.setLagsFromVector(lag_codes);

        // Configure lag multipliers (different time scales for each series)
        std::vector<int> lag_multipliers = {1, 5, 2}; // Series 0: ×1, Series 1: ×5, Series 2: ×2
        hyperparams.setLagMultiplier(lag_multipliers);
        hyperparams.setMaxLagMultiplier(10);

        // Configure network architecture using code-based generation
        hyperparams.setMaxNumberOfHiddenNodes(128);
        hyperparams.setMaxNumberOfHiddenLayers(4);
        hyperparams.setHiddenLayersFromCode(16450L, 32);  // Generate architecture from code

        // Configure training parameters
        hyperparams.setActivationFunction("relu");
        hyperparams.setNumEpochs(100);
        hyperparams.setBatchSize(32);
        hyperparams.setLearningRate(0.001);
        hyperparams.setTrainTestSplit(0.8);

        // Validate and display configuration
        if (!hyperparams.isValid()) {
            throw std::runtime_error("Invalid hyperparameter configuration");
        }

        std::cout << "HyperParameters configuration:" << std::endl;
        std::cout << hyperparams.toString() << std::endl;

        // Step 3: Create and initialize neural network
        std::cout << "\n3. Initializing neural network..." << std::endl;
        NeuralNetworkWrapper net;

        // Initialize network using hyperparameters (1 output)
        net.initializeNetwork(&hyperparams, 1);

        std::cout << "Network initialized with " << net.getTotalParameters() << " parameters" << std::endl;

        // Step 4: Prepare training and test data using hyperparameters
        std::cout << "\n4. Preparing data using hyperparameters..." << std::endl;

        double t_start = 0.0;
        double t_end = 100.0;
        double dt = 0.1;
        double split_ratio = hyperparams.getTrainTestSplit();
        double split_time = t_start + split_ratio * (t_end - t_start);

        // Set training data using hyperparameter configuration
        net.setInputDataFromHyperParams(DataType::Train, input_data, t_start, split_time, dt);
        net.setTargetData(DataType::Train, target_series, t_start, split_time, dt);

        // Set test data using hyperparameter configuration
        net.setInputDataFromHyperParams(DataType::Test, input_data, split_time, t_end, dt);
        net.setTargetData(DataType::Test, target_series, split_time, t_end, dt);

        std::cout << "Training data: " << net.getInputData(DataType::Train).size(0) << " samples" << std::endl;
        std::cout << "Test data: " << net.getInputData(DataType::Test).size(0) << " samples" << std::endl;

        // Write input data for verification
        TimeSeriesSet<double> training_data_for_check = TimeSeriesSet<double>::fromTensor(
            net.getInputData(DataType::Train), t_start, split_time, net.generateInputFeatureNames());
        training_data_for_check.write("training_data_for_check.txt");

        // Step 5: Train the network using hyperparameters
        std::cout << "\n5. Training network..." << std::endl;

        std::vector<double> training_losses = net.train(
            hyperparams.getNumEpochs(),
            hyperparams.getBatchSize(),
            hyperparams.getLearningRate()
            );

        std::cout << "Training completed. Final loss: " << training_losses.back() << std::endl;

        // Step 6: Evaluate performance
        std::cout << "\n6. Evaluating performance..." << std::endl;

        // Make predictions on test data
        TimeSeriesSet<double> test_predictions = net.predict(
            DataType::Test, split_time, t_end, dt, {"predicted_output"});

        std::cout << "Test predictions generated: " << test_predictions[0].size() << " points" << std::endl;

        // Calculate metrics using the evaluate method
        auto metrics = net.evaluate();
        std::cout << "Test MSE: " << metrics["mse"] << std::endl;
        std::cout << "Test RMSE: " << metrics["rmse"] << std::endl;
        std::cout << "Test MAE: " << metrics["mae"] << std::endl;
        std::cout << "Test R²: " << metrics["r_squared"] << std::endl;

        // Step 7: Save results
        std::cout << "\n7. Saving results..." << std::endl;

        // Save the trained model
        net.saveModel("trained_model.pt");

        // Save predictions
        test_predictions.write("test_predictions.csv");

        // Save hyperparameters configuration
        std::ofstream hyperparams_file("hyperparameters_config.txt");
        hyperparams_file << hyperparams.toString() << std::endl;
        hyperparams_file.close();

        // Save training history
        std::cout << "Training loss history (last 10 epochs):" << std::endl;
        for (size_t i = std::max(0, static_cast<int>(training_losses.size()) - 10);
             i < training_losses.size(); ++i) {
            std::cout << "  Epoch " << (i + 1) << ": " << training_losses[i] << std::endl;
        }

        // Step 8: Demonstrate hyperparameter optimization potential
        std::cout << "\n8. Demonstrating optimization potential..." << std::endl;

        // Show how to iterate through different configurations
        std::cout << "Example optimization iterations:" << std::endl;

        int max_series_selections = HyperParameters::getMaxSelectionCode(3);
        std::cout << "  Possible series selections: 1 to " << max_series_selections << std::endl;

        long int max_lag_code = hyperparams.getMaxLagCode();
        std::cout << "  Max lag code per series: " << max_lag_code << std::endl;

        long int max_arch_code = hyperparams.getMaxArchitectureCode();
        std::cout << "  Max architecture code: " << max_arch_code << std::endl;

        long int max_multiplier_code = hyperparams.getMaxLagMultiplierCode(3);
        std::cout << "  Max lag multiplier code: " << max_multiplier_code << std::endl;

        // Example: test a few different configurations
        std::cout << "  Example series selection configurations:" << std::endl;
        for (int selection_code = 1; selection_code <= 3 && selection_code <= max_series_selections; selection_code++) {
            HyperParameters test_params = hyperparams;  // Copy current config
            test_params.setSelectedSeriesFromBinary(selection_code, 3);

            std::cout << "    Config " << selection_code << " would select series: [";
            auto selected = test_params.getSelectedSeriesIds();
            for (size_t i = 0; i < selected.size(); i++) {
                std::cout << selected[i];
                if (i < selected.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }

        // Example lag multiplier configurations
        std::cout << "  Example lag multiplier configurations:" << std::endl;
        for (long int mult_code = 0; mult_code <= 2; mult_code++) {
            HyperParameters test_params = hyperparams;  // Copy current config
            test_params.setLagMultipliersFromCode(mult_code, 3);

            auto multipliers = test_params.getLagMultiplier();
            std::cout << "    Multiplier code " << mult_code << " generates: [";
            for (size_t i = 0; i < multipliers.size(); i++) {
                std::cout << multipliers[i];
                if (i < multipliers.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }

        // Show total optimization space
        long int total_configs = max_series_selections *
                                 std::pow(max_lag_code + 1, 3) *
                                 max_arch_code *
                                 max_multiplier_code;
        std::cout << "  Total discrete configuration space: " << total_configs << " combinations" << std::endl;

        // Step 9: Demonstrate model loading with hyperparameters
        std::cout << "\n9. Demonstrating model save/load with hyperparameters..." << std::endl;

        // Create a new network with same hyperparameters
        NeuralNetworkWrapper loaded_net;
        loaded_net.initializeNetwork(&hyperparams, 1);

        // Load the saved model
        loaded_net.loadModel("trained_model.pt");

        // Verify it works by making a prediction
        loaded_net.setInputDataFromHyperParams(DataType::Test, input_data, split_time, t_end, dt);
        TimeSeriesSet<double> loaded_predictions = loaded_net.predict(DataType::Test, split_time, t_end, dt);

        std::cout << "Loaded model predictions: " << loaded_predictions[0].size() << " points" << std::endl;

        std::cout << "\n=== HyperParameters example completed successfully! ===" << std::endl;
        std::cout << "Configuration saved to: hyperparameters_config.txt" << std::endl;

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
    for (double t = -10.0; t <= 100.0; t += 0.1) {  // Extended range for larger lags
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
        double target = 0.3 * synthetic_input[0].interpol(t - 0.0) +     // lag 0
                        0.2 * synthetic_input[0].interpol(t - 0.1) +     // lag 1
                        0.2 * synthetic_input[1].interpol(t - 0.5) +     // lag 5
                        0.2 * synthetic_input[2].interpol(t - 0.2) +     // lag 2
                        0.1 * (static_cast<double>(rand()) / RAND_MAX - 0.5); // noise
        synthetic_target.addPoint(t, target);
    }
    synthetic_target.writefile("target_output.txt");

    std::cout << "Synthetic data created successfully!" << std::endl;
}
