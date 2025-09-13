#include <QCoreApplication>
#include <iostream>
#include <vector>
#include "neuralnetworkwrapper.h"
#include "TimeSeriesSet.h"
#include "TimeSeries.h"
#include "TestHyperParameters.h"
#include "hyperparameters.h"
#include "ga.h"
#include <ATen/ATen.h>
#include <omp.h>

void createSyntheticData();
void testIndividualParameters();
std::vector<unsigned long int> binaryStringToParameters(const std::string& binary_str,
                                                        const std::vector<int>& split_locations);

int _main(int argc, char *argv[]) {
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

        std::cout << "Training performance:" << std::endl;
        double train_r2 = net.calculateR2(DataType::Train);
        std::cout << "  Training R²: " << std::fixed << std::setprecision(4) << train_r2 << std::endl;


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
        std::cout<< "Optimization space: " << hyperparams.getOptimizationSpaceInfo() << std::endl;

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

void createSyntheticGATestData();

int main(int argc, char *argv[]) {
    QCoreApplication app(argc, argv);

    at::set_num_threads(1);         // intra-op threads
    at::set_num_interop_threads(1); // inter-op threads

    // Optionally decide GA threads explicitly:
    omp_set_num_threads(std::max(1, (int)std::thread::hardware_concurrency() - 1));

#ifdef _OPENMP
    std::cout << "OpenMP ON, threads: " << omp_get_max_threads() << "\n";
#else
    std::cout << "OpenMP OFF\n";
#endif

    //testIndividualParameters();

    try {
        std::cout << "=== Genetic Algorithm Neural Network Optimization Test ===" << std::endl;

        // Step 1: Create synthetic data for GA testing
        std::cout << "\n1. Creating synthetic test data..." << std::endl;
        //createSyntheticGATestData();

        // Step 2: Load test data
        TimeSeriesSet<double> training_data;
        training_data.read("ga_test_input.csv", true);
        TimeSeries<double> target_data;
        target_data.readfile("ga_test_target.txt");
        std::cout << "Loaded " << training_data.size() << " input time series" << std::endl;
        std::cout << "Target series has " << target_data.size() << " points" << std::endl;

        // Step 3: Create and configure your NeuralNetworkWrapper as the model
        NeuralNetworkWrapper base_model;

        // ADD THIS SECTION HERE - Configure GA data
        std::cout << "\n2.5. Configuring GA data for neural network..." << std::endl;
        double t_start = 0.0;
        double t_end = 100.0;    // Adjust based on your data range
        double dt = 0.1;         // Adjust based on your data time step
        double split_ratio = 0.8;
        int available_series_count = training_data.size(); // Use actual number of series

        base_model.setTimeSeriesData(training_data, target_data);
        base_model.setTimeRange(t_start, t_end, dt, split_ratio);
        base_model.setAvailableSeriesCount(available_series_count);

        std::cout << "  Time range: " << t_start << " to " << t_end << " (dt=" << dt << ")" << std::endl;
        std::cout << "  Train/test split: " << split_ratio << std::endl;
        std::cout << "  Available series: " << available_series_count << std::endl;



        // Step 4: Configure GA settings
        GeneticAlgorithmsettings ga_settings;
        ga_settings.totalpopulation = 40;
        ga_settings.generations = 40;
        ga_settings.mutation_probability = 0.03;
        ga_settings.outputpath = ".";
        ga_settings.MSE_optimization = true;

        std::cout << "\n3. Configuring GA with settings:" << std::endl;
        std::cout << "  Population size: " << ga_settings.totalpopulation << std::endl;
        std::cout << "  Generations: " << ga_settings.generations << std::endl;
        std::cout << "  Mutation rate: " << ga_settings.mutation_probability << std::endl;
        std::cout << "  MSE optimization: " << (ga_settings.MSE_optimization ? "Test only" : "Test + Train") << std::endl;

        HyperParameters constraints;
        constraints.setMaxNumberOfHiddenNodes(6);
        constraints.setMaxNumberOfHiddenLayers(3);
        constraints.setMaxLags(10);
        constraints.setLagSelectionOdd(3);
        constraints.setMaxLagMultiplier(10);

        base_model.setHyperParameters(constraints);

        // Step 5: Create and run GA
        std::cout << "\n4. Running Genetic Algorithm..." << std::endl;
        GeneticAlgorithm<NeuralNetworkWrapper> ga;
        ga.model = base_model;  // Now the base model has GA data configured
        ga.Settings = ga_settings;

        // Run optimization
        ga.Initialize();
        NeuralNetworkWrapper best_model = ga.Optimize();

        std::cout << "\n5. GA optimization completed!" << std::endl;

        std::cout << "\n6. Saving the best model ... " << std::endl;

        best_model.saveModel("best_model.pt");

        std::cout << "Returned model config: " << best_model.ParametersToString() << std::endl;

        try {
            if (best_model.hasInputData(DataType::Test) && best_model.hasTargetData(DataType::Test)) {
                auto test_metrics = best_model.evaluate();
                std::cout << "Final model test performance:" << std::endl;
                std::cout << "MSE: " << test_metrics["mse"] << std::endl;
                std::cout << "R²: " << test_metrics["r_squared"] << std::endl;
            } else {
                std::cout << "Model is trained but test data not available for evaluation" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << "Evaluation error: " << e.what() << std::endl;

            // If evaluation fails, the model might need data to be set up again
            // This can happen if the model reference loses its data context
            std::cout << "Model is initialized: " << best_model.isInitialized() << std::endl;
            std::cout << "Model config: " << best_model.ParametersToString() << std::endl;
        }


        // Test on same data used during GA to verify consistency
        auto verify_metrics = best_model.evaluate();
        std::cout << "Verification R²: " << verify_metrics["r_squared"] << std::endl;


        // Save optimal model results using TimeSeriesSet
        std::cout << "\nSaving optimal model results..." << std::endl;

        try {
            // Generate predictions for training data
            torch::Tensor train_targets = best_model.getTargetData(DataType::Train);
            torch::Tensor train_predictions;
            {
                torch::NoGradGuard no_grad;
                train_predictions = best_model.forward(DataType::Train);
            }

            // Generate predictions for test data
            torch::Tensor test_targets = best_model.getTargetData(DataType::Test);
            torch::Tensor test_predictions;
            {
                torch::NoGradGuard no_grad;
                test_predictions = best_model.forward(DataType::Test);
            }

            // Combine targets and predictions into single tensors
            torch::Tensor train_combined = torch::cat({train_targets, train_predictions}, 1);  // [samples, 2]
            torch::Tensor test_combined = torch::cat({test_targets, test_predictions}, 1);     // [samples, 2]

            // Convert to TimeSeriesSet using your existing function
            double train_start = 0.0;
            double train_end = 80.0;  // 80% of 100.0
            std::vector<std::string> train_names = {"targets", "predictions"};

            TimeSeriesSet<double> train_results = TimeSeriesSet<double>::fromTensor(
                train_combined, train_start, train_end, train_names);

            double test_start = 80.0;  // Start where training ended
            double test_end = 100.0;
            std::vector<std::string> test_names = {"targets", "predictions"};

            TimeSeriesSet<double> test_results = TimeSeriesSet<double>::fromTensor(
                test_combined, test_start, test_end, test_names);

            // Write to CSV files
            train_results.write("optimal_training_results.csv");
            test_results.write("optimal_test_results.csv");

            // Calculate and save summary statistics
            auto train_metrics = best_model.evaluate(best_model.getInputData(DataType::Train), train_targets);
            auto test_metrics = best_model.evaluate(best_model.getInputData(DataType::Test), test_targets);

            std::ofstream summary_file("optimal_model_summary.txt");
            summary_file << "=== OPTIMAL MODEL PERFORMANCE SUMMARY ===" << std::endl;
            summary_file << best_model.ParametersToString() << std::endl;
            summary_file << "\nTraining Metrics:" << std::endl;
            summary_file << "  MSE: " << train_metrics["mse"] << std::endl;
            summary_file << "  R²:  " << train_metrics["r_squared"] << std::endl;
            summary_file << "\nTest Metrics:" << std::endl;
            summary_file << "  MSE: " << test_metrics["mse"] << std::endl;
            summary_file << "  R²:  " << test_metrics["r_squared"] << std::endl;
            summary_file << "\nGeneralization:" << std::endl;
            summary_file << "  Test/Train MSE Ratio: " << test_metrics["mse"] / train_metrics["mse"] << std::endl;
            summary_file.close();

            std::cout << "Optimal model results saved:" << std::endl;
            std::cout << "  optimal_training_results.csv - Training targets and predictions with time" << std::endl;
            std::cout << "  optimal_test_results.csv - Test targets and predictions with time" << std::endl;
            std::cout << "  optimal_model_summary.txt - Performance summary" << std::endl;

            std::cout << "\nOptimal Model Performance:" << std::endl;
            std::cout << "  Training R²: " << train_metrics["r_squared"] << std::endl;
            std::cout << "  Test R²: " << test_metrics["r_squared"] << std::endl;

        } catch (const std::exception& e) {
            std::cout << "Error saving optimal model results: " << e.what() << std::endl;
        }



        // Step 6: Display results
        std::cout << "\n6. Best solution results:" << std::endl;
        if (!ga.Individuals.empty()) {
            auto best_individual = ga.Individuals[ga.getRanks()[0]];
            std::cout << "Best individual fitness: " << best_individual.fitness << std::endl;
            std::cout << "Binary representation: " << best_individual.toBinary().getBinary() << std::endl;
            for (const auto& measure : best_individual.fitness_measures) {
                std::cout << measure.first << ": " << measure.second << std::endl;
            }
        }

        std::cout << "\n=== GA Test completed successfully! ===" << std::endl;
        std::cout << "Check GA_Output.txt for detailed results." << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
}

void createSyntheticGATestData() {
    std::cout << "Creating stationary synthetic GA test data..." << std::endl;

    // Seed for reproducible results
    srand(42);

    // Create synthetic input data (5 stationary time series)
    TimeSeriesSet<double> synthetic_input(5);

    for (double t = -2.0; t <= 100.0; t += 0.1) {
        // Series 0: Pure sine wave (stationary)
        synthetic_input[0].addPoint(t, 2.0 * std::sin(0.1 * t) + 0.5 * std::sin(0.3 * t));

        // Series 1: Cosine wave without trend (stationary)
        synthetic_input[1].addPoint(t, 1.5 * std::cos(0.15 * t) + 0.8 * std::cos(0.4 * t));

        // Series 2: White noise around constant mean (stationary)
        double noise1 = 0.5 * (static_cast<double>(rand()) / RAND_MAX - 0.5);
        synthetic_input[2].addPoint(t, 1.0 + noise1);

        // Series 3: AR(1) process (stationary) - autoregressive with mean reversion
        static double prev_val = 0.0;
        double noise2 = 0.3 * (static_cast<double>(rand()) / RAND_MAX - 0.5);
        double ar_val = 0.7 * prev_val + noise2;  // AR(1) with coefficient 0.7 < 1
        synthetic_input[3].addPoint(t, ar_val);
        prev_val = ar_val;

        // Series 4: Multiple periodic components (stationary)
        synthetic_input[4].addPoint(t,
                                    1.2 * std::sin(0.2 * t) +
                                        0.8 * std::cos(0.5 * t) +
                                        0.4 * std::sin(0.8 * t));
    }

    synthetic_input.setSeriesName(0, "temperature");
    synthetic_input.setSeriesName(1, "pressure");
    synthetic_input.setSeriesName(2, "flow_rate");
    synthetic_input.setSeriesName(3, "concentration");
    synthetic_input.setSeriesName(4, "velocity");
    synthetic_input.write("ga_test_input.csv");

    // Create stationary target data (linear combination with lags)
    TimeSeries<double> synthetic_target;

    for (double t = 0.0; t <= 100.0; t += 0.1) {
        double target = 0.4 * synthetic_input[0].interpol(t - 0.1) +    // Series 0, lag 1
                        0.3 * synthetic_input[1].interpol(t - 0.3) +     // Series 1, lag 3
                        0.2 * synthetic_input[3].interpol(t - 0.2) +     // Series 3, lag 2
                        0.1 * synthetic_input[4].interpol(t - 0.5) +     // Series 4, lag 5
                        0.05 * (static_cast<double>(rand()) / RAND_MAX - 0.5); // Small noise

        synthetic_target.addPoint(t, target);
    }

    // Write target data
    std::ofstream target_file("ga_test_target.txt");
    if (target_file.is_open()) {
        for (size_t i = 0; i < synthetic_target.size(); ++i) {
            target_file << std::fixed << std::setprecision(6)
            << synthetic_target.getTime(i) << ","
            << synthetic_target.getValue(i) << std::endl;
        }
        target_file.close();
    }

    // Print statistics to verify stationarity
    std::cout << "Stationary synthetic data created successfully!" << std::endl;
    std::cout << "Target combines: series 0 (lag1), series 1 (lag3), series 3 (lag2), series 4 (lag5)" << std::endl;

    // Calculate and display statistics for each series to verify stationarity
    std::cout << "\nSeries Statistics (should be roughly constant across time periods):" << std::endl;

    for (int series = 0; series < 5; series++) {
        // Calculate mean and std for first half vs second half
        std::vector<double> first_half, second_half;

        for (size_t i = 0; i < synthetic_input[series].size(); i++) {
            double time = synthetic_input[series].getTime(i);
            double value = synthetic_input[series].getValue(i);

            if (time < 50.0) {
                first_half.push_back(value);
            } else {
                second_half.push_back(value);
            }
        }

        auto calc_stats = [](const std::vector<double>& data) {
            double mean = 0.0, var = 0.0;
            for (double val : data) mean += val;
            mean /= data.size();

            for (double val : data) var += (val - mean) * (val - mean);
            var /= data.size();

            return std::make_pair(mean, std::sqrt(var));
        };

        auto stats1 = calc_stats(first_half);
        auto stats2 = calc_stats(second_half);

        std::cout << "  Series " << series << " - First half: mean=" << std::fixed << std::setprecision(3)
                  << stats1.first << ", std=" << stats1.second
                  << " | Second half: mean=" << stats2.first << ", std=" << stats2.second << std::endl;
    }

    // Calculate target statistics
    std::vector<double> target_first_half, target_second_half;
    for (size_t i = 0; i < synthetic_target.size(); i++) {
        double time = synthetic_target.getTime(i);
        double value = synthetic_target.getValue(i);

        if (time < 50.0) {
            target_first_half.push_back(value);
        } else {
            target_second_half.push_back(value);
        }
    }

    auto calc_target_stats = [](const std::vector<double>& data) {
        double mean = 0.0, var = 0.0;
        for (double val : data) mean += val;
        mean /= data.size();

        for (double val : data) var += (val - mean) * (val - mean);
        var /= data.size();

        return std::make_pair(mean, std::sqrt(var));
    };

    auto target_stats1 = calc_target_stats(target_first_half);
    auto target_stats2 = calc_target_stats(target_second_half);

    std::cout << "\nTarget Statistics:" << std::endl;
    std::cout << "  First half: mean=" << std::fixed << std::setprecision(3)
              << target_stats1.first << ", std=" << target_stats1.second << std::endl;
    std::cout << "  Second half: mean=" << target_stats2.first << ", std=" << target_stats2.second << std::endl;

    double mean_diff = std::abs(target_stats1.first - target_stats2.first);
    std::cout << "  Mean difference between halves: " << mean_diff << std::endl;

    if (mean_diff < 0.2) {
        std::cout << "  ✓ Target appears stationary (small mean difference)" << std::endl;
    } else {
        std::cout << "  ⚠ Target may still have trend (large mean difference)" << std::endl;
    }
}

// Alternative version with even more explicit stationarity control
void createStrictlyStationarySyntheticData() {

    std::cout << "Creating strictly stationary synthetic data..." << std::endl;

    srand(42);
    TimeSeriesSet<double> synthetic_input(5);

    // Generate stationary series with explicit mean centering
    for (double t = -2.0; t <= 100.0; t += 0.1) {
        // All series designed to have zero mean and constant variance

        // Series 0: Sum of sines with different frequencies
        synthetic_input[0].addPoint(t,
                                    std::sin(0.1 * t) + 0.5 * std::sin(0.25 * t) + 0.3 * std::sin(0.7 * t));

        // Series 1: Sum of cosines
        synthetic_input[1].addPoint(t,
                                    std::cos(0.12 * t) + 0.6 * std::cos(0.3 * t) + 0.4 * std::cos(0.8 * t));

        // Series 2: White noise
        synthetic_input[2].addPoint(t,
                                    (static_cast<double>(rand()) / RAND_MAX - 0.5) * 2.0);

        // Series 3: ARMA process (stationary by design)
        static std::vector<double> ar_history = {0.0, 0.0};
        double noise = (static_cast<double>(rand()) / RAND_MAX - 0.5) * 0.5;
        double ar_val = 0.6 * ar_history[0] - 0.2 * ar_history[1] + noise;
        synthetic_input[3].addPoint(t, ar_val);
        ar_history[1] = ar_history[0];
        ar_history[0] = ar_val;

        // Series 4: Mixed frequencies
        synthetic_input[4].addPoint(t,
                                    0.8 * std::sin(0.15 * t) + 0.6 * std::cos(0.4 * t) + 0.2 * std::sin(1.1 * t));
    }

    synthetic_input.setSeriesName(0, "temp_stat");
    synthetic_input.setSeriesName(1, "press_stat");
    synthetic_input.setSeriesName(2, "flow_stat");
    synthetic_input.setSeriesName(3, "conc_stat");
    synthetic_input.setSeriesName(4, "vel_stat");
    synthetic_input.write("ga_test_input.csv");

    // Create target as linear combination (inherently stationary if inputs are stationary)
    TimeSeries<double> synthetic_target;

    for (double t = 0.0; t <= 100.0; t += 0.1) {
        double target = 0.4 * synthetic_input[0].interpol(t - 0.1) +
                        0.3 * synthetic_input[1].interpol(t - 0.3) +
                        0.2 * synthetic_input[3].interpol(t - 0.2) +
                        0.1 * synthetic_input[4].interpol(t - 0.5) +
                        0.02 * (static_cast<double>(rand()) / RAND_MAX - 0.5);

        synthetic_target.addPoint(t, target);
    }

    std::ofstream target_file("ga_test_target.txt");
    if (target_file.is_open()) {
        for (size_t i = 0; i < synthetic_target.size(); ++i) {
            target_file << std::fixed << std::setprecision(6)
            << synthetic_target.getTime(i) << ","
            << synthetic_target.getValue(i) << std::endl;
        }
        target_file.close();
    }

    std::cout << "Strictly stationary data created with zero-mean components!" << std::endl;
}

// Code snippet to test specific individual parameters and save predictions
void testIndividualParameters() {
    // Create HyperParameters directly and set the exact configuration
    createSyntheticGATestData();
    HyperParameters hyperparams;
    hyperparams.setVerbose(true);

    // Set the exact configuration from Individual 2
    std::vector<int> selected_series = {0,1, 3, 4};  // Columns: [1,3,4]
    hyperparams.setSelectedSeriesIds(selected_series);

    // Set the exact lags from Individual 2
    std::vector<std::vector<int>> lags = {
        {1},    // Series 0: lag 1 (0.1 time units)
        {3},    // Series 1: lag 3 (0.3 time units)
        {},     // Series 2: not used
        {2},    // Series 3: lag 2 (0.2 time units)
        {5}     // Series 4: lag 5 (0.5 time units)
    };
    hyperparams.setLags(lags);

    // Set the exact architecture from Individual 2
    std::vector<int> hidden_layers = {16};
    hyperparams.setNumEpochs(50);

    // Set other parameters to match your GA defaults
    hyperparams.setActivationFunction("relu");
    hyperparams.setNumEpochs(100);
    hyperparams.setBatchSize(32);
    hyperparams.setLearningRate(0.001);
    hyperparams.setTrainTestSplit(0.8);

    // Set lag multipliers (you'll need to figure out what these were)
    std::vector<int> lag_multipliers = {1, 1, 1, 1, 1};  // Assuming all 1s
    hyperparams.setLagMultiplier(lag_multipliers);

    // Create NeuralNetworkWrapper
    NeuralNetworkWrapper model;
    model.setVerbose(true);

    // Set up your time series data (replace with your actual data)
    // TimeSeriesSet<double> input_data = ...; // Your input data
    // TimeSeries<double> target_data = ...;   // Your target data

    try {
        // Initialize network with hyperparameters directly
        model.initializeNetwork(&hyperparams, 1);  // 1 output

        // YOU MUST SET YOUR ACTUAL DATA HERE - replace these with your data
        // Example setup (replace with your actual values):
        double t_start = 0.0;     // Your actual start time
        double t_end = 100.0;     // Your actual end time
        double dt = 0.1;          // Your actual time step
        double split_ratio = 0.8; // 80% training, 20% test
        double split_time = t_start + split_ratio * (t_end - t_start);


        TimeSeriesSet<double> input_data;
        input_data.read("ga_test_input.csv", true);  // CSV with multiple time series

        TimeSeries<double> target_data;
        target_data.readfile("ga_test_target.txt");  // Single target series

        model.setInputDataFromHyperParams(DataType::Train, input_data, t_start, split_time, dt);
        model.setTargetData(DataType::Train, target_data, t_start, split_time, dt);
        model.setInputDataFromHyperParams(DataType::Test, input_data, split_time, t_end, dt);
        model.setTargetData(DataType::Test, target_data, split_time, t_end, dt);

        // Train the model
        std::cout << "Training model..." << std::endl;
        auto training_history = model.Fitness();

        // Print fitness results
        std::cout << "\nFitness Results:" << std::endl;
        for (const auto& metric : training_history) {
            std::cout << metric.first << ": " << metric.second << std::endl;
        }

        // Get predictions for training data
        std::cout << "\nGenerating training predictions..." << std::endl;
        torch::Tensor train_inputs = model.getInputData(DataType::Train);
        torch::Tensor train_targets = model.getTargetData(DataType::Train);
        torch::Tensor train_predictions;
        {
            torch::NoGradGuard no_grad;
            train_predictions = model.forward(DataType::Train);
        }

        // Get predictions for test data
        std::cout << "Generating test predictions..." << std::endl;
        torch::Tensor test_inputs = model.getInputData(DataType::Test);
        torch::Tensor test_targets = model.getTargetData(DataType::Test);
        torch::Tensor test_predictions;
        {
            torch::NoGradGuard no_grad;
            test_predictions = model.forward(DataType::Test);
        }

        // Save training results to file
        std::ofstream train_file("training_results.csv");
        train_file << "Sample,Target,Prediction,Error,AbsError\n";

        int train_samples = train_targets.size(0);
        for (int i = 0; i < train_samples; i++) {
            double target = train_targets[i][0].item<double>();
            double prediction = train_predictions[i][0].item<double>();
            double error = target - prediction;
            double abs_error = std::abs(error);

            train_file << i << "," << target << "," << prediction << ","
                       << error << "," << abs_error << "\n";
        }
        train_file.close();

        // Save test results to file
        std::ofstream test_file("test_results.csv");
        test_file << "Sample,Target,Prediction,Error,AbsError\n";

        int test_samples = test_targets.size(0);
        for (int i = 0; i < test_samples; i++) {
            double target = test_targets[i][0].item<double>();
            double prediction = test_predictions[i][0].item<double>();
            double error = target - prediction;
            double abs_error = std::abs(error);

            test_file << i << "," << target << "," << prediction << ","
                      << error << "," << abs_error << "\n";
        }
        test_file.close();

        // Calculate and print detailed statistics
        torch::Tensor train_errors = train_targets - train_predictions;
        torch::Tensor test_errors = test_targets - test_predictions;

        double train_mse = torch::mean(train_errors * train_errors).item<double>();
        double test_mse = torch::mean(test_errors * test_errors).item<double>();

        double train_mae = torch::mean(torch::abs(train_errors)).item<double>();
        double test_mae = torch::mean(torch::abs(test_errors)).item<double>();

        // R-squared calculations
        torch::Tensor train_target_mean = torch::mean(train_targets);
        torch::Tensor test_target_mean = torch::mean(test_targets);

        torch::Tensor train_ss_res = torch::sum(train_errors * train_errors);
        torch::Tensor train_ss_tot = torch::sum((train_targets - train_target_mean) * (train_targets - train_target_mean));
        double train_r2 = 1.0 - (train_ss_res.item<double>() / train_ss_tot.item<double>());

        torch::Tensor test_ss_res = torch::sum(test_errors * test_errors);
        torch::Tensor test_ss_tot = torch::sum((test_targets - test_target_mean) * (test_targets - test_target_mean));
        double test_r2 = 1.0 - (test_ss_res.item<double>() / test_ss_tot.item<double>());

        std::cout << "\n=== Detailed Statistics ===" << std::endl;
        std::cout << "Training Data (" << train_samples << " samples):" << std::endl;
        std::cout << "  MSE: " << train_mse << std::endl;
        std::cout << "  MAE: " << train_mae << std::endl;
        std::cout << "  R²:  " << train_r2 << std::endl;
        std::cout << "  Target range: [" << torch::min(train_targets).item<double>()
                  << ", " << torch::max(train_targets).item<double>() << "]" << std::endl;
        std::cout << "  Prediction range: [" << torch::min(train_predictions).item<double>()
                  << ", " << torch::max(train_predictions).item<double>() << "]" << std::endl;

        std::cout << "\nTest Data (" << test_samples << " samples):" << std::endl;
        std::cout << "  MSE: " << test_mse << std::endl;
        std::cout << "  MAE: " << test_mae << std::endl;
        std::cout << "  R²:  " << test_r2 << std::endl;
        std::cout << "  Target range: [" << torch::min(test_targets).item<double>()
                  << ", " << torch::max(test_targets).item<double>() << "]" << std::endl;
        std::cout << "  Prediction range: [" << torch::min(test_predictions).item<double>()
                  << ", " << torch::max(test_predictions).item<double>() << "]" << std::endl;

        // Check for potential issues
        std::cout << "\n=== Diagnostic Information ===" << std::endl;
        std::cout << "Network Architecture: [117, 127, 22, 52] hidden layers" << std::endl;
        std::cout << "Selected Series: [1, 3, 4]" << std::endl;
        std::cout << "Input Features: " << model.generateInputFeatureNames().size() << std::endl;

        // Check for common overfitting indicators
        double performance_gap = test_mse / train_mse;
        std::cout << "Test/Train MSE Ratio: " << performance_gap << std::endl;
        if (performance_gap > 10.0) {
            std::cout << "WARNING: Severe overfitting detected (Test MSE >> Train MSE)" << std::endl;
        }

        if (test_r2 < -1.0) {
            std::cout << "WARNING: Test R² < -1 indicates model performs much worse than mean prediction" << std::endl;
        }

        std::cout << "\nResults saved to:" << std::endl;
        std::cout << "  simple_training_results.csv - Training data targets and predictions" << std::endl;
        std::cout << "  simple_test_results.csv - Test data targets and predictions" << std::endl;

        std::cout << "\n=== SIMPLE MODEL CONFIGURATION ===" << std::endl;
        std::cout << "Selected Series: [1, 3] (was [1, 3, 4])" << std::endl;
        std::cout << "Input Features: 2 (was 10)" << std::endl;
        std::cout << "Hidden Layers: [16] (was [117, 127, 22, 52])" << std::endl;
        std::cout << "Total Parameters: ~50 (was 20,338)" << std::endl;
        std::cout << "Training Epochs: 50 (was 100)" << std::endl;

    } catch (const std::exception& e) {
        std::cout << "Error testing individual: " << e.what() << std::endl;
    }
}

// Helper function to convert binary string to parameters
std::vector<unsigned long int> binaryStringToParameters(const std::string& binary_str,
                                                        const std::vector<int>& split_locations) {
    std::vector<unsigned long int> params;
    int start_pos = 0;

    for (int split_len : split_locations) {
        std::string param_binary = binary_str.substr(start_pos, split_len);
        unsigned long int param_value = 0;

        // Convert binary string to decimal
        for (int i = 0; i < param_binary.length(); i++) {
            if (param_binary[i] == '1') {
                param_value += (1UL << (param_binary.length() - 1 - i));
            }
        }

        params.push_back(param_value);
        start_pos += split_len;
    }

    return params;
}
