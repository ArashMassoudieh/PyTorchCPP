#include <QCoreApplication>
#include <iostream>
#include <vector>
#include "neuralnetworkwrapper.h"
#include "TimeSeriesSet.h"
#include "TimeSeries.h"

void createSyntheticData();

int main(int argc, char *argv[]) {
    QCoreApplication app(argc, argv);

    try {
        std::cout << "=== Neural Network Wrapper Example ===" << std::endl;
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

        // Step 2: Create and configure the neural network
        std::cout << "\n2. Configuring neural network..." << std::endl;
        NeuralNetworkWrapper net;

        // Configure lag structure - use lags from multiple input series
        std::vector<std::vector<int>> lags = {
            {1, 2, 3},    // Series 0: use values from 1, 2, and 3 time steps ago
            {1, 5, 10},   // Series 1: use values from 1, 5, and 10 time steps ago
            {2, 4}        // Series 2: use values from 2 and 4 time steps ago
        };
        net.setLags(lags);

        // Configure network architecture
        std::vector<int> hidden_layers = {64, 32, 16};
        net.setHiddenLayers(hidden_layers);

        // Initialize network (8 inputs from lags: 3+3+2, 1 output)
        net.initializeNetwork(1, "relu");

        std::cout << "Network initialized with " << net.getTotalParameters() << " parameters" << std::endl;

        // Step 3: Prepare training and test data
        std::cout << "\n3. Preparing data..." << std::endl;

        // Set training data (first 80% of time range)
        double t_start = 0.0;
        double t_end = 100.0;
        double dt = 0.1;
        double split_time = 80.0;  // 80% for training, 20% for testing

        net.setInputData(DataType::Train, input_data, t_start, split_time, dt);
        net.setTargetData(DataType::Train, target_series, t_start, split_time, dt);

        // Set test data (last 20% of time range)
        net.setInputData(DataType::Test, input_data, split_time, t_end, dt);
        net.setTargetData(DataType::Test, target_series, split_time, t_end, dt);

        std::cout << "Training data: " << net.getInputData(DataType::Train).size(0) << " samples" << std::endl;
        std::cout << "Test data: " << net.getInputData(DataType::Test).size(0) << " samples" << std::endl;

        // Writing the inputdata for check
        // Checking the training data
        TimeSeriesSet<double> training_data_for_check = TimeSeriesSet<double>::fromTensor(net.getInputData(DataType::Train),t_start, t_end, net.generateInputFeatureNames());
        training_data_for_check.write("training_data_for_check.txt");

        // Step 4: Train the network
        std::cout << "\n4. Training network..." << std::endl;

        int num_epochs = 100;
        int batch_size = 32;
        double learning_rate = 0.001;

        std::vector<double> training_losses = net.train(num_epochs, batch_size, learning_rate);

        std::cout << "Training completed. Final loss: " << training_losses.back() << std::endl;

        // Step 5: Evaluate performance
        std::cout << "\n5. Evaluating performance..." << std::endl;

        // Make predictions on test data
        TimeSeriesSet<double> test_predictions = net.predict(DataType::Test, split_time, t_end, dt, {"predicted_output"});

        std::cout << "Test predictions generated: " << test_predictions[0].size() << " points" << std::endl;

        // You can also get raw tensor predictions for custom metrics
        torch::Tensor test_pred_tensor = net.forward(DataType::Test);
        torch::Tensor test_target_tensor = net.getTargetData(DataType::Test);

        // Calculate simple metrics
        torch::Tensor mse = torch::mse_loss(test_pred_tensor, test_target_tensor);
        torch::Tensor mae = torch::mean(torch::abs(test_pred_tensor - test_target_tensor));

        std::cout << "Test MSE: " << mse.item<double>() << std::endl;
        std::cout << "Test MAE: " << mae.item<double>() << std::endl;
        std::cout << "Test R2:" << net.calculateR2(DataType::Test) << std::endl;
        // Step 6: Save results
        std::cout << "\n6. Saving results..." << std::endl;

        // Save the trained model
        net.saveModel("trained_model.pt");

        // Save predictions
        test_predictions.write("test_predictions.csv");

        // Save training history
        std::cout << "Training loss history (last 10 epochs):" << std::endl;
        for (size_t i = std::max(0, static_cast<int>(training_losses.size()) - 10);
             i < training_losses.size(); ++i) {
            std::cout << "  Epoch " << (i + 1) << ": " << training_losses[i] << std::endl;
        }

        // Step 7: Demonstrate model loading (optional)
        std::cout << "\n7. Demonstrating model save/load..." << std::endl;

        // Create a new network with same architecture
        NeuralNetworkWrapper loaded_net;
        loaded_net.setLags(lags);
        loaded_net.setHiddenLayers(hidden_layers);
        loaded_net.initializeNetwork(1, "relu");

        // Load the saved model
        loaded_net.loadModel("trained_model.pt");

        // Verify it works by making a prediction
        loaded_net.setInputData(DataType::Test, input_data, split_time, t_end, dt);
        TimeSeriesSet<double> loaded_predictions = loaded_net.predict(DataType::Test, split_time, t_end, dt);

        std::cout << "Loaded model predictions: " << loaded_predictions[0].size() << " points" << std::endl;

        std::cout << "\n=== Example completed successfully! ===" << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
}

// Additional helper function for creating synthetic data if you don't have real data
void createSyntheticData() {
    std::cout << "Creating synthetic data for testing..." << std::endl;

    // Create synthetic input data (3 time series)
    TimeSeriesSet<double> synthetic_input(3);

    // Extend the time range to include negative values for proper lag handling
    for (double t = -2.0; t <= 100.0; t += 0.1) {
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

    // Create synthetic target data (combination of inputs with lags)
    TimeSeries<double> synthetic_target;
    for (double t = 0.0; t <= 100.0; t += 0.1) {
        double target = 0.5 * synthetic_input[0].interpol(t - 0.1) +  // Lag 1 from series 0
                        0.3 * synthetic_input[1].interpol(t - 0.5) +   // Lag 5 from series 1
                        0.2 * synthetic_input[2].interpol(t - 0.2) +   // Lag 2 from series 2
                        0.1 * (static_cast<double>(rand()) / RAND_MAX - 0.5); // Add some noise
        synthetic_target.addPoint(t, target);
    }
    synthetic_target.writefile("target_output.txt");

    std::cout << "Synthetic data created successfully!" << std::endl;
}
