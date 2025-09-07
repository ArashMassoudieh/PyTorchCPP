#include <QCoreApplication>
#include <iostream>
#include <torch/torch.h>
#include <iomanip>
#include <cmath>

// Simple feedforward network for regression
struct RegressionNet : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};

    RegressionNet(int input_size, int hidden_size, int output_size) {
        fc1 = register_module("fc1", torch::nn::Linear(input_size, hidden_size));
        fc2 = register_module("fc2", torch::nn::Linear(hidden_size, hidden_size));
        fc3 = register_module("fc3", torch::nn::Linear(hidden_size, output_size));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        x = fc3->forward(x);  // No activation on output for regression
        return x;
    }
};

// Generate synthetic data: output = f(input)
std::pair<torch::Tensor, torch::Tensor> generateData(int num_samples) {
    torch::Tensor inputs = torch::randn({num_samples, 1}) * 2.0;  // Random inputs

    // Define relationship: output = sin(input) + 0.5 * input^2 + noise
    torch::Tensor outputs = torch::sin(inputs) + 0.5 * torch::pow(inputs, 2) +
                            0.1 * torch::randn({num_samples, 1});  // Add noise

    return {inputs, outputs};
}

void printSeparator() {
    std::cout << "=====================================" << std::endl;
}

int main(int argc, char *argv[])
{
    QCoreApplication app(argc, argv);

    try {
        printSeparator();
        std::cout << "Simple Neural Network Regression Example" << std::endl;
        std::cout << "Task: Predict output from current input (lag 0)" << std::endl;
        printSeparator();

        // Set random seed for reproducibility
        torch::manual_seed(42);

        // Parameters
        const int input_size = 1;
        const int hidden_size = 64;
        const int output_size = 1;
        const int num_train_samples = 1000;
        const int num_test_samples = 200;

        std::cout << "Network architecture: " << input_size << " -> "
                  << hidden_size << " -> " << hidden_size << " -> " << output_size << std::endl;
        std::cout << "Training samples: " << num_train_samples << std::endl;
        std::cout << "Test samples: " << num_test_samples << std::endl;

        // Generate training and test data
        auto [train_inputs, train_outputs] = generateData(num_train_samples);
        auto [test_inputs, test_outputs] = generateData(num_test_samples);

        std::cout << "\nGenerated synthetic data" << std::endl;
        std::cout << "Relationship: output = sin(input) + 0.5 * input^2 + noise" << std::endl;

        // Show sample data
        std::cout << "\nSample training data:" << std::endl;
        std::cout << "Input  -> Output" << std::endl;
        std::cout << "-------|--------" << std::endl;
        for (int i = 0; i < 5; ++i) {
            float input_val = train_inputs[i][0].item<float>();
            float output_val = train_outputs[i][0].item<float>();
            std::cout << std::fixed << std::setprecision(3)
                      << std::setw(6) << input_val << " -> "
                      << std::setw(6) << output_val << std::endl;
        }

        // Create network
        auto net = std::make_shared<RegressionNet>(input_size, hidden_size, output_size);

        // Create optimizer
        torch::optim::Adam optimizer(net->parameters(), torch::optim::AdamOptions(0.001));

        std::cout << "\nOptimizer: Adam with learning rate 0.001" << std::endl;

        printSeparator();
        std::cout << "Starting training..." << std::endl;

        // Training loop
        int num_epochs = 200;
        const int batch_size = 32;

        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            float total_loss = 0.0f;
            int num_batches = 0;

            // Mini-batch training
            for (int i = 0; i < num_train_samples; i += batch_size) {
                int current_batch_size = std::min(batch_size, num_train_samples - i);

                // Get batch
                torch::Tensor batch_inputs = train_inputs.slice(0, i, i + current_batch_size);
                torch::Tensor batch_outputs = train_outputs.slice(0, i, i + current_batch_size);

                // Zero gradients
                optimizer.zero_grad();

                // Forward pass
                torch::Tensor predictions = net->forward(batch_inputs);

                // Compute loss
                torch::Tensor loss = torch::mse_loss(predictions, batch_outputs);

                // Backward pass
                loss.backward();

                // Update weights
                optimizer.step();

                total_loss += loss.item<float>();
                num_batches++;
            }

            float avg_loss = total_loss / num_batches;

            // Print progress
            if ((epoch + 1) % 20 == 0) {
                std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs
                          << "], Average Loss: " << std::fixed << std::setprecision(6)
                          << avg_loss << std::endl;
            }
        }

        printSeparator();
        std::cout << "Training completed! Evaluating on test data..." << std::endl;

        // Test the network
        {
            torch::NoGradGuard no_grad;

            torch::Tensor test_predictions = net->forward(test_inputs);
            torch::Tensor test_loss = torch::mse_loss(test_predictions, test_outputs);

            std::cout << "Test Loss (MSE): " << std::fixed << std::setprecision(6)
                      << test_loss.item<float>() << std::endl;

            // Calculate additional metrics
            torch::Tensor squared_errors = torch::pow(test_predictions - test_outputs, 2);
            torch::Tensor absolute_errors = torch::abs(test_predictions - test_outputs);

            float rmse = std::sqrt(torch::mean(squared_errors).item<float>());
            float mae = torch::mean(absolute_errors).item<float>();

            std::cout << "Root Mean Square Error (RMSE): " << rmse << std::endl;
            std::cout << "Mean Absolute Error (MAE): " << mae << std::endl;

            // Calculate R-squared
            torch::Tensor target_mean = torch::mean(test_outputs);
            torch::Tensor ss_res = torch::sum(torch::pow(test_outputs - test_predictions, 2));
            torch::Tensor ss_tot = torch::sum(torch::pow(test_outputs - target_mean, 2));
            float r_squared = 1.0f - (ss_res.item<float>() / ss_tot.item<float>());

            std::cout << "R-squared: " << std::fixed << std::setprecision(4)
                      << r_squared << std::endl;

            // Show sample predictions
            std::cout << "\nSample predictions vs actual values:" << std::endl;
            std::cout << "Input    | Predicted | Actual    | Error" << std::endl;
            std::cout << "---------|-----------|-----------|----------" << std::endl;

            for (int i = 0; i < std::min(15, num_test_samples); ++i) {
                float input_val = test_inputs[i][0].item<float>();
                float predicted = test_predictions[i][0].item<float>();
                float actual = test_outputs[i][0].item<float>();
                float error = std::abs(predicted - actual);

                std::cout << std::fixed << std::setprecision(3)
                          << std::setw(8) << input_val << " | "
                          << std::setw(8) << predicted << "  | "
                          << std::setw(8) << actual << "  | "
                          << std::setw(8) << error << std::endl;
            }

            // Performance assessment
            std::cout << "\nPerformance Assessment:" << std::endl;
            if (r_squared > 0.9) {
                std::cout << "Excellent fit! Network learned the relationship very well." << std::endl;
            } else if (r_squared > 0.7) {
                std::cout << "Good fit! Network captured most of the relationship." << std::endl;
            } else if (r_squared > 0.5) {
                std::cout << "Moderate fit. Network learned some patterns." << std::endl;
            } else {
                std::cout << "Poor fit. Network may need more training or different architecture." << std::endl;
            }
        }

        printSeparator();

        // Show network info
        int total_params = 0;
        for (const auto& param : net->parameters()) {
            total_params += param.numel();
        }
        std::cout << "Network has " << total_params << " trainable parameters" << std::endl;

        std::cout << "\nRegression demonstration completed!" << std::endl;
        std::cout << "The network learned to map input values directly to output values." << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
        return -1;
    }
}
