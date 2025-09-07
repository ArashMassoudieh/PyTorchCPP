#include <QCoreApplication>
#include <iostream>
#include <torch/torch.h>
#include <iomanip>
#include <cmath>
#include "TimeSeries.h"

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

#include <torch/torch.h>
#include <iostream>
#include <iomanip>

// Utility function to print comprehensive tensor information
void printTensorInfo(const torch::Tensor& tensor, const std::string& name = "Tensor") {
    std::cout << "=== " << name << " Information ===" << std::endl;

    // Basic properties
    std::cout << "Shape/Sizes:     " << tensor.sizes() << std::endl;
    std::cout << "Dimensions:      " << tensor.dim() << std::endl;
    std::cout << "Total elements:  " << tensor.numel() << std::endl;

    // Data type and device
    std::cout << "Data type:       " << tensor.dtype() << std::endl;
    std::cout << "Device:          " << tensor.device() << std::endl;

    // Memory layout
    std::cout << "Strides:         " << tensor.strides() << std::endl;
    std::cout << "Is contiguous:   " << (tensor.is_contiguous() ? "Yes" : "No") << std::endl;

    // Gradient information
    std::cout << "Requires grad:   " << (tensor.requires_grad() ? "Yes" : "No") << std::endl;
    std::cout << "Has grad_fn:     " << (tensor.grad_fn() ? "Yes" : "No") << std::endl;

    // Storage information
    std::cout << "Storage offset:  " << tensor.storage_offset() << std::endl;

    // Memory usage (approximate)
    size_t memory_bytes = tensor.numel() * tensor.element_size();
    std::cout << "Memory usage:    " << memory_bytes << " bytes";
    if (memory_bytes > 1024) {
        std::cout << " (" << std::fixed << std::setprecision(2)
        << static_cast<double>(memory_bytes) / 1024.0 << " KB)";
    }
    if (memory_bytes > 1024 * 1024) {
        std::cout << " (" << std::fixed << std::setprecision(2)
        << static_cast<double>(memory_bytes) / (1024.0 * 1024.0) << " MB)";
    }
    std::cout << std::endl;

    // Value range (for small tensors or summary for large ones)
    if (tensor.numel() > 0) {
        if (tensor.numel() <= 20) {
            std::cout << "Values:          " << tensor << std::endl;
        } else {
            // Statistical summary for large tensors
            auto flattened = tensor.flatten();
            std::cout << "Min value:       " << torch::min(flattened).item<double>() << std::endl;
            std::cout << "Max value:       " << torch::max(flattened).item<double>() << std::endl;
            std::cout << "Mean value:      " << torch::mean(flattened.to(torch::kFloat)).item<double>() << std::endl;
            std::cout << "First 5 values:  " << flattened.slice(0, 0, 5) << std::endl;
            std::cout << "Last 5 values:   " << flattened.slice(0, -5, flattened.size(0)) << std::endl;
        }
    }

    std::cout << "=================================" << std::endl << std::endl;
}

// Compact version for quick debugging
void tensorInfo(const torch::Tensor& tensor, const std::string& name = "") {
    std::cout << (name.empty() ? "Tensor" : name)
    << ": shape=" << tensor.sizes()
    << ", dtype=" << tensor.dtype()
    << ", device=" << tensor.device()
    << ", numel=" << tensor.numel() << std::endl;
}

// Comparison function for two tensors
void compareTensors(const torch::Tensor& tensor1, const torch::Tensor& tensor2,
                    const std::string& name1 = "Tensor1", const std::string& name2 = "Tensor2") {
    std::cout << "=== Tensor Comparison ===" << std::endl;
    std::cout << name1 << " shape: " << tensor1.sizes() << std::endl;
    std::cout << name2 << " shape: " << tensor2.sizes() << std::endl;
    std::cout << "Same shape: " << (tensor1.sizes() == tensor2.sizes() ? "Yes" : "No") << std::endl;
    std::cout << "Same dtype: " << (tensor1.dtype() == tensor2.dtype() ? "Yes" : "No") << std::endl;
    std::cout << "Same device: " << (tensor1.device() == tensor2.device() ? "Yes" : "No") << std::endl;

    if (tensor1.sizes() == tensor2.sizes() && tensor1.dtype() == tensor2.dtype()) {
        std::cout << "Tensors equal: " << (torch::equal(tensor1, tensor2) ? "Yes" : "No") << std::endl;
        if (!torch::equal(tensor1, tensor2)) {
            auto diff = tensor1 - tensor2;
            std::cout << "Max difference: " << torch::max(torch::abs(diff)).item<double>() << std::endl;
        }
    }
    std::cout << "=========================" << std::endl << std::endl;
}


// Generate synthetic data: output = f(input)
std::pair<torch::Tensor, torch::Tensor> generateData(int num_samples) {

    TimeSeries<double> inputTS;
    inputTS.readfile("/home/arash/Dropbox/Aquifolium/input_for_pytorch.txt");


    TimeSeries<double> outputTS;
    outputTS.readfile("/home/arash/Dropbox/Aquifolium/output_for_pytorch.txt");

    torch::Tensor inputs = inputTS.toTensor(false);

    // Define relationship: output = sin(input) + 0.5 * input^2 + noise
    torch::Tensor outputs = outputTS.toTensor(false);

    printTensorInfo(inputs,"Input");

    printTensorInfo(outputs,"Output");

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

        auto [train_inputs, train_outputs] = generateData(0);
        auto [test_inputs, test_outputs] = generateData(0);

        // Parameters
        const int input_size = 1;
        const int hidden_size = 64;
        const int output_size = 1;
        const int num_train_samples = train_inputs.size(0);
        const int num_test_samples = test_inputs.size(0);

        std::cout << "Network architecture: " << input_size << " -> "
                  << hidden_size << " -> " << hidden_size << " -> " << output_size << std::endl;
        std::cout << "Training samples: " << num_train_samples << std::endl;
        std::cout << "Test samples: " << num_test_samples << std::endl;

        // Generate training and test data


        std::cout << "\nGenerated synthetic data" << std::endl;
        std::cout << "Relationship: output = sin(input) + 0.5 * input^2 + noise" << std::endl;

        // Show sample data
        std::cout << "\nSample training data:" << std::endl;
        std::cout << "Input  -> Output" << std::endl;
        std::cout << "-------|--------" << std::endl;
        for (int i = 0; i < 5; ++i) {
            float input_val = train_inputs[i].item<float>();
            float output_val = train_outputs[i].item<float>();
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
                float input_val = test_inputs[i].item<float>();
                float predicted = test_predictions[i].item<float>();
                float actual = test_outputs[i].item<float>();
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
