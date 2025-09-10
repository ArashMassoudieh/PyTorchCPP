// NeuralNetworkWrapper.cpp
#include "neuralnetworkwrapper.h"
#include <stdexcept>
#include <iostream>
#include <cmath>
#include "hyperparameters.h"

// In constructor:
NeuralNetworkWrapper::NeuralNetworkWrapper()
    : current_loss_(0.0), is_initialized_(false),
    activation_function_("relu"), input_size_(0), output_size_(0),
    layers_(torch::nn::ModuleList()),
    train_input_data_(torch::Tensor()), train_target_data_(torch::Tensor()),
    test_input_data_(torch::Tensor()), test_target_data_(torch::Tensor()),
    original_series_names_(), hyperparams_(),
    ga_t_start_(0.0), ga_t_end_(100.0), ga_dt_(0.1), ga_split_ratio_(0.8),
    ga_available_series_count_(3), ga_data_configured_(false) {
}

void NeuralNetworkWrapper::clear() {
    training_history_.clear();
    current_loss_ = 0.0;
    is_initialized_ = false;
    activation_function_ = "relu";
    input_size_ = 0;
    output_size_ = 0;
    train_input_data_ = torch::Tensor();
    train_target_data_ = torch::Tensor();
    test_input_data_ = torch::Tensor();
    test_target_data_ = torch::Tensor();
    original_series_names_.clear();
    hyperparams_.reset();  // Reset pointer

    layers_ = torch::nn::ModuleList();
}

NeuralNetworkWrapper::~NeuralNetworkWrapper() {
    // Destructor implementation
}

void NeuralNetworkWrapper::initializeNetwork(int output_size,
                                             const std::string& activation_function) {
    // Calculate input size based on lags configuration
    int actual_input_size = 0;
    for (const auto& series_lags : lags_) {
        actual_input_size += series_lags.size(); // Each lag creates one input feature
    }

    if (actual_input_size == 0) {
        throw std::runtime_error("No lags configured. Use setLags() before initializing network.");
    }

    if (hidden_layers_.empty()) {
        throw std::runtime_error("No hidden layers configured. Use setHiddenLayers() before initializing network.");
    }

    // Store configuration
    activation_function_ = activation_function;
    input_size_ = actual_input_size;
    output_size_ = output_size;

    // Create a new ModuleList - PyTorch will handle the replacement
    layers_ = torch::nn::ModuleList();

    // Build layer sizes: [input] + [hidden_layers] + [output]
    std::vector<int> layer_sizes;
    layer_sizes.push_back(actual_input_size);
    layer_sizes.insert(layer_sizes.end(), hidden_layers_.begin(), hidden_layers_.end());
    layer_sizes.push_back(output_size);

    // Create and add linear layers
    for (size_t i = 0; i < layer_sizes.size() - 1; ++i) {
        auto linear_layer = torch::nn::Linear(layer_sizes[i], layer_sizes[i + 1]);
        layers_->push_back(linear_layer);
    }

    // Re-register the ModuleList (this replaces the old one)
    register_module("layers", layers_);

    // Initialize weights
    initializeWeights("xavier");

    is_initialized_ = true;
}

void NeuralNetworkWrapper::initializeNetwork(HyperParameters* hyperparams, int output_size) {
    if (hyperparams == nullptr) {
        throw std::runtime_error("HyperParameters pointer cannot be null.");
    }

    if (!hyperparams->isValid()) {
        throw std::runtime_error("Invalid HyperParameters configuration provided.");
    }

    if (output_size <= 0) {
        throw std::runtime_error("Output size must be positive.");
    }

    // Copy hyperparameters to internal storage
    hyperparams_ = *hyperparams;

    // Calculate input size based on selected series and their lags
    const auto& selected_series = hyperparams->getSelectedSeriesIds();
    const auto& all_lags = hyperparams->getLags();
    const auto& lag_multipliers = hyperparams->getLagMultiplier();


    if (selected_series.empty()) {
        throw std::runtime_error("No time series selected in hyperparameters.");
    }

    // Validate that we have lag configuration for all potentially available series
    if (all_lags.empty()) {
        throw std::runtime_error("No lag configuration found in hyperparameters.");
    }

    // Calculate total input features from selected series only
    int actual_input_size = 0;
    for (int series_idx : selected_series) {
        if (series_idx < 0 || series_idx >= static_cast<int>(all_lags.size())) {
            throw std::runtime_error("Selected series index " + std::to_string(series_idx) +
                                     " is out of range for lag configuration (size: " +
                                     std::to_string(all_lags.size()) + ").");
        }

        // Add number of lags for this selected series
        actual_input_size += all_lags[series_idx].size();
    }

    if (actual_input_size == 0) {
        throw std::runtime_error("No lag features configured for selected time series.");
    }

    // Get network architecture from hyperparameters
    const auto& hidden_layers = hyperparams->getHiddenLayers();
    if (hidden_layers.empty()) {
        throw std::runtime_error("No hidden layers configured in hyperparameters.");
    }

    // Store configuration
    activation_function_ = hyperparams->getActivationFunction();
    input_size_ = actual_input_size;
    output_size_ = output_size;

    // Create a new ModuleList
    layers_ = torch::nn::ModuleList();

    // Build layer sizes: [input] + [hidden_layers] + [output]
    std::vector<int> layer_sizes;
    layer_sizes.push_back(actual_input_size);
    layer_sizes.insert(layer_sizes.end(), hidden_layers.begin(), hidden_layers.end());
    layer_sizes.push_back(output_size);

    // Create and add linear layers
    for (size_t i = 0; i < layer_sizes.size() - 1; ++i) {
        auto linear_layer = torch::nn::Linear(layer_sizes[i], layer_sizes[i + 1]);
        layers_->push_back(linear_layer);
    }

    // Re-register the ModuleList
    register_module("layers", layers_);

    // Initialize weights
    initializeWeights("xavier");

    is_initialized_ = true;

    // Print network summary
    std::cout << "Network initialized with HyperParameters:" << std::endl;
    std::cout << "  Selected series: [";
    for (size_t i = 0; i < selected_series.size(); i++) {
        std::cout << selected_series[i];
        if (i < selected_series.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "  Total input features: " << actual_input_size << std::endl;
    std::cout << "  Hidden layers: [";
    for (size_t i = 0; i < hidden_layers.size(); i++) {
        std::cout << hidden_layers[i];
        if (i < hidden_layers.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "  Output size: " << output_size << std::endl;
    std::cout << "  Activation: " << activation_function_ << std::endl;
    std::cout << "  Total parameters: " << getTotalParameters() << std::endl;

    std::cout << "]" << std::endl;

    // Print lag details for selected series
    std::cout << "  Lag configuration for selected series:" << std::endl;
    for (int series_idx : selected_series) {
        const auto& base_lags = all_lags[series_idx];
        int multiplier = lag_multipliers[series_idx];
        std::cout << "    Series " << series_idx << " (multiplier " << multiplier << "): [";
        for (size_t j = 0; j < base_lags.size(); j++) {
            int actual_lag = base_lags[j] * multiplier;  // This line is crucial
            std::cout << actual_lag;  // Should print multiplied values
            if (j < base_lags.size() - 1) std::cout << ", ";
        }
        std::cout << "] (" << base_lags.size() << " features)" << std::endl;
    }
}

torch::Tensor NeuralNetworkWrapper::forward(DataType data_type) {
    if (!is_initialized_) {
        throw std::runtime_error("Network must be initialized before forward pass. Call initializeNetwork() first.");
    }

    if (layers_->size() == 0) {
        throw std::runtime_error("No layers defined in the network.");
    }

    // Get the appropriate input data based on data type
    if (!hasInputData(data_type)) {
        std::string data_type_str = (data_type == DataType::Train) ? "training" : "test";
        throw std::runtime_error("No " + data_type_str + " input data available. Use setInputData() first.");
    }

    torch::Tensor input = getInputData(data_type);

    // Validate input dimensions
    if (input.dim() != 2) {
        throw std::runtime_error("Input must be a 2D tensor with shape [batch_size, features].");
    }

    if (input.size(1) != input_size_) {
        throw std::runtime_error("Input feature size (" + std::to_string(input.size(1)) +
                                 ") doesn't match network input size (" + std::to_string(input_size_) + ").");
    }

    torch::Tensor x = input;

    // Forward pass through all layers except the last one (with activation)
    for (size_t i = 0; i < layers_->size() - 1; ++i) {
        // Apply linear transformation
        x = layers_[i]->as<torch::nn::Linear>()->forward(x);

        // Apply activation function
        if (activation_function_ == "relu") {
            x = torch::relu(x);
        } else if (activation_function_ == "tanh") {
            x = torch::tanh(x);
        } else if (activation_function_ == "sigmoid") {
            x = torch::sigmoid(x);
        } else {
            throw std::runtime_error("Unknown activation function: " + activation_function_);
        }
    }

    // Final layer without activation (for regression)
    if (layers_->size() > 0) {
        x = layers_[layers_->size() - 1]->as<torch::nn::Linear>()->forward(x);
    }

    return x;
}
std::vector<double> NeuralNetworkWrapper::train(int num_epochs,
                                                int batch_size,
                                                double learning_rate) {
    if (!is_initialized_) {
        throw std::runtime_error("Network must be initialized before training. Call initializeNetwork() first.");
    }

    if (!hasInputData(DataType::Train) || !hasTargetData(DataType::Train)) {
        throw std::runtime_error("Training data not available. Use setInputData() and setOutputData() with DataType::Train first.");
    }

    // Get training data
    torch::Tensor train_inputs = getInputData(DataType::Train);
    torch::Tensor train_targets = getTargetData(DataType::Train);

    // Validate data compatibility
    if (train_inputs.size(0) != train_targets.size(0)) {
        throw std::runtime_error("Training input and output data must have the same number of samples.");
    }

    const int num_train_samples = train_inputs.size(0);

    // Create optimizer
    torch::optim::Adam optimizer(this->parameters(), torch::optim::AdamOptions(learning_rate));

    // Clear previous training history
    training_history_.clear();
    training_history_.reserve(num_epochs);

    std::cout << "Starting training..." << std::endl;
    std::cout << "Epochs: " << num_epochs << ", Batch size: " << batch_size
              << ", Learning rate: " << learning_rate << std::endl;
    std::cout << "Training samples: " << num_train_samples << std::endl;

    // Training loop
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        double total_loss = 0.0;
        int num_batches = 0;

        // Mini-batch training
        for (int i = 0; i < num_train_samples; i += batch_size) {
            int current_batch_size = std::min(batch_size, num_train_samples - i);

            // Get batch
            torch::Tensor batch_inputs = train_inputs.slice(0, i, i + current_batch_size);
            torch::Tensor batch_targets = train_targets.slice(0, i, i + current_batch_size);

            // Zero gradients
            optimizer.zero_grad();

            // Forward pass
            torch::Tensor predictions = this->forward_internal(batch_inputs);

            // Compute loss
            torch::Tensor loss = torch::mse_loss(predictions, batch_targets);

            // Backward pass
            loss.backward();

            // Update weights
            optimizer.step();

            total_loss += loss.item<double>();
            num_batches++;
        }

        double avg_loss = total_loss / num_batches;
        current_loss_ = avg_loss;
        training_history_.push_back(avg_loss);

        // Print progress
        if ((epoch + 1) % 20 == 0 || epoch == 0) {
            std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs
                      << "], Average Loss: " << std::fixed << std::setprecision(6)
                      << avg_loss << std::endl;
        }
    }

    std::cout << "Training completed!" << std::endl;
    return training_history_;
}

void NeuralNetworkWrapper::saveModel(const std::string& filepath) {
    if (!is_initialized_) {
        throw std::runtime_error("Cannot save uninitialized network. Call initializeNetwork() first.");
    }

    try {
        // Create an output archive
        torch::serialize::OutputArchive archive;

        // Save all named parameters
        for (const auto& pair : this->named_parameters()) {
            archive.write(pair.key(), pair.value());
        }

        // Save to file
        archive.save_to(filepath);

        std::cout << "Model saved successfully to: " << filepath << std::endl;
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to save model to " + filepath + ": " + e.what());
    }
}

void NeuralNetworkWrapper::loadModel(const std::string& filepath) {
    if (!is_initialized_) {
        throw std::runtime_error("Cannot load into uninitialized network. Call initializeNetwork() first.");
    }

    try {
        // Create an input archive
        torch::serialize::InputArchive archive;
        archive.load_from(filepath);

        // Load parameters into the model
        torch::NoGradGuard no_grad;
        for (const auto& pair : this->named_parameters()) {
            const std::string& name = pair.key();
            auto& param = pair.value();

            torch::Tensor loaded_param;
            archive.read(name, loaded_param);
            param.copy_(loaded_param);
        }

        std::cout << "Model loaded successfully from: " << filepath << std::endl;
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to load model from " + filepath + ": " + e.what());
    }
}

int NeuralNetworkWrapper::getTotalParameters() const {
    // TODO: Implement parameter counting
    // This will sum up all trainable parameters
    int total = 0;
    for (const auto& param : this->parameters()) {
        total += param.numel();
    }
    return total;
}

double NeuralNetworkWrapper::getCurrentLoss() const {
    return current_loss_;
}

std::vector<double> NeuralNetworkWrapper::getTrainingHistory() const {
    return training_history_;
}

bool NeuralNetworkWrapper::isInitialized() const {
    return is_initialized_;
}

// Getters and Setters for Lag Configuration
const std::vector<std::vector<int>>& NeuralNetworkWrapper::getLags() const {
    return lags_;
}

void NeuralNetworkWrapper::setLags(const std::vector<std::vector<int>>& lags) {
    lags_ = lags;
    // Mark as uninitialized since architecture may need to change
    is_initialized_ = false;
}

// Getters and Setters for Network Architecture
const std::vector<int>& NeuralNetworkWrapper::getHiddenLayers() const {
    return hidden_layers_;
}

void NeuralNetworkWrapper::setHiddenLayers(const std::vector<int>& hidden_layers) {
    hidden_layers_ = hidden_layers;
    // Mark as uninitialized since architecture needs to change
    is_initialized_ = false;
}

void NeuralNetworkWrapper::initializeWeights(const std::string& strategy) {
    // TODO: Implement weight initialization
    // This will initialize weights using Xavier, He, or other strategies
    if (strategy == "xavier") {
        // Xavier/Glorot initialization
    } else if (strategy == "he") {
        // He initialization
    } else if (strategy == "normal") {
        // Normal distribution initialization
    }
}

torch::nn::AnyModule NeuralNetworkWrapper::createActivation(const std::string& activation_name) {
    // TODO: Implement activation function creation
    // This will return the appropriate activation function
    if (activation_name == "relu") {
        return torch::nn::AnyModule(torch::nn::ReLU());
    } else if (activation_name == "tanh") {
        return torch::nn::AnyModule(torch::nn::Tanh());
    } else if (activation_name == "sigmoid") {
        return torch::nn::AnyModule(torch::nn::Sigmoid());
    } else {
        throw std::invalid_argument("Unknown activation function: " + activation_name);
    }
}

std::map<std::string, double> NeuralNetworkWrapper::computeMetrics(const torch::Tensor& predictions,
                                                                   const torch::Tensor& targets) {
    // TODO: Implement metrics computation
    // This will compute MSE, MAE, R-squared, etc.
    std::map<std::string, double> metrics;

    // Mean Squared Error
    torch::Tensor mse = torch::mse_loss(predictions, targets);
    metrics["mse"] = mse.item<double>();

    // Root Mean Squared Error
    metrics["rmse"] = std::sqrt(metrics["mse"]);

    // Mean Absolute Error
    torch::Tensor mae = torch::mean(torch::abs(predictions - targets));
    metrics["mae"] = mae.item<double>();

    // R-squared
    torch::Tensor target_mean = torch::mean(targets);
    torch::Tensor ss_res = torch::sum(torch::pow(targets - predictions, 2));
    torch::Tensor ss_tot = torch::sum(torch::pow(targets - target_mean, 2));
    double r_squared = 1.0 - (ss_res.item<double>() / ss_tot.item<double>());
    metrics["r_squared"] = r_squared;

    return metrics;
}

void NeuralNetworkWrapper::setInputData(DataType data_type, const TimeSeriesSet<double>& time_series_set,
                                        double t_start, double t_end, double dt) {
    if (lags_.empty()) {
        throw std::runtime_error("No lag configuration set. Use setLags() before setting input data.");
    }

    setOriginalSeriesNames(time_series_set.getSeriesNames());

    // Calculate total number of features based on lags
    int total_features = 0;
    for (const auto& series_lags : lags_) {
        total_features += series_lags.size();
    }

    if (total_features == 0) {
        throw std::runtime_error("No lag features configured. Check lag configuration.");
    }

    // Calculate number of time steps
    int num_time_steps = static_cast<int>((t_end - t_start) / dt) + 1;

    // Create output tensor [num_samples, total_features]
    torch::Tensor input_tensor = torch::zeros({num_time_steps, total_features}, torch::kFloat32);

    int feature_index = 0;

    // Process each TimeSeries with its corresponding lags
    for (size_t series_idx = 0; series_idx < lags_.size(); ++series_idx) {
        if (series_idx >= time_series_set.size()) {
            throw std::runtime_error("Lag configuration references TimeSeries index " +
                                     std::to_string(series_idx) + " but TimeSeriesSet only has " +
                                     std::to_string(time_series_set.size()) + " series.");
        }

        const auto& current_series = time_series_set[series_idx];
        const auto& series_lags = lags_[series_idx];

        // For each lag in this series
        for (int lag : series_lags) {
            // Fill the feature column for this lag
            for (int time_idx = 0; time_idx < num_time_steps; ++time_idx) {
                double current_time = t_start + time_idx * dt;
                double lag_time = current_time - lag * dt;  // Apply lag

                // Interpolate value at lagged time
                double lagged_value = current_series.interpol(lag_time);
                input_tensor[time_idx][feature_index] = static_cast<float>(lagged_value);
            }
            feature_index++;
        }
    }

    // Store the processed input data based on data type
    if (data_type == DataType::Train) {
        train_input_data_ = input_tensor;
    } else {
        test_input_data_ = input_tensor;
    }

    // Validate input dimensions if network is initialized
    if (is_initialized_ && total_features != input_size_) {
        throw std::runtime_error("Generated input features (" + std::to_string(total_features) +
                                 ") don't match network input size (" + std::to_string(input_size_) +
                                 "). Network may need to be reinitialized.");
    }
}

void NeuralNetworkWrapper::setTargetData(DataType data_type, const TimeSeriesSet<double>& time_series_set,
                                         const std::vector<int>& output_indices,
                                         double t_start, double t_end, double dt) {
    if (output_indices.empty()) {
        throw std::runtime_error("No output indices specified. Provide at least one TimeSeries index.");
    }

    // Validate all indices are within bounds
    for (int idx : output_indices) {
        if (idx < 0 || idx >= static_cast<int>(time_series_set.size())) {
            throw std::runtime_error("Output index " + std::to_string(idx) +
                                     " is out of range. TimeSeriesSet has " +
                                     std::to_string(time_series_set.size()) + " series.");
        }
    }

    // Calculate number of time steps and output features
    int num_time_steps = static_cast<int>((t_end - t_start) / dt) + 1;
    int num_outputs = output_indices.size();

    // Create output tensor [num_samples, num_outputs]
    torch::Tensor output_tensor = torch::zeros({num_time_steps, num_outputs}, torch::kFloat32);

    // Fill tensor with interpolated values from selected TimeSeries
    for (int time_idx = 0; time_idx < num_time_steps; ++time_idx) {
        double current_time = t_start + time_idx * dt;

        for (int output_idx = 0; output_idx < num_outputs; ++output_idx) {
            int series_index = output_indices[output_idx];
            const auto& series = time_series_set[series_index];

            double interpolated_value = series.interpol(current_time);
            output_tensor[time_idx][output_idx] = static_cast<float>(interpolated_value);
        }
    }

    // Store the processed output data based on data type
    if (data_type == DataType::Train) {
        train_target_data_ = output_tensor;
    } else {
        test_target_data_ = output_tensor;
    }

    // Validate output dimensions if network is initialized
    if (is_initialized_ && num_outputs != output_size_) {
        throw std::runtime_error("Generated output features (" + std::to_string(num_outputs) +
                                 ") don't match network output size (" + std::to_string(output_size_) +
                                 "). Network may need to be reinitialized.");
    }

    // Validate sample count compatibility with input data
    if (hasInputData(data_type)) {
        torch::Tensor input_data = getInputData(data_type);
        if (num_time_steps != input_data.size(0)) {
            throw std::runtime_error("Output data sample count (" + std::to_string(num_time_steps) +
                                     ") doesn't match input data sample count (" + std::to_string(input_data.size(0)) + ")");
        }
    }
}

void NeuralNetworkWrapper::setTargetData(DataType data_type, const TimeSeries<double>& output_series,
                                         double t_start, double t_end, double dt) {
    // Calculate number of time steps
    int num_time_steps = static_cast<int>((t_end - t_start) / dt) + 1;

    // Create output tensor [num_samples, 1] for single output
    torch::Tensor output_tensor = torch::zeros({num_time_steps, 1}, torch::kFloat32);

    // Fill tensor with interpolated values from the single TimeSeries
    for (int time_idx = 0; time_idx < num_time_steps; ++time_idx) {
        double current_time = t_start + time_idx * dt;
        double interpolated_value = output_series.interpol(current_time);
        output_tensor[time_idx][0] = static_cast<float>(interpolated_value);
    }

    // Store the processed output data based on data type
    if (data_type == DataType::Train) {
        train_target_data_ = output_tensor;
    } else {
        test_target_data_ = output_tensor;
    }

    // Validate output dimensions if network is initialized
    if (is_initialized_ && 1 != output_size_) {
        throw std::runtime_error("Single output doesn't match network output size (" +
                                 std::to_string(output_size_) +
                                 "). Network may need to be reinitialized.");
    }

    // Validate sample count compatibility with input data
    if (hasInputData(data_type)) {
        torch::Tensor input_data = getInputData(data_type);
        if (num_time_steps != input_data.size(0)) {
            throw std::runtime_error("Output data sample count (" + std::to_string(num_time_steps) +
                                     ") doesn't match input data sample count (" + std::to_string(input_data.size(0)) + ")");
        }
    }
}

const torch::Tensor& NeuralNetworkWrapper::getInputData(DataType data_type) const {
    if (data_type == DataType::Train) {
        if (!hasInputData(DataType::Train)) {
            throw std::runtime_error("No training input data has been set. Use setInputData(DataType::Train, ...) first.");
        }
        return train_input_data_;
    } else {
        if (!hasInputData(DataType::Test)) {
            throw std::runtime_error("No test input data has been set. Use setInputData(DataType::Test, ...) first.");
        }
        return test_input_data_;
    }
}

const torch::Tensor& NeuralNetworkWrapper::getTargetData(DataType data_type) const {
    if (data_type == DataType::Train) {
        if (!hasTargetData(DataType::Train)) {
            throw std::runtime_error("No training target data has been set. Use setOutputData(DataType::Train, ...) first.");
        }
        return train_target_data_;
    } else {
        if (!hasTargetData(DataType::Test)) {
            throw std::runtime_error("No test target data has been set. Use setOutputData(DataType::Test, ...) first.");
        }
        return test_target_data_;
    }
}

bool NeuralNetworkWrapper::hasInputData(DataType data_type) const {
    if (data_type == DataType::Train) {
        return train_input_data_.defined() && train_input_data_.numel() > 0;
    } else {
        return test_input_data_.defined() && test_input_data_.numel() > 0;
    }
}

bool NeuralNetworkWrapper::hasTargetData(DataType data_type) const {
    if (data_type == DataType::Train) {
        return train_target_data_.defined() && train_target_data_.numel() > 0;
    } else {
        return test_target_data_.defined() && test_target_data_.numel() > 0;
    }
}

torch::Tensor NeuralNetworkWrapper::forward_internal(torch::Tensor input) {
    torch::Tensor x = input;

    // Forward pass through all layers except the last one (with activation)
    for (size_t i = 0; i < layers_->size() - 1; ++i) {
        x = layers_[i]->as<torch::nn::Linear>()->forward(x);

        // Apply activation function
        if (activation_function_ == "relu") {
            x = torch::relu(x);
        } else if (activation_function_ == "tanh") {
            x = torch::tanh(x);
        } else if (activation_function_ == "sigmoid") {
            x = torch::sigmoid(x);
        }
    }

    // Final layer without activation
    if (layers_->size() > 0) {
        x = layers_[layers_->size() - 1]->as<torch::nn::Linear>()->forward(x);
    }

    return x;
}

TimeSeriesSet<double> NeuralNetworkWrapper::predict(DataType data_type,
                                                    double t_start, double t_end, double dt,
                                                    const std::vector<std::string>& output_names) {
    if (!is_initialized_) {
        throw std::runtime_error("Network must be initialized before prediction. Call initializeNetwork() first.");
    }

    if (!hasInputData(data_type)) {
        std::string data_type_str = (data_type == DataType::Train) ? "training" : "test";
        throw std::runtime_error("No " + data_type_str + " input data available. Use setInputData() first.");
    }

    // Get predictions from the network
    torch::Tensor predictions;
    {
        torch::NoGradGuard no_grad;  // Disable gradient computation for inference
        predictions = forward(data_type);
    }

    // Calculate number of time steps
    int num_time_steps = static_cast<int>((t_end - t_start) / dt) + 1;

    // Validate that predictions match expected time steps
    if (predictions.size(0) != num_time_steps) {
        throw std::runtime_error("Prediction tensor size (" + std::to_string(predictions.size(0)) +
                                 ") doesn't match expected time steps (" + std::to_string(num_time_steps) + ")");
    }

    // Get number of output features
    int num_outputs = predictions.size(1);

    // Create TimeSeriesSet for results
    TimeSeriesSet<double> result(num_outputs);

    // Set names for each output TimeSeries
    for (int output_idx = 0; output_idx < num_outputs; ++output_idx) {
        std::string name;
        if (output_idx < static_cast<int>(output_names.size()) && !output_names[output_idx].empty()) {
            name = output_names[output_idx];
        } else {
            name = "output_" + std::to_string(output_idx);
        }
        result.setSeriesName(output_idx, name);
    }

    // Fill TimeSeries with predicted values
    for (int time_idx = 0; time_idx < num_time_steps; ++time_idx) {
        double current_time = t_start + time_idx * dt;
        for (int output_idx = 0; output_idx < num_outputs; ++output_idx) {
            double predicted_value = predictions[time_idx][output_idx].item<double>();
            result[output_idx].addPoint(current_time, predicted_value);
        }
    }

    return result;
}

double NeuralNetworkWrapper::calculateR2(DataType data_type) {
    if (!is_initialized_) {
        throw std::runtime_error("Network must be initialized before calculating R². Call initializeNetwork() first.");
    }

    if (!hasInputData(data_type) || !hasTargetData(data_type)) {
        std::string data_type_str = (data_type == DataType::Train) ? "training" : "test";
        throw std::runtime_error("No " + data_type_str + " data available for R² calculation. Use setInputData() and setOutputData() first.");
    }

    // Get predictions and targets
    torch::Tensor predictions;
    torch::Tensor targets = getTargetData(data_type);

    {
        torch::NoGradGuard no_grad;  // Disable gradients for inference
        predictions = forward(data_type);
    }

    // Validate tensor dimensions
    if (predictions.sizes() != targets.sizes()) {
        throw std::runtime_error("Prediction and target tensor dimensions don't match.");
    }

    // Calculate R² = 1 - (SS_res / SS_tot)
    torch::Tensor target_mean = torch::mean(targets);
    torch::Tensor ss_res = torch::sum(torch::pow(targets - predictions, 2));
    torch::Tensor ss_tot = torch::sum(torch::pow(targets - target_mean, 2));

    // Handle edge case where all target values are the same
    if (ss_tot.item<double>() == 0.0) {
        return (ss_res.item<double>() == 0.0) ? 1.0 : -std::numeric_limits<double>::infinity();
    }

    double r_squared = 1.0 - (ss_res.item<double>() / ss_tot.item<double>());

    return r_squared;
}

void NeuralNetworkWrapper::setOriginalSeriesNames(const std::vector<std::string>& series_names) {
    original_series_names_ = series_names;
}

const std::vector<std::string>& NeuralNetworkWrapper::getOriginalSeriesNames() const {
    return original_series_names_;
}

std::vector<std::string> NeuralNetworkWrapper::generateInputFeatureNames() const {
    std::vector<std::string> feature_names;

    if (lags_.empty()) {
        return feature_names; // No lags configured
    }

    // Calculate total number of features
    int total_features = 0;
    for (const auto& series_lags : lags_) {
        total_features += series_lags.size();
    }

    feature_names.reserve(total_features);

    // Generate names for each lag feature
    for (size_t series_idx = 0; series_idx < lags_.size(); ++series_idx) {
        // Get series name
        std::string series_name;
        if (series_idx < original_series_names_.size() && !original_series_names_[series_idx].empty()) {
            series_name = original_series_names_[series_idx];
        } else {
            series_name = "series_" + std::to_string(series_idx);
        }

        // Generate name for each lag in this series
        const auto& series_lags = lags_[series_idx];
        for (int lag : series_lags) {
            std::string feature_name = series_name + "_lag" + std::to_string(lag);
            feature_names.push_back(feature_name);
        }
    }

    return feature_names;
}

std::map<std::string, double> NeuralNetworkWrapper::evaluate(const torch::Tensor& test_inputs,
                                                             const torch::Tensor& test_targets) {
    if (!is_initialized_) {
        throw std::runtime_error("Network must be initialized before evaluation.");
    }

    // Validate input tensors
    if (!test_inputs.defined() || !test_targets.defined()) {
        throw std::runtime_error("Input tensors must be defined for evaluation.");
    }

    if (test_inputs.size(0) != test_targets.size(0)) {
        throw std::runtime_error("Input and target tensors must have the same number of samples.");
    }

    // Ensure we're in evaluation mode and no gradients are computed
    this->eval();

    torch::Tensor predictions;
    {
        torch::NoGradGuard no_grad;  // Critical: disable gradients completely
        predictions = forward_internal(test_inputs.detach());  // Detach inputs from computation graph
    }

    // Compute metrics without gradients
    std::map<std::string, double> metrics;

    // Detach all tensors before computing metrics
    torch::Tensor pred_detached = predictions.detach();
    torch::Tensor target_detached = test_targets.detach();

    // Mean Squared Error
    torch::Tensor mse = torch::mse_loss(pred_detached, target_detached);
    metrics["mse"] = mse.item<double>();

    // Root Mean Squared Error
    metrics["rmse"] = std::sqrt(metrics["mse"]);

    // Mean Absolute Error
    torch::Tensor mae = torch::mean(torch::abs(pred_detached - target_detached));
    metrics["mae"] = mae.item<double>();

    // R-squared
    torch::Tensor target_mean = torch::mean(target_detached);
    torch::Tensor ss_res = torch::sum(torch::pow(target_detached - pred_detached, 2));
    torch::Tensor ss_tot = torch::sum(torch::pow(target_detached - target_mean, 2));
    double r_squared = 1.0 - (ss_res.item<double>() / ss_tot.item<double>());
    metrics["r_squared"] = r_squared;

    metrics["sample_count"] = static_cast<double>(test_inputs.size(0));

    std::cout << "Evaluation completed on " << test_inputs.size(0) << " samples:" << std::endl;
    std::cout << "  MSE:  " << std::fixed << std::setprecision(6) << metrics["mse"] << std::endl;
    std::cout << "  RMSE: " << std::fixed << std::setprecision(6) << metrics["rmse"] << std::endl;
    std::cout << "  MAE:  " << std::fixed << std::setprecision(6) << metrics["mae"] << std::endl;
    std::cout << "  R²:   " << std::fixed << std::setprecision(6) << metrics["r_squared"] << std::endl;

    return metrics;
}

// And add the convenience method:
std::map<std::string, double> NeuralNetworkWrapper::evaluate() {
    if (!hasInputData(DataType::Test) || !hasTargetData(DataType::Test)) {
        throw std::runtime_error("No test data available for evaluation. Use setInputData() and setTargetData() with DataType::Test first.");
    }

    return evaluate(getInputData(DataType::Test), getTargetData(DataType::Test));
}

std::string NeuralNetworkWrapper::parametersToString() const {
    std::string out;

    // Number of hidden layers
    out += "Number of hidden layers: " + std::to_string(hidden_layers_.size());

    // Hidden layer nodes
    out += ", Number of nodes: [";
    for (size_t i = 0; i < hidden_layers_.size(); i++) {
        out += std::to_string(hidden_layers_[i]);
        if (i < hidden_layers_.size() - 1) {
            out += ",";
        }
    }
    out += "]";

    // Input/Output sizes (if network is initialized)
    if (is_initialized_) {
        out += ", Input size: " + std::to_string(input_size_);
        out += ", Output size: " + std::to_string(output_size_);
    }

    // Activation function
    out += ", Activation: " + activation_function_;

    // Lags configuration
    out += ", Lags: [";
    for (size_t i = 0; i < lags_.size(); i++) {
        out += "[";
        for (size_t j = 0; j < lags_[i].size(); j++) {
            out += std::to_string(lags_[i][j]);
            if (j < lags_[i].size() - 1) {
                out += ",";
            }
        }
        out += "]";
        if (i < lags_.size() - 1) {
            out += ",";
        }
    }
    out += "]";

    // Original series names (if available)
    if (!original_series_names_.empty()) {
        out += ", Series names: [";
        for (size_t i = 0; i < original_series_names_.size(); i++) {
            out += "\"" + original_series_names_[i] + "\"";
            if (i < original_series_names_.size() - 1) {
                out += ",";
            }
        }
        out += "]";
    }

    // Total parameters (if network is initialized)
    if (is_initialized_) {
        out += ", Total parameters: " + std::to_string(getTotalParameters());
    }

    // Training status
    out += ", Initialized: " + std::string(is_initialized_ ? "true" : "false");

    return out;
}

void NeuralNetworkWrapper::setInputDataFromHyperParams(DataType data_type,
                                                       const TimeSeriesSet<double>& time_series_set,
                                                       double t_start, double t_end, double dt) {
    if (!is_initialized_) {
        throw std::runtime_error("Network must be initialized before setting input data.");
    }

    const auto& selected_series = hyperparams_.getSelectedSeriesIds();
    const auto& all_lags = hyperparams_.getLags();
    const auto& lag_multipliers = hyperparams_.getLagMultiplier();

    if (selected_series.empty()) {
        throw std::runtime_error("No time series selected in hyperparameters.");
    }

    // Validate that time_series_set has enough series
    for (int series_idx : selected_series) {
        if (series_idx >= static_cast<int>(time_series_set.size())) {
            throw std::runtime_error("Selected series index " + std::to_string(series_idx) +
                                     " exceeds TimeSeriesSet size (" + std::to_string(time_series_set.size()) + ").");
        }
    }

    // Store original series names
    setOriginalSeriesNames(time_series_set.getSeriesNames());

    // Calculate total features from selected series
    int total_features = 0;
    for (int series_idx : selected_series) {
        total_features += all_lags[series_idx].size();
    }

    if (total_features != input_size_) {
        throw std::runtime_error("Calculated input features (" + std::to_string(total_features) +
                                 ") don't match network input size (" + std::to_string(input_size_) + ").");
    }

    // Calculate number of time steps
    int num_time_steps = static_cast<int>((t_end - t_start) / dt) + 1;

    // Create input tensor [num_samples, total_features]
    torch::Tensor input_tensor = torch::zeros({num_time_steps, total_features}, torch::kFloat32);

    int feature_index = 0;

    // Process each selected time series with its lags and multipliers
    for (int series_idx : selected_series) {
        const auto& current_series = time_series_set[series_idx];
        const auto& base_lags = all_lags[series_idx];
        int multiplier = lag_multipliers[series_idx];

        // For each lag in this series, apply the multiplier
        for (int base_lag : base_lags) {
            int actual_lag = base_lag * multiplier;

            // Fill the feature column for this actual lag
            for (int time_idx = 0; time_idx < num_time_steps; ++time_idx) {
                double current_time = t_start + time_idx * dt;
                double lag_time = current_time - actual_lag * dt;  // Apply actual lag

                // Interpolate value at lagged time
                double lagged_value = current_series.interpol(lag_time);
                input_tensor[time_idx][feature_index] = static_cast<float>(lagged_value);
            }
            feature_index++;
        }
    }

    // Store the processed input data
    if (data_type == DataType::Train) {
        train_input_data_ = input_tensor;
    } else {
        test_input_data_ = input_tensor;
    }

    std::cout << "Input data created: " << num_time_steps << " samples, "
              << total_features << " features" << std::endl;
}


// GA Interface implementations:
int NeuralNetworkWrapper::ParametersSize() const {
    return 3 + ga_available_series_count_;
}

long int NeuralNetworkWrapper::MaxParameter(int index) const {
    auto bounds = HyperParameters::getOptimizationBounds(ga_available_series_count_);
    if (index < 0 || index >= static_cast<int>(bounds.size())) {
        throw std::runtime_error("Parameter index out of range");
    }
    return bounds[index];
}

void NeuralNetworkWrapper::AssignParameters(const std::vector<unsigned long int>& parameters) {
    if (!ga_data_configured_) {
        throw std::runtime_error("GA data must be configured before assigning parameters");
    }

    // Convert to long int and configure hyperparameters
    std::vector<long int> params_long(parameters.begin(), parameters.end());

    // Create temporary hyperparameters object and configure it
    HyperParameters temp_hyperparams;
    temp_hyperparams.setFromOptimizationParameters(params_long, ga_available_series_count_);

    // Store the configured hyperparameters
    hyperparams_ = temp_hyperparams;
}

void NeuralNetworkWrapper::CreateModel() {
    try {
        // Clear any existing state
        clear();

        // Initialize network with current hyperparameters
        initializeNetwork(&hyperparams_, 1); // 1 output

        // Calculate split time
        double split_time = ga_t_start_ + ga_split_ratio_ * (ga_t_end_ - ga_t_start_);

        // Set training data
        setInputDataFromHyperParams(DataType::Train, ga_input_data_,
                                    ga_t_start_, split_time, ga_dt_);
        setTargetData(DataType::Train, ga_target_data_,
                      ga_t_start_, split_time, ga_dt_);

        // Set test data
        setInputDataFromHyperParams(DataType::Test, ga_input_data_,
                                    split_time, ga_t_end_, ga_dt_);
        setTargetData(DataType::Test, ga_target_data_,
                      split_time, ga_t_end_, ga_dt_);

    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to create model: " + std::string(e.what()));
    }
}

std::map<std::string, double> NeuralNetworkWrapper::Fitness() {
    std::map<std::string, double> fitness_map;

    try {
        // Train the network using hyperparameter settings
        train(hyperparams_.getNumEpochs(),
              hyperparams_.getBatchSize(),
              hyperparams_.getLearningRate());

        // Calculate training metrics
        double train_r2 = calculateR2(DataType::Train);
        auto train_metrics = evaluate(getInputData(DataType::Train),
                                      getTargetData(DataType::Train));

        // Calculate test metrics
        auto test_metrics = evaluate();

        // Store metrics with GA naming convention
        fitness_map["MSE_Train_0"] = train_metrics["mse"];
        fitness_map["R2_Train_0"] = train_r2;
        fitness_map["MSE_Test_0"] = test_metrics["mse"];
        fitness_map["R2_Test_0"] = test_metrics["r_squared"];

    } catch (const std::exception& e) {
        // Return bad fitness if training fails
        fitness_map["MSE_Train_0"] = 1e12;
        fitness_map["R2_Train_0"] = -1e12;
        fitness_map["MSE_Test_0"] = 1e12;
        fitness_map["R2_Test_0"] = -1e12;

        std::cout << "Training failed: " << e.what() << std::endl;
    }

    return fitness_map;
}

void NeuralNetworkWrapper::setTimeSeriesData(const TimeSeriesSet<double>& input_data,
                                             const TimeSeries<double>& target_data) {
    ga_input_data_ = input_data;
    ga_target_data_ = target_data;
    ga_data_configured_ = true;
}

void NeuralNetworkWrapper::setTimeRange(double t_start, double t_end, double dt, double split_ratio) {
    ga_t_start_ = t_start;
    ga_t_end_ = t_end;
    ga_dt_ = dt;
    ga_split_ratio_ = split_ratio;
}

void NeuralNetworkWrapper::setAvailableSeriesCount(int count) {
    ga_available_series_count_ = count;
}
