// HyperParameters.cpp
#include "hyperparameters.h"
#include <stdexcept>
#include <algorithm>
#include <iomanip>  // for std::scientific, std::fixed, std::setprecision
#include <sstream>  // for std::ostringstream
#include <cmath>
#include <iostream>


// Constructor
HyperParameters::HyperParameters()
    : selected_series_ids_(),
    hidden_layers_{64, 32},
    input_activation_function_("sigmoid"),         // NEW: default linear input
    hidden_activation_function_("sigmoid"),    // NEW: default hidden relu
    output_activation_function_(""),        // NEW: default linear output,
    max_number_of_hidden_nodes_(128),
    max_number_of_hidden_layers_(5),
    lags_(),
    lag_multiplier_{1},
    max_lag_multiplier_(10),
    max_lags_(10),        // Default max nodes per layer
    lag_selection_odd_(3),         // Default max layers
    num_epochs_(100),
    batch_size_(32),
    learning_rate_(0.001),
    train_test_split_(0.8) {
}
// Time Series Selection
const std::vector<int>& HyperParameters::getSelectedSeriesIds() const {
    return selected_series_ids_;
}

void HyperParameters::setSelectedSeriesIds(const std::vector<int>& selected_ids) {
    validateSelectedSeriesIds(selected_ids);
    selected_series_ids_ = selected_ids;
}

long int HyperParameters::getMaxSelectionCode(int available_series_count) {
    if (available_series_count <= 0) {
        throw std::runtime_error("Available series count must be positive.");
    }

    if (available_series_count > 32) {
        throw std::runtime_error("Maximum 32 time series supported for binary selection.");
    }

    // Maximum value is 2^n - 1 (all bits set)
    // We exclude 0 since it would select no time series
    return (1 << available_series_count) - 1;
}

void HyperParameters::clearSelectedSeriesIds() {
    selected_series_ids_.clear();
}

void HyperParameters::setSelectedSeriesFromBinary(long int selection_code, int available_series_count) {
    if (available_series_count <= 0) {
        throw std::runtime_error("Available series count must be positive.");
    }

    if (available_series_count > 32) {
        throw std::runtime_error("Maximum 32 time series supported for binary selection.");
    }

    if (selection_code < 0) {
        throw std::runtime_error("Selection code must be non-negative.");
    }

    std::vector<int> selected_ids;

    // Convert integer to binary and store corresponding indices
    for (int i = 0; i < available_series_count; i++) {
        if ((selection_code >> i) & 1) {
            selected_ids.push_back(i);
        }
    }

    if (selected_ids.empty()) {
        throw std::runtime_error("Selection code " + std::to_string(selection_code) +
                                 " results in no time series being selected.");
    }

    selected_series_ids_ = selected_ids;
}

// Network Architecture
const std::vector<int>& HyperParameters::getHiddenLayers() const {
    return hidden_layers_;
}

void HyperParameters::setHiddenLayers(const std::vector<int>& hidden_layers) {
    validateHiddenLayers(hidden_layers);
    hidden_layers_ = hidden_layers;
}

// === Input Activation ===
const std::string& HyperParameters::getInputActivation() const {
    return input_activation_function_;
}
void HyperParameters::setInputActivation(const std::string& activation) {
    input_activation_function_ = activation;
}

// === Hidden Activation ===
const std::string& HyperParameters::getHiddenActivation() const {
    return hidden_activation_function_;
}
void HyperParameters::setHiddenActivation(const std::string& activation) {
    if (activation != "relu" && activation != "tanh" && activation != "sigmoid") {
        throw std::runtime_error("Invalid hidden activation: " + activation);
    }
    hidden_activation_function_ = activation;
}

// === Output Activation ===
const std::string& HyperParameters::getOutputActivation() const {
    return output_activation_function_;
}
void HyperParameters::setOutputActivation(const std::string& activation) {
    if (!activation.empty() &&
        activation != "relu" && activation != "tanh" && activation != "sigmoid") {
        throw std::runtime_error("Invalid output activation: " + activation);
    }
    output_activation_function_ = activation;
}


// Lag Configuration
const std::vector<std::vector<int>>& HyperParameters::getLags() const {
    return lags_;
}

void HyperParameters::setLags(const std::vector<std::vector<int>>& lags) {
    lags_ = lags;
}

// Training Parameters
int HyperParameters::getNumEpochs() const {
    return num_epochs_;
}

void HyperParameters::setNumEpochs(int num_epochs) {
    if (num_epochs <= 0) {
        throw std::runtime_error("Number of epochs must be positive.");
    }
    num_epochs_ = num_epochs;
}

int HyperParameters::getBatchSize() const {
    return batch_size_;
}

void HyperParameters::setBatchSize(int batch_size) {
    if (batch_size <= 0) {
        throw std::runtime_error("Batch size must be positive.");
    }
    batch_size_ = batch_size;
}

double HyperParameters::getLearningRate() const {
    return learning_rate_;
}

void HyperParameters::setLearningRate(double learning_rate) {
    if (learning_rate <= 0.0) {
        throw std::runtime_error("Learning rate must be positive.");
    }
    learning_rate_ = learning_rate;
}

// Data Configuration
double HyperParameters::getTrainTestSplit() const {
    return train_test_split_;
}

void HyperParameters::setTrainTestSplit(double train_test_split) {
    if (train_test_split < 0.0 || train_test_split > 1.0) {
        throw std::runtime_error("Train/test split must be between 0.0 and 1.0.");
    }
    train_test_split_ = train_test_split;
}

std::string HyperParameters::toString() const {
    std::string out;

    // Selected series
    out += "Selected series: [";
    for (size_t i = 0; i < selected_series_ids_.size(); i++) {
        out += std::to_string(selected_series_ids_[i]);
        if (i < selected_series_ids_.size() - 1) out += ",";
    }
    out += "]";

    // Hidden layers
    out += ", Hidden layers: [";
    for (size_t i = 0; i < hidden_layers_.size(); i++) {
        out += std::to_string(hidden_layers_[i]);
        if (i < hidden_layers_.size() - 1) out += ",";
    }
    out += "]";

    // Activation
    out += ", Input Activation: " + input_activation_function_;
    out += ", Hidden Activation: " + hidden_activation_function_;
    out += ", Output Activation: " + output_activation_function_;

    // Lags
    out += ", Lags: [";
    for (size_t i = 0; i < lags_.size(); i++) {
        out += "[";
        for (size_t j = 0; j < lags_[i].size(); j++) {
            out += std::to_string(lags_[i][j]);
            if (j < lags_[i].size() - 1) out += ",";
        }
        out += "]";
        if (i < lags_.size() - 1) out += ",";
    }
    out += "]";

    // Lag multipliers
    out += ", Lag multipliers: [";
    for (size_t i = 0; i < lag_multiplier_.size(); i++) {
        out += std::to_string(lag_multiplier_[i]);
        if (i < lag_multiplier_.size() - 1) out += ",";
    }
    out += "]";

    // Max lag multiplier
    out += ", Max lag multiplier: " + std::to_string(max_lag_multiplier_);
    out += ", Multiplier generation base: " + std::to_string(max_lag_multiplier_ + 1);


    // Max lags - THIS WAS MISSING
    out += ", Max lags: " + std::to_string(max_lags_);

    // Lag selection odd
    out += ", Lag selection odd: " + std::to_string(lag_selection_odd_);

    // Training parameters
    out += ", Epochs: " + std::to_string(num_epochs_);
    out += ", Batch size: " + std::to_string(batch_size_);
    out += ", Learning rate: " + std::to_string(learning_rate_);
    out += ", Train/test split: " + std::to_string(train_test_split_);

    out += ", Hidden layers: [";
    for (size_t i = 0; i < hidden_layers_.size(); i++) {
        out += std::to_string(hidden_layers_[i]);
        if (i < hidden_layers_.size() - 1) out += ",";
    }
    out += "]";

    // Add count of layers and total nodes
    out += ", Number of hidden layers: " + std::to_string(hidden_layers_.size());

    int total_hidden_nodes = 0;
    for (int nodes : hidden_layers_) {
        total_hidden_nodes += nodes;
    }
    out += ", Total hidden nodes: " + std::to_string(total_hidden_nodes);

    out += ", Max hidden nodes: " + std::to_string(max_number_of_hidden_nodes_);
    out += ", Max hidden layers: " + std::to_string(max_number_of_hidden_layers_);



    return out;
}
bool HyperParameters::isValid() const {
    // Check selected series
    if (selected_series_ids_.empty()) return false;

    // Check for negative series IDs
    for (int id : selected_series_ids_) {
        if (id < 0) return false;
    }

    // Check hidden layers
    if (hidden_layers_.empty()) return false;
    for (int nodes : hidden_layers_) {
        if (nodes <= 0) return false;
    }

    // Check training parameters
    if (num_epochs_ <= 0) return false;
    if (batch_size_ <= 0) return false;
    if (learning_rate_ <= 0.0) return false;
    if (train_test_split_ < 0.0 || train_test_split_ > 1.0) return false;

    // Check lag multipliers (now a vector)
    if (lag_multiplier_.empty()) return false;
    for (int multiplier : lag_multiplier_) {
        if (multiplier <= 0) return false;
        if (multiplier > max_lag_multiplier_) return false;
    }

    // Check max lag multiplier
    if (max_lag_multiplier_ <= 0) return false;

    // Check max lags
    if (max_lags_ <= 0) return false;

    // Check lag selection odd
    if (lag_selection_odd_ <= 1) return false;

    // Check activation function
    if (input_activation_function_ != "relu" &&
        input_activation_function_ != "tanh" &&
        input_activation_function_ != "sigmoid") {
        return false;
    }
    if (hidden_activation_function_ != "relu" &&
        hidden_activation_function_ != "tanh" &&
        hidden_activation_function_ != "sigmoid") {
        return false;
    }
    if (output_activation_function_ != "relu" &&
        output_activation_function_ != "tanh" &&
        output_activation_function_ != "sigmoid") {
        return false;
    }

    return true;
}

void HyperParameters::reset() {
    selected_series_ids_.clear();
    hidden_layers_ = {64, 32};
    input_activation_function_= "sigmoid";     // NEW
    hidden_activation_function_ = "sigmoid";   // NEW
    output_activation_function_.clear();    // NEW
    lags_.clear();
    lag_multiplier_ = {1};
    max_lag_multiplier_ = 10;
    max_lags_ = 10;
    lag_selection_odd_ = 3;
    max_number_of_hidden_nodes_ = 128;
    max_number_of_hidden_layers_ = 5;
    num_epochs_ = 100;
    batch_size_ = 32;
    learning_rate_ = 0.001;
    train_test_split_ = 0.8;
}

// Private helper methods
void HyperParameters::validateSelectedSeriesIds(const std::vector<int>& selected_ids) const {
    if (selected_ids.empty()) {
        throw std::runtime_error("At least one time series must be selected.");
    }

    // Validate no duplicate IDs
    std::vector<int> sorted_ids = selected_ids;
    std::sort(sorted_ids.begin(), sorted_ids.end());
    if (std::adjacent_find(sorted_ids.begin(), sorted_ids.end()) != sorted_ids.end()) {
        throw std::runtime_error("Duplicate time series IDs are not allowed.");
    }

    // Validate all IDs are non-negative
    for (int id : selected_ids) {
        if (id < 0) {
            throw std::runtime_error("Time series ID must be non-negative: " + std::to_string(id));
        }
    }
}

void HyperParameters::validateHiddenLayers(const std::vector<int>& hidden_layers) const {
    if (hidden_layers.empty()) {
        throw std::runtime_error("At least one hidden layer must be specified.");
    }

    for (int nodes : hidden_layers) {
        if (nodes <= 0) {
            throw std::runtime_error("Hidden layer size must be positive: " + std::to_string(nodes));
        }
    }
}

void HyperParameters::validateActivationFunction(const std::string& activation_function) const {
    if (activation_function != "relu" &&
        activation_function != "tanh" &&
        activation_function != "sigmoid") {
        throw std::runtime_error("Invalid activation function: " + activation_function +
                                 ". Supported: relu, tanh, sigmoid");
    }
}

void HyperParameters::setMaxLags(int max_lags) {
    if (max_lags <= 0) {
        throw std::runtime_error("Max lags must be positive.");
    }
    max_lags_ = max_lags;
}

int HyperParameters::getMaxLags() const {
    return max_lags_;
}

void HyperParameters::setLagSelectionOdd(int lag_selection_odd) {
    if (lag_selection_odd <= 1) {
        throw std::runtime_error("Lag selection odd must be greater than 1.");
    }
    lag_selection_odd_ = lag_selection_odd;
}

int HyperParameters::getLagSelectionOdd() const {
    return lag_selection_odd_;
}

void HyperParameters::setLagsFromVector(const std::vector<long int>& lag_codes) {
    if (lag_codes.empty()) {
        throw std::runtime_error("Lag codes vector cannot be empty.");
    }

    lags_.clear();
    lags_.resize(lag_codes.size());

    // Process each time series
    for (size_t series_idx = 0; series_idx < lag_codes.size(); series_idx++) {
        long int code = lag_codes[series_idx];
        std::vector<int> selected_lags;

        // Check each lag from 0 to max_lags-1
        for (int lag = 0; lag < max_lags_; lag++) {
            // Convert to base lag_selection_odd_ and check if remainder is 0
            if ((code % lag_selection_odd_) == 0) {
                selected_lags.push_back(lag);
            }
            code /= lag_selection_odd_;  // Move to next "digit"
        }

        lags_[series_idx] = selected_lags;
    }
}

long int HyperParameters::getMaxLagCode() const {
    if (max_lags_ <= 0) {
        throw std::runtime_error("Max lags must be set to a positive value.");
    }

    if (lag_selection_odd_ <= 1) {
        throw std::runtime_error("Lag selection odd must be set to a value greater than 1.");
    }

    // Maximum code is when all lags are selected (remainder 0 for all positions)
    // This happens when code is 0, lag_selection_odd_^1, lag_selection_odd_^2, etc.
    // The maximum useful value is lag_selection_odd_^max_lags - 1
    long int max_code = 1L;
    for (int i = 0; i < max_lags_; i++) {
        max_code *= lag_selection_odd_;
    }
    return max_code - 1L;
}

void HyperParameters::setLagMultiplier(const std::vector<int>& lag_multipliers) {
    if (lag_multipliers.empty()) {
        throw std::runtime_error("Lag multipliers vector cannot be empty.");
    }

    for (int multiplier : lag_multipliers) {
        if (multiplier <= 0) {
            throw std::runtime_error("All lag multipliers must be positive.");
        }
        if (multiplier > max_lag_multiplier_) {
            throw std::runtime_error("Lag multiplier " + std::to_string(multiplier) +
                                     " exceeds maximum allowed value " + std::to_string(max_lag_multiplier_));
        }
    }

    lag_multiplier_ = lag_multipliers;
}

const std::vector<int>& HyperParameters::getLagMultiplier() const {
    return lag_multiplier_;
}

void HyperParameters::setMaxLagMultiplier(int max_lag_multiplier) {
    if (max_lag_multiplier <= 0) {
        throw std::runtime_error("Max lag multiplier must be positive.");
    }
    max_lag_multiplier_ = max_lag_multiplier;
}

int HyperParameters::getMaxLagMultiplier() const {
    return max_lag_multiplier_;
}

void HyperParameters::setMaxNumberOfHiddenNodes(int max_nodes) {
    if (max_nodes <= 0) {
        throw std::runtime_error("Max number of hidden nodes must be positive.");
    }
    max_number_of_hidden_nodes_ = max_nodes;
}

int HyperParameters::getMaxNumberOfHiddenNodes() const {
    return max_number_of_hidden_nodes_;
}

void HyperParameters::setMaxNumberOfHiddenLayers(int max_layers) {
    if (max_layers <= 0) {
        throw std::runtime_error("Max number of hidden layers must be positive.");
    }
    max_number_of_hidden_layers_ = max_layers;
}

int HyperParameters::getMaxNumberOfHiddenLayers() const {
    return max_number_of_hidden_layers_;
}

void HyperParameters::setHiddenLayersFromCode(long int architecture_code, int min_nodes_per_layer) {
    if (architecture_code < 0) {
        throw std::runtime_error("Architecture code must be non-negative.");
    }

    if (min_nodes_per_layer <= 0) {
        throw std::runtime_error("Minimum nodes per layer must be positive.");
    }

    if (min_nodes_per_layer > max_number_of_hidden_nodes_) {
        throw std::runtime_error("Minimum nodes per layer cannot exceed maximum nodes per layer.");
    }

    // Base is max_nodes + 1 to allow 0 to max_nodes
    int base = max_number_of_hidden_nodes_ + 1;
    long int code = architecture_code;

    std::vector<int> temp_layers;

    // Extract up to max_number_of_hidden_layers_ digits
    for (int layer = 0; layer < max_number_of_hidden_layers_; layer++) {
        int digit = code % base;
        code /= base;

        // If digit > 0, add this layer (skip layers with 0 nodes)
        if (digit > 0) {
            // Map digit 1 to min_nodes_per_layer, digit max_nodes to max_nodes
            // Available range: min_nodes_per_layer to max_number_of_hidden_nodes_
            int range = max_number_of_hidden_nodes_ - min_nodes_per_layer + 1;
            int actual_nodes = min_nodes_per_layer + ((digit - 1) % range);
            temp_layers.push_back(actual_nodes);
        }

        // If code becomes 0, we've processed all meaningful digits
        if (code == 0) break;
    }

    // Reverse to get correct layer order (first extracted = last layer)
    std::reverse(temp_layers.begin(), temp_layers.end());

    // Ensure at least one layer exists
    if (temp_layers.empty()) {
        temp_layers.push_back(min_nodes_per_layer);
    }

    hidden_layers_ = temp_layers;
}

long int HyperParameters::getMaxArchitectureCode() const {
    int base = max_number_of_hidden_nodes_ + 1;
    return static_cast<long int>(std::pow(base, max_number_of_hidden_layers_)) - 1;
}

void HyperParameters::setLagMultipliersFromCode(long int multiplier_code, int num_series) {
    if (multiplier_code < 0) {
        throw std::runtime_error("Multiplier code must be non-negative.");
    }

    if (num_series <= 0) {
        throw std::runtime_error("Number of series must be positive.");
    }

    if (max_lag_multiplier_ <= 0) {
        throw std::runtime_error("Max lag multiplier must be set before generating multipliers from code.");
    }

    // Base is max_lag_multiplier + 1 to allow values 0 to max_lag_multiplier
    int base = max_lag_multiplier_ + 1;
    long int code = multiplier_code;

    std::vector<int> multipliers;
    multipliers.reserve(num_series);

    // Extract digits for each series
    for (int i = 0; i < num_series; i++) {
        int digit = code % base;
        code /= base;

        // Map digit 0 to 1 (minimum multiplier), digit max_lag_multiplier to max_lag_multiplier
        int actual_multiplier = (digit == 0) ? 1 : digit;
        multipliers.push_back(actual_multiplier);
    }

    // Set the generated multipliers
    lag_multiplier_ = multipliers;
}

long int HyperParameters::getMaxLagMultiplierCode(int num_series) const {
    if (num_series <= 0) {
        throw std::runtime_error("Number of series must be positive.");
    }

    if (max_lag_multiplier_ <= 0) {
        throw std::runtime_error("Max lag multiplier must be set to calculate maximum code.");
    }

    // Base is max_lag_multiplier + 1
    int base = max_lag_multiplier_ + 1;

    // Maximum code when all series have maximum multiplier
    long int max_code = 1L;
    for (int i = 0; i < num_series; i++) {
        max_code *= base;
    }
    return max_code - 1L;
}

std::vector<long int> HyperParameters::getOptimizationBounds(int num_available_series) {
    if (num_available_series <= 0) {
        throw std::runtime_error("Number of available series must be positive.");
    }

    if (num_available_series > 63) {
        throw std::runtime_error("Maximum 63 time series supported for optimization.");
    }

    std::vector<long int> bounds;

    // Parameter 0: Series selection code
    // Range: 1 to 2^num_available_series - 1
    bounds.push_back(getMaxSelectionCode(num_available_series));

    // Parameters 1 to num_available_series: Lag codes for each potential series
    // Range: 0 to (lag_selection_odd^max_lags - 1) for each series
    // We'll use default values for calculation
    int default_max_lags = 10;
    int default_lag_selection_odd = 3;
    long int max_lag_code = static_cast<long int>(std::pow(default_lag_selection_odd, default_max_lags)) - 1L;

    for (int i = 0; i < num_available_series; i++) {
        bounds.push_back(max_lag_code);
    }

    // Parameter (num_available_series + 1): Lag multiplier code
    // Range: 0 to (max_lag_multiplier + 1)^num_available_series - 1
    int default_max_lag_multiplier = 10;
    long int max_multiplier_code = static_cast<long int>(std::pow(default_max_lag_multiplier + 1, num_available_series)) - 1L;
    bounds.push_back(max_multiplier_code);

    // Parameter (num_available_series + 2): Architecture code
    // Range: 0 to (max_hidden_nodes + 1)^max_hidden_layers - 1
    int default_max_hidden_nodes = 128;
    int default_max_hidden_layers = 5;
    long int max_arch_code = static_cast<long int>(std::pow(default_max_hidden_nodes + 1, default_max_hidden_layers)) - 1L;
    bounds.push_back(max_arch_code);

    return bounds;
}

std::vector<std::string> HyperParameters::getOptimizationParameterNames(int num_available_series) {
    if (num_available_series <= 0) {
        throw std::runtime_error("Number of available series must be positive.");
    }

    std::vector<std::string> names;

    // Series selection
    names.push_back("series_selection_code");

    // Lag codes for each series
    for (int i = 0; i < num_available_series; i++) {
        names.push_back("lag_code_series_" + std::to_string(i));
    }

    // Lag multiplier code
    names.push_back("lag_multiplier_code");

    // Architecture code
    names.push_back("architecture_code");

    return names;
}

void HyperParameters::setFromOptimizationParameters(const std::vector<long int>& params, int num_available_series) {
    if (verbose_) {
        std::cout << "DEBUG setFromOptimizationParameters called with params: [";
        for (size_t i = 0; i < params.size(); i++) {
            std::cout << params[i];
            if (i < params.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << "num_available_series: " << num_available_series << std::endl;
    }

    // Expected parameter count: 1 (series) + num_available_series (lags) + 1 (multipliers) + 1 (architecture)
    int expected_size = 3 + num_available_series;
    if (static_cast<int>(params.size()) != expected_size) {
        throw std::runtime_error("Parameter vector size (" + std::to_string(params.size()) +
                                 ") doesn't match expected size (" + std::to_string(expected_size) +
                                 ") for " + std::to_string(num_available_series) + " series.");
    }

    // Set defaults first
    setMaxLags(10);
    setLagSelectionOdd(3);
    setMaxLagMultiplier(10);
    setMaxNumberOfHiddenNodes(128);
    setMaxNumberOfHiddenLayers(5);

    int param_idx = 0;

    // Parameter 0: Series selection
    long int series_code = params[param_idx];
    if (verbose_) {
        std::cout << "DEBUG Processing series selection code: " << series_code << std::endl;
    }
    if (series_code == 0) {
        if (verbose_) std::cout << "DEBUG Series code is 0 - clearing selection" << std::endl;
        selected_series_ids_.clear();
    } else {
        setSelectedSeriesFromBinary(series_code, num_available_series);
    }
    param_idx++;

    // Parameters 1 to num_available_series: Lag codes for each potential series
    if (verbose_) {
        std::cout << "DEBUG Processing lag codes..." << std::endl;
    }
    std::vector<long int> lag_codes;
    for (int i = 0; i < num_available_series; i++) {
        lag_codes.push_back(params[param_idx]);
        if (verbose_) {
            std::cout << "DEBUG Lag code for series " << i << ": " << params[param_idx] << std::endl;
        }
        param_idx++;
    }
    setLagsFromVector(lag_codes);

    // Parameter (num_available_series + 1): Lag multiplier code
    if (verbose_) {
        std::cout << "DEBUG Processing lag multiplier code: " << params[param_idx] << std::endl;
    }
    setLagMultipliersFromCode(params[param_idx], num_available_series);
    param_idx++;

    // Parameter (num_available_series + 2): Architecture code
    long int arch_code = params[param_idx];
    if (verbose_) {
        std::cout << "DEBUG Processing architecture code: " << arch_code << std::endl;
    }
    setHiddenLayersFromCode(arch_code, 16);

    if (verbose_) {
        std::cout << "DEBUG After setHiddenLayersFromCode, hidden_layers size: " << hidden_layers_.size() << std::endl;
        std::cout << "DEBUG Hidden layers: [";
        for (size_t i = 0; i < hidden_layers_.size(); i++) {
            std::cout << hidden_layers_[i];
            if (i < hidden_layers_.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        std::cout << "DEBUG Final lags size: " << lags_.size() << std::endl;
        std::cout << "DEBUG Final selected series size: " << selected_series_ids_.size() << std::endl;
    }

    // Set other default training parameters
    setInputActivation("relu");
    setHiddenActivation("relu");
    setOutputActivation("relu");
    setNumEpochs(100);
    setBatchSize(32);
    setLearningRate(0.001);
    setTrainTestSplit(0.8);
}

std::string HyperParameters::getOptimizationSpaceInfo() const {
    // Use the selected series to determine available series count
    int num_available_series = 0;
    if (!selected_series_ids_.empty()) {
        // Find the maximum series ID to determine available series count
        num_available_series = *std::max_element(selected_series_ids_.begin(), selected_series_ids_.end()) + 1;
    } else {
        // If no series selected, assume minimum 3 series for calculation
        num_available_series = 3;
    }

    std::string result;
    result += "=== Optimization Parameter Space ===\n";
    result += "Number of parameters: " + std::to_string(3 + num_available_series) + "\n";
    result += "Available time series: " + std::to_string(num_available_series) + "\n";
    result += "\n";

    // Calculate bounds using actual configured values
    std::vector<std::string> param_names;
    std::vector<long int> bounds;

    // Parameter 0: Series selection
    param_names.push_back("series_selection_code");
    long int max_series_code = getMaxSelectionCode(num_available_series);
    bounds.push_back(max_series_code);

    // Parameters 1 to num_available_series: Lag codes
    for (int i = 0; i < num_available_series; i++) {
        param_names.push_back("lag_code_series_" + std::to_string(i));
        long int max_lag_code = static_cast<long int>(std::pow(lag_selection_odd_, max_lags_)) - 1L;
        bounds.push_back(max_lag_code);
    }

    // Parameter (num_available_series + 1): Lag multiplier code
    param_names.push_back("lag_multiplier_code");
    long int max_multiplier_code = static_cast<long int>(std::pow(max_lag_multiplier_ + 1, num_available_series)) - 1L;
    bounds.push_back(max_multiplier_code);

    // Parameter (num_available_series + 2): Architecture code
    param_names.push_back("architecture_code");
    long int max_arch_code = static_cast<long int>(std::pow(max_number_of_hidden_nodes_ + 1, max_number_of_hidden_layers_)) - 1L;
    bounds.push_back(max_arch_code);

    // Display parameter details
    for (size_t i = 0; i < bounds.size(); i++) {
        result += "Parameter " + std::to_string(i) + ": " + param_names[i] + "\n";
        result += "  Range: 0 to " + std::to_string(bounds[i]) + "\n";
        result += "  Space size: " + std::to_string(bounds[i] + 1) + "\n";
        result += "\n";
    }

    // Calculate total combinations using logarithms to avoid overflow
    double log_total = 0.0;
    for (long int bound : bounds) {
        if (bound > 0) {
            log_total += std::log10(bound + 1);
        }
    }

    result += "Configuration used:\n";
    result += "  Max lags: " + std::to_string(max_lags_) + "\n";
    result += "  Lag selection odd: " + std::to_string(lag_selection_odd_) + "\n";
    result += "  Max lag multiplier: " + std::to_string(max_lag_multiplier_) + "\n";
    result += "  Max hidden nodes: " + std::to_string(max_number_of_hidden_nodes_) + "\n";
    result += "  Max hidden layers: " + std::to_string(max_number_of_hidden_layers_) + "\n";
    result += "\n";

    result += "Total configuration space: ";
    if (log_total > 15) {
        result += "~10^" + std::to_string(static_cast<int>(log_total)) + " combinations (extremely large)";
    } else {
        long double total_combinations = std::pow(10.0L, log_total);
        std::ostringstream oss;
        oss << std::scientific << std::setprecision(2) << total_combinations;
        result += oss.str() + " combinations";
    }
    result += "\n";
    result += "===============================\n";

    return result;
}

std::string HyperParameters::ParametersToString() const {
    std::string out;

    // Number of hidden layers
    out += "Number of hidden layers:" + std::to_string(hidden_layers_.size());

    // Number of nodes
    out += ", Number of nodes: [";
    for (size_t i = 0; i < hidden_layers_.size(); i++) {
        out += std::to_string(hidden_layers_[i]);
        if (i < hidden_layers_.size() - 1) {
            out += ",";
        }
    }
    out += "]";

    // Selected series (columns)
    out += ", Columns: [";
    for (size_t i = 0; i < selected_series_ids_.size(); i++) {
        out += std::to_string(selected_series_ids_[i]);
        if (i < selected_series_ids_.size() - 1) {
            out += ",";
        }
    }
    out += "]";

    // Lags
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

    // Lag multipliers
    out += ", Lag multipliers: [";
    for (size_t i = 0; i < lag_multiplier_.size(); i++) {
        out += std::to_string(lag_multiplier_[i]);
        if (i < lag_multiplier_.size() - 1) {
            out += ",";
        }
    }
    out += "]";

    return out;
}
bool HyperParameters::ValidLags() const {
    for (size_t i = 0; i < lags_.size(); i++) {
        if (lags_[i].size() > 0) {
            return true;
        }
    }
    return false;
}

void HyperParameters::setVerbose(bool verbose) {
    verbose_ = verbose;
}

bool HyperParameters::getVerbose() const {
    return verbose_;
}
