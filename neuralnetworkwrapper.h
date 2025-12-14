// NeuralNetworkWrapper.h
#pragma once

#include <torch/torch.h>
#include <memory>
#include <string>
#include <vector>
#include "TimeSeriesSet.h"
#include "hyperparameters.h"

/**
 * @brief Enum to specify whether data is for training or testing.
 */
enum class DataType {
    Train,
    Test
};

class HyperParameters;

/**
 * @brief A flexible wrapper class for neural networks that inherits from torch::nn::Module.
 *
 * This class provides a convenient interface for creating, training, and using
 * neural networks with PyTorch C++. It encapsulates common neural network
 * operations and provides easy-to-use methods for training and inference.
 */

class NeuralNetworkWrapper {

public:
    // Constructors and Destructor
    NeuralNetworkWrapper();
    ~NeuralNetworkWrapper();

    // Copy and move constructors/assignment operators
    NeuralNetworkWrapper(const NeuralNetworkWrapper& other);
    NeuralNetworkWrapper& operator=(const NeuralNetworkWrapper& other);
    NeuralNetworkWrapper(NeuralNetworkWrapper&& other) = default;
    NeuralNetworkWrapper& operator=(NeuralNetworkWrapper&& other) = default;

    /**
     * @brief Get the current hyperparameters configuration.
     * @return Const reference to the hyperparameters object
     */
    const HyperParameters& getHyperParameters() const;

    void setHyperParameters(const HyperParameters& hyperparams);

    // Network Architecture
    /**
     * @brief Initialize the network with specified architecture.
     * Uses member variables lags_ and hidden_layers_ to determine input size and hidden architecture.
     * @param output_size Number of output features
     * @param activation_function Activation function name ("relu", "tanh", "sigmoid")
     */
    void initializeNetwork(int output_size,
                           const std::string& activation_function = "relu");

    // Forward pass
    /**
     * @brief Forward pass through the network using stored data.
     * @param data_type Specify whether to use training or test data
     * @return Output tensor from the forward pass
     */
    torch::Tensor forward(DataType data_type);

    /**
     * @brief Get the number of output time series (output features).
     * @return Number of output features/time series
     */
    int getOutputSize() const;

    /**
     * @brief Check if at least one time series has lags configured.
     * Delegates to HyperParameters if available, otherwise checks internal lags.
     * @return True if at least one time series has one or more lags, false otherwise
     */
    bool ValidLags() const;

    // Training
    /**
     * @brief Train the network using stored training data.
     * @param num_epochs Number of training epochs
     * @param batch_size Batch size for mini-batch training
     * @param learning_rate Learning rate for optimizer
     * @return Vector of loss values per epoch
     */
    std::vector<double> train(int num_epochs,
                              int batch_size = 32,
                              double learning_rate = 0.001);

    // Evaluation
    /**
     * @brief Evaluate the network on test data.
     * @param test_inputs Test input data
     * @param test_targets Test target data
     * @return Dictionary of evaluation metrics
     */
    std::map<std::string, double> evaluate(const torch::Tensor& test_inputs,
                                           const torch::Tensor& test_targets);


    /**
     * @brief Evaluate the network using internally stored test data.
     * @return Dictionary of evaluation metrics
     */
    std::map<std::string, double> evaluate();
    /**
     * @brief Make predictions and return results as TimeSeriesSet.
     * @param data_type Specify whether to use training or test input data
     * @param t_start Start time for the output time series
     * @param t_end End time for the output time series
     * @param dt Time step for the output time series
     * @param output_names Optional vector of names for output TimeSeries (default: "output_0", "output_1", etc.)
     * @return TimeSeriesSet containing predicted values as TimeSeries
     */
    TimeSeriesSet<double> predict(DataType data_type,
                                  double t_start, double t_end, double dt,
                                  const std::vector<std::string>& output_names = {});

    // Model Management
    /**
     * @brief Save the model to a file.
     * @param filepath Path to save the model
     */
    void saveModel(const std::string& filepath);

    /**
     * @brief Load a model from a file.
     * @param filepath Path to load the model from
     */
    void loadModel(const std::string& filepath);

    // Getters
    /**
     * @brief Get the total number of trainable parameters.
     * @return Total number of parameters
     */
    int getTotalParameters();

    /**
     * @brief Get the current loss value.
     * @return Current loss value
     */
    double getCurrentLoss() const;

    /**
     * @brief Get training history.
     * @return Vector of loss values from training
     */
    std::vector<double> getTrainingHistory() const;

    /**
     * @brief Check if the network is initialized.
     * @return True if network is initialized, false otherwise
     */
    bool isInitialized() const;

    void setInitialized(bool initialized) { is_initialized_ = initialized; }

    // Getters and Setters for Lag Configuration
    const std::vector<std::vector<int>>& getLags() const;
    void setLags(const std::vector<std::vector<int>>& lags);
    // Getters and Setters for Network Architecture
    const std::vector<int>& getHiddenLayers() const;
    void setHiddenLayers(const std::vector<int>& hidden_layers);
    void clear();

    /**
     * @brief Set input data from TimeSeriesSet using lag configuration.
     * @param data_type Specify whether this is training or test data
     * @param time_series_set Input TimeSeriesSet containing the raw time series data
     * @param t_start Start time for data extraction
     * @param t_end End time for data extraction
     * @param dt Time step for uniform sampling
     */
    void setInputData(DataType data_type, const TimeSeriesSet<double>& time_series_set,
                      double t_start, double t_end, double dt);

    /**
     * @brief Set output data from multiple TimeSeries columns.
     * @param data_type Specify whether this is training or test data
     * @param time_series_set TimeSeriesSet containing output data
     * @param output_indices Vector of indices specifying which TimeSeries to use as outputs
     * @param t_start Start time for data extraction
     * @param t_end End time for data extraction
     * @param dt Time step for uniform sampling
     */
    void setTargetData(DataType data_type, const TimeSeriesSet<double>& time_series_set,
                       const std::vector<int>& output_indices,
                       double t_start, double t_end, double dt);

    /**
     * @brief Set output data from a single TimeSeries.
     * @param data_type Specify whether this is training or test data
     * @param output_series Single TimeSeries to use as output
     * @param t_start Start time for data extraction
     * @param t_end End time for data extraction
     * @param dt Time step for uniform sampling
     */
    void setTargetData(DataType data_type, const TimeSeries<double>& output_series,
                       double t_start, double t_end, double dt);

    /**
     * @brief Get input data tensor for specified data type.
     * @param data_type Specify whether to get training or test data
     * @return Reference to the requested input data tensor
     */
    const torch::Tensor& getInputData(DataType data_type) const;

    /**
     * @brief Get output data tensor for specified data type.
     * @param data_type Specify whether to get training or test data
     * @return Reference to the requested output data tensor
     */
    const torch::Tensor& getTargetData(DataType data_type) const;

    /**
     * @brief Check if input data is available for specified data type.
     * @param data_type Specify whether to check training or test data
     * @return True if input data is available
     */
    bool hasInputData(DataType data_type) const;

    /**
     * @brief Check if output data is available for specified data type.
     * @param data_type Specify whether to check training or test data
     * @return True if output data is available
     */
    bool hasTargetData(DataType data_type) const;

    /**
     * @brief Calculate R² (coefficient of determination) for specified data type.
     * @param data_type Specify whether to calculate for training or test data
     * @return R² value
     */
    double calculateR2(DataType data_type);

    /**
     * @brief Set the names of the original TimeSeries for feature naming.
     * @param series_names Vector of original TimeSeries names
     */
    void setOriginalSeriesNames(const std::vector<std::string>& series_names);

    /**
     * @brief Get the names of the original TimeSeries.
     * @return Vector of original TimeSeries names
     */
    const std::vector<std::string>& getOriginalSeriesNames() const;

    /**
     * @brief Generate feature names for lag-based input data.
     * @return Vector of feature names in format "seriesname_lag"
     */
    std::vector<std::string> generateInputFeatureNames() const;

    /**
     * @brief Generate a string representation of the network parameters and configuration.
     * @return String describing the network architecture and configuration
     */
    std::string ParametersToString() const;

    /**
     * @brief Initialize the network using HyperParameters configuration.
     * Uses the HyperParameters object to set up the complete network architecture,
     * lag configuration, and other settings.
     * @param hyperparams Pointer to HyperParameters object containing configuration
     * @param output_size Number of output features
     */
    void initializeNetwork(HyperParameters* hyperparams, int output_size);

    /**
     * @brief Create input data tensor from TimeSeriesSet using HyperParameters configuration.
     * Uses the selected series and lag configuration from hyperparameters to create
     * the input tensor for training or testing.
     * @param data_type Specify whether this is training or test data
     * @param time_series_set Input TimeSeriesSet containing all available time series
     * @param t_start Start time for data extraction
     * @param t_end End time for data extraction
     * @param dt Time step for uniform sampling
     */
    void setInputDataFromHyperParams(DataType data_type, const TimeSeriesSet<double>& time_series_set,
                                     double t_start, double t_end, double dt);


    // Add these public methods to the existing class:

    // ============================================================================
    /// @name Genetic Algorithm Interface
    /// @{

    /**
    * @brief Get number of optimization parameters for GA.
    * @return Number of parameters needed for complete model specification
    */
    int ParametersSize() const;

    /**
     * @brief Get maximum value for a specific parameter index.
     * @param index Parameter index (0 to ParametersSize()-1)
     * @return Maximum allowed value for the parameter
     */
    long int MaxParameter(int index) const;

    /**
     * @brief Configure hyperparameters from GA chromosome.
     * @param parameters Vector of parameter values from GA
     */
    void AssignParameters(const std::vector<unsigned long int>& parameters);

    /**
     * @brief Create and initialize the neural network model.
     * Must be called after AssignParameters() and data configuration.
     */
    void CreateModel();

    /**
     * @brief Train network and return fitness metrics.
     * @return Map containing MSE and R² values for train/test data
     */
    std::map<std::string, double> Fitness();

    /**
     * @brief Configure time series data for training/testing.
     * @param input_data TimeSeriesSet containing input time series
     * @param target_data TimeSeries containing target values
     */
    void setTimeSeriesData(const TimeSeriesSet<double>& input_data,
                           const TimeSeries<double>& target_data);

    /**
     * @brief Configure time range and data split for training/testing.
     * @param t_start Start time
     * @param t_end End time
     * @param dt Time step
     * @param split_ratio Fraction for training (0.0-1.0, default 0.8)
     */
    void setTimeRange(double t_start, double t_end, double dt, double split_ratio = 0.8);

    /**
     * @brief Set number of available time series for optimization bounds.
     * @param count Total number of time series available in dataset
     */
    void setAvailableSeriesCount(int count);

    /// @}

    // Setter for verbose mode
    void setVerbose(bool verbose);
    bool getVerbose() const;

    // In neuralnetworkwrapper.h - add this declaration around line 92:
    /**
 * @brief Continue training the network for additional epochs from current state.
 * @param additional_epochs Number of additional epochs to train
 * @param batch_size Batch size for mini-batch training
 * @param learning_rate Learning rate for optimizer
 * @return Vector of loss values for the additional epochs only
 */
    std::vector<double> trainMore(int additional_epochs,
                                  int batch_size = 32,
                                  double learning_rate = 0.001);
private:
    // Member variables
    std::vector<std::vector<int>> lags_;                  ///< Lag configuration for each TimeSeries
    std::vector<int> hidden_layers_;                      ///< Number of nodes in each hidden layer
    HyperParameters  hyperparams_;
    std::vector<std::string> original_series_names_;     ///< Names of original TimeSeries from input data

    // Training state
    std::vector<double> training_history_;               ///< Loss history during training
    double current_loss_;                                ///< Current loss value
    bool is_initialized_;                                ///< Network initialization status

    // Input/Output data state
    torch::Tensor train_input_data_;                     ///< Training input tensor with lag features
    torch::Tensor train_target_data_;                    ///< Training target output tensor
    torch::Tensor test_input_data_;                      ///< Test input tensor with lag features
    torch::Tensor test_target_data_;                     ///< Test target output tensor

    // Network components
    std::vector<torch::nn::Linear> layers_;      ///< All linear layers
    std::string input_activation_function_;      ///< Activation for input layer
    std::string hidden_activation_function_;     ///< Activation for hidden layers
    std::string output_activation_function_;     ///< Activation for output layer
    int input_size_;                             ///< Input layer size
    int output_size_;

    bool verbose_ = false;

    // Network components (to be defined based on architecture)
    // torch::nn::ModuleList layers_; // Will be added when implementing network structure

    // Private helper methods
    /**
     * @brief Initialize weights using specified strategy.
     * @param strategy Weight initialization strategy
     */
    void initializeWeights(const std::string& strategy = "xavier");

    /**
     * @brief Create activation function from string.
     * @param activation_name Name of activation function
     * @return Activation function module
     */
    torch::nn::AnyModule createActivation(const std::string& activation_name);

    /**
     * @brief Compute various metrics for evaluation.
     * @param predictions Predicted values
     * @param targets True values
     * @return Map of metric names to values
     */
    std::map<std::string, double> computeMetrics(const torch::Tensor& predictions,
                                                 const torch::Tensor& targets);

    /**
     * @brief Internal forward pass that takes tensor input directly.
     * @param input Input tensor
     * @return Output tensor
     */
    torch::Tensor forward_internal(torch::Tensor input);

    /**
     * @brief Copy weights from another network instance.
     * @param other Source network to copy weights from
     */
    void copyWeightsFrom(const NeuralNetworkWrapper& other);



    // GA Interface data
    TimeSeriesSet<double> ga_input_data_;        ///< Input data for GA optimization
    TimeSeries<double> ga_target_data_;          ///< Target data for GA optimization
    double ga_t_start_, ga_t_end_, ga_dt_;       ///< Time configuration for GA
    double ga_split_ratio_;                      ///< Train/test split ratio for GA
    int ga_available_series_count_;              ///< Number of available time series
    bool ga_data_configured_;                    ///< Whether GA data has been set
};
