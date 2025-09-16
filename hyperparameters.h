// HyperParameters.h
#pragma once

#include <vector>
#include <string>

/**
 * @brief Class to hold and manage all hyperparameters for neural network optimization.
 */
class HyperParameters {
public:
    // Constructors
    HyperParameters();
    HyperParameters(const HyperParameters& other) = default;
    HyperParameters& operator=(const HyperParameters& other) = default;
    ~HyperParameters() = default;

    // Time Series Selection
    const std::vector<int>& getSelectedSeriesIds() const;
    void setSelectedSeriesIds(const std::vector<int>& selected_ids);
    void clearSelectedSeriesIds();

    // Network Architecture
    const std::vector<int>& getHiddenLayers() const;
    void setHiddenLayers(const std::vector<int>& hidden_layers);

    // Activation functions
    const std::string& getInputActivation() const;
    void setInputActivation(const std::string& activation);

    const std::string& getHiddenActivation() const;
    void setHiddenActivation(const std::string& activation);

    const std::string& getOutputActivation() const;
    void setOutputActivation(const std::string& activation);

    // Lag Configuration
    const std::vector<std::vector<int>>& getLags() const;
    void setLags(const std::vector<std::vector<int>>& lags);

    /**
     * @brief Set lag multipliers for each time series.
     * @param lag_multipliers Vector of multipliers (one per time series)
     */
    void setLagMultiplier(const std::vector<int>& lag_multipliers);

    /**
     * @brief Get lag multipliers for all time series.
     * @return Vector of lag multipliers
     */
    const std::vector<int>& getLagMultiplier() const;

    /**
     * @brief Set maximum allowed value for lag multipliers.
     * @param max_lag_multiplier Maximum value for any lag multiplier
     */
    void setMaxLagMultiplier(int max_lag_multiplier);

    /**
     * @brief Get maximum allowed value for lag multipliers.
     * @return Maximum lag multiplier value
     */
    int getMaxLagMultiplier() const;

    // Training Parameters
    int getNumEpochs() const;
    void setNumEpochs(int num_epochs);

    int getBatchSize() const;
    void setBatchSize(int batch_size);

    double getLearningRate() const;
    void setLearningRate(double learning_rate);

    // Data Configuration
    double getTrainTestSplit() const;
    void setTrainTestSplit(double train_test_split);

    // Utility Methods
    /**
     * @brief Convert hyperparameters to string representation.
     * @return String describing all hyperparameters
     */
    std::string toString() const;

    /**
     * @brief Validate hyperparameters.
     * @return True if valid, false otherwise
     */
    bool isValid() const;

    /**
     * @brief Reset all parameters to default values.
     */
    void reset();

    /**
     * @brief Set selected time series using binary encoding.
     * @param selection_code Long integer for binary encoding
     * @param available_series_count Total available series
     */
    void setSelectedSeriesFromBinary(long int selection_code, int available_series_count);

    /**
     * @brief Get maximum selection code for given number of available series.
     * @param available_series_count Total available series
     * @return Maximum long integer value for binary encoding (2^n - 1)
     */
    static long int getMaxSelectionCode(int available_series_count);

    /**
     * @brief Set the maximum number of lags available for selection.
     * @param max_lags Maximum number of lags (lags will be numbered 0 to max_lags-1)
     */
    void setMaxLags(int max_lags);

    /**
     * @brief Get the maximum number of lags.
     * @return Maximum number of lags
     */
    int getMaxLags() const;

    /**
     * @brief Set the base number used for lag selection conversion.
     * @param lag_selection_odd Base number for converting lag codes (must be > 1)
     */
    void setLagSelectionOdd(int lag_selection_odd);

    /**
     * @brief Get the base number used for lag selection conversion.
     * @return Base number for lag selection conversion
     */
    int getLagSelectionOdd() const;

    /**
     * @brief Set lag configuration from vector of codes using base conversion.
     * Each element in lag_codes corresponds to a time series. The code is converted
     * to base lag_selection_odd_, and lags where the remainder is 0 are selected.
     * @param lag_codes Vector of long integers for lag selection (one per time series)
     */
    void setLagsFromVector(const std::vector<long int>& lag_codes);

    /**
     * @brief Get maximum possible value for lag codes given current configuration.
     * This represents the highest value where at least one lag would be selected
     * using the current max_lags and lag_selection_odd settings.
     * @return Maximum possible lag code value
     */
    long int getMaxLagCode() const;

    // Add to public section:
    /**
     * @brief Set maximum number of nodes allowed in any hidden layer.
     * @param max_nodes Maximum nodes per layer
     */
    void setMaxNumberOfHiddenNodes(int max_nodes);

    /**
     * @brief Get maximum number of nodes allowed in any hidden layer.
     * @return Maximum nodes per layer
     */
    int getMaxNumberOfHiddenNodes() const;

    /**
     * @brief Set maximum number of hidden layers allowed.
     * @param max_layers Maximum number of hidden layers
     */
    void setMaxNumberOfHiddenLayers(int max_layers);

    /**
     * @brief Get maximum number of hidden layers allowed.
     * @return Maximum number of hidden layers
     */
    int getMaxNumberOfHiddenLayers() const;

    /**
     * @brief Set hidden layer architecture from a single code using base conversion.
     * Layers with zero nodes are automatically excluded from the architecture.
     * @param architecture_code Single number encoding the entire architecture
     * @param min_nodes_per_layer Minimum nodes per layer (default: 1)
     */
    void setHiddenLayersFromCode(long int architecture_code, int min_nodes_per_layer = 1);

    /**
     * @brief Get maximum possible architecture code.
     * @return Maximum architecture code value
     */
    long int getMaxArchitectureCode() const;


    // Add to public section:
    /**
     * @brief Set lag multipliers from a single code using base conversion.
     * Each digit in base (max_lag_multiplier + 1) represents the multiplier for that series.
     * @param multiplier_code Single integer encoding all lag multipliers
     * @param num_series Number of time series (determines how many digits to extract)
     */
    void setLagMultipliersFromCode(long int multiplier_code, int num_series);

    /**
     * @brief Get maximum possible lag multiplier code for given number of series.
     * @param num_series Number of time series
     * @return Maximum multiplier code value
     */
    long int getMaxLagMultiplierCode(int num_series) const;

    /**
     * @brief Get optimization parameter bounds for genetic algorithm.
     * Returns maximum values for all parameters needed to fully specify the model.
     * @param num_available_series Total number of time series available in dataset
     * @return Vector containing maximum value for each optimization parameter
     */
    static std::vector<long int> getOptimizationBounds(int num_available_series);

    /**
     * @brief Get parameter names corresponding to optimization bounds.
     * @param num_available_series Total number of time series available in dataset
     * @return Vector of parameter names in same order as getOptimizationBounds()
     */
    static std::vector<std::string> getOptimizationParameterNames(int num_available_series);

    /**
     * @brief Configure HyperParameters from optimization parameter vector.
     * Sets all model structure parameters from a vector of integer values.
     * @param params Vector of parameter values (must match getOptimizationBounds() size)
     * @param num_available_series Total number of time series available in dataset
     */
    void setFromOptimizationParameters(const std::vector<long int>& params, int num_available_series);

    /**
     * @brief Get optimization space information as string for debugging.
     * @param num_available_series Total number of time series available in dataset
     * @return String containing optimization space details
     */
    std::string getOptimizationSpaceInfo() const;

    /**
     * @brief Generate a string representation matching the Qt CModelStructure style.
     * @return String describing the model structure parameters
     */
    std::string ParametersToString() const;

    /**
     * @brief Check if at least one time series has lags configured.
     * @return True if at least one time series has one or more lags, false otherwise
     */
    bool ValidLags() const;

    // Setter for verbose mode
    void setVerbose(bool verbose);
    bool getVerbose() const;

private:
    // Time Series Selection
    std::vector<int> selected_series_ids_;        ///< IDs of selected time series from input

    // Network Architecture
    std::vector<int> hidden_layers_;              ///< Number of nodes in each hidden layer
    std::string input_activation_function_;   ///< Activation for input pre-processing
    std::string hidden_activation_function_;  ///< Activation for hidden layers
    std::string output_activation_function_;  ///< Activation for output layer
    int max_number_of_hidden_nodes_;               ///< Maximum nodes allowed in any hidden layer
    int max_number_of_hidden_layers_;              ///< Maximum number of hidden layers allowed


    // Lag Configuration
    std::vector<std::vector<int>> lags_;          ///< Lag configuration for each selected time series
    std::vector<int> lag_multiplier_;              ///< Multiplier for lag optimization (one per time series)
    int max_lag_multiplier_;                       ///< Maximum value for lag multipliers
    int max_lags_;                                 ///< Maximum number of lags (lags are 0 to max_lags-1)
    int lag_selection_odd_;                        ///< Base number for lag selection conversion


    // Training Parameters
    int num_epochs_;                              ///< Number of training epochs
    int batch_size_;                              ///< Training batch size
    double learning_rate_;                        ///< Learning rate for optimizer

    // Data Configuration
    double train_test_split_;                     ///< Ratio for train/test split (0.0-1.0)

    bool verbose_ = false;

    // Private helper methods
    void validateSelectedSeriesIds(const std::vector<int>& selected_ids) const;
    void validateHiddenLayers(const std::vector<int>& hidden_layers) const;
    void validateActivationFunction(const std::string& activation_function) const;
};
