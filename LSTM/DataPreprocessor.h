/**
 * @file DataPreprocessor.h
 * @brief Data loading and preprocessing for time series
 * 
 * This class handles:
 * - Loading CSV data
 * - Creating sequences for LSTM input
 * - Normalization (standardization)
 * - Train/test splitting
 * 
 * @author Generated for Time Series Prediction
 * @date 2025
 */

#ifndef DATAPREPROCESSOR_H
#define DATAPREPROCESSOR_H

#include <torch/torch.h>
#include <QString>
#include <vector>
#include <memory>

namespace lstm_predictor {

/**
 * @struct ProcessedData
 * @brief Container for preprocessed training and testing data
 */
struct ProcessedData {
    torch::Tensor XTrain;          ///< Training input sequences
    torch::Tensor yTrain;          ///< Training targets
    torch::Tensor XTest;           ///< Testing input sequences
    torch::Tensor yTest;           ///< Testing targets
    torch::Tensor XAll;            ///< All sequences (for chronological prediction)
    torch::Tensor yAll;            ///< All targets (for chronological prediction)
    std::vector<int> trainTestMask; ///< Mask: 0=train, 1=test
    torch::Tensor meanX;           ///< Mean of input features (for normalization)
    torch::Tensor stdX;            ///< Std dev of input features
    torch::Tensor meanY;           ///< Mean of target (for denormalization)
    torch::Tensor stdY;            ///< Std dev of target
};

/**
 * @class DataPreprocessor
 * @brief Handles all data preprocessing operations
 */
class DataPreprocessor {
public:
    /**
     * @brief Constructor
     * @param seqLength Length of input sequences for LSTM
     * @param testSize Fraction of data to use for testing (0.0 to 1.0)
     */
    DataPreprocessor(int seqLength = 10, double testSize = 0.2);

    /**
     * @brief Load and preprocess data from CSV file
     * 
     * @param filePath Path to the CSV file
     * @param randomSplit If true, use random train/test split; if false, sequential
     * @return ProcessedData structure containing all preprocessed tensors
     */
    ProcessedData loadAndPreprocess(const QString& filePath, bool randomSplit = true);

    /**
     * @brief Denormalize target values back to original scale
     * 
     * @param normalizedY Normalized target values
     * @param meanY Mean used for normalization
     * @param stdY Standard deviation used for normalization
     * @return Denormalized values in original scale
     */
    static torch::Tensor denormalize(const torch::Tensor& normalizedY,
                                     const torch::Tensor& meanY,
                                     const torch::Tensor& stdY);

private:
    /**
     * @brief Read CSV file and return as tensor
     * 
     * @param filePath Path to CSV file
     * @return Tensor containing all data
     */
    torch::Tensor readCSV(const QString& filePath);

    /**
     * @brief Create sequences from time series data
     * 
     * @param data Input data tensor
     * @return Pair of (sequences, targets)
     */
    std::pair<torch::Tensor, torch::Tensor> createSequences(const torch::Tensor& data);

    /**
     * @brief Normalize tensor using z-score standardization
     * 
     * @param data Input tensor
     * @return Tuple of (normalized data, mean, std)
     */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> 
        normalize(const torch::Tensor& data);

    int seqLength_;     ///< Sequence length for LSTM input
    double testSize_;   ///< Fraction of data for testing
};

} // namespace lstm_predictor

#endif // DATAPREPROCESSOR_H
