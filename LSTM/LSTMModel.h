/**
 * @file LSTMModel.h
 * @brief LSTM neural network model for time series prediction
 * 
 * This class implements an LSTM-based neural network using LibTorch (PyTorch C++).
 * The model consists of LSTM layers followed by dropout and a fully connected layer.
 * 
 * @author Generated for Time Series Prediction
 * @date 2025
 */

#ifndef LSTMMODEL_H
#define LSTMMODEL_H

#include <torch/torch.h>
#include <memory>

namespace lstm_predictor {

/**
 * @class LSTMModelImpl
 * @brief Implementation of LSTM model for time series prediction
 * 
 * This class defines the architecture of the LSTM network including:
 * - Multi-layer LSTM
 * - Dropout regularization
 * - Fully connected output layer
 */
class LSTMModelImpl : public torch::nn::Module {
public:
    /**
     * @brief Constructor for LSTM model
     * 
     * @param inputSize Number of input features per timestep
     * @param hiddenSize Number of hidden units in LSTM layers
     * @param numLayers Number of stacked LSTM layers
     * @param outputSize Number of output features (typically 1 for regression)
     * @param dropout Dropout probability for regularization (default: 0.3)
     */
    LSTMModelImpl(int64_t inputSize, int64_t hiddenSize, int64_t numLayers, 
                  int64_t outputSize, double dropout = 0.3);

    /**
     * @brief Forward pass through the network
     * 
     * @param x Input tensor of shape [batch_size, seq_length, input_size]
     * @return Output tensor of shape [batch_size, output_size]
     */
    torch::Tensor forward(torch::Tensor x);

    /**
     * @brief Get number of trainable parameters
     * @return Total number of parameters in the model
     */
    int64_t getNumParameters() const;

private:
    int64_t hiddenSize_;    ///< Number of hidden units
    int64_t numLayers_;     ///< Number of LSTM layers
    
    torch::nn::LSTM lstm_{nullptr};          ///< LSTM layer
    torch::nn::Dropout dropout_{nullptr};    ///< Dropout layer
    torch::nn::Linear fc_{nullptr};          ///< Fully connected output layer
};

// Register module with torch
TORCH_MODULE(LSTMModel);

} // namespace lstm_predictor

#endif // LSTMMODEL_H
