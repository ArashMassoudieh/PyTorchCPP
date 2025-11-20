/**
 * @file LSTMModel.cpp
 * @brief Implementation of LSTM neural network model
 */

#include "LSTMModel.h"

namespace lstm_predictor {

LSTMModelImpl::LSTMModelImpl(int64_t inputSize, int64_t hiddenSize, 
                             int64_t numLayers, int64_t outputSize, double dropout)
    : hiddenSize_(hiddenSize), numLayers_(numLayers) {
    
    // Configure LSTM options
    torch::nn::LSTMOptions lstmOptions(inputSize, hiddenSize);
    lstmOptions.num_layers(numLayers)
               .batch_first(true);  // Input shape: [batch, seq, feature]
    
    // Add dropout between LSTM layers if more than 1 layer
    if (numLayers > 1) {
        lstmOptions.dropout(dropout);
    }
    
    // Create LSTM layer
    lstm_ = register_module("lstm", torch::nn::LSTM(lstmOptions));
    
    // Create dropout layer
    dropout_ = register_module("dropout", torch::nn::Dropout(dropout));
    
    // Create fully connected output layer
    fc_ = register_module("fc", torch::nn::Linear(hiddenSize, outputSize));
}

torch::Tensor LSTMModelImpl::forward(torch::Tensor x) {
    // Input shape: [batch_size, seq_length, input_size]
    
    // Initialize hidden state and cell state with zeros
    auto batchSize = x.size(0);
    auto h0 = torch::zeros({numLayers_, batchSize, hiddenSize_}).to(x.device());
    auto c0 = torch::zeros({numLayers_, batchSize, hiddenSize_}).to(x.device());
    
    // Forward propagate LSTM
    // output shape: [batch_size, seq_length, hidden_size]
    auto lstmOut = lstm_->forward(x, std::make_tuple(h0, c0));
    auto output = std::get<0>(lstmOut);
    
    // Take the output from the last time step
    // Shape: [batch_size, hidden_size]
    output = output.select(1, -1);
    
    // Apply dropout
    output = dropout_->forward(output);
    
    // Apply fully connected layer
    // Shape: [batch_size, output_size]
    output = fc_->forward(output);
    
    return output;
}

int64_t LSTMModelImpl::getNumParameters() const {
    int64_t numParams = 0;
    for (const auto& param : parameters()) {
        numParams += param.numel();
    }
    return numParams;
}

} // namespace lstm_predictor
