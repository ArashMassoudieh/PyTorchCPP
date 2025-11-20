/**
 * @file TimeSeriesDataset.h
 * @brief Custom dataset class for time series data
 * 
 * This class provides a PyTorch-compatible dataset interface for loading
 * and accessing time series sequences.
 * 
 * @author Generated for Time Series Prediction
 * @date 2025
 */

#ifndef TIMESERIESDATASET_H
#define TIMESERIESDATASET_H

#include <torch/torch.h>
#include <vector>

namespace lstm_predictor {

/**
 * @class TimeSeriesDataset
 * @brief Dataset class for time series data compatible with torch::data::Dataset
 * 
 * This class stores sequences of time series data and provides access
 * through the torch dataset interface.
 */
class TimeSeriesDataset : public torch::data::Dataset<TimeSeriesDataset> {
public:
    /**
     * @brief Constructor
     * 
     * @param X Input sequences tensor [num_samples, seq_length, num_features]
     * @param y Target values tensor [num_samples]
     */
    TimeSeriesDataset(torch::Tensor X, torch::Tensor y);

    /**
     * @brief Get a single sample from the dataset
     * 
     * @param index Index of the sample to retrieve
     * @return Example containing input sequence and target value
     */
    torch::data::Example<> get(size_t index) override;

    /**
     * @brief Get the size of the dataset
     * @return Number of samples in the dataset
     */
    torch::optional<size_t> size() const override;

    /**
     * @brief Get all input sequences
     * @return Tensor containing all input sequences
     */
    torch::Tensor getX() const { return X_; }

    /**
     * @brief Get all target values
     * @return Tensor containing all target values
     */
    torch::Tensor getY() const { return y_; }

private:
    torch::Tensor X_;  ///< Input sequences [num_samples, seq_length, num_features]
    torch::Tensor y_;  ///< Target values [num_samples]
};

} // namespace lstm_predictor

#endif // TIMESERIESDATASET_H
