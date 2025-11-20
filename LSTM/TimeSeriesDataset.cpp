/**
 * @file TimeSeriesDataset.cpp
 * @brief Implementation of TimeSeriesDataset class
 */

#include "TimeSeriesDataset.h"

namespace lstm_predictor {

TimeSeriesDataset::TimeSeriesDataset(torch::Tensor X, torch::Tensor y)
    : X_(X), y_(y) {
    // Validate input shapes
    TORCH_CHECK(X_.size(0) == y_.size(0), 
                "Number of samples in X and y must match");
}

torch::data::Example<> TimeSeriesDataset::get(size_t index) {
    // Return the sequence and target at the given index
    return {X_[index], y_[index]};
}

torch::optional<size_t> TimeSeriesDataset::size() const {
    return X_.size(0);
}

} // namespace lstm_predictor
