#pragma once

#include "../models/hydro_run_types.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

inline void populateHydroMetrics(HydroRunResult& result,
                                 const std::vector<double>& observed,
                                 const std::vector<double>& predicted) {
    const size_t n = std::min(observed.size(), predicted.size());
    if (n == 0) return;
    double mean = 0.0;
    for (size_t i = 0; i < n; ++i) mean += observed[i];
    mean /= static_cast<double>(n);
    double predictedMean = 0.0;
    for (size_t i = 0; i < n; ++i) predictedMean += predicted[i];
    predictedMean /= static_cast<double>(n);
    double squared = 0.0, absolute = 0.0, denominator = 0.0, signedError = 0.0;
    double predictedVariance = 0.0, covariance = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double error = predicted[i] - observed[i];
        squared += error * error;
        absolute += std::abs(error);
        signedError += error;
        const double centered = observed[i] - mean;
        denominator += centered * centered;
        const double predictedCentered = predicted[i] - predictedMean;
        predictedVariance += predictedCentered * predictedCentered;
        covariance += centered * predictedCentered;
    }
    result.mse = squared / static_cast<double>(n);
    result.rmse = std::sqrt(result.mse);
    result.mae = absolute / static_cast<double>(n);
    result.nse = denominator > 0.0 ? 1.0 - squared / denominator
                                   : std::numeric_limits<double>::quiet_NaN();
    double observedSum = 0.0;
    for (size_t i = 0; i < n; ++i) observedSum += observed[i];
    result.pbias = std::abs(observedSum) > 0.0 ? 100.0 * signedError / observedSum
                                               : std::numeric_limits<double>::quiet_NaN();
    result.volume_error_percent = result.pbias;
    result.correlation = denominator > 0.0 && predictedVariance > 0.0
                             ? covariance / std::sqrt(denominator * predictedVariance)
                             : std::numeric_limits<double>::quiet_NaN();
    const double observedStd = n > 1 ? std::sqrt(denominator / static_cast<double>(n - 1)) : 0.0;
    const double predictedStd = n > 1 ? std::sqrt(predictedVariance / static_cast<double>(n - 1)) : 0.0;
    if (std::isfinite(result.correlation) && std::abs(mean) > 0.0 && observedStd > 0.0) {
        const double alpha = predictedStd / observedStd;
        const double beta = predictedMean / mean;
        result.kge = 1.0 - std::sqrt((result.correlation - 1.0) * (result.correlation - 1.0) +
                                     (alpha - 1.0) * (alpha - 1.0) +
                                     (beta - 1.0) * (beta - 1.0));
    } else {
        result.kge = std::numeric_limits<double>::quiet_NaN();
    }
}

inline bool hydroMetricsAreFinite(const HydroRunResult& result) {
    return std::isfinite(result.mse) && std::isfinite(result.rmse) &&
           std::isfinite(result.mae);
}

inline void populateHydroPeakMetrics(HydroRunResult& result) {
    const size_t n = std::min(result.x.size(), std::min(result.y_true.size(), result.y_pred.size()));
    if (n == 0) return;
    size_t observedPeak = n;
    size_t predictedPeak = n;
    for (size_t i = 0; i < n; ++i) {
        if (i < result.split.size() && result.split[i] != "test") continue;
        if (observedPeak == n || result.y_true[i] > result.y_true[observedPeak]) observedPeak = i;
        if (predictedPeak == n || result.y_pred[i] > result.y_pred[predictedPeak]) predictedPeak = i;
    }
    if (observedPeak == n || predictedPeak == n) return;
    result.peak_timing_error = result.x[predictedPeak] - result.x[observedPeak];
    const double observedMagnitude = result.y_true[observedPeak];
    if (std::abs(observedMagnitude) > 0.0) {
        result.peak_magnitude_error_percent =
            100.0 * (result.y_pred[predictedPeak] - observedMagnitude) / std::abs(observedMagnitude);
    }
}
