#pragma once

#include "../models/hydro_run_types.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

inline void populateHydroMetrics(HydroRunResult& result,
                                 const std::vector<double>& observed,
                                 const std::vector<double>& predicted) {
    if (observed.empty()) throw std::invalid_argument("Hydro metrics require at least one observation.");
    if (observed.size() != predicted.size()) {
        throw std::invalid_argument("Hydro metric vectors must have matching lengths.");
    }
    const size_t n = observed.size();
    double mean = 0.0;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(observed[i]) || !std::isfinite(predicted[i])) {
            throw std::invalid_argument("Hydro metrics require finite observed and predicted values.");
        }
        mean += observed[i];
    }
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
    // ML coefficient of determination on the same held-out series. With this
    // standard 1-SSE/SST definition it is numerically identical to NSE; both
    // names are retained because they serve different ML/hydrology audiences.
    result.r2 = result.nse;
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
    if (result.x.empty()) throw std::invalid_argument("Hydro peak metrics require at least one sample.");
    if (result.x.size() != result.y_true.size() || result.x.size() != result.y_pred.size() ||
        result.x.size() != result.split.size()) {
        throw std::invalid_argument("Hydro peak metric series and split labels must have matching lengths.");
    }
    const size_t n = result.x.size();
    std::vector<size_t> testIndices;
    testIndices.reserve(n);
    size_t observedPeak = n;
    size_t predictedPeak = n;
    for (size_t i = 0; i < n; ++i) {
        if (result.split[i] != "test") continue;
        if (!std::isfinite(result.x[i]) || !std::isfinite(result.y_true[i]) || !std::isfinite(result.y_pred[i])) {
            throw std::invalid_argument("Hydro peak metrics require finite test samples.");
        }
        testIndices.push_back(i);
        if (observedPeak == n || result.y_true[i] > result.y_true[observedPeak]) observedPeak = i;
        if (predictedPeak == n || result.y_pred[i] > result.y_pred[predictedPeak]) predictedPeak = i;
    }
    if (observedPeak == n || predictedPeak == n) {
        throw std::invalid_argument("Hydro peak metrics require at least one test sample.");
    }
    result.peak_timing_error = result.x[predictedPeak] - result.x[observedPeak];
    const double observedMagnitude = result.y_true[observedPeak];
    if (std::abs(observedMagnitude) > 0.0) {
        result.peak_magnitude_error_percent =
            100.0 * (result.y_pred[predictedPeak] - observedMagnitude) / std::abs(observedMagnitude);
    }
    std::sort(testIndices.begin(), testIndices.end(), [&](const size_t left, const size_t right) {
        return result.y_true[left] < result.y_true[right];
    });
    const size_t tailCount = std::max<size_t>(1, static_cast<size_t>(std::ceil(testIndices.size() * 0.1)));
    double lowSquaredError = 0.0;
    double highSquaredError = 0.0;
    for (size_t rank = 0; rank < tailCount; ++rank) {
        const size_t low = testIndices[rank];
        const size_t high = testIndices[testIndices.size() - 1 - rank];
        const double lowError = result.y_pred[low] - result.y_true[low];
        const double highError = result.y_pred[high] - result.y_true[high];
        lowSquaredError += lowError * lowError;
        highSquaredError += highError * highError;
    }
    result.low_flow_rmse = std::sqrt(lowSquaredError / static_cast<double>(tailCount));
    result.high_flow_rmse = std::sqrt(highSquaredError / static_cast<double>(tailCount));
}

inline void populateHydroPhysicsResidualMetrics(HydroRunResult& result) {
    if (result.physics_residual.empty()) {
        throw std::invalid_argument("Physics residual metrics require at least one residual.");
    }
    if (!result.x.empty() && result.x.size() != result.physics_residual.size()) {
        throw std::invalid_argument("Physics residuals and timestamps must have matching lengths.");
    }
    if (!result.x.empty()) {
        for (size_t i = 0; i < result.x.size(); ++i) {
            if (!std::isfinite(result.x[i]) || (i > 0 && result.x[i] <= result.x[i - 1])) {
                throw std::invalid_argument("Physics residual timestamps must be finite and strictly increasing.");
            }
        }
    }
    double sum = 0.0;
    double squared = 0.0;
    double cumulative = 0.0;
    size_t count = 0;
    for (size_t i = 0; i < result.physics_residual.size(); ++i) {
        const double residual = result.physics_residual[i];
        if (!std::isfinite(residual)) continue;
        sum += residual;
        squared += residual * residual;
        ++count;
        if (i > 0 && !result.x.empty()) {
            cumulative += residual * (result.x[i] - result.x[i - 1]);
        } else if (result.x.empty()) {
            cumulative += residual;
        }
    }
    if (count == 0) return;
    result.physics_residual_mean = sum / static_cast<double>(count);
    result.physics_residual_rmse = std::sqrt(squared / static_cast<double>(count));
    result.cumulative_physics_residual = cumulative;
}
