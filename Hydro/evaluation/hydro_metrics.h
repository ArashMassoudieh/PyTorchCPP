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
    double squared = 0.0, absolute = 0.0, denominator = 0.0, signedError = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double error = predicted[i] - observed[i];
        squared += error * error;
        absolute += std::abs(error);
        signedError += error;
        const double centered = observed[i] - mean;
        denominator += centered * centered;
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
}

inline bool hydroMetricsAreFinite(const HydroRunResult& result) {
    return std::isfinite(result.mse) && std::isfinite(result.rmse) &&
           std::isfinite(result.mae);
}
