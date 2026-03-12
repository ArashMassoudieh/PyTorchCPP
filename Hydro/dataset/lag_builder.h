#pragma once

#include <vector>

/**
 * @file lag_builder.h
 * @brief Utilities for creating lag-index configurations.
 */

/**
 * @brief Constructs lag definitions for time-series feature expansion.
 */
class LagBuilder {
public:
    /**
     * @brief Build a default lag layout per input series.
     * @param seriesCount Number of time series.
     * @return Vector of lag lists for each series.
     */
    std::vector<std::vector<int>> buildDefaultLags(int seriesCount) const;
};
