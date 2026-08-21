#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>

struct ChronologicalSplit {
    int64_t train_end = 0;
    int64_t validation_end = 0;
    int64_t total = 0;
};

inline ChronologicalSplit makeChronologicalSplit(int64_t total,
                                                 double trainRatio,
                                                 double validationRatio) {
    if (total < 3) {
        throw std::invalid_argument("A train/validation/test split requires at least three samples.");
    }
    const double train = std::clamp(trainRatio, 0.1, 0.9);
    const double validation = std::clamp(validationRatio, 0.01, 0.4);
    if (train + validation >= 0.99) {
        throw std::invalid_argument("Train and validation ratios must leave a non-empty test partition.");
    }
    int64_t trainEnd = std::clamp<int64_t>(static_cast<int64_t>(total * train), 1, total - 2);
    int64_t validationCount = std::max<int64_t>(1, static_cast<int64_t>(total * validation));
    int64_t validationEnd = std::min<int64_t>(total - 1, trainEnd + validationCount);
    return {trainEnd, validationEnd, total};
}
