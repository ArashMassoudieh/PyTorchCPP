#pragma once
#include "chronological_split.h"
#include "resolve_test_end_x_tensor.h"
#include "../models/hydro_run_types.h"

inline ChronologicalSplit configuredChronologicalSplit(
        const torch::Tensor& endpoints, const HydroRunConfig& config) {
    const auto total = endpoints.numel();
    const auto testEnd = resolveTestEndSequenceCount(endpoints, config.test_end_x);
    const bool exactTrain = config.train_end_x >= 0;
    const bool exactValidation = config.validation_end_x >= 0;
    if (exactTrain != exactValidation) {
        throw std::invalid_argument("Set both train_end_x and validation_end_x for an exact temporal split.");
    }
    if (exactTrain) {
        if (!(config.train_end_x < config.validation_end_x &&
              config.validation_end_x < config.test_end_x)) {
            throw std::invalid_argument("Exact temporal split requires train_end_x < validation_end_x < test_end_x.");
        }
        return makeExactChronologicalSplit(total,
            resolveTestEndSequenceCount(endpoints, config.train_end_x),
            resolveTestEndSequenceCount(endpoints, config.validation_end_x), testEnd);
    }
    return makeChronologicalSplit(total, config.train_split_ratio,
        config.validation_split_ratio, config.test_end_split_ratio, testEnd);
}
