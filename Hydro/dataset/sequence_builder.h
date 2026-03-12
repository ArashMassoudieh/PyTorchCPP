#pragma once

#include <cstddef>

/**
 * @file sequence_builder.h
 * @brief Sequence-sizing helper for sequential model datasets.
 */

/**
 * @brief Validates sample sufficiency for sequence windows.
 */
class SequenceBuilder {
public:
    /**
     * @brief Check whether enough samples exist for the requested sequence length.
     * @param sampleCount Number of available samples.
     * @param sequenceLength Required sequence/window length.
     * @return True when sampleCount is at least sequenceLength.
     */
    bool hasEnoughSamples(std::size_t sampleCount, std::size_t sequenceLength) const;
};
