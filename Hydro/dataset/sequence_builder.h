#pragma once

#include <cstddef>

class SequenceBuilder {
public:
    bool hasEnoughSamples(std::size_t sampleCount, std::size_t sequenceLength) const;
};
