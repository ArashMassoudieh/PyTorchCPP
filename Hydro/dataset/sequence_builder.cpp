#include "sequence_builder.h"

bool SequenceBuilder::hasEnoughSamples(std::size_t sampleCount, std::size_t sequenceLength) const {
    return sampleCount >= sequenceLength;
}
