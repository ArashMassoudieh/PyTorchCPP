#pragma once

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

// Pure, torch-free core of the regional-pretraining pooled-batch sampler.
// Produces one epoch's worth of (basinIndex, withinBasinSequenceIndex)
// pairs, drawing exactly `perBasinCount` sequences from every basin
// regardless of how many sequences that basin actually has available, then
// shuffling the combined list across basins.
//
// Why this exists: pooling basins' sequences naively (concatenate then
// shuffle) lets a long-record basin dominate gradient updates in proportion
// to its own length, rather than contributing roughly equally the way
// multi-basin LSTM literature (Kratzert et al. 2019) intends. Each basin
// contributing the SAME count per epoch removes that length-dependence.
//
// Sampling per basin: a basin with >= perBasinCount sequences is sampled
// WITHOUT replacement (a fresh shuffle of its own index range, truncated to
// perBasinCount) - no sequence repeats within the same epoch. A basin with
// FEWER sequences than perBasinCount is cycled WITH repetition (full passes
// through its own shuffled index range, then a final partial pass) so every
// basin still contributes exactly perBasinCount regardless of its own
// record length, instead of silently contributing less.
//
// Determinism: entirely driven by the caller-supplied RNG (seed it once per
// epoch, e.g. from HydroRunConfig::random_seed combined with the epoch
// number, for a reproducible but epoch-varying shuffle) - no hidden global
// random state.
inline std::vector<std::pair<int64_t, int64_t>> balancedShuffledEpochIndices(
    const std::vector<int64_t>& basinSizes, int64_t perBasinCount, std::mt19937_64& rng) {
    if (basinSizes.empty()) throw std::invalid_argument("balancedShuffledEpochIndices requires at least one basin.");
    if (perBasinCount <= 0) throw std::invalid_argument("perBasinCount must be positive.");
    for (const int64_t size : basinSizes) {
        if (size <= 0) throw std::invalid_argument("Every basin must have at least one available sequence.");
    }

    std::vector<std::pair<int64_t, int64_t>> combined;
    combined.reserve(basinSizes.size() * static_cast<std::size_t>(perBasinCount));

    for (std::size_t basinIndex = 0; basinIndex < basinSizes.size(); ++basinIndex) {
        const int64_t size = basinSizes[basinIndex];
        std::vector<int64_t> order(static_cast<std::size_t>(size));
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), rng);

        if (size >= perBasinCount) {
            for (int64_t i = 0; i < perBasinCount; ++i) {
                combined.emplace_back(static_cast<int64_t>(basinIndex), order[static_cast<std::size_t>(i)]);
            }
        } else {
            // Cycle through full shuffled passes, reshuffling between passes
            // so repeats are not always in the same relative order, then a
            // final partial pass for the remainder.
            int64_t remaining = perBasinCount;
            while (remaining > 0) {
                const int64_t take = std::min(remaining, size);
                for (int64_t i = 0; i < take; ++i) {
                    combined.emplace_back(static_cast<int64_t>(basinIndex), order[static_cast<std::size_t>(i)]);
                }
                remaining -= take;
                if (remaining > 0) std::shuffle(order.begin(), order.end(), rng);
            }
        }
    }

    std::shuffle(combined.begin(), combined.end(), rng);
    return combined;
}
