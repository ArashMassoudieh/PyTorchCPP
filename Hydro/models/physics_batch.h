#pragma once
#include <algorithm>
#include <cstdint>

// Include the preceding prediction for the boundary transition, while losses
// on observations and nonnegativity count only [dataBegin, dataEnd).
struct PhysicsBatch {
    int64_t dataBegin;
    int64_t dataEnd;
    int64_t contextBegin;
    int64_t offset() const { return dataBegin - contextBegin; }
};
inline PhysicsBatch physicsBatch(int64_t start, int64_t total, int64_t batchSize) {
    return {start, std::min(start + batchSize, total), std::max<int64_t>(0, start - 1)};
}
