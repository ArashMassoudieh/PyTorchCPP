#pragma once

#include "resolve_test_end_x.h"

#include <torch/torch.h>

#include <cstring>

// Thin torch::Tensor convenience wrapper around
// resolveTestEndSequenceCountFromSorted (resolve_test_end_x.h) - kept in a
// separate header so the pure lookup logic stays unit-testable without a
// torch dependency (see scientific_core_test.cpp's resolveTestEndSequenceCount
// assertions). See resolve_test_end_x.h's doc comment for why this lookup
// exists and must be used by every wrapper, windowed or not.
inline int64_t resolveTestEndSequenceCount(const torch::Tensor& sequenceEndpointX,
                                           int64_t testEndXCoordinate) {
    if (testEndXCoordinate < 0) return -1;
    const auto flat = sequenceEndpointX.reshape({-1}).to(torch::kDouble).contiguous();
    std::vector<double> values(static_cast<std::size_t>(flat.size(0)));
    std::memcpy(values.data(), flat.data_ptr<double>(), values.size() * sizeof(double));
    return resolveTestEndSequenceCountFromSorted(values, testEndXCoordinate);
}
