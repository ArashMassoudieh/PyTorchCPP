#pragma once

#include <torch/torch.h>
#include <cmath>
#include <stdexcept>

// Interval-end discharge for piecewise-constant inflow. Initial storage is zero.
// The affine parallel scan preserves gradients without a serial time loop.
inline torch::Tensor routeReservoirTensor(const torch::Tensor& runoff,
                                          double rateTimesDt, double fraction,
                                          bool exponential = true) {
    if (!std::isfinite(rateTimesDt) || rateTimesDt < 0.0 ||
        !std::isfinite(fraction) || fraction < 0.0 || fraction > 1.0) {
        throw std::invalid_argument("Reservoir routing requires finite nonnegative k*dt and fraction in [0,1].");
    }
    const double gain = exponential ? -std::expm1(-rateTimesDt) : rateTimesDt;
    const double decay = exponential ? std::exp(-rateTimesDt) : 1.0 - rateTimesDt;
    const int64_t n = runoff.size(0);
    torch::Tensor a = torch::full_like(runoff, decay);
    torch::Tensor b = (gain * fraction) * runoff;
    for (int64_t offset = 1; offset < n; offset *= 2) {
        torch::Tensor aShift = torch::ones_like(a);
        torch::Tensor bShift = torch::zeros_like(b);
        aShift.slice(0, offset, n) = a.slice(0, 0, n - offset);
        bShift.slice(0, offset, n) = b.slice(0, 0, n - offset);
        b = a * bShift + b;
        a = a * aShift;
    }
    return b;
}
