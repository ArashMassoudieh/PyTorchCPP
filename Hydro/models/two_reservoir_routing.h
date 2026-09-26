#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

// Shared exact forward-simulation of the fast/slow two-reservoir routing used
// by the standalone PINN (pinn_wrapper.cpp) and the residual-correction
// hybrids (residual_pinn_wrapper.h), so both stay exactly consistent instead
// of maintaining two copies of the same recurrence.
struct TwoReservoirRoutingParams {
    double fastK = 1.0;
    double slowK = 0.1;
    double alpha = 0.5;
    double runoffCoefficient = 1.0;
    double dt = 1.0;
    std::int64_t lagSteps = 0;
    // K scales with (flow/initial_flow)^flowExponent instead of being fixed;
    // 0.0 recovers plain constant-K routing. See pinn_wrapper.cpp for the
    // physical motivation (kinematic-wave / differentiable Muskingum-Cunge).
    double flowExponent = 0.0;
};

inline std::vector<double> simulateTwoReservoirBaseline(
    const std::vector<double>& peff, double q0, const TwoReservoirRoutingParams& p) {
    const auto n = static_cast<std::int64_t>(peff.size());
    std::vector<double> qFast(static_cast<std::size_t>(n));
    std::vector<double> qSlow(static_cast<std::size_t>(n));
    std::vector<double> predicted(static_cast<std::size_t>(n));
    if (n == 0) return predicted;
    qFast[0] = p.alpha * q0;
    qSlow[0] = (1.0 - p.alpha) * q0;
    predicted[0] = q0;
    const double flowReference = std::max(q0, 1.0e-3);
    const double flowFloor = 1.0e-3;
    for (std::int64_t i = 1; i < n; ++i) {
        const std::int64_t forcingIdx = std::max<std::int64_t>(0, i - p.lagSteps);
        const double forcing = p.runoffCoefficient * peff[static_cast<std::size_t>(forcingIdx)];
        const auto prev = static_cast<std::size_t>(i - 1);
        const auto cur = static_cast<std::size_t>(i);
        double nonlinearFactor = 1.0;
        if (p.flowExponent != 0.0) {
            const double qPrevTotal = std::max(qFast[prev] + qSlow[prev], 0.0) + flowFloor;
            nonlinearFactor = std::pow(qPrevTotal / flowReference, p.flowExponent);
            nonlinearFactor = std::min(std::max(nonlinearFactor, 0.05), 20.0);
        }
        const double effectiveFastK = std::min(p.fastK * nonlinearFactor, 0.99 / p.dt);
        const double effectiveSlowK = std::min(p.slowK * nonlinearFactor, 0.99 / p.dt);
        qFast[cur] = qFast[prev] + p.dt * effectiveFastK * (p.alpha * forcing - qFast[prev]);
        qSlow[cur] = qSlow[prev] + p.dt * effectiveSlowK * ((1.0 - p.alpha) * forcing - qSlow[prev]);
        predicted[cur] = qFast[cur] + qSlow[cur];
    }
    return predicted;
}
