#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include <stdexcept>

// Shared exponential forward-simulation of the fast/slow two-reservoir routing used
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
    if (!std::isfinite(p.dt) || p.dt <= 0.0 ||
        !std::isfinite(p.fastK) || p.fastK <= 0.0 ||
        !std::isfinite(p.slowK) || p.slowK <= 0.0 ||
        !std::isfinite(p.alpha) || p.alpha < 0.0 || p.alpha > 1.0 ||
        !std::isfinite(p.runoffCoefficient) || p.runoffCoefficient < 0.0 ||
        !std::isfinite(p.flowExponent) || !std::isfinite(q0) || q0 < 0.0 || p.lagSteps < 0) {
        throw std::invalid_argument("Invalid two-reservoir routing parameters.");
    }
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
        // Exact for constant rates; freeze state-dependent rates per interval.
        const double effectiveFastK = p.fastK * nonlinearFactor;
        const double effectiveSlowK = p.slowK * nonlinearFactor;
        qFast[cur] = std::exp(-p.dt * effectiveFastK) * qFast[prev] +
                     (-std::expm1(-p.dt * effectiveFastK)) * p.alpha * forcing;
        qSlow[cur] = std::exp(-p.dt * effectiveSlowK) * qSlow[prev] +
                     (-std::expm1(-p.dt * effectiveSlowK)) * (1.0 - p.alpha) * forcing;
        predicted[cur] = qFast[cur] + qSlow[cur];
    }
    return predicted;
}
