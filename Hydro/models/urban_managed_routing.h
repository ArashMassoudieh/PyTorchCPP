#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

// Reduced conceptual routing for a regulated urban basin.  It is deliberately
// small enough to calibrate inside a train/validation split, while representing
// three structures omitted by the legacy two-reservoir baseline:
//   (1) connected-impervious quickflow,
//   (2) pervious/groundwater recession, and
//   (3) a managed/detention branch followed by downstream channel routing.
// Fractions describe spatial/input partitioning and therefore sum through the
// branches without creating or deleting water.  The rates are per-hour when
// dtHours is in hours.
struct UrbanManagedRoutingParams {
    double quickK = 1.0;
    double slowK = 0.04;
    double detentionK = 0.08;
    double channelK = 1.0;
    double quickFraction = 0.26;
    double detentionCatchmentFraction = 0.183;
    double runoffCoefficient = 1.0;
    // Opt-in intensity-dependent redistribution; total routed input is unchanged.
    double intensityQuickStrength = 0.0;
    double intensityReferenceMmH = 1.0;
    double dtHours = 1.0;
    std::int64_t lagSteps = 0;
};

inline std::vector<double> simulateUrbanManagedBaseline(
    const std::vector<double>& effectivePrecipitation,
    double q0,
    const UrbanManagedRoutingParams& p) {
    const auto positiveFinite = [](double x) { return std::isfinite(x) && x > 0.0; };
    if (!positiveFinite(p.quickK) || !positiveFinite(p.slowK) ||
        !positiveFinite(p.detentionK) || !positiveFinite(p.channelK) ||
        !positiveFinite(p.dtHours) ||
        !std::isfinite(p.quickFraction) || p.quickFraction < 0.0 || p.quickFraction > 1.0 ||
        !std::isfinite(p.detentionCatchmentFraction) ||
        p.detentionCatchmentFraction < 0.0 || p.detentionCatchmentFraction > 1.0 ||
        !std::isfinite(p.runoffCoefficient) || p.runoffCoefficient < 0.0 ||
        !std::isfinite(p.intensityQuickStrength) || p.intensityQuickStrength < 0.0 || p.intensityQuickStrength > 1.0 ||
        !positiveFinite(p.intensityReferenceMmH) ||
        !std::isfinite(q0) || q0 < 0.0 || p.lagSteps < 0) {
        throw std::invalid_argument("Invalid urban managed-routing parameters.");
    }

    const std::size_t n = effectivePrecipitation.size();
    std::vector<double> predicted(n, 0.0);
    if (n == 0) return predicted;

    const double freeFraction = 1.0 - p.detentionCatchmentFraction;
    double quickOut = q0 * freeFraction * p.quickFraction;
    double slowOut = q0 * freeFraction * (1.0 - p.quickFraction);
    double detentionOut = q0 * p.detentionCatchmentFraction;
    double channelOut = q0;
    predicted[0] = q0;

    const auto updateLinearReservoir = [&](double previousOut, double targetIn, double k) {
        const double retention = std::exp(-p.dtHours * k);
        return retention * previousOut + (1.0 - retention) * targetIn;
    };

    for (std::size_t i = 1; i < n; ++i) {
        const std::size_t forcingIndex = i > static_cast<std::size_t>(p.lagSteps)
            ? i - static_cast<std::size_t>(p.lagSteps) : 0;
        const double forcing = p.runoffCoefficient *
            std::max(0.0, effectivePrecipitation[forcingIndex]);

        double quickFraction = p.quickFraction;
        if (p.intensityQuickStrength > 0.0) {
            const double rain = effectivePrecipitation[forcingIndex];
            if (!std::isfinite(rain)) {
                throw std::invalid_argument("Nonfinite rainfall in intensity-dependent routing");
            }
            const double nonnegativeRain = std::max(0.0, rain);
            const double saturation = nonnegativeRain / (nonnegativeRain + p.intensityReferenceMmH);
            quickFraction += p.intensityQuickStrength * (1.0 - p.quickFraction) * saturation;
        }
        quickOut = updateLinearReservoir(
            quickOut, freeFraction * quickFraction * forcing, p.quickK);
        slowOut = updateLinearReservoir(
            slowOut, freeFraction * (1.0 - quickFraction) * forcing, p.slowK);
        detentionOut = updateLinearReservoir(
            detentionOut, p.detentionCatchmentFraction * forcing, p.detentionK);

        const double routedIn = quickOut + slowOut + detentionOut;
        channelOut = updateLinearReservoir(channelOut, routedIn, p.channelK);
        predicted[i] = channelOut;
    }
    return predicted;
}

inline std::vector<double> simulateUrbanManagedBaselinePerSegment(
    const std::vector<double>& effectivePrecipitation,
    const std::vector<double>& observed,
    const std::vector<std::int64_t>& segmentId,
    const UrbanManagedRoutingParams& p,
    double initialFlow = -1.0) {
    if (effectivePrecipitation.size() != observed.size() ||
        effectivePrecipitation.size() != segmentId.size()) {
        throw std::invalid_argument("Urban routing inputs must have equal lengths.");
    }
    std::vector<double> predicted(effectivePrecipitation.size(), 0.0);
    std::size_t begin = 0;
    while (begin < effectivePrecipitation.size()) {
        std::size_t end = begin + 1;
        while (end < segmentId.size() && segmentId[end] == segmentId[begin]) ++end;
        const std::vector<double> segmentForcing(
            effectivePrecipitation.begin() + static_cast<std::ptrdiff_t>(begin),
            effectivePrecipitation.begin() + static_cast<std::ptrdiff_t>(end));
        const double segmentQ0 = initialFlow >= 0.0 ? initialFlow : observed[begin];
        const auto segmentPrediction = simulateUrbanManagedBaseline(segmentForcing, segmentQ0, p);
        std::copy(segmentPrediction.begin(), segmentPrediction.end(),
                  predicted.begin() + static_cast<std::ptrdiff_t>(begin));
        begin = end;
    }
    return predicted;
}
