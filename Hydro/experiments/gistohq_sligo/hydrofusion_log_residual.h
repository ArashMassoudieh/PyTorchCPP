#pragma once

// Opt-in log-residual / direct-log-Q target formulation and reconstruction
// (CLAUDE_HYDROFUSION_IMPLEMENTATION_TASKS_20261115.md, Task 3), built on
// hydrofusion_feature_builder.h's Sample/ProvenanceRecord types from
// Task 2. Torch-free, same convention as Task 2. Direct-log-Q and
// log-residual modes share the SAME feature builder (hydrofusion_
// feature_builder.h) and would share the same model architecture once
// one exists -- this header changes only the TARGET representation and
// its inverse, never the input features. No model/training code exists
// yet; this is the target/reconstruction math and its round-trip proof
// only.
//
// z(Q)        = log1p(Q / q_ref)
// r_target    = z(Q(t+1)) - z(Q(t))                 (log-residual mode)
// direct      = z(Q(t+1))                           (direct-log-Q mode)
// Qhat(t+1)   = max(0, q_ref * expm1(z(Q(t)) + rhat))   (residual; multiplier fixed at 1, not a parameter)
// Qhat(t+1)   = max(0, q_ref * expm1(zhat))                      (direct)
//
// These are a transparent proposed HydroPINN adaptation, not a claimed
// verbatim equation from the source paper's unavailable supplement
// tables (CLAUDE_HYDROFUSION_IMPLEMENTATION_TASKS_20261115.md Task 3).

#include "hydrofusion_feature_builder.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace hydrofusion {

enum class TargetMode { DirectLogQ, LogResidual };

// Independent-review correction: validates qRef on its own, callable
// before branching on mode, so a negative/nonpositive/nonfinite qRef
// cannot slip past a code path (e.g. reconstructQ's direct-log-Q branch)
// that does not otherwise call zTransform.
inline void validateQRef(double qRef) {
    if (!std::isfinite(qRef) || qRef <= 0.0) {
        throw std::invalid_argument("validateQRef: q_ref must be finite and strictly positive.");
    }
}

// Shared by computeTrainingTarget and reconstructQ so neither can treat
// an unrecognized TargetMode as LogResidual by falling through an
// `else`. Independent-review correction: an earlier version of
// computeTrainingTarget had exactly that fallthrough bug even after
// reconstructQ was fixed to reject it explicitly -- both call sites must
// use the same explicit, exhaustive check.
inline void validateTargetMode(TargetMode mode) {
    if (mode != TargetMode::DirectLogQ && mode != TargetMode::LogResidual) {
        throw std::invalid_argument("validateTargetMode: unknown TargetMode.");
    }
}

// z(Q) = log1p(Q / q_ref). Rejects nonfinite/negative Q and a nonfinite or
// nonpositive q_ref consistently, independent of any validity flag.
// Independent-review correction: finite inputs do not guarantee a finite
// result (Q/qRef can overflow to +inf even when both are finite, e.g.
// Q=DBL_MAX, qRef=DBL_MIN) -- the computed z itself is re-checked before
// being returned, matching the same "derived values can overflow" rule
// already applied to the trailing-window arithmetic in Task 2.
inline double zTransform(double q, double qRef) {
    if (!std::isfinite(q) || q < 0.0) {
        throw std::invalid_argument("zTransform: Q must be finite and nonnegative.");
    }
    validateQRef(qRef);
    const double z = std::log1p(q / qRef);
    if (!std::isfinite(z)) {
        throw std::overflow_error("zTransform: Q/q_ref overflowed before log1p could be applied.");
    }
    return z;
}

// q_ref = median of the positive values among each Train-split, included
// record's own TARGET reading (observed_target_q), each used exactly
// once. Independent-review correction: an earlier version additionally
// pooled each record's origin reading (persistence_prediction), which
// double-counts every interior hourly observation (hour H's reading
// appears once as sample H's origin and once as sample H-1's target) and
// counts only the series' two boundary hours once -- an unintended,
// undocumented weighting, not an equivalent recipe. Using each eligible
// training TARGET observation exactly once removes that bias.
//
// Also per review: an `included` record's masks are its own guarantee
// that observed_target_q is finite and nonnegative (target_native_mask's
// contract in hydrofusion_feature_builder.h) -- a violation of that
// guarantee here is an upstream invariant failure and throws, it is not
// silently treated as "just not positive." A genuine zero reading is a
// legitimate value that is still excluded from a POSITIVE median, not an
// error.
//
// Independent-review correction: an earlier version trusted a record's
// self-reported `split`/`split_assigned` fields alone -- a fabricated or
// corrupted record could claim `split == Train` while its own
// origin_x/target_x do not actually belong there (or are at/past the
// reserved-period seal entirely, as the review's probe demonstrated with
// origin_x=77015, target_x=77016). This version requires an explicit,
// independently-validated `cutoffs` context and re-derives each record's
// partition from its own origin_x/target_x via assignSampleSplit (the
// same function runPersistenceBenchmark itself uses) -- the record's own
// `split` label must agree with that re-derivation, or the record is
// rejected outright rather than trusted. Chronological/preseal domain
// consistency (target_x == origin_x + 1, both within [0, kSealStartX))
// is checked before any of that re-derivation or value access. A
// duplicate target_x across distinct records is also rejected, enforcing
// "each eligible training observation used exactly once."
inline double computeQRefFromTrainingSamples(const std::vector<ProvenanceRecord>& records,
                                               const SplitCutoffs& cutoffs) {
    validateSplitCutoffs(cutoffs);
    std::vector<double> positive;
    std::set<int64_t> seenTargetX;
    for (const auto& r : records) {
        if (!r.included) continue;
        if (!r.split_assigned) {
            throw std::invalid_argument(
                "computeQRefFromTrainingSamples: an included record must have split_assigned true.");
        }
        if (!r.predictor_availability_mask || !r.target_native_mask) {
            throw std::invalid_argument(
                "computeQRefFromTrainingSamples: an included record must have both masks true.");
        }
        if (r.origin_x < 0 || r.origin_x >= kSealStartX || r.target_x < 0 ||
            r.target_x >= kSealStartX || r.target_x != r.origin_x + 1) {
            throw std::invalid_argument(
                "computeQRefFromTrainingSamples: included record's origin_x/target_x fail "
                "chronological/preseal consistency.");
        }
        const auto actualSplit = assignSampleSplit(r.origin_x, r.target_x, cutoffs);
        const bool actuallyTrain = actualSplit.has_value() && *actualSplit == SplitPartition::Train;
        if (!actualSplit.has_value() || *actualSplit == SplitPartition::BeyondTestEnd ||
            r.split != *actualSplit) {
            throw std::invalid_argument(
                "computeQRefFromTrainingSamples: record's split label disagrees with the "
                "independently re-derived partition for its own origin_x/target_x.");
        }
        if (!actuallyTrain) continue;  // genuinely not a Train-split record; skip quietly

        if (!seenTargetX.insert(r.target_x).second) {
            throw std::invalid_argument(
                "computeQRefFromTrainingSamples: duplicate target_x among eligible training "
                "records -- each observation must be used exactly once.");
        }
        const double targetQ = r.observed_target_q;
        if (!std::isfinite(targetQ) || targetQ < 0.0) {
            throw std::invalid_argument(
                "computeQRefFromTrainingSamples: included record's observed_target_q violates "
                "target_native_mask's own finite/nonnegative guarantee.");
        }
        if (targetQ > 0.0) {
            positive.push_back(targetQ);
        }
    }
    if (positive.empty()) {
        throw std::runtime_error(
            "computeQRefFromTrainingSamples: no positive discharge value exists among the "
            "training-eligible target observations; q_ref is undefined.");
    }
    std::sort(positive.begin(), positive.end());
    const std::size_t n = positive.size();
    double median;
    if (n % 2 == 1) {
        median = positive[n / 2];
    } else {
        const double a = positive[n / 2 - 1];
        const double b = positive[n / 2];
        // Overflow-safe midpoint: a + 0.5*(b-a), not 0.5*(a+b), which can
        // overflow to +inf when a and b are both large finite values
        // (e.g. a == b == DBL_MAX, where a+b overflows but b-a == 0).
        median = a + 0.5 * (b - a);
    }
    if (!std::isfinite(median) || median <= 0.0) {
        throw std::overflow_error(
            "computeQRefFromTrainingSamples: median computation produced a non-finite or "
            "nonpositive q_ref.");
    }
    return median;
}

// Builds the training-target scalar for one sample, in the chosen mode's
// space. Both modes read the exact same (originQ, targetQ, qRef) inputs;
// only the returned representation differs. targetQ is read here exactly
// once, as a label -- never as a feature, threshold, origin flag, or
// model input, matching the contract's rule for future Q.
inline double computeTrainingTarget(TargetMode mode, double originQ, double targetQ, double qRef) {
    validateTargetMode(mode);
    const double zTarget = zTransform(targetQ, qRef);
    if (mode == TargetMode::DirectLogQ) {
        return zTarget;
    }
    const double zOrigin = zTransform(originQ, qRef);
    return zTarget - zOrigin;
}

// Reconstructs Qhat from a prediction in the chosen mode's space.
// Independent-review correction: the residual multiplier is fixed at
// exactly 1 for this frozen prototype -- not a caller-adjustable
// parameter (an earlier version exposed `alpha` as an argument, which a
// caller could set to any finite value, defeating the "fixed, not a
// tuning knob" requirement). There is no sweep/grid utility anywhere in
// this file. qRef is validated before branching on mode, so a
// negative/nonpositive/nonfinite qRef cannot slip past the direct-log-Q
// branch, which does not otherwise call zTransform. expm1 overflow and
// the final nonnegative clip are both explicit, not assumed from the
// max(0, ...) formula alone (max(0, +inf) is still +inf, not a safe
// value).
inline double reconstructQ(TargetMode mode, double originQ, double predicted, double qRef) {
    validateTargetMode(mode);
    validateQRef(qRef);
    if (!std::isfinite(predicted)) {
        throw std::invalid_argument("reconstructQ: predicted value must be finite.");
    }
    double zHat;
    if (mode == TargetMode::DirectLogQ) {
        zHat = predicted;
    } else {
        const double zOrigin = zTransform(originQ, qRef);
        zHat = zOrigin + predicted;  // multiplier fixed at 1, not a parameter
    }
    if (!std::isfinite(zHat)) {
        throw std::invalid_argument("reconstructQ: computed z-space value is not finite.");
    }
    const double expanded = std::expm1(zHat);
    if (!std::isfinite(expanded)) {
        throw std::overflow_error("reconstructQ: expm1 overflowed for the given z-space value.");
    }
    const double reconstructed = qRef * expanded;
    if (!std::isfinite(reconstructed)) {
        throw std::overflow_error("reconstructQ: reconstruction overflowed.");
    }
    return std::max(0.0, reconstructed);
}

// Minimal, self-contained export/reload for q_ref. No model/checkpoint
// exists yet in this prototype, so this proves the actual export/reload
// round-trip the task requires for a single scalar, rather than pulling
// in a JSON library dependency for one field or merely asserting the
// round-trip informally.
inline std::string serializeQRef(double qRef) {
    validateQRef(qRef);
    std::ostringstream oss;
    oss.precision(17);  // sufficient for an exact IEEE-754 double round-trip
    oss << "q_ref=" << qRef << "\n";
    return oss.str();
}

// Independent-review correction: an earlier version used std::stod's
// prefix-parsing behavior directly on the remainder of the string, which
// silently accepted trailing garbage ("q_ref=2.0junk\n" parsed as 2.0).
// This version requires the record to be exactly one newline-terminated
// line, with the numeric token between the prefix and the newline fully
// consumed by std::stod (no unconsumed trailing characters) -- rejecting
// trailing junk and any additional record/line. NaN/infinity/nonpositive
// results are still rejected by the existing finite/positive check.
inline double deserializeQRef(const std::string& serialized) {
    const std::string prefix = "q_ref=";
    if (serialized.rfind(prefix, 0) != 0) {
        throw std::invalid_argument("deserializeQRef: malformed serialized q_ref (missing prefix).");
    }
    const std::size_t newlinePos = serialized.find('\n');
    if (newlinePos == std::string::npos || newlinePos != serialized.size() - 1) {
        throw std::invalid_argument(
            "deserializeQRef: expected exactly one newline-terminated record, nothing more.");
    }
    const std::string numericToken = serialized.substr(prefix.size(), newlinePos - prefix.size());
    if (numericToken.empty()) {
        throw std::invalid_argument("deserializeQRef: missing numeric value.");
    }
    double qRef;
    std::size_t consumed = 0;
    try {
        qRef = std::stod(numericToken, &consumed);
    } catch (const std::exception&) {
        throw std::invalid_argument("deserializeQRef: could not parse q_ref value.");
    }
    if (consumed != numericToken.size()) {
        throw std::invalid_argument("deserializeQRef: trailing characters after the numeric value.");
    }
    if (!std::isfinite(qRef) || qRef <= 0.0) {
        throw std::invalid_argument("deserializeQRef: deserialized q_ref must be finite and positive.");
    }
    return qRef;
}

}  // namespace hydrofusion
