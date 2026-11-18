#pragma once

// Opt-in HydroFusion discharge-memory feature builder and persistence
// benchmark (CLAUDE_HYDROFUSION_IMPLEMENTATION_TASKS_20261115.md, Task 2),
// built against the frozen contract in hydrofusion_forecast_contract.md /
// hydrofusion_forecast_contract_schema.json. Deliberately torch-free so it
// can be built and tested without the research-training build, matching
// the task's "this can be built/tested without training" instruction.
//
// Nothing in this header is wired into the existing six-feature frozen
// model, HydroRunConfig, or any checkpoint format. It is new, standalone,
// and not called from anywhere yet.

#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <vector>

namespace hydrofusion {

// Reserved-period seal, duplicated per this codebase's own existing
// convention (shared_scale_inference.cpp, verify_stageb_checkpoint_
// reproduction_parsing.h each define their own copy; no shared header
// exports one).
inline constexpr int64_t kSealStartX = 77016;

// Safe upper domain bound for x, well below std::numeric_limits<int64_t>::max()
// (~9.223e18) with margin so no double-precision rounding near the exact
// boundary can matter. A finite, integer-valued double can still exceed
// int64_t's representable range -- static_cast<int64_t> of such a value is
// undefined behavior, not merely "a large number" -- so this must be
// checked explicitly before any cast, separately from the finite/integer/
// nonnegative checks already in place.
inline constexpr double kMaxRepresentableX = 9.0e18;

// One raw hourly row, field names matching GisToOhqHourlyRow exactly so
// a caller can populate this directly from the real harmonizer output.
struct HourlyRow {
    double x = 0.0;
    double precipitation_mm_per_hour = 0.0;
    double temperature_c = 0.0;
    double relative_humidity_percent = 0.0;
    double wind_speed_m_per_second = 0.0;
    double solar_energy_mj_per_m2_per_hour = 0.0;
    double pet_mm_per_hour = 0.0;
    double observed_runoff_mm_per_hour = 0.0;
    bool all_forcings_valid = false;
    bool discharge_valid = false;
};

// Twenty-feature allowlist from the frozen contract, section 4. Field
// order matches the schema's input_allowlist arrays exactly.
struct Features {
    double q_lag_0h = 0.0;
    double q_lag_1h = 0.0;
    double q_lag_3h = 0.0;
    double q_lag_6h = 0.0;
    double q_lag_12h = 0.0;
    double q_lag_24h = 0.0;
    double q_trailing_mean_6h = 0.0;
    double q_trailing_mean_24h = 0.0;
    double q_trailing_mean_72h = 0.0;
    double q_trailing_var_6h = 0.0;
    double q_trailing_var_24h = 0.0;
    double q_trailing_var_72h = 0.0;
    double q_last_hourly_change = 0.0;
    double q_past_slope_6h = 0.0;
    double met_precip_mm_per_h = 0.0;
    double met_temperature_c = 0.0;
    double met_relative_humidity_pct = 0.0;
    double met_wind_speed_m_per_s = 0.0;
    double met_solar_mj_per_m2_per_h = 0.0;
    double met_pet_mm_per_h = 0.0;
};

struct Sample {
    int64_t origin_x = 0;
    int64_t target_x = 0;
    bool predictor_availability_mask = false;
    bool target_native_mask = false;
    Features features{};
    double persistence_prediction = 0.0;  // meaningful only if predictor_availability_mask
    double observed_target_q = 0.0;       // meaningful only if target_native_mask
};

enum class SplitPartition { Train, Validation, Test, BeyondTestEnd };

struct SplitCutoffs {
    int64_t train_end_x = 0;
    int64_t validation_end_x = 0;
    int64_t test_end_x = 0;
};

// Chronological-gap-derived index: rejects non-finite/non-integer/negative
// x, duplicate hours, and non-monotonic ordering explicitly (matches
// daily_history_builder.h's buildHourIndex discipline). Segment id
// increments whenever the elapsed-hour gap between consecutive rows
// is not exactly 1 -- never inferred from row position.
//
// x must be >= 0: this program's x coordinate is an hours-since-epoch
// convention (see kSealStartX's own comment elsewhere in this codebase),
// so a negative value has no valid domain meaning here and is rejected
// before any arithmetic is done on it, not merely before the int64 cast.
class HourlyIndex {
public:
    explicit HourlyIndex(const std::vector<HourlyRow>& rows) {
        int64_t previousHour = 0;
        bool first = true;
        int64_t segment = 0;
        for (std::size_t i = 0; i < rows.size(); ++i) {
            const double x = rows[i].x;
            if (!std::isfinite(x) || x != std::floor(x)) {
                throw std::invalid_argument("HourlyIndex: non-finite or non-integer x encountered.");
            }
            if (x < 0.0) {
                throw std::invalid_argument("HourlyIndex: negative x has no valid domain meaning here.");
            }
            if (x > kMaxRepresentableX) {
                throw std::invalid_argument(
                    "HourlyIndex: x exceeds the safe int64-representable domain bound.");
            }
            const int64_t hour = static_cast<int64_t>(x);
            if (hourToRow_.count(hour) != 0) {
                throw std::invalid_argument("HourlyIndex: duplicate hour value encountered.");
            }
            if (!first && hour <= previousHour) {
                throw std::invalid_argument("HourlyIndex: non-monotonic timestamp ordering encountered.");
            }
            if (!first && (hour - previousHour) != 1) {
                ++segment;
            }
            hourToRow_[hour] = i;
            segmentId_.push_back(segment);
            previousHour = hour;
            first = false;
        }
    }

    std::optional<std::size_t> rowForHour(int64_t hour) const {
        auto it = hourToRow_.find(hour);
        if (it == hourToRow_.end()) return std::nullopt;
        return it->second;
    }

    int64_t segmentFor(std::size_t rowIndex) const { return segmentId_.at(rowIndex); }

private:
    std::map<int64_t, std::size_t> hourToRow_;
    std::vector<int64_t> segmentId_;
};

namespace detail {

// Looks up one lagged/trailing discharge hour, requiring it to exist,
// be discharge_valid, share originSegment, AND carry a finite,
// nonnegative value. The discharge_valid flag alone is not trusted as a
// numeric guarantee -- a source that marks a row valid while its value is
// NaN/Inf (or physically impossible, negative discharge) must still void
// the group; flags and numeric domain checks are independent safeguards.
// Returns std::nullopt on any failure -- the caller treats that as voiding
// the whole group (no substitution with an older value, no zero padding).
inline std::optional<double> readDischargeHour(const HourlyIndex& idx, const std::vector<HourlyRow>& rows,
                                                int64_t hour, int64_t originSegment) {
    auto rowIdx = idx.rowForHour(hour);
    if (!rowIdx.has_value()) return std::nullopt;
    if (idx.segmentFor(*rowIdx) != originSegment) return std::nullopt;
    const HourlyRow& row = rows[*rowIdx];
    if (!row.discharge_valid) return std::nullopt;
    const double q = row.observed_runoff_mm_per_hour;
    if (!std::isfinite(q) || q < 0.0) return std::nullopt;
    return q;
}

// Met-origin values are only trusted when every one of the six fields is
// finite; fields with a physically nonnegative domain (precipitation,
// wind speed, solar energy, PET, relative humidity) are additionally
// rejected if negative. Temperature has no such domain constraint.
inline bool metRowIsNumericallySound(const HourlyRow& row) {
    if (!std::isfinite(row.precipitation_mm_per_hour) || row.precipitation_mm_per_hour < 0.0) return false;
    if (!std::isfinite(row.temperature_c)) return false;
    if (!std::isfinite(row.relative_humidity_percent) || row.relative_humidity_percent < 0.0) return false;
    if (!std::isfinite(row.wind_speed_m_per_second) || row.wind_speed_m_per_second < 0.0) return false;
    if (!std::isfinite(row.solar_energy_mj_per_m2_per_hour) || row.solar_energy_mj_per_m2_per_hour < 0.0)
        return false;
    if (!std::isfinite(row.pet_mm_per_hour) || row.pet_mm_per_hour < 0.0) return false;
    return true;
}

// Complete-window trailing mean/variance over [originHour - windowHours + 1, originHour].
// Every hour in the window must resolve via readDischargeHour or the whole
// window is void (std::nullopt for both outputs). Every individual value
// readDischargeHour returns is already finite, but the SUM/MEAN/VARIANCE
// arithmetic over a window of large-but-finite values can still overflow
// to +-inf (e.g. every value near DBL_MAX) -- finite inputs do not
// guarantee finite derived results, so the derived mean and variance are
// explicitly re-checked for finiteness before being trusted.
inline std::optional<std::pair<double, double>> trailingMeanAndVariance(const HourlyIndex& idx,
                                                                          const std::vector<HourlyRow>& rows,
                                                                          int64_t originHour, int64_t originSegment,
                                                                          int windowHours) {
    std::vector<double> values;
    values.reserve(static_cast<std::size_t>(windowHours));
    for (int k = windowHours - 1; k >= 0; --k) {
        auto v = readDischargeHour(idx, rows, originHour - k, originSegment);
        if (!v.has_value()) return std::nullopt;
        values.push_back(*v);
    }
    double mean = 0.0;
    for (double v : values) mean += v;
    mean /= static_cast<double>(values.size());
    double variance = 0.0;
    for (double v : values) variance += (v - mean) * (v - mean);
    variance /= static_cast<double>(values.size());
    if (!std::isfinite(mean) || !std::isfinite(variance)) return std::nullopt;
    return std::make_pair(mean, variance);
}

}  // namespace detail

// Builds one HydroFusion sample for the given origin hour, or returns
// std::nullopt if the origin row itself does not exist or either the
// origin or target index is at/past the reserved-period seal. The seal
// check happens before any value is read for either row, per the
// contract's section 3 ordering requirement. sealStartX has no mechanism
// here to widen it beyond kSealStartX -- it is a plain parameter with
// that default, never raised internally; this module has no holdoutMode-
// style override at all, by design, in this development-only prototype.
inline std::optional<Sample> buildSample(const HourlyIndex& idx, const std::vector<HourlyRow>& rows,
                                          int64_t originHour, int64_t sealStartX = kSealStartX) {
    // Origin domain validated BEFORE the origin+1 arithmetic below, not
    // after -- an out-of-range originHour must never reach the addition
    // (let alone anything that reads a value), matching the "validate
    // before arithmetic" discipline applied everywhere else in this file.
    if (originHour < 0 || originHour > kMaxRepresentableX - 1) {
        return std::nullopt;
    }
    const int64_t targetHour = originHour + 1;
    if (originHour >= sealStartX || targetHour >= sealStartX) {
        return std::nullopt;
    }

    auto originRowIdx = idx.rowForHour(originHour);
    if (!originRowIdx.has_value()) {
        return std::nullopt;
    }
    const HourlyRow& originRow = rows[*originRowIdx];
    const int64_t originSegment = idx.segmentFor(*originRowIdx);

    Sample sample;
    sample.origin_x = originHour;
    sample.target_x = targetHour;

    bool dischargeGroupComplete = true;
    Features f;

    auto lag = [&](int k) -> std::optional<double> {
        return detail::readDischargeHour(idx, rows, originHour - k, originSegment);
    };
    auto q0 = lag(0);
    auto q1 = lag(1);
    auto q3 = lag(3);
    auto q6 = lag(6);
    auto q12 = lag(12);
    auto q24 = lag(24);
    if (!q0 || !q1 || !q3 || !q6 || !q12 || !q24) {
        dischargeGroupComplete = false;
    } else {
        const double lastHourlyChange = *q0 - *q1;
        const double pastSlope6h = (*q0 - *q6) / 6.0;
        // Each input is individually finite, but the difference/division
        // derived from two large-but-finite values can still overflow --
        // checked explicitly, not assumed from finite inputs alone.
        if (!std::isfinite(lastHourlyChange) || !std::isfinite(pastSlope6h)) {
            dischargeGroupComplete = false;
        } else {
            f.q_lag_0h = *q0;
            f.q_lag_1h = *q1;
            f.q_lag_3h = *q3;
            f.q_lag_6h = *q6;
            f.q_lag_12h = *q12;
            f.q_lag_24h = *q24;
            f.q_last_hourly_change = lastHourlyChange;
            f.q_past_slope_6h = pastSlope6h;
        }
    }

    if (dischargeGroupComplete) {
        auto w6 = detail::trailingMeanAndVariance(idx, rows, originHour, originSegment, 6);
        auto w24 = detail::trailingMeanAndVariance(idx, rows, originHour, originSegment, 24);
        auto w72 = detail::trailingMeanAndVariance(idx, rows, originHour, originSegment, 72);
        if (!w6 || !w24 || !w72) {
            dischargeGroupComplete = false;
        } else {
            f.q_trailing_mean_6h = w6->first;
            f.q_trailing_var_6h = w6->second;
            f.q_trailing_mean_24h = w24->first;
            f.q_trailing_var_24h = w24->second;
            f.q_trailing_mean_72h = w72->first;
            f.q_trailing_var_72h = w72->second;
        }
    }

    const bool metGroupComplete = originRow.all_forcings_valid && detail::metRowIsNumericallySound(originRow);
    if (metGroupComplete) {
        f.met_precip_mm_per_h = originRow.precipitation_mm_per_hour;
        f.met_temperature_c = originRow.temperature_c;
        f.met_relative_humidity_pct = originRow.relative_humidity_percent;
        f.met_wind_speed_m_per_s = originRow.wind_speed_m_per_second;
        f.met_solar_mj_per_m2_per_h = originRow.solar_energy_mj_per_m2_per_hour;
        f.met_pet_mm_per_h = originRow.pet_mm_per_hour;
    }

    sample.predictor_availability_mask = dischargeGroupComplete && metGroupComplete;
    sample.features = f;

    auto targetRowIdx = idx.rowForHour(targetHour);
    if (targetRowIdx.has_value() && rows[*targetRowIdx].discharge_valid) {
        const double targetQ = rows[*targetRowIdx].observed_runoff_mm_per_hour;
        if (std::isfinite(targetQ) && targetQ >= 0.0) {
            sample.target_native_mask = true;
            sample.observed_target_q = targetQ;
        }
    }

    // Persistence rides on the same two masks as every learned arm, per
    // the contract's masks policy -- not a separately looser check.
    if (sample.predictor_availability_mask && dischargeGroupComplete) {
        sample.persistence_prediction = f.q_lag_0h;
    }

    return sample;
}

// IMPORTANT -- this is a narrowly-scoped, independent, torch-free adapter,
// NOT a call into the real configuredChronologicalSplit/
// resolveTestEndSequenceCount path (dataset/configured_chronological_split.h),
// which operates on torch tensors and windowed-sequence array positions
// that do not exist at this builder's raw-per-row granularity. Because
// this module intentionally stays torch-free (per Task 2's "built/tested
// without training" requirement), it reimplements only the cutoff
// comparison, not the full split machinery. Its exact-cutoff, inclusive-
// of-the-cutoff-row semantics are verified for parity against
// resolveTestEndSequenceCountFromSorted (dataset/resolve_test_end_x.h) --
// the same pure, torch-free core the production path itself calls -- in
// test_hydrofusion_feature_builder.cpp's testPartitionForXParityWithProduction
// test, including across a gapped sequence where array position and x
// diverge. This is the documented trade-off, not an unverified reimplementation.
inline SplitPartition partitionForX(int64_t x, const SplitCutoffs& cutoffs) {
    if (x <= cutoffs.train_end_x) return SplitPartition::Train;
    if (x <= cutoffs.validation_end_x) return SplitPartition::Validation;
    if (x <= cutoffs.test_end_x) return SplitPartition::Test;
    return SplitPartition::BeyondTestEnd;
}

// Enforces the chronological ordering every caller of partitionForX/
// assignSampleSplit/runPersistenceBenchmark must satisfy:
// 0 <= train_end_x < validation_end_x < test_end_x < sealStartX. Without
// this, e.g. a validation_end_x below train_end_x silently produces a
// nonsensical partition assignment rather than failing -- this must be
// checked before any sample work is done with the cutoffs, not left
// implicit in partitionForX's three sequential comparisons.
inline void validateSplitCutoffs(const SplitCutoffs& cutoffs, int64_t sealStartX = kSealStartX) {
    if (!(0 <= cutoffs.train_end_x && cutoffs.train_end_x < cutoffs.validation_end_x &&
          cutoffs.validation_end_x < cutoffs.test_end_x && cutoffs.test_end_x < sealStartX)) {
        throw std::invalid_argument(
            "validateSplitCutoffs: cutoffs must satisfy 0 <= train_end_x < validation_end_x < "
            "test_end_x < sealStartX.");
    }
}

// A sample straddling a split cutoff (origin and target resolve to
// different partitions) is discarded entirely, per contract section 7.
inline std::optional<SplitPartition> assignSampleSplit(int64_t originX, int64_t targetX,
                                                         const SplitCutoffs& cutoffs) {
    const SplitPartition originPartition = partitionForX(originX, cutoffs);
    const SplitPartition targetPartition = partitionForX(targetX, cutoffs);
    if (originPartition != targetPartition) return std::nullopt;
    return originPartition;
}

struct ProvenanceRecord {
    int64_t origin_x = 0;
    int64_t target_x = 0;
    bool predictor_availability_mask = false;
    bool target_native_mask = false;
    bool split_assigned = false;
    SplitPartition split = SplitPartition::BeyondTestEnd;
    // included = predictor_availability_mask && target_native_mask &&
    // split_assigned && split != BeyondTestEnd. A straddling sample
    // (split_assigned == false) or one past test_end is never included,
    // regardless of what the two masks say -- the masks and the split
    // assignment are independent provenance fields, kept separately above,
    // but "included" itself requires all of them to hold.
    bool included = false;
    double persistence_prediction = 0.0;
    double observed_target_q = 0.0;
};

// Runs the persistence benchmark (Qhat(t+1) = Q(t)) over every hour
// present as an origin candidate, saving per-sample provenance rather
// than only an aggregate score, per the contract and Task 2's explicit
// instruction.
inline std::vector<ProvenanceRecord> runPersistenceBenchmark(const HourlyIndex& idx,
                                                               const std::vector<HourlyRow>& rows,
                                                               const SplitCutoffs& cutoffs,
                                                               int64_t sealStartX = kSealStartX) {
    validateSplitCutoffs(cutoffs, sealStartX);  // before any sample work, per the contract
    std::vector<ProvenanceRecord> records;
    records.reserve(rows.size());
    for (const HourlyRow& row : rows) {
        if (!std::isfinite(row.x) || row.x != std::floor(row.x)) continue;
        if (row.x < 0.0 || row.x > kMaxRepresentableX) continue;
        const int64_t originHour = static_cast<int64_t>(row.x);
        auto sample = buildSample(idx, rows, originHour, sealStartX);
        if (!sample.has_value()) continue;

        ProvenanceRecord record;
        record.origin_x = sample->origin_x;
        record.target_x = sample->target_x;
        record.predictor_availability_mask = sample->predictor_availability_mask;
        record.target_native_mask = sample->target_native_mask;
        record.persistence_prediction = sample->persistence_prediction;
        record.observed_target_q = sample->observed_target_q;

        auto split = assignSampleSplit(sample->origin_x, sample->target_x, cutoffs);
        record.split_assigned = split.has_value();
        if (split.has_value()) record.split = *split;

        // included requires BOTH masks AND a valid, non-BeyondTestEnd split
        // assignment -- a straddling or past-test-end sample is excluded
        // here even if the masks alone would have allowed it.
        record.included = sample->predictor_availability_mask && sample->target_native_mask &&
                           record.split_assigned && record.split != SplitPartition::BeyondTestEnd;

        records.push_back(record);
    }
    return records;
}

}  // namespace hydrofusion
