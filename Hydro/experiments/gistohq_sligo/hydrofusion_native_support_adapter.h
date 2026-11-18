#pragma once

#include "hydrofusion_feature_builder.h"
#include "../../dataset/gistohq_hourly_harmonizer.h"
#include <string>

namespace hydrofusion {

// Development-only boundary between an upstream package loader and HydroFusion.
// This deliberately does NOT open a package, infer native support from a finite
// interpolated value, or enable real-data training in FFNHydroFusionWrapper.
struct NativeSupportEvidence {
    int64_t epoch_seconds = 0;
    bool native_observation = false;
    bool interpolated = false;
    double native_coverage = 0.0;
    std::string source_id;
};

struct NativeAdapterContext {
    int64_t epoch_zero_seconds = 1420070400; // 2015-01-01T00:00:00Z
    int64_t first_x = 0;
    int64_t end_x_exclusive = 0;
    double catchment_area_m2 = 0.0;
    double minimum_native_coverage = 0.75;
    std::string discharge_unit = "m3/s";
    std::string precipitation_unit = "mm/h";
    std::string expected_source_id;
};

inline std::vector<HourlyRow> adaptNativeHourlyRows(
    const std::vector<GisToOhqHourlyRow>& input,
    const std::vector<NativeSupportEvidence>& evidence,
    const NativeAdapterContext& context) {
    if (context.epoch_zero_seconds != 1420070400 || context.first_x < 0 ||
        context.end_x_exclusive <= context.first_x || context.end_x_exclusive > kSealStartX ||
        !std::isfinite(context.catchment_area_m2) || context.catchment_area_m2 <= 0 ||
        !std::isfinite(context.minimum_native_coverage) || context.minimum_native_coverage <= 0 ||
        context.minimum_native_coverage > 1 || context.discharge_unit != "m3/s" ||
        context.precipitation_unit != "mm/h" || context.expected_source_id.empty()) {
        throw std::invalid_argument("Invalid native adapter epoch, development interval, units, area or provenance.");
    }
    // Inspect ALL timestamps before reading any discharge/forcing value. A caller
    // must request a presealed slice upstream, not pass full data and filter here.
    auto checkedX = [&](int64_t epoch) {
        const int64_t lower = context.epoch_zero_seconds + context.first_x * 3600;
        const int64_t upper = context.epoch_zero_seconds + context.end_x_exclusive * 3600;
        if (epoch < lower || epoch >= upper || (epoch - context.epoch_zero_seconds) % 3600 != 0)
            throw std::invalid_argument("Native adapter timestamp outside requested preseal hourly interval.");
        return (epoch - context.epoch_zero_seconds) / 3600;
    };
    int64_t previous = -1;
    for (const auto& row : input) {
        const auto x = checkedX(row.epoch_seconds);
        if (x <= previous) throw std::invalid_argument("Native adapter rows must be unique and sorted.");
        previous = x;
    }
    std::map<int64_t, const NativeSupportEvidence*> support;
    for (const auto& entry : evidence) {
        checkedX(entry.epoch_seconds);
        if (!support.emplace(entry.epoch_seconds, &entry).second)
            throw std::invalid_argument("Duplicate native support identity.");
        if (!std::isfinite(entry.native_coverage) || entry.native_coverage < 0 || entry.native_coverage > 1 ||
            entry.source_id != context.expected_source_id ||
            (entry.native_observation && entry.interpolated))
            throw std::invalid_argument("Invalid or contradictory native support evidence.");
    }
    if (support.size() != input.size())
        throw std::invalid_argument("Native support must identify every input row exactly once.");
    std::vector<HourlyRow> output;
    output.reserve(input.size());
    for (const auto& row : input) {
        const auto found = support.find(row.epoch_seconds);
        if (found == support.end()) throw std::invalid_argument("Missing native support identity.");
        const auto& entry = *found->second;
        HourlyRow out;
        out.x = static_cast<double>(checkedX(row.epoch_seconds));
        out.precipitation_mm_per_hour = row.precipitation_mm_per_hour;
        out.temperature_c = row.temperature_c;
        out.relative_humidity_percent = row.relative_humidity_percent;
        out.wind_speed_m_per_second = row.wind_speed_m_per_second;
        out.solar_energy_mj_per_m2_per_hour = row.solar_energy_mj_per_m2_per_hour;
        out.pet_mm_per_hour = row.pet_mm_per_hour;
        out.all_forcings_valid = row.all_forcings_valid && row.precipitation_valid && row.temperature_valid &&
            row.relative_humidity_valid && row.wind_valid && row.solar_valid && row.pet_valid &&
            std::isfinite(out.precipitation_mm_per_hour) && out.precipitation_mm_per_hour >= 0 &&
            std::isfinite(out.temperature_c) && std::isfinite(out.relative_humidity_percent) &&
            out.relative_humidity_percent >= 0 && out.relative_humidity_percent <= 100 &&
            std::isfinite(out.wind_speed_m_per_second) && out.wind_speed_m_per_second >= 0 &&
            std::isfinite(out.solar_energy_mj_per_m2_per_hour) && out.solar_energy_mj_per_m2_per_hour >= 0 &&
            std::isfinite(out.pet_mm_per_hour) && out.pet_mm_per_hour >= 0;
        // Convert using the package's actual area, never the public display-area
        // constant. Do not trust an independently precomputed runoff column.
        const double q = row.observed_discharge_m3_per_second;
        const double runoff = q * (3600000.0 / context.catchment_area_m2);
        out.discharge_valid = entry.native_observation && !entry.interpolated &&
            entry.native_coverage >= context.minimum_native_coverage && row.discharge_valid &&
            std::isfinite(q) && q >= 0 && std::isfinite(runoff) && runoff >= 0;
        out.observed_runoff_mm_per_hour = out.discharge_valid ? runoff : 0.0;
        output.push_back(out);
    }
    return output;
}

} // namespace hydrofusion
