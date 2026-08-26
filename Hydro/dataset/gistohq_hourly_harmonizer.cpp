#include "gistohq_hourly_harmonizer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
constexpr std::int64_t kHourSeconds = 3600;
constexpr std::int64_t kDaySeconds = 86400;
constexpr double kCubicFeetToCubicMetres = 0.028316846592;

double missing() { return std::numeric_limits<double>::quiet_NaN(); }

void validateSeries(const std::vector<GisToOhqTimedValue>& values, const char* name) {
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (!std::isfinite(values[i].value)) {
            throw std::invalid_argument(std::string(name) + " contains a non-finite value.");
        }
        if (i > 0 && values[i].epoch_seconds <= values[i - 1].epoch_seconds) {
            throw std::invalid_argument(std::string(name) + " timestamps must be strictly increasing and unique.");
        }
    }
}

struct Aggregate {
    double value = 0.0;
    std::size_t count = 0;
};

std::vector<Aggregate> aggregateSamples(const std::vector<GisToOhqTimedValue>& values,
                                        const GisToOhqHourlyConfig& config,
                                        const bool sum) {
    const auto bins = static_cast<std::size_t>((config.end_epoch_seconds - config.start_epoch_seconds) / kHourSeconds);
    std::vector<Aggregate> result(bins);
    for (const auto& sample : values) {
        if (sample.epoch_seconds < config.start_epoch_seconds || sample.epoch_seconds >= config.end_epoch_seconds) continue;
        const auto index = static_cast<std::size_t>((sample.epoch_seconds - config.start_epoch_seconds) / kHourSeconds);
        result[index].value += sample.value;
        ++result[index].count;
    }
    if (!sum) {
        for (auto& item : result) if (item.count > 0) item.value /= static_cast<double>(item.count);
    }
    return result;
}

void requireRange(const double value, const double minimum, const double maximum, const char* name) {
    if (value < minimum || value > maximum) {
        throw std::invalid_argument(std::string(name) + " is outside its supported range.");
    }
}
}

std::vector<GisToOhqHourlyRow> harmonizeGisToOhqHourly(
    const GisToOhqHourlyInputs& inputs,
    const GisToOhqHourlyConfig& config) {
    if (config.start_epoch_seconds % kHourSeconds != 0 || config.end_epoch_seconds % kHourSeconds != 0 ||
        config.end_epoch_seconds <= config.start_epoch_seconds) {
        throw std::invalid_argument("GIStoOHQ handoff interval must use increasing whole UTC hours.");
    }
    if (!std::isfinite(config.catchment_area_m2) || config.catchment_area_m2 <= 0.0 ||
        !std::isfinite(config.discharge_sample_support_seconds) || config.discharge_sample_support_seconds <= 0.0 ||
        config.discharge_sample_support_seconds > kHourSeconds ||
        !std::isfinite(config.minimum_discharge_coverage) || config.minimum_discharge_coverage < 0.0 ||
        config.minimum_discharge_coverage > 1.0 || !std::isfinite(config.latent_heat_mj_per_kg) ||
        config.latent_heat_mj_per_kg <= 0.0) {
        throw std::invalid_argument("GIStoOHQ hourly harmonization configuration is invalid.");
    }
    validateSeries(inputs.precipitation_mm_per_day, "PRECTOTCORR");
    validateSeries(inputs.temperature_c, "T2M");
    validateSeries(inputs.relative_humidity_percent, "RH2M");
    validateSeries(inputs.wind_speed_m_per_second, "WS2M");
    validateSeries(inputs.solar_energy_mj_per_m2, "ALLSKY_SFC_SW_DWN");
    validateSeries(inputs.pet_energy_mj_per_m2_per_day, "EVPTRNS");
    validateSeries(inputs.discharge_ft3_per_second, "00060");

    const auto precipitation = aggregateSamples(inputs.precipitation_mm_per_day, config, false);
    const auto temperature = aggregateSamples(inputs.temperature_c, config, false);
    const auto humidity = aggregateSamples(inputs.relative_humidity_percent, config, false);
    const auto wind = aggregateSamples(inputs.wind_speed_m_per_second, config, false);
    const auto solar = aggregateSamples(inputs.solar_energy_mj_per_m2, config, true);
    const auto count = precipitation.size();
    std::vector<GisToOhqHourlyRow> rows(count);
    for (std::size_t i = 0; i < count; ++i) {
        auto& row = rows[i];
        row.epoch_seconds = config.start_epoch_seconds + static_cast<std::int64_t>(i) * kHourSeconds;
        row.precipitation_mm_per_hour = row.temperature_c = row.relative_humidity_percent =
            row.wind_speed_m_per_second = row.solar_energy_mj_per_m2_per_hour = row.pet_mm_per_hour =
            row.observed_discharge_m3_per_second = row.observed_runoff_mm_per_hour = missing();
        if (precipitation[i].count) {
            requireRange(precipitation[i].value, 0.0, std::numeric_limits<double>::max(), "PRECTOTCORR");
            row.precipitation_mm_per_hour = precipitation[i].value / 24.0;
            row.precipitation_valid = true;
        }
        if (temperature[i].count) { row.temperature_c = temperature[i].value; row.temperature_valid = true; }
        if (humidity[i].count) {
            requireRange(humidity[i].value, 0.0, 100.0, "RH2M");
            row.relative_humidity_percent = humidity[i].value; row.relative_humidity_valid = true;
        }
        if (wind[i].count) {
            requireRange(wind[i].value, 0.0, std::numeric_limits<double>::max(), "WS2M");
            row.wind_speed_m_per_second = wind[i].value; row.wind_valid = true;
        }
        if (solar[i].count) {
            requireRange(solar[i].value, 0.0, std::numeric_limits<double>::max(), "ALLSKY_SFC_SW_DWN");
            row.solar_energy_mj_per_m2_per_hour = solar[i].value; row.solar_valid = true;
        }
    }

    for (const auto& daily : inputs.pet_energy_mj_per_m2_per_day) {
        if (daily.epoch_seconds % kDaySeconds != 0) {
            throw std::invalid_argument("EVPTRNS timestamps must identify a UTC day boundary.");
        }
        requireRange(daily.value, 0.0, std::numeric_limits<double>::max(), "EVPTRNS");
        const double hourly = daily.value / config.latent_heat_mj_per_kg / 24.0;
        for (std::int64_t hour = 0; hour < 24; ++hour) {
            const auto timestamp = daily.epoch_seconds + hour * kHourSeconds;
            if (timestamp < config.start_epoch_seconds || timestamp >= config.end_epoch_seconds) continue;
            auto& row = rows[static_cast<std::size_t>((timestamp - config.start_epoch_seconds) / kHourSeconds)];
            if (row.pet_valid) throw std::invalid_argument("EVPTRNS daily values overlap.");
            row.pet_mm_per_hour = hourly;
            row.pet_valid = true;
        }
    }

    std::vector<double> dischargeVolume(count, 0.0);
    std::vector<double> dischargeCoveredSeconds(count, 0.0);
    for (std::size_t i = 0; i < inputs.discharge_ft3_per_second.size(); ++i) {
        const auto& sample = inputs.discharge_ft3_per_second[i];
        requireRange(sample.value, 0.0, std::numeric_limits<double>::max(), "00060");
        const double next = i + 1 < inputs.discharge_ft3_per_second.size()
                                ? static_cast<double>(inputs.discharge_ft3_per_second[i + 1].epoch_seconds)
                                : static_cast<double>(sample.epoch_seconds) + config.discharge_sample_support_seconds;
        const double supportedEnd = std::min(next, static_cast<double>(sample.epoch_seconds) +
                                                       config.discharge_sample_support_seconds);
        double cursor = std::max(static_cast<double>(sample.epoch_seconds),
                                 static_cast<double>(config.start_epoch_seconds));
        const double end = std::min(supportedEnd, static_cast<double>(config.end_epoch_seconds));
        while (cursor < end) {
            const auto index = static_cast<std::size_t>((static_cast<std::int64_t>(cursor) -
                                                         config.start_epoch_seconds) / kHourSeconds);
            const double binEnd = static_cast<double>(config.start_epoch_seconds +
                static_cast<std::int64_t>(index + 1) * kHourSeconds);
            const double duration = std::min(end, binEnd) - cursor;
            dischargeVolume[index] += sample.value * duration;
            dischargeCoveredSeconds[index] += duration;
            cursor += duration;
        }
    }
    for (std::size_t i = 0; i < count; ++i) {
        auto& row = rows[i];
        row.discharge_coverage = std::min(1.0, dischargeCoveredSeconds[i] / static_cast<double>(kHourSeconds));
        if (row.discharge_coverage >= config.minimum_discharge_coverage) {
            const double ft3s = dischargeVolume[i] / dischargeCoveredSeconds[i];
            row.observed_discharge_m3_per_second = ft3s * kCubicFeetToCubicMetres;
            row.observed_runoff_mm_per_hour = row.observed_discharge_m3_per_second * 3600.0 * 1000.0 /
                                             config.catchment_area_m2;
            row.discharge_valid = true;
        }
        row.all_forcings_valid = row.precipitation_valid && row.temperature_valid &&
            row.relative_humidity_valid && row.wind_valid && row.solar_valid && row.pet_valid;
    }
    return rows;
}
