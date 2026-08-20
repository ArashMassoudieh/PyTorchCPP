#pragma once

#include "ddrr_loader.h"
#include "forecast_alignment.h"
#include "../models/hydro_run_types.h"

#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

inline bool loadHydroPackageTensors(const HydroRunConfig& config,
                                    torch::Tensor& x,
                                    torch::Tensor& y,
                                    torch::Tensor& plotX) {
    if (!config.use_hydro_package) return false;
    if (config.hydro_package_path.empty()) throw std::runtime_error("Hydro package path is empty.");
    if (config.hydro_catchment_id.empty()) throw std::runtime_error("Hydro package catchment ID is empty.");
    const bool waterBalance = config.hydro_package_profile == "water-balance";
    if (!waterBalance && config.hydro_package_profile != "rainfall-runoff") {
        throw std::runtime_error("Unknown Hydro package profile: " + config.hydro_package_profile);
    }
    DDRRLoader loader;
    const auto dataset = loader.loadPackageDirectory(
        config.hydro_package_path,
        waterBalance ? HydroDatasetContract::waterBalanceV1() : HydroDatasetContract::rainfallRunoffV1());
    const auto found = dataset.observations_by_catchment.find(config.hydro_catchment_id);
    if (found == dataset.observations_by_catchment.end()) {
        throw std::runtime_error("Catchment not found in Hydro package: " + config.hydro_catchment_id);
    }
    const auto& rows = found->second;
    std::optional<AlignedForecastFeature> forecastFeature;
    if (config.use_hydro_forecast_feature) {
        const auto manifest = loader.loadManifest(config.hydro_package_path + "/manifest.json");
        if (manifest.forecast_file.empty()) throw std::runtime_error("Hydro package has no forecast_file for the requested forecast feature.");
        std::vector<std::string> validTimes;
        validTimes.reserve(rows.size());
        for (const auto& row : rows) validTimes.push_back(row.timestamp);
        const auto forecasts = loader.loadForecasts(config.hydro_package_path + "/" + manifest.forecast_file, {});
        forecastFeature = buildAlignedForecastFeature(
            forecasts, validTimes, config.hydro_catchment_id, config.hydro_forecast_variable,
            config.hydro_forecast_lead_hours, config.hydro_forecast_ensemble_member);
    }
    const int64_t featureCount = config.use_hydro_forecast_feature ? 6 : 5;
    std::vector<float> features;
    std::vector<float> targets;
    std::vector<float> times;
    features.reserve(rows.size() * static_cast<std::size_t>(featureCount));
    targets.reserve(rows.size());
    times.reserve(rows.size());
    for (std::size_t rowIndex = 0; rowIndex < rows.size(); ++rowIndex) {
        const auto& row = rows[rowIndex];
        if (waterBalance && !row.storage_mm.has_value()) {
            throw std::runtime_error("Water-balance package row is missing storage.");
        }
        features.push_back(static_cast<float>(row.elapsed_hours));
        features.push_back(static_cast<float>(row.precipitation_mm_per_hour));
        features.push_back(static_cast<float>(row.potential_et_mm_per_hour));
        features.push_back(0.0f); // reserved temperature slot for current wrapper layout
        features.push_back(static_cast<float>(row.storage_mm.value_or(0.0)));
        if (forecastFeature) features.push_back(static_cast<float>(forecastFeature->values.at(rowIndex)));
        targets.push_back(static_cast<float>(row.observed_runoff_mm_per_hour));
        times.push_back(static_cast<float>(row.elapsed_hours));
    }
    const auto n = static_cast<int64_t>(rows.size());
    x = torch::from_blob(features.data(), {n, featureCount}, torch::kFloat32).clone();
    y = torch::from_blob(targets.data(), {n, 1}, torch::kFloat32).clone();
    plotX = torch::from_blob(times.data(), {n, 1}, torch::kFloat32).clone();
    return true;
}

inline double regularPhysicalTimeStep(const torch::Tensor& inputs,
                                      double relativeTolerance = 1.0e-6) {
    if (!inputs.defined() || inputs.dim() != 2 || inputs.size(0) < 2 || inputs.size(1) < 1) {
        throw std::runtime_error("Physical timestep inference requires inputs [N,F] with N >= 2.");
    }
    auto time = inputs.slice(1, 0, 1).reshape({-1});
    auto intervals = time.slice(0, 1, time.size(0)) - time.slice(0, 0, time.size(0) - 1);
    if ((intervals <= 0).any().item<bool>()) throw std::runtime_error("Physical timestamps must be strictly increasing.");
    const double dt = intervals[0].item<double>();
    const double tolerance = std::max(1.0e-12, std::abs(dt) * relativeTolerance);
    if ((torch::abs(intervals - dt) > tolerance).any().item<bool>()) {
        throw std::runtime_error("Current PINN backends require a regular package timestep; irregular timestamps need interval-aware training.");
    }
    return dt;
}
