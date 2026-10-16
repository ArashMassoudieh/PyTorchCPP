#pragma once

#include "ddrr_loader.h"
#include "forecast_alignment.h"
#include "gistohq_package_adapter.h"
#include "../models/hydro_run_types.h"

#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <vector>

inline bool loadHydroPackageTensors(const HydroRunConfig& config,
                                    torch::Tensor& x,
                                    torch::Tensor& y,
                                    torch::Tensor& plotX,
                                    torch::Tensor& segmentId);

inline bool loadHydroPackageTensors(const HydroRunConfig& config,
                                    torch::Tensor& x,
                                    torch::Tensor& y,
                                    torch::Tensor& plotX) {
    torch::Tensor segmentId;
    return loadHydroPackageTensors(config, x, y, plotX, segmentId);
}

// segmentId is an int64 [N] tensor, one entry per row of x/y/plotX, equal to
// the originating GIStoOHQ contiguous-block id (0, 1, 2, ...) - rows from
// different blocks are never implicitly adjacent even though they sit next
// to each other in the concatenated tensor. Sequence/lag builders MUST use
// it to avoid bridging a gap (see makeSequences in lstm_pinn_wrapper.cpp for
// the reference implementation; the already-tested, already-correct pattern
// this mirrors is buildGisToOhqSequenceTensors in gistohq_tensor_builder.h).
inline bool loadHydroPackageTensors(const HydroRunConfig& config,
                                    torch::Tensor& x,
                                    torch::Tensor& y,
                                    torch::Tensor& plotX,
                                    torch::Tensor& segmentId) {
    if (!config.use_hydro_package) return false;
    if (config.hydro_package_path.empty()) throw std::runtime_error("Hydro package path is empty.");
    const bool waterBalance = config.hydro_package_profile == "water-balance";
    if (!waterBalance && config.hydro_package_profile != "rainfall-runoff") {
        throw std::runtime_error("Unknown Hydro package profile: " + config.hydro_package_profile);
    }
    DDRRLoader loader;
    const auto packageRoot = resolveHydroPackageDirectory(config.hydro_package_path);
    if (isGisToOhqHydroPinnExport(packageRoot)) {
        if (config.use_hydro_forecast_feature) {
            throw std::runtime_error("GIStoOHQ temporal exports do not contain forecast assets.");
        }
        const auto prepared = prepareGisToOhqPackage(packageRoot, true);
        if (prepared.model_rows.empty()) throw std::runtime_error("GIStoOHQ package has no contiguous model segment.");
        std::vector<float> features, targets, times;
        std::vector<std::int64_t> segments;

        // Every valid row from every contiguous block is kept - NOT just the
        // single longest block (the prior behavior here, which matched the
        // original "all five paper methods use the same uninterrupted domain"
        // intent but did so by discarding every row outside the longest block,
        // silently dropping any other usable year whenever a real forcing or
        // discharge gap split the record - exactly what the NSE>=0.8 recovery
        // program's Phase 2 Step 2 flags as disallowed. Sequence/lag builders
        // downstream use segmentId to keep never bridging a block boundary, so
        // "same uninterrupted domain per block" is preserved without discarding
        // any block.
        if (config.use_latent_storage_physics) {
            features.reserve(prepared.model_rows.size() * 8);
        } else {
            features.reserve(prepared.model_rows.size() * 6);
        }
        targets.reserve(prepared.model_rows.size());
        times.reserve(prepared.model_rows.size());
        segments.reserve(prepared.model_rows.size());
        const std::int64_t featureWidth = config.use_latent_storage_physics ? 8 : 6;
        for (const auto& row : prepared.model_rows) {
            const double time = row.elapsed_hours;
            if (config.use_latent_storage_physics) {
                const double precipitation = std::max(0.0, row.features[0]);
                const double pet = std::max(0.0, row.features[5]);
                const double effectiveInput = std::max(0.0, precipitation - pet);
                features.push_back(static_cast<float>(time));
                features.push_back(static_cast<float>(effectiveInput));
                features.push_back(static_cast<float>(precipitation));
                features.push_back(static_cast<float>(pet));
                features.push_back(static_cast<float>(row.features[1]));
                features.push_back(static_cast<float>(row.features[2]));
                features.push_back(static_cast<float>(row.features[3]));
                features.push_back(static_cast<float>(row.features[4]));
            } else {
                for (const auto value : row.features) features.push_back(static_cast<float>(value));
            }
            targets.push_back(static_cast<float>(row.target_runoff_mm_per_hour));
            times.push_back(static_cast<float>(time));
            segments.push_back(static_cast<std::int64_t>(row.segment_id));
        }
        if (targets.empty()) throw std::runtime_error("GIStoOHQ package contains no usable supervised rows.");
        const auto n = static_cast<int64_t>(targets.size());
        x = torch::from_blob(features.data(), {n, featureWidth}, torch::kFloat32).clone();
        y = torch::from_blob(targets.data(), {n, 1}, torch::kFloat32).clone();
        plotX = torch::from_blob(times.data(), {n, 1}, torch::kFloat32).clone();
        segmentId = torch::from_blob(segments.data(), {n}, torch::kInt64).clone();
        return true;
    }
    const auto dataset = loader.loadPackageDirectory(
        packageRoot,
        waterBalance ? HydroDatasetContract::waterBalanceV1() : HydroDatasetContract::rainfallRunoffV1());
    const std::string catchmentId = resolveHydroCatchmentId(dataset, config.hydro_catchment_id);
    const auto found = dataset.observations_by_catchment.find(catchmentId);
    const auto& rows = found->second;
    std::optional<AlignedForecastFeature> forecastFeature;
    if (config.use_hydro_forecast_feature) {
        const auto manifest = loader.loadManifest(packageRoot + "/manifest.json");
        if (manifest.forecast_file.empty()) throw std::runtime_error("Hydro package has no forecast_file for the requested forecast feature.");
        std::vector<std::string> validTimes;
        validTimes.reserve(rows.size());
        for (const auto& row : rows) validTimes.push_back(row.timestamp);
        const auto forecasts = loader.loadForecasts(packageRoot + "/" + manifest.forecast_file, {});
        forecastFeature = buildAlignedForecastFeature(
            forecasts, validTimes, catchmentId, config.hydro_forecast_variable,
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
        features.push_back(0.0f);
        features.push_back(static_cast<float>(row.storage_mm.value_or(0.0)));
        if (forecastFeature) features.push_back(static_cast<float>(forecastFeature->values.at(rowIndex)));
        targets.push_back(static_cast<float>(row.observed_runoff_mm_per_hour));
        times.push_back(static_cast<float>(row.elapsed_hours));
    }
    const auto n = static_cast<int64_t>(rows.size());
    x = torch::from_blob(features.data(), {n, featureCount}, torch::kFloat32).clone();
    y = torch::from_blob(targets.data(), {n, 1}, torch::kFloat32).clone();
    plotX = torch::from_blob(times.data(), {n, 1}, torch::kFloat32).clone();
    // The generic DDRR loader path has no multi-block concept of its own
    // (unlike the GIStoOHQ path above) - a single segment preserves its
    // existing behavior exactly rather than silently asserting a gap-free
    // guarantee this path has never actually checked.
    segmentId = torch::zeros({n}, torch::kInt64);
    return true;
}

inline double regularPhysicalTimeStepFromTime(const torch::Tensor& physicalTime,
                                              double relativeTolerance = 1.0e-6) {
    if (!physicalTime.defined() || physicalTime.numel() < 2) {
        throw std::runtime_error("Physical timestep inference requires at least two timestamps.");
    }
    auto time = physicalTime.reshape({-1});
    auto intervals = time.slice(0, 1, time.size(0)) - time.slice(0, 0, time.size(0) - 1);
    if ((intervals <= 0).any().item<bool>()) throw std::runtime_error("Physical timestamps must be strictly increasing.");
    const double dt = intervals[0].item<double>();
    const double tolerance = std::max(1.0e-12, std::abs(dt) * relativeTolerance);
    if ((torch::abs(intervals - dt) > tolerance).any().item<bool>()) {
        throw std::runtime_error("Current PINN backends require a regular package timestep; irregular timestamps need interval-aware training.");
    }
    return dt;
}

inline double regularPhysicalTimeStep(const torch::Tensor& inputs,
                                      double relativeTolerance = 1.0e-6) {
    if (!inputs.defined() || inputs.dim() != 2 || inputs.size(1) < 1) {
        throw std::runtime_error("Physical timestep inference requires inputs [N,F] with N >= 2.");
    }
    return regularPhysicalTimeStepFromTime(inputs.slice(1, 0, 1), relativeTolerance);
}
