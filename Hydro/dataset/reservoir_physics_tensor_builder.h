#pragma once

#include "hydro_tensor_builder.h"
#include "csv_tensor_builder.h"
#include "../models/hydro_run_types.h"

#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

/**
 * Unified forcing contract for the reduced runoff-reservoir physics used by
 * FFN+PINN, LSTM+PINN, and standalone PINN:
 *
 *   x[:,0] = physical time
 *   x[:,1] = Peff = max(P - PET, 0)
 *   optional remaining columns are explanatory forcings only
 *   y[:,0] = observed/synthetic runoff used for hybrid data loss or evaluation
 *
 * No storage state is reconstructed from the same equation used by the physics
 * residual. This prevents the former Q ~= kS algebraic circularity.
 */
inline void loadReservoirPhysicsCsvTensors(const HydroRunConfig& config,
                                           torch::Tensor& x,
                                           torch::Tensor& y,
                                           torch::Tensor& plotX) {
    if (config.csv_path.empty()) throw std::invalid_argument("CSV path is empty.");
    if (config.csv_x_column != 0) {
        throw std::invalid_argument(
            "Reduced-reservoir CSV physics requires column 0=time, column 1=precipitation, "
            "column 2=PET; set csv_x_column=0 and csv_y_column to the runoff target column.");
    }
    if (config.csv_y_column < 3) {
        throw std::invalid_argument(
            "Reduced-reservoir CSV physics requires at least [time, precipitation, PET, runoff].");
    }

    std::ifstream file(config.csv_path);
    if (!file) throw std::runtime_error("Unable to open CSV file: " + config.csv_path);

    std::vector<float> features;
    std::vector<float> targets;
    std::vector<float> times;
    std::string line;
    bool firstLine = true;
    std::size_t lineNumber = 0;
    std::size_t featureCount = 0;

    while (std::getline(file, line)) {
        ++lineNumber;
        if (line.empty()) continue;
        if (firstLine && config.csv_has_header) { firstLine = false; continue; }
        firstLine = false;
        const auto columns = parseHydroCsvRow(line);
        if (static_cast<int>(columns.size()) <= config.csv_y_column || columns.size() < 4) {
            throw std::runtime_error("CSV row " + std::to_string(lineNumber) +
                                     " does not satisfy reduced-reservoir physics contract.");
        }
        const float time = parseHydroCsvNumber(columns[0]);
        const float precipitation = std::max(0.0f, parseHydroCsvNumber(columns[1]));
        const float pet = std::max(0.0f, parseHydroCsvNumber(columns[2]));
        const float peff = std::max(0.0f, precipitation - pet);
        const float target = parseHydroCsvNumber(columns[config.csv_y_column]);

        std::vector<float> row = {time, peff, precipitation, pet};
        for (int c = 3; c < static_cast<int>(columns.size()); ++c) {
            if (c == config.csv_y_column) continue;
            row.push_back(parseHydroCsvNumber(columns[c]));
        }
        if (featureCount == 0) featureCount = row.size();
        if (row.size() != featureCount) throw std::runtime_error("CSV physics feature width changes between rows.");
        features.insert(features.end(), row.begin(), row.end());
        targets.push_back(target);
        times.push_back(time);
    }

    if (targets.size() < 10) throw std::runtime_error("CSV physics parsing yielded too few samples (<10).");
    const int64_t n = static_cast<int64_t>(targets.size());
    x = torch::from_blob(features.data(), {n, static_cast<int64_t>(featureCount)}, torch::kFloat32).clone();
    y = torch::from_blob(targets.data(), {n, 1}, torch::kFloat32).clone();
    plotX = torch::from_blob(times.data(), {n, 1}, torch::kFloat32).clone();
    regularPhysicalTimeStepFromTime(plotX);
}

inline void buildReducedReservoirSyntheticTensors(const HydroRunConfig& config,
                                                   torch::Tensor& x,
                                                   torch::Tensor& y,
                                                   torch::Tensor& plotX) {
    const int samples = std::max(32, config.sample_count);
    const double t0 = config.t_start;
    const double t1 = config.t_end;
    const double dt = (samples > 1) ? (t1 - t0) / static_cast<double>(samples - 1) : 1.0;
    if (!(dt > 0.0)) throw std::invalid_argument("Synthetic reservoir physics requires t_end > t_start.");

    // IMPORTANT: truth k is intentionally independent from the candidate model
    // k (lambda_decay/storage_coeff). A physics sweep must never regenerate a
    // different target hydrograph for each candidate.
    const double truthK = std::max(1.0e-8, config.synthetic_reservoir_truth_k);
    constexpr double pi = 3.14159265358979323846;
    std::vector<float> features;
    std::vector<float> targets;
    std::vector<float> times;
    features.reserve(static_cast<std::size_t>(samples) * 4);
    targets.reserve(static_cast<std::size_t>(samples));
    times.reserve(static_cast<std::size_t>(samples));

    double q = 0.15;
    for (int i = 0; i < samples; ++i) {
        const double t = t0 + dt * static_cast<double>(i);
        const double r = samples > 1 ? static_cast<double>(i) / static_cast<double>(samples - 1) : 0.0;
        const double storm1 = 1.6 * std::exp(-0.5 * std::pow((r - 0.25) / 0.055, 2.0));
        const double storm2 = 1.1 * std::exp(-0.5 * std::pow((r - 0.62) / 0.085, 2.0));
        const double precipitation = storm1 + storm2 + 0.12 * std::max(0.0, std::sin(6.0 * pi * r));
        const double pet = 0.035 + 0.02 * (1.0 + std::sin(2.0 * pi * r - 0.5));
        const double peff = std::max(0.0, precipitation - pet);
        if (i > 0) q += dt * truthK * (peff - q);
        q = std::max(0.0, q);

        features.push_back(static_cast<float>(t));
        features.push_back(static_cast<float>(peff));
        features.push_back(static_cast<float>(precipitation));
        features.push_back(static_cast<float>(pet));
        targets.push_back(static_cast<float>(q));
        times.push_back(static_cast<float>(t));
    }

    x = torch::from_blob(features.data(), {samples, 4}, torch::kFloat32).clone();
    y = torch::from_blob(targets.data(), {samples, 1}, torch::kFloat32).clone();
    plotX = torch::from_blob(times.data(), {samples, 1}, torch::kFloat32).clone();
}

inline bool loadReservoirPhysicsTensors(const HydroRunConfig& config,
                                        torch::Tensor& x,
                                        torch::Tensor& y,
                                        torch::Tensor& plotX) {
    if (config.use_hydro_package) {
        if (!loadHydroPackageTensors(config, x, y, plotX)) return false;
        if (x.dim() != 2 || x.size(1) < 2) {
            throw std::runtime_error("Hydro physics input requires at least [time, forcing].");
        }
        if (!config.use_latent_storage_physics && x.size(1) >= 3) {
            torch::Tensor time = x.slice(1, 0, 1);
            torch::Tensor precipitation = torch::clamp_min(x.slice(1, 1, 2), 0.0);
            torch::Tensor pet = torch::clamp_min(x.slice(1, 2, 3), 0.0);
            torch::Tensor peff = torch::clamp_min(precipitation - pet, 0.0);
            torch::Tensor remainder = x.size(1) > 3 ? x.slice(1, 3, x.size(1)) : torch::Tensor();
            x = remainder.defined()
                    ? torch::cat({time, peff, precipitation, pet, remainder}, 1).contiguous()
                    : torch::cat({time, peff, precipitation, pet}, 1).contiguous();
        }
        regularPhysicalTimeStepFromTime(plotX);
        return true;
    }
    if (config.use_csv_data) {
        loadReservoirPhysicsCsvTensors(config, x, y, plotX);
        return true;
    }

    buildReducedReservoirSyntheticTensors(config, x, y, plotX);
    return true;
}
