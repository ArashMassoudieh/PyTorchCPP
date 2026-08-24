#pragma once

#include "../models/hydro_run_types.h"

#include <torch/torch.h>

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

inline void loadHydroCsvTensors(const HydroRunConfig& config,
                                torch::Tensor& inputs,
                                torch::Tensor& targets,
                                torch::Tensor& plotX) {
    if (config.csv_path.empty()) throw std::invalid_argument("CSV path is empty.");
    std::ifstream file(config.csv_path);
    if (!file) throw std::runtime_error("Unable to open CSV file: " + config.csv_path);
    std::vector<float> flatInputs;
    std::vector<float> targetValues;
    std::vector<float> plotValues;
    std::size_t featureCount = 0;
    std::string line;
    bool firstLine = true;
    const int requiredColumn = std::max(config.csv_x_column, config.csv_y_column);
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        if (firstLine && config.csv_has_header) { firstLine = false; continue; }
        firstLine = false;
        std::vector<std::string> columns;
        std::stringstream row(line);
        std::string cell;
        while (std::getline(row, cell, ',')) columns.push_back(cell);
        if (requiredColumn < 0 || static_cast<int>(columns.size()) <= requiredColumn) continue;
        try {
            std::vector<float> rowInputs;
            float rowPlot = 0.0f;
            if (config.synthetic_profile == "neuroforge_inputs_target") {
                for (int column = 0; column < static_cast<int>(columns.size()); ++column) {
                    if (column != config.csv_y_column) rowInputs.push_back(static_cast<float>(std::stod(columns[column])));
                }
                if (rowInputs.empty()) continue;
                rowPlot = config.csv_x_column != config.csv_y_column
                              ? static_cast<float>(std::stod(columns[config.csv_x_column]))
                              : rowInputs.front();
            } else {
                rowPlot = static_cast<float>(std::stod(columns[config.csv_x_column]));
                rowInputs.push_back(rowPlot);
            }
            const float target = static_cast<float>(std::stod(columns[config.csv_y_column]));
            if (featureCount == 0) featureCount = rowInputs.size();
            if (rowInputs.size() != featureCount) throw std::runtime_error("CSV input feature width changes between rows.");
            flatInputs.insert(flatInputs.end(), rowInputs.begin(), rowInputs.end());
            targetValues.push_back(target);
            plotValues.push_back(rowPlot);
        } catch (const std::invalid_argument&) {
            continue;
        } catch (const std::out_of_range&) {
            continue;
        }
    }
    if (targetValues.size() < 10 || featureCount == 0) {
        throw std::runtime_error("CSV parsing yielded too few numeric samples (<10).");
    }
    const auto samples = static_cast<int64_t>(targetValues.size());
    inputs = torch::from_blob(flatInputs.data(), {samples, static_cast<int64_t>(featureCount)}, torch::kFloat32).clone();
    targets = torch::from_blob(targetValues.data(), {samples, 1}, torch::kFloat32).clone();
    plotX = torch::from_blob(plotValues.data(), {samples, 1}, torch::kFloat32).clone();
}
