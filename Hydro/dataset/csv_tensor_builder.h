#pragma once

#include "../models/hydro_run_types.h"

#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

inline std::vector<std::string> parseHydroCsvRow(const std::string& line) {
    std::vector<std::string> columns;
    std::string cell;
    bool quoted = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char character = line[i];
        if (character == '"') {
            if (quoted && i + 1 < line.size() && line[i + 1] == '"') {
                cell.push_back('"');
                ++i;
            } else {
                quoted = !quoted;
            }
        } else if (character == ',' && !quoted) {
            columns.push_back(cell);
            cell.clear();
        } else if (character != '\r' || i + 1 != line.size()) {
            cell.push_back(character);
        }
    }
    if (quoted) throw std::runtime_error("CSV row contains an unterminated quoted field.");
    columns.push_back(cell);
    return columns;
}

inline float parseHydroCsvNumber(const std::string& value) {
    std::size_t consumed = 0;
    const double parsed = std::stod(value, &consumed);
    if (consumed != value.size() || !std::isfinite(parsed)) {
        throw std::invalid_argument("CSV field is not a finite number.");
    }
    const float result = static_cast<float>(parsed);
    if (!std::isfinite(result)) throw std::out_of_range("CSV field exceeds the supported numeric range.");
    return result;
}

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
        const auto columns = parseHydroCsvRow(line);
        if (requiredColumn < 0 || static_cast<int>(columns.size()) <= requiredColumn) continue;
        try {
            std::vector<float> rowInputs;
            float rowPlot = 0.0f;
            if (config.synthetic_profile == "neuroforge_inputs_target") {
                for (int column = 0; column < static_cast<int>(columns.size()); ++column) {
                    if (column != config.csv_y_column) rowInputs.push_back(parseHydroCsvNumber(columns[column]));
                }
                if (rowInputs.empty()) continue;
                rowPlot = config.csv_x_column != config.csv_y_column
                              ? parseHydroCsvNumber(columns[config.csv_x_column])
                              : rowInputs.front();
            } else {
                rowPlot = parseHydroCsvNumber(columns[config.csv_x_column]);
                rowInputs.push_back(rowPlot);
            }
            const float target = parseHydroCsvNumber(columns[config.csv_y_column]);
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
