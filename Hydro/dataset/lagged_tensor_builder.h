#pragma once

#include <torch/torch.h>

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct HydroLaggedTensor {
    torch::Tensor inputs;
    int64_t leading_rows = 0;
};

inline HydroLaggedTensor buildHydroLaggedTensor(const torch::Tensor& inputs,
                                                const std::string& lagSpecification) {
    if (!inputs.defined() || inputs.dim() != 2 || inputs.size(0) <= 1) {
        throw std::invalid_argument("Lagged inputs require a 2D tensor with more than one sample.");
    }
    std::vector<std::vector<int>> lags;
    std::stringstream groups(lagSpecification);
    std::string group;
    while (std::getline(groups, group, ';')) {
        std::vector<int> featureLags;
        std::stringstream tokens(group);
        std::string token;
        while (std::getline(tokens, token, ',')) {
            try {
                const int lag = std::stoi(token);
                if (lag > 0) featureLags.push_back(lag);
            } catch (...) {}
        }
        if (!featureLags.empty()) lags.push_back(std::move(featureLags));
    }
    if (lags.empty()) lags.push_back({1});
    while (lags.size() < static_cast<std::size_t>(inputs.size(1))) lags.push_back(lags.front());
    lags.resize(static_cast<std::size_t>(inputs.size(1)));

    int64_t maxLag = 0;
    for (const auto& featureLags : lags) {
        for (const int lag : featureLags) maxLag = std::max(maxLag, static_cast<int64_t>(lag));
    }
    if (inputs.size(0) <= maxLag) {
        throw std::invalid_argument("Input series is shorter than the configured FFN lag horizon.");
    }
    std::vector<torch::Tensor> columns;
    for (int64_t feature = 0; feature < inputs.size(1); ++feature) {
        columns.push_back(inputs.slice(0, maxLag, inputs.size(0)).slice(1, feature, feature + 1));
        for (const int lag : lags[static_cast<std::size_t>(feature)]) {
            columns.push_back(inputs.slice(0, maxLag - lag, inputs.size(0) - lag)
                                  .slice(1, feature, feature + 1));
        }
    }
    return {torch::cat(columns, 1).contiguous(), maxLag};
}
