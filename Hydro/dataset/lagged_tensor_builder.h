#pragma once

#include <torch/torch.h>

#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct HydroLaggedTensor {
    torch::Tensor inputs;
    int64_t leading_rows = 0;
};

inline std::vector<std::vector<int>> parseHydroLagSpecification(
    const std::string& lagSpecification, const int64_t featureCount) {
    if (featureCount <= 0) throw std::invalid_argument("Lag configuration requires at least one feature.");
    if (lagSpecification.empty()) {
        return std::vector<std::vector<int>>(static_cast<std::size_t>(featureCount), {1});
    }
    if (lagSpecification.back() == ';' || lagSpecification.back() == ',') {
        throw std::invalid_argument("FFN lag configuration cannot end with an empty token.");
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
                std::size_t consumed = 0;
                const long long parsed = std::stoll(token, &consumed);
                if (consumed != token.size() || parsed <= 0 || parsed > std::numeric_limits<int>::max()) {
                    throw std::invalid_argument("invalid lag");
                }
                const int lag = static_cast<int>(parsed);
                if (std::find(featureLags.begin(), featureLags.end(), lag) != featureLags.end()) {
                    throw std::invalid_argument("duplicate lag");
                }
                featureLags.push_back(lag);
            } catch (...) {
                throw std::invalid_argument("Invalid FFN lag token: " + token);
            }
        }
        if (featureLags.empty()) throw std::invalid_argument("FFN lag groups cannot be empty.");
        lags.push_back(std::move(featureLags));
    }
    if (lags.size() > static_cast<std::size_t>(featureCount)) {
        throw std::invalid_argument("FFN lag configuration has more groups than input features.");
    }
    while (lags.size() < static_cast<std::size_t>(featureCount)) lags.push_back(lags.front());
    return lags;
}

inline int hydroCurrentFeatureColumn(const std::vector<std::vector<int>>& lags, const int featureIndex) {
    int column = 0;
    for (int feature = 0; feature < featureIndex; ++feature)
        column += 1 + static_cast<int>(lags.at(static_cast<std::size_t>(feature)).size());
    return column;
}

inline HydroLaggedTensor buildHydroLaggedTensor(const torch::Tensor& inputs,
                                                const std::string& lagSpecification) {
    if (!inputs.defined() || inputs.dim() != 2 || inputs.size(0) <= 1) {
        throw std::invalid_argument("Lagged inputs require a 2D tensor with more than one sample.");
    }
    const auto lags = parseHydroLagSpecification(lagSpecification, inputs.size(1));

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
