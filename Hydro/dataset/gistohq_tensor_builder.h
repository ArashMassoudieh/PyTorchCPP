#pragma once

#include "gistohq_model_rows.h"

#include <torch/torch.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

struct GisToOhqTensorTable {
    torch::Tensor features;      // [rows, 6]
    torch::Tensor targets;       // [rows, 1], NaN when unavailable
    torch::Tensor target_valid;  // [rows], bool
    torch::Tensor epoch_seconds; // [rows], int64
    torch::Tensor elapsed_hours; // [rows, 1]
    torch::Tensor segment_ids;   // [rows], int64
};

struct GisToOhqSequenceTensors {
    torch::Tensor features;      // [windows, sequence_length, 6]
    torch::Tensor targets;       // [windows, 1]
    torch::Tensor target_valid;  // [windows], bool
    torch::Tensor epoch_seconds; // endpoint timestamps
    torch::Tensor segment_ids;   // [windows], int64
};

inline GisToOhqTensorTable buildGisToOhqTensorTable(const std::vector<GisToOhqModelRow>& rows) {
    if (rows.empty()) throw std::invalid_argument("GIStoOHQ model rows are empty.");
    std::vector<double> features;
    std::vector<double> targets;
    std::vector<std::uint8_t> valid;
    std::vector<std::int64_t> timestamps;
    std::vector<double> elapsed;
    std::vector<std::int64_t> segments;
    features.reserve(rows.size() * 6);
    targets.reserve(rows.size()); valid.reserve(rows.size()); timestamps.reserve(rows.size());
    elapsed.reserve(rows.size()); segments.reserve(rows.size());
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const bool validFirst = i != 0 || (rows[i].segment_id == 0 && rows[i].index_in_segment == 0);
        const bool validContinuation = i == 0 ||
            (rows[i].segment_id == rows[i - 1].segment_id &&
             rows[i].index_in_segment == rows[i - 1].index_in_segment + 1);
        const bool validNewSegment = i > 0 && rows[i].segment_id == rows[i - 1].segment_id + 1 &&
                                     rows[i].index_in_segment == 0;
        if (!validFirst || (!validContinuation && !validNewSegment)) {
            throw std::invalid_argument("GIStoOHQ model row segment metadata is inconsistent.");
        }
        features.insert(features.end(), rows[i].features.begin(), rows[i].features.end());
        targets.push_back(rows[i].target_runoff_mm_per_hour);
        valid.push_back(rows[i].target_valid ? 1 : 0);
        timestamps.push_back(rows[i].epoch_seconds);
        elapsed.push_back(rows[i].elapsed_hours);
        segments.push_back(static_cast<std::int64_t>(rows[i].segment_id));
    }
    const auto count = static_cast<std::int64_t>(rows.size());
    GisToOhqTensorTable table;
    table.features = torch::from_blob(features.data(), {count, 6}, torch::kFloat64).clone();
    table.targets = torch::from_blob(targets.data(), {count, 1}, torch::kFloat64).clone();
    table.target_valid = torch::from_blob(valid.data(), {count}, torch::kUInt8).to(torch::kBool).clone();
    table.epoch_seconds = torch::from_blob(timestamps.data(), {count}, torch::kInt64).clone();
    table.elapsed_hours = torch::from_blob(elapsed.data(), {count, 1}, torch::kFloat64).clone();
    table.segment_ids = torch::from_blob(segments.data(), {count}, torch::kInt64).clone();
    return table;
}

inline GisToOhqSequenceTensors buildGisToOhqSequenceTensors(
    const GisToOhqTensorTable& table, const std::int64_t sequenceLength,
    const bool requireObservedTarget) {
    if (sequenceLength < 2) throw std::invalid_argument("GIStoOHQ sequence length must be at least two.");
    if (!table.features.defined() || table.features.dim() != 2 || table.features.size(1) != 6 ||
        !table.targets.defined() || table.targets.dim() != 2 || table.targets.size(0) != table.features.size(0) ||
        table.targets.size(1) != 1 || !table.target_valid.defined() || table.target_valid.dim() != 1 ||
        table.target_valid.size(0) != table.features.size(0) || !table.epoch_seconds.defined() ||
        table.epoch_seconds.dim() != 1 || table.epoch_seconds.size(0) != table.features.size(0) ||
        !table.segment_ids.defined() || table.segment_ids.dim() != 1 ||
        table.segment_ids.size(0) != table.features.size(0)) {
        throw std::invalid_argument("GIStoOHQ tensor table has inconsistent shapes.");
    }
    std::vector<torch::Tensor> featureWindows, targetChunks, validChunks, timestampChunks, segmentChunks;
    std::int64_t begin = 0;
    while (begin < table.features.size(0)) {
        const auto segment = table.segment_ids[begin].item<std::int64_t>();
        std::int64_t end = begin + 1;
        while (end < table.features.size(0) && table.segment_ids[end].item<std::int64_t>() == segment) ++end;
        if (end - begin >= sequenceLength) {
            auto windows = table.features.slice(0, begin, end).unfold(0, sequenceLength, 1).transpose(1, 2).contiguous();
            auto endpoints = torch::arange(begin + sequenceLength - 1, end, torch::kInt64);
            if (requireObservedTarget) {
                const auto keep = table.target_valid.index_select(0, endpoints);
                windows = windows.index({keep});
                endpoints = endpoints.index({keep});
            }
            if (endpoints.numel() > 0) {
                featureWindows.push_back(windows);
                targetChunks.push_back(table.targets.index_select(0, endpoints));
                validChunks.push_back(table.target_valid.index_select(0, endpoints));
                timestampChunks.push_back(table.epoch_seconds.index_select(0, endpoints));
                segmentChunks.push_back(table.segment_ids.index_select(0, endpoints));
            }
        }
        begin = end;
    }
    if (featureWindows.empty()) throw std::runtime_error("GIStoOHQ segments contain no eligible sequence windows.");
    return {torch::cat(featureWindows), torch::cat(targetChunks), torch::cat(validChunks),
            torch::cat(timestampChunks), torch::cat(segmentChunks)};
}
