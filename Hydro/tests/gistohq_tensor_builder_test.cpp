#include "../dataset/gistohq_tensor_builder.h"

#include <cassert>
#include <cmath>
#include <limits>
#include <vector>

int main() {
    std::vector<GisToOhqModelRow> rows;
    for (std::size_t i = 0; i < 5; ++i) {
        GisToOhqModelRow row;
        row.epoch_seconds = static_cast<std::int64_t>(i) * 3600;
        row.elapsed_hours = static_cast<double>(i);
        row.features = {double(i), 10.0, 50.0, 2.0, 0.5, 0.1};
        row.target_runoff_mm_per_hour = double(i);
        row.target_valid = true;
        row.segment_id = i < 3 ? 0 : 1;
        row.index_in_segment = i < 3 ? i : i - 3;
        rows.push_back(row);
    }
    rows[2].target_valid = false;
    rows[2].target_runoff_mm_per_hour = std::numeric_limits<double>::quiet_NaN();
    const auto table = buildGisToOhqTensorTable(rows);
    assert(table.features.sizes() == torch::IntArrayRef({5, 6}));
    assert(table.targets.sizes() == torch::IntArrayRef({5, 1}));
    assert(!table.target_valid[2].item<bool>());
    assert(table.segment_ids[3].item<std::int64_t>() == 1);

    const auto inference = buildGisToOhqSequenceTensors(table, 2, false);
    assert(inference.features.sizes() == torch::IntArrayRef({3, 2, 6}));
    assert(inference.epoch_seconds[0].item<std::int64_t>() == 3600);
    assert(inference.epoch_seconds[1].item<std::int64_t>() == 7200);
    assert(inference.epoch_seconds[2].item<std::int64_t>() == 14400);
    assert(!inference.target_valid[1].item<bool>());

    const auto supervised = buildGisToOhqSequenceTensors(table, 2, true);
    assert(supervised.features.sizes() == torch::IntArrayRef({2, 2, 6}));
    assert(supervised.epoch_seconds[0].item<std::int64_t>() == 3600);
    assert(supervised.epoch_seconds[1].item<std::int64_t>() == 14400);
    assert(supervised.target_valid.all().item<bool>());

    const std::string lags = "1;1;1;1;1;1";
    const auto laggedInference = buildGisToOhqLaggedTensors(table, lags, false);
    assert(laggedInference.features.sizes() == torch::IntArrayRef({3, 12}));
    assert(laggedInference.epoch_seconds[0].item<std::int64_t>() == 3600);
    assert(laggedInference.epoch_seconds[1].item<std::int64_t>() == 7200);
    assert(laggedInference.epoch_seconds[2].item<std::int64_t>() == 14400);
    assert(laggedInference.segment_ids[1].item<std::int64_t>() == 0);
    assert(laggedInference.segment_ids[2].item<std::int64_t>() == 1);
    assert(!laggedInference.target_valid[1].item<bool>());
    const auto laggedSupervised = buildGisToOhqLaggedTensors(table, lags, true);
    assert(laggedSupervised.features.sizes() == torch::IntArrayRef({2, 12}));
    assert(laggedSupervised.epoch_seconds[0].item<std::int64_t>() == 3600);
    assert(laggedSupervised.epoch_seconds[1].item<std::int64_t>() == 14400);
    assert(laggedSupervised.target_valid.all().item<bool>());
    return 0;
}
