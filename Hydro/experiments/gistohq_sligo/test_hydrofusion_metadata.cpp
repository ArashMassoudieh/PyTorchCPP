// Pure synthetic metadata/domain tests; no model training or package I/O.
#include "hydrofusion_metadata_sidecar.h"
#include <cassert>
#include <functional>
#include <iostream>
#include <limits>
#include <unistd.h>

using namespace hydrofusion;

int main() {
    const auto root = std::filesystem::temp_directory_path() /
        ("hydrofusion_metadata_codex_" + std::to_string(getpid()));
    if (!std::filesystem::create_directory(root)) throw std::runtime_error("Test directory exists.");
    const auto path = root / "metadata.txt";
    const auto map = root / "map.csv";
    HydroFusionMetadata meta;
    meta.qRef = 2;
    meta.cutoffs = {100, 150, 180};
    meta.modelSha256 = std::string(64, 'a');
    meta.scalersSha256 = std::string(64, 'b');
    const auto reject = [](const std::function<void()>& fn) {
        bool threw = false;
        try { fn(); } catch (const std::exception&) { threw = true; }
        assert(threw);
    };
    exportHydroFusionMetadata(meta, path.string());
    const auto restored = reloadHydroFusionMetadata(path.string());
    assert(restored.qRef == 2 && restored.modelSha256 == meta.modelSha256);
    validateMetadataBoundToBundle(restored, meta.modelSha256, meta.scalersSha256);
    reject([&] { validateMetadataBoundToBundle(restored, std::string(64, 'c'), meta.scalersSha256); });
    reject([&] { validateMetadataBoundToBundle(restored, meta.modelSha256, std::string(64, 'c')); });

    std::ifstream in(path);
    const std::string valid((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    for (const auto& extra : {std::string("\n"), std::string("mode=direct_log_q\n")}) {
        { std::ofstream out(path); out << valid << extra; }
        reject([&] { reloadHydroFusionMetadata(path.string()); });
    }
    for (const auto& field : {std::string("feature_order="), std::string("feature_units=")}) {
        std::string broken = valid;
        auto end = broken.find('\n', broken.find(field));
        broken.insert(end, ",");
        { std::ofstream out(path); out << broken; }
        reject([&] { reloadHydroFusionMetadata(path.string()); });
    }
    auto bad = meta;
    bad.modelSha256 = std::string(64, 'z');
    reject([&] { exportHydroFusionMetadata(bad, path.string()); });
    bad = meta; bad.qRef = -1;
    reject([&] { exportHydroFusionMetadata(bad, path.string()); });

    writeOriginTargetMapCsv(map, {10, 11}, {11, 12}, 1);
    const auto pairs = readOriginTargetMapCsv(map);
    validateOriginTargetMap(pairs, 1);
    assert(pairs.size() == 2);
    reject([&] { validateOriginTargetPair(std::numeric_limits<int64_t>::max(), 0, 1); });
    reject([&] { validateOriginTargetPair(77015, 77016, 1); });
    reject([&] { validateOriginTargetPair(-1, 0, 1); });
    reject([&] { validateOriginTargetPair(1, 2, std::numeric_limits<int64_t>::max()); });
    reject([&] { validateOriginTargetMap({{10, 11}, {10, 11}}, 1); });
    reject([&] { validateOriginTargetMap({{11, 12}, {10, 11}}, 1); });
    reject([&] { writeOriginTargetMapCsv(map, {10, 10}, {11, 11}, 1); });
    for (const std::string row : {"10,11,", "10,11,extra", "10junk,11", "77015,77016", "10,11\n10,11"}) {
        { std::ofstream out(map); out << "origin_x,target_x\n" << row << '\n'; }
        reject([&] { readOriginTargetMapCsv(map); });
    }
    std::filesystem::remove_all(root);  // unique directory created by this test only
    std::cout << "HydroFusion metadata roundtrip, binding, strict parsing and 20 invalid cases: PASS\n";
}
