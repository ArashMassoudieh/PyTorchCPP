#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

inline std::filesystem::path temporaryHydroCheckpointPath(const std::string& prefix) {
    const auto stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    const auto thread = std::hash<std::thread::id>{}(std::this_thread::get_id());
    return std::filesystem::temp_directory_path() /
        (prefix + "_" + std::to_string(stamp) + "_" + std::to_string(thread) + ".pt");
}

inline std::vector<std::uint8_t> readHydroCheckpoint(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Unable to read model checkpoint: " + path.string());
    return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}
