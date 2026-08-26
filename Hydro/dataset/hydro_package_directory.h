#pragma once

#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

/** Resolves a package root or an unambiguous parent containing one package. */
inline std::string resolveHydroPackageDirectory(const std::string& selectedDirectory) {
    std::filesystem::path root(selectedDirectory);
    if (!std::filesystem::is_directory(root)) {
        throw std::runtime_error("Hydro package directory does not exist: " + selectedDirectory);
    }
    if (!std::filesystem::is_regular_file(root / "manifest.json")) {
        std::vector<std::filesystem::path> candidates;
        for (const auto& entry : std::filesystem::directory_iterator(root)) {
            if (entry.is_directory() && std::filesystem::is_regular_file(entry.path() / "manifest.json")) {
                candidates.push_back(entry.path());
            }
        }
        if (candidates.size() != 1) {
            throw std::runtime_error("Selected Hydro package directory '" + selectedDirectory +
                                     "' is missing manifest.json and contains " +
                                     std::to_string(candidates.size()) +
                                     " immediate child packages; select a package root.");
        }
        root = candidates.front();
    }
    return root.string();
}
