#pragma once

#include <string>

/**
 * @file ddrr_loader.h
 * @brief Loader stub for DDRR-style hydrology datasets.
 */

/**
 * @brief Reads dataset resources used by HydroPINN experiments.
 */
class DDRRLoader {
public:
    /**
     * @brief Load dataset content from a file path.
     * @param path Input file path.
     * @return True if the input path is considered valid by the loader.
     */
    bool load(const std::string& path);
};
