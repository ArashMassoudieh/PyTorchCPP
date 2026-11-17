#pragma once

// Pure parsing/validation logic for verify_stageb_checkpoint_reproduction.cpp,
// extracted to a header (no torch dependency) so it can be exercised by a
// lightweight standalone test binary without linking libtorch.
//
// Fail-closed guards added per independent review (2026-11-18, checkpoint69):
// readExistingPredictions previously parsed x/observed/predicted with no
// validation at all -- a non-finite, non-integer, negative, sealed
// (x>=77016), or duplicate x would have been silently accepted. Every row's
// x is now validated BEFORE the approach filter (mirroring
// compute_stageb_matched_support_metrics.py's read_split_rows_fail_closed
// discipline: a corrupted row belonging to a different approach must not be
// silently skipped past these checks), and the header line is verified
// against the exact expected schema rather than unconditionally discarded.

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

inline constexpr double kReservedPeriodXCutoff = 77016.0;
inline const char* kExpectedPredictionsHeader = "approach,index,split,x,observed,predicted,residual";

struct ExistingRow {
    double x = 0.0;
    double observed = 0.0;
    std::string split;
    double predicted = 0.0;
};

inline std::string readFile(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open " + path);
    return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

inline std::vector<ExistingRow> readExistingPredictions(const std::string& predictionsCsvPath,
                                                         const std::string& approach) {
    std::ifstream f(predictionsCsvPath);
    if (!f) throw std::runtime_error("Cannot open " + predictionsCsvPath);
    std::string header;
    if (!std::getline(f, header)) {
        throw std::runtime_error(predictionsCsvPath + ": empty file, no header row");
    }
    // Trailing \r tolerance (CRLF), but the schema itself must match exactly.
    while (!header.empty() && (header.back() == '\r' || header.back() == '\n')) header.pop_back();
    if (header != kExpectedPredictionsHeader) {
        throw std::runtime_error(predictionsCsvPath + ": header schema mismatch -- expected '" +
                                  std::string(kExpectedPredictionsHeader) + "', got '" + header + "'");
    }

    std::vector<ExistingRow> rows;
    std::map<double, bool> seenXForApproach;
    std::string line;
    std::size_t lineNo = 1;
    while (std::getline(f, line)) {
        ++lineNo;
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string rowApproach, indexTok, split, xTok, observedTok, predictedTok, residualTok;
        if (!std::getline(ss, rowApproach, ',') || !std::getline(ss, indexTok, ',') ||
            !std::getline(ss, split, ',') || !std::getline(ss, xTok, ',') ||
            !std::getline(ss, observedTok, ',') || !std::getline(ss, predictedTok, ',') ||
            !std::getline(ss, residualTok, ',')) {
            throw std::runtime_error(predictionsCsvPath + ": line " + std::to_string(lineNo) +
                                      " does not have the expected 7 comma-separated fields");
        }

        double x;
        try {
            x = std::stod(xTok);
        } catch (const std::exception&) {
            throw std::runtime_error(predictionsCsvPath + ": line " + std::to_string(lineNo) +
                                      " has a non-numeric x=" + xTok);
        }

        // Guards apply to EVERY row, before the approach filter -- a
        // corrupted/sealed/negative row belonging to a DIFFERENT approach
        // sharing this same file must not be silently skipped past these
        // checks just because it is not the approach being verified.
        if (!std::isfinite(x)) {
            throw std::runtime_error(predictionsCsvPath + ": line " + std::to_string(lineNo) +
                                      " has a non-finite x=" + xTok + " (approach=" + rowApproach + ")");
        }
        if (x != std::trunc(x)) {
            throw std::runtime_error(predictionsCsvPath + ": line " + std::to_string(lineNo) +
                                      " has a non-integer x=" + xTok + " (approach=" + rowApproach + ")");
        }
        if (x < 0.0) {
            throw std::runtime_error(predictionsCsvPath + ": line " + std::to_string(lineNo) +
                                      " has a negative x=" + xTok + " (approach=" + rowApproach + ")");
        }
        if (x >= kReservedPeriodXCutoff) {
            throw std::runtime_error(predictionsCsvPath + ": line " + std::to_string(lineNo) +
                                      " has x=" + xTok + " at or past the reserved-period seal (" +
                                      std::to_string(kReservedPeriodXCutoff) + "), approach=" + rowApproach +
                                      " -- refusing to process regardless of approach");
        }

        if (rowApproach != approach) continue;  // this file may hold other approaches' rows too

        if (seenXForApproach.count(x)) {
            throw std::runtime_error(predictionsCsvPath + ": duplicate x=" + xTok +
                                      " for approach " + approach + " (line " + std::to_string(lineNo) + ")");
        }
        seenXForApproach[x] = true;

        ExistingRow row;
        row.split = split;
        row.x = x;
        try {
            row.observed = std::stod(observedTok);
            row.predicted = std::stod(predictedTok);
        } catch (const std::exception&) {
            throw std::runtime_error(predictionsCsvPath + ": line " + std::to_string(lineNo) +
                                      " has a non-numeric observed/predicted value");
        }
        if (!std::isfinite(row.observed) || !std::isfinite(row.predicted)) {
            throw std::runtime_error(predictionsCsvPath + ": line " + std::to_string(lineNo) +
                                      " has a non-finite observed/predicted value (x=" + xTok + ")");
        }
        rows.push_back(row);
    }
    if (rows.empty()) {
        throw std::runtime_error(predictionsCsvPath + ": no rows found for approach " + approach);
    }
    return rows;
}

// Mirrors experiment_loader.cpp's own integer-field regex convention
// exactly, for the fields this tool needs out of JSON sidecars/configs.
inline int64_t extractInt64(const std::string& json, const std::string& key) {
    const std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*(-?(?:0|[1-9][0-9]*))\\s*[,}]");
    std::smatch match;
    if (!std::regex_search(json, match, pattern)) {
        throw std::runtime_error("Missing integer field: " + key);
    }
    return std::stoll(match[1].str());
}

inline std::string extractString(const std::string& json, const std::string& key) {
    const std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*\\\"((?:\\\\.|[^\\\"])*)\\\"\\s*[,}]");
    std::smatch match;
    if (!std::regex_search(json, match, pattern)) {
        throw std::runtime_error("Missing string field: " + key);
    }
    return match[1].str();
}

// Parses a 'ROOT: <dir>' + '<sha256>  <path>' frozen-manifest-hash file
// (the same format STAGED_MANIFEST_HASHES.txt uses; see
// compute_stageb_matched_support_metrics.py's _parse_frozen_manifest_hashes
// for the Python-side counterpart this mirrors) into
// {resolved_absolute_path: expected_sha256}.
inline std::map<std::string, std::string> parseFrozenManifestHashes(const std::string& path) {
    const std::string content = readFile(path);
    std::istringstream stream(content);
    std::string line;
    std::vector<std::string> lines;
    while (std::getline(stream, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
        if (!line.empty() && line[0] != '#') lines.push_back(line);
    }
    if (lines.empty() || lines[0].rfind("ROOT:", 0) != 0) {
        throw std::runtime_error(path + " must start with a 'ROOT: <dir>' line");
    }
    std::filesystem::path root = std::filesystem::path(lines[0].substr(5)).lexically_normal();
    // Trim leading whitespace left by "ROOT:" prefix removal.
    {
        std::string rootStr = lines[0].substr(5);
        std::size_t start = rootStr.find_first_not_of(" \t");
        root = std::filesystem::path(start == std::string::npos ? "" : rootStr.substr(start));
    }
    root = std::filesystem::weakly_canonical(root);

    std::map<std::string, std::string> expected;
    for (std::size_t i = 1; i < lines.size(); ++i) {
        const auto sep = lines[i].find("  ");
        if (sep == std::string::npos) continue;
        const std::string digest = lines[i].substr(0, sep);
        const std::string rawPath = lines[i].substr(sep + 2);
        std::filesystem::path candidate(rawPath);
        if (!candidate.is_absolute()) candidate = root / candidate;
        expected[std::filesystem::weakly_canonical(candidate).string()] = digest;
    }
    return expected;
}
