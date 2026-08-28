#include "../evaluation/experiment_exporter.h"
#include "../evaluation/experiment_loader.h"
#include "../models/ffn_wrapper.h"
#include "../models/lstm_wrapper.h"

#include <torch/torch.h>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {
struct BatchJob { std::string mode; fs::path config_path; };

std::string trim(const std::string& value) {
    const auto begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) return {};
    const auto end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

std::string stripUtf8Bom(std::string value) {
    if (value.size() >= 3 && static_cast<unsigned char>(value[0]) == 0xEF &&
        static_cast<unsigned char>(value[1]) == 0xBB && static_cast<unsigned char>(value[2]) == 0xBF) {
        value.erase(0, 3);
    }
    return value;
}

fs::path findRepositoryRoot(const fs::path& start) {
    fs::path current = fs::absolute(start);
    if (!fs::is_directory(current)) current = current.parent_path();
    while (!current.empty()) {
        if (fs::exists(current / "HydroPINN.pro") || fs::exists(current / "HydroBatch.pro")) return current;
        const fs::path parent = current.parent_path();
        if (parent == current) break;
        current = parent;
    }
    throw std::runtime_error("Unable to locate PyTorchCPP repository root from: " + start.string());
}

void resolveConfigPaths(HydroRunConfig& config, const fs::path& repository_root) {
    if (config.use_hydro_package && !config.hydro_package_path.empty()) {
        fs::path p(config.hydro_package_path);
        if (p.is_relative()) p = repository_root / p;
        config.hydro_package_path = fs::weakly_canonical(p).string();
    }
    if (config.use_csv_data && !config.csv_path.empty()) {
        fs::path p(config.csv_path);
        if (p.is_relative()) p = repository_root / p;
        config.csv_path = fs::weakly_canonical(p).string();
    }
}

std::vector<BatchJob> readBatchFile(const fs::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Unable to open Hydro batch file: " + path.string());
    std::vector<BatchJob> jobs;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (line_number == 1) line = stripUtf8Bom(line);
        const auto comment = line.find('#');
        if (comment != std::string::npos) line.erase(comment);
        const std::string stripped = trim(line);
        if (stripped.empty()) continue;
        std::istringstream parser(stripped);
        std::string mode, config;
        if (!(parser >> mode >> config)) {
            throw std::runtime_error("Invalid Hydro batch entry at line " + std::to_string(line_number) +
                                     "; expected: <ffn|lstm> <config.json>; line='" + stripped + "'");
        }
        std::string extra;
        if (parser >> extra) {
            throw std::runtime_error("Unexpected extra token in Hydro batch entry at line " +
                                     std::to_string(line_number) + "; line='" + stripped + "'");
        }
        if (mode != "ffn" && mode != "lstm") {
            throw std::runtime_error("Unsupported supervised Hydro batch mode at line " +
                                     std::to_string(line_number) + ": " + mode);
        }
        fs::path config_path(config);
        if (config_path.is_relative()) config_path = path.parent_path() / config_path;
        config_path = fs::weakly_canonical(config_path);
        if (!fs::exists(config_path)) {
            throw std::runtime_error("Hydro batch config does not exist at line " +
                                     std::to_string(line_number) + ": " + config_path.string());
        }
        jobs.push_back({mode, config_path});
    }
    if (jobs.empty()) throw std::runtime_error("Hydro batch file contains no jobs.");
    return jobs;
}

HydroRunResult runJob(const std::string& mode, const HydroRunConfig& config) {
    if (mode == "ffn") { FFNWrapper runner; return runner.train(config); }
    if (mode == "lstm") { LSTMWrapper runner; return runner.train(config); }
    throw std::runtime_error("Unsupported Hydro batch mode: " + mode);
}

void printMetrics(const std::string& experiment_id, const std::string& mode,
                  const HydroRunConfig& config, const HydroRunResult& result) {
    std::cout << std::setprecision(9)
              << "[batch] experiment=" << experiment_id
              << " mode=" << mode
              << " success=" << (result.success ? "yes" : "no")
              << " sequence_length=" << config.lstm_sequence_length
              << " normalization=" << config.normalization
              << " test_mse=" << result.mse
              << " rmse=" << result.rmse
              << " mae=" << result.mae
              << " r2=" << result.r2
              << " nse=" << result.nse
              << " kge=" << result.kge
              << " pbias=" << result.pbias << '\n';
}

void appendSummaryHeaderIfNeeded(const fs::path& summary_path) {
    if (fs::exists(summary_path) && fs::file_size(summary_path) > 0) return;
    std::ofstream out(summary_path, std::ios::app);
    out << "experiment_id,mode,lstm_sequence_length,normalization,success,final_loss,validation_mse,test_mse,rmse,mae,r2,nse,kge,correlation,pbias,volume_error_percent,peak_timing_error,peak_magnitude_error_percent,high_flow_rmse,low_flow_rmse\n";
}

void appendSummary(const fs::path& summary_path, const std::string& experiment_id,
                   const std::string& mode, const HydroRunConfig& config, const HydroRunResult& r) {
    std::ofstream out(summary_path, std::ios::app);
    out << std::setprecision(12)
        << experiment_id << ',' << mode << ',' << config.lstm_sequence_length << ',' << config.normalization << ','
        << (r.success ? "true" : "false") << ',' << r.final_loss << ',' << r.validation_mse << ',' << r.mse << ','
        << r.rmse << ',' << r.mae << ',' << r.r2 << ',' << r.nse << ',' << r.kge << ',' << r.correlation << ',' << r.pbias << ','
        << r.volume_error_percent << ',' << r.peak_timing_error << ',' << r.peak_magnitude_error_percent << ','
        << r.high_flow_rmse << ',' << r.low_flow_rmse << '\n';
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            std::cerr << "Usage: HydroBatch <batch-file> <output-directory>\n"
                      << "Batch format: one '<ffn|lstm> <config.json>' entry per line.\n";
            return 2;
        }
        torch::set_num_threads(1);
        torch::set_num_interop_threads(1);
        const fs::path batch_path = fs::weakly_canonical(fs::path(argv[1]));
        const fs::path repository_root = findRepositoryRoot(batch_path);
        const fs::path output_root = fs::absolute(fs::path(argv[2]));
        fs::create_directories(output_root);
        const fs::path summary_path = output_root / "batch_summary.csv";
        appendSummaryHeaderIfNeeded(summary_path);
        const auto jobs = readBatchFile(batch_path);
        std::cout << "[batch] loaded " << jobs.size() << " job(s) from " << batch_path << '\n';
        std::cout << "[batch] repository_root=" << repository_root << '\n';
        int failures = 0;
        std::size_t index = 0;
        for (const auto& job : jobs) {
            ++index;
            std::cout << "[batch] " << index << '/' << jobs.size() << " loading " << job.config_path << '\n';
            try {
                const auto loaded = HydroExperimentLoader().loadConfig(job.config_path.string());
                HydroRunConfig config = loaded.config;
                resolveConfigPaths(config, repository_root);
                std::cout << "[batch] starting experiment=" << loaded.experiment_id
                          << " mode=" << job.mode
                          << " sequence_length=" << config.lstm_sequence_length
                          << " normalization=" << config.normalization << '\n';
                if (config.use_hydro_package) std::cout << "[batch] hydro_package_path=" << config.hydro_package_path << '\n';
                HydroRunResult result = runJob(job.mode, config);
                printMetrics(loaded.experiment_id, job.mode, config, result);
                appendSummary(summary_path, loaded.experiment_id, job.mode, config, result);
                std::map<std::string, HydroRunResult> results;
                results.emplace(job.mode, result);
                HydroExperimentExporter().exportRun(output_root.string(), loaded.experiment_id, config, results);
                if (!result.success) ++failures;
            } catch (const std::exception& error) {
                ++failures;
                std::cerr << "[batch] job failed: " << error.what() << '\n';
            }
        }
        std::cout << "[batch] complete: jobs=" << jobs.size() << " failures=" << failures
                  << " summary=" << summary_path << '\n';
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Hydro batch error: " << error.what() << '\n';
        return 2;
    }
}
