#include "experiment_exporter.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace {
std::string escapeJson(const std::string& value) {
    std::string out;
    for (const char c : value) {
        if (c == '\\' || c == '"') out.push_back('\\');
        if (c == '\n') out += "\\n";
        else out.push_back(c);
    }
    return out;
}

void requireStream(const std::ofstream& stream, const std::filesystem::path& path) {
    if (!stream) throw std::runtime_error("Unable to write experiment artifact: " + path.string());
}
}

void HydroExperimentExporter::exportRun(const std::string& outputDirectory,
                                        const std::string& experimentId,
                                        const HydroRunConfig& config,
                                        const std::map<std::string, HydroRunResult>& results) const {
    if (experimentId.empty()) throw std::invalid_argument("Experiment ID cannot be empty.");
    const std::filesystem::path root = std::filesystem::path(outputDirectory) / experimentId;
    std::filesystem::create_directories(root);

    const auto configPath = root / "experiment_config.json";
    std::ofstream configOut(configPath);
    requireStream(configOut, configPath);
    configOut << std::setprecision(17)
              << "{\n"
              << "  \"experiment_id\": \"" << escapeJson(experimentId) << "\",\n"
              << "  \"epochs\": " << config.epochs << ",\n"
              << "  \"batch_size\": " << config.batch_size << ",\n"
              << "  \"learning_rate\": " << config.learning_rate << ",\n"
              << "  \"random_seed\": " << config.random_seed << ",\n"
              << "  \"train_fraction\": " << config.train_split_ratio << ",\n"
              << "  \"validation_fraction\": " << config.validation_split_ratio << ",\n"
              << "  \"normalization\": \"" << escapeJson(config.normalization) << "\",\n"
              << "  \"hidden_layers\": \"" << escapeJson(config.hidden_layers_csv) << "\",\n"
              << "  \"input_lags\": \"" << escapeJson(config.input_lags_csv) << "\",\n"
              << "  \"lstm_sequence_length\": " << config.lstm_sequence_length << ",\n"
              << "  \"physics_profile\": \"" << escapeJson(config.pinn_physics_profile) << "\",\n"
              << "  \"data_weight\": " << config.data_weight << ",\n"
              << "  \"physics_weight\": " << config.physics_weight << ",\n"
              << "  \"hydro_package_path\": \"" << escapeJson(config.hydro_package_path) << "\",\n"
              << "  \"hydro_catchment_id\": \"" << escapeJson(config.hydro_catchment_id) << "\",\n"
              << "  \"hydro_package_profile\": \"" << escapeJson(config.hydro_package_profile) << "\"\n"
              << "}\n";

    const auto metricsPath = root / "metrics.csv";
    std::ofstream metrics(metricsPath);
    requireStream(metrics, metricsPath);
    metrics << "approach,success,final_loss,validation_mse,test_mse,rmse,mae,nse,kge,correlation,pbias,volume_error_percent,physics_loss\n";
    metrics << std::setprecision(17);
    for (const auto& entry : results) {
        const auto& r = entry.second;
        metrics << entry.first << ',' << (r.success ? 1 : 0) << ',' << r.final_loss << ',' << r.validation_mse << ','
                << r.mse << ',' << r.rmse << ',' << r.mae << ',' << r.nse << ',' << r.kge << ',' << r.correlation << ','
                << r.pbias << ',' << r.volume_error_percent << ',' << r.physics_loss << '\n';
    }

    const auto predictionsPath = root / "predictions.csv";
    std::ofstream predictions(predictionsPath);
    requireStream(predictions, predictionsPath);
    predictions << "approach,index,x,observed,predicted,residual\n" << std::setprecision(17);
    for (const auto& entry : results) {
        const auto& r = entry.second;
        const size_t n = std::min(r.x.size(), std::min(r.y_true.size(), r.y_pred.size()));
        for (size_t i = 0; i < n; ++i) {
            predictions << entry.first << ',' << i << ',' << r.x[i] << ',' << r.y_true[i] << ',' << r.y_pred[i]
                        << ',' << (r.y_pred[i] - r.y_true[i]) << '\n';
        }
    }
}
