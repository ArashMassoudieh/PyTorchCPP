#include "experiment_exporter.h"
#include "../dataset/hydro_checksum.h"

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
              << "  \"lambda_decay\": " << config.lambda_decay << ",\n"
              << "  \"optimizer\": \"" << escapeJson(config.optimizer) << "\",\n"
              << "  \"weight_decay\": " << config.weight_decay << ",\n"
              << "  \"momentum\": " << config.momentum << ",\n"
              << "  \"random_seed\": " << config.random_seed << ",\n"
              << "  \"train_fraction\": " << config.train_split_ratio << ",\n"
              << "  \"validation_fraction\": " << config.validation_split_ratio << ",\n"
              << "  \"shuffle_training\": " << (config.shuffle_training ? "true" : "false") << ",\n"
              << "  \"evaluate_metrics\": " << (config.evaluate_metrics ? "true" : "false") << ",\n"
              << "  \"normalization\": \"" << escapeJson(config.normalization) << "\",\n"
              << "  \"hidden_layers\": \"" << escapeJson(config.hidden_layers_csv) << "\",\n"
              << "  \"input_lags\": \"" << escapeJson(config.input_lags_csv) << "\",\n"
              << "  \"activation\": \"" << escapeJson(config.activation) << "\",\n"
              << "  \"use_time_lagged_ffn\": " << (config.use_time_lagged_ffn ? "true" : "false") << ",\n"
              << "  \"lstm_sequence_length\": " << config.lstm_sequence_length << ",\n"
              << "  \"physics_profile\": \"" << escapeJson(config.pinn_physics_profile) << "\",\n"
              << "  \"data_weight\": " << config.data_weight << ",\n"
              << "  \"physics_weight\": " << config.physics_weight << ",\n"
              << "  \"physics_dt\": " << config.physics_dt << ",\n"
              << "  \"forcing_gain\": " << config.forcing_gain << ",\n"
              << "  \"runoff_coeff\": " << config.runoff_coeff << ",\n"
              << "  \"storage_coeff\": " << config.storage_coeff << ",\n"
              << "  \"pinn_collocation_points\": " << config.pinn_collocation_points << ",\n"
              << "  \"use_hydro_package\": " << (config.use_hydro_package ? "true" : "false") << ",\n"
              << "  \"use_csv_data\": " << (config.use_csv_data ? "true" : "false") << ",\n"
              << "  \"csv_path\": \"" << escapeJson(config.csv_path) << "\",\n"
              << "  \"csv_x_column\": " << config.csv_x_column << ",\n"
              << "  \"csv_y_column\": " << config.csv_y_column << ",\n"
              << "  \"csv_has_header\": " << (config.csv_has_header ? "true" : "false") << ",\n"
              << "  \"synthetic_profile\": \"" << escapeJson(config.synthetic_profile) << "\",\n"
              << "  \"sample_count\": " << config.sample_count << ",\n"
              << "  \"t_start\": " << config.t_start << ",\n"
              << "  \"t_end\": " << config.t_end << ",\n"
              << "  \"hydro_package_path\": \"" << escapeJson(config.hydro_package_path) << "\",\n"
              << "  \"hydro_catchment_id\": \"" << escapeJson(config.hydro_catchment_id) << "\",\n"
              << "  \"hydro_package_profile\": \"" << escapeJson(config.hydro_package_profile) << "\",\n"
              << "  \"use_hydro_forecast_feature\": " << (config.use_hydro_forecast_feature ? "true" : "false") << ",\n"
              << "  \"hydro_forecast_variable\": \"" << escapeJson(config.hydro_forecast_variable) << "\",\n"
              << "  \"hydro_forecast_lead_hours\": " << config.hydro_forecast_lead_hours << ",\n"
              << "  \"hydro_forecast_ensemble_member\": \"" << escapeJson(config.hydro_forecast_ensemble_member) << "\"\n"
              << "}\n";

    const auto environmentPath = root / "environment.json";
    std::ofstream environment(environmentPath);
    requireStream(environment, environmentPath);
    environment << "{\n"
                << "  \"compiler\": \"" << escapeJson(__VERSION__) << "\",\n"
                << "  \"cplusplus\": " << __cplusplus << ",\n"
                << "  \"build_date\": \"" << __DATE__ << "\",\n"
                << "  \"build_time\": \"" << __TIME__ << "\"\n"
                << "}\n";

    if (config.use_hydro_package && !config.hydro_package_path.empty()) {
        const auto sourceManifest = std::filesystem::path(config.hydro_package_path) / "manifest.json";
        if (!std::filesystem::is_regular_file(sourceManifest)) {
            throw std::runtime_error("Cannot export package-backed experiment without source manifest.json.");
        }
        std::filesystem::copy_file(sourceManifest, root / "dataset_manifest.json",
                                   std::filesystem::copy_options::overwrite_existing);
        const auto provenancePath = root / "provenance.json";
        std::ofstream provenance(provenancePath);
        requireStream(provenance, provenancePath);
        provenance << "{\n"
                   << "  \"fingerprint_algorithm\": \"sha256\",\n"
                   << "  \"dataset_manifest_sha256\": \"" << sha256File(sourceManifest.string()) << "\"\n"
                   << "}\n";
    }

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

    const auto physicsPath = root / "physics_residuals.csv";
    std::ofstream physics(physicsPath);
    requireStream(physics, physicsPath);
    physics << "approach,index,split,x,physics_residual\n" << std::setprecision(17);
    for (const auto& entry : results) {
        const auto& r = entry.second;
        const size_t n = std::min(r.x.size(), r.physics_residual.size());
        for (size_t i = 0; i < n; ++i) {
            const std::string split = i < r.split.size() ? r.split[i] : "unknown";
            physics << entry.first << ',' << i << ',' << split << ',' << r.x[i] << ',' << r.physics_residual[i] << '\n';
        }
    }

    const auto predictionsPath = root / "predictions.csv";
    std::ofstream predictions(predictionsPath);
    requireStream(predictions, predictionsPath);
    predictions << "approach,index,split,x,observed,predicted,residual\n" << std::setprecision(17);
    for (const auto& entry : results) {
        const auto& r = entry.second;
        const size_t n = std::min(r.x.size(), std::min(r.y_true.size(), r.y_pred.size()));
        for (size_t i = 0; i < n; ++i) {
            const std::string split = i < r.split.size() ? r.split[i] : "unknown";
            predictions << entry.first << ',' << i << ',' << split << ',' << r.x[i] << ',' << r.y_true[i] << ',' << r.y_pred[i]
                        << ',' << (r.y_pred[i] - r.y_true[i]) << '\n';
        }
    }

    const auto historyPath = root / "training_history.csv";
    std::ofstream history(historyPath);
    requireStream(history, historyPath);
    history << "approach,epoch,training_loss\n" << std::setprecision(17);
    for (const auto& entry : results) {
        for (size_t epoch = 0; epoch < entry.second.training_loss_history.size(); ++epoch) {
            history << entry.first << ',' << (epoch + 1) << ',' << entry.second.training_loss_history[epoch] << '\n';
        }
    }
}
