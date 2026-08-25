#include "experiment_exporter.h"
#include "../dataset/hydro_checksum.h"
#include "hydro_metrics.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <utility>

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

void finalizeStream(std::ofstream& stream, const std::filesystem::path& path) {
    stream.close();
    if (!stream) throw std::runtime_error("Unable to finalize experiment artifact: " + path.string());
}

class StagedExperimentDirectory {
public:
    explicit StagedExperimentDirectory(std::filesystem::path destination)
        : destination_(std::move(destination)), staging_(destination_.string() + ".tmp") {
        if (std::filesystem::exists(destination_)) {
            throw std::runtime_error("Experiment export destination already exists: " + destination_.string());
        }
        std::filesystem::remove_all(staging_);
        std::filesystem::create_directories(staging_);
    }

    ~StagedExperimentDirectory() {
        if (!committed_) {
            std::error_code ignored;
            std::filesystem::remove_all(staging_, ignored);
        }
    }

    const std::filesystem::path& path() const { return staging_; }

    void commit() {
        std::filesystem::rename(staging_, destination_);
        committed_ = true;
    }

private:
    std::filesystem::path destination_;
    std::filesystem::path staging_;
    bool committed_ = false;
};

std::string safeFileStem(const std::string& value) {
    std::string stem;
    for (const unsigned char c : value) stem.push_back(std::isalnum(c) ? static_cast<char>(std::tolower(c)) : '_');
    if (stem.empty()) throw std::runtime_error("Approach name cannot produce an empty checkpoint filename.");
    return stem;
}

bool supportedApproach(const std::string& approach) {
    return approach == "ffn" || approach == "ffn_pinn" || approach == "pinn" ||
           approach == "lstm" || approach == "lstm_pinn";
}

void validateExportResult(const std::string& approach, const HydroRunResult& result) {
    if (!supportedApproach(approach)) throw std::invalid_argument("Unsupported Hydro export approach: " + approach);
    if (!result.success) throw std::invalid_argument("Cannot export unsuccessful Hydro result: " + approach);
    if (result.x.empty() || result.x.size() != result.y_true.size() || result.x.size() != result.y_pred.size() ||
        result.x.size() != result.split.size()) {
        throw std::invalid_argument("Hydro export prediction series are empty or misaligned for: " + approach);
    }
    for (std::size_t i = 0; i < result.x.size(); ++i) {
        if (!std::isfinite(result.x[i]) || !std::isfinite(result.y_true[i]) || !std::isfinite(result.y_pred[i]) ||
            (result.split[i] != "train" && result.split[i] != "validation" && result.split[i] != "test")) {
            throw std::invalid_argument("Hydro export contains invalid prediction data for: " + approach);
        }
    }
    if (!result.physics_residual.empty() && result.physics_residual.size() != result.x.size()) {
        throw std::invalid_argument("Hydro export physics residuals do not align for: " + approach);
    }
    if (result.model_checkpoint.empty() ||
        (result.model_checkpoint_format != "neuralnetworkwrapper-v1" &&
         result.model_checkpoint_format != "torch-module-v1")) {
        throw std::invalid_argument("Hydro export checkpoint is missing or has an unsupported format for: " + approach);
    }
    const bool recurrent = approach == "lstm" || approach == "lstm_pinn";
    const std::string expectedFormat = recurrent ? "torch-module-v1" : "neuralnetworkwrapper-v1";
    if (result.model_checkpoint_format != expectedFormat) {
        throw std::invalid_argument("Hydro export checkpoint format does not match approach: " + approach);
    }
    const auto validateScaler = [&](const HydroScalerState& state, const char* kind) {
        if (state.offset.empty() || state.offset.size() != state.scale.size() || state.shape.empty()) {
            throw std::invalid_argument("Hydro export has incomplete " + std::string(kind) + " scaler for: " + approach);
        }
        if (state.method != "none" && state.method != "standardize" && state.method != "minmax") {
            throw std::invalid_argument("Hydro export has unsupported scaler method for: " + approach);
        }
        std::size_t expected = 1;
        for (const auto extent : state.shape) {
            if (extent <= 0 || expected > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(extent)) {
                throw std::invalid_argument("Hydro export scaler shape is invalid for: " + approach);
            }
            expected *= static_cast<std::size_t>(extent);
        }
        if (expected != state.offset.size()) {
            throw std::invalid_argument("Hydro export scaler shape does not match values for: " + approach);
        }
        for (std::size_t i = 0; i < state.offset.size(); ++i) {
            if (!std::isfinite(state.offset[i]) || !std::isfinite(state.scale[i]) || state.scale[i] == 0.0) {
                throw std::invalid_argument("Hydro export scaler values are invalid for: " + approach);
            }
        }
    };
    validateScaler(result.input_scaler, "input");
    validateScaler(result.target_scaler, "target");
    if (!result.training_loss_history.empty() &&
        (result.best_epoch < 1 || static_cast<std::size_t>(result.best_epoch) > result.training_loss_history.size())) {
        throw std::invalid_argument("Hydro export best epoch is outside training history for: " + approach);
    }
}
}

void HydroExperimentExporter::exportRun(const std::string& outputDirectory,
                                        const std::string& experimentId,
                                        const HydroRunConfig& config,
                                        const std::map<std::string, HydroRunResult>& results) const {
    if (experimentId.empty()) throw std::invalid_argument("Experiment ID cannot be empty.");
    if (results.empty()) throw std::invalid_argument("Hydro experiment export requires at least one result.");
    for (const auto& entry : results) validateExportResult(entry.first, entry.second);
    auto exportResults = results;
    for (auto& entry : exportResults) {
        std::vector<double> observedTest;
        std::vector<double> predictedTest;
        for (std::size_t i = 0; i < entry.second.split.size(); ++i) {
            if (entry.second.split[i] != "test") continue;
            observedTest.push_back(entry.second.y_true[i]);
            predictedTest.push_back(entry.second.y_pred[i]);
        }
        if (observedTest.empty()) throw std::invalid_argument("Hydro export requires held-out test samples for: " + entry.first);
        populateHydroMetrics(entry.second, observedTest, predictedTest);
        populateHydroPeakMetrics(entry.second);
        if (!entry.second.physics_residual.empty()) populateHydroPhysicsResidualMetrics(entry.second);
    }
    StagedExperimentDirectory staged(std::filesystem::path(outputDirectory) / experimentId);
    const std::filesystem::path& root = staged.path();

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
    metrics << "approach,success,final_loss,validation_mse,test_mse,rmse,mae,nse,kge,correlation,pbias,volume_error_percent,peak_timing_error,peak_magnitude_error_percent,high_flow_rmse,low_flow_rmse,physics_residual_mean,physics_residual_rmse,cumulative_physics_residual,physics_loss\n";
    metrics << std::setprecision(17);
    for (const auto& entry : exportResults) {
        const auto& r = entry.second;
        metrics << entry.first << ',' << (r.success ? 1 : 0) << ',' << r.final_loss << ',' << r.validation_mse << ','
                << r.mse << ',' << r.rmse << ',' << r.mae << ',' << r.nse << ',' << r.kge << ',' << r.correlation << ','
                << r.pbias << ',' << r.volume_error_percent << ',' << r.peak_timing_error << ','
                << r.peak_magnitude_error_percent << ',' << r.high_flow_rmse << ',' << r.low_flow_rmse << ','
                << r.physics_residual_mean << ',' << r.physics_residual_rmse << ','
                << r.cumulative_physics_residual << ',' << r.physics_loss << '\n';
    }

    const auto modelsDirectory = root / "models";
    std::filesystem::create_directories(modelsDirectory);
    const auto modelManifestPath = root / "models.csv";
    std::ofstream modelManifest(modelManifestPath);
    requireStream(modelManifest, modelManifestPath);
    modelManifest << "approach,file,format,size_bytes,sha256\n";
    for (const auto& entry : exportResults) {
        if (entry.second.model_checkpoint.empty()) continue;
        const std::string filename = safeFileStem(entry.first) + ".pt";
        const auto modelPath = modelsDirectory / filename;
        std::ofstream modelFile(modelPath, std::ios::binary);
        requireStream(modelFile, modelPath);
        modelFile.write(reinterpret_cast<const char*>(entry.second.model_checkpoint.data()),
                        static_cast<std::streamsize>(entry.second.model_checkpoint.size()));
        finalizeStream(modelFile, modelPath);
        modelManifest << entry.first << ",models/" << filename << ',' << entry.second.model_checkpoint_format << ','
                      << entry.second.model_checkpoint.size() << ',' << sha256File(modelPath.string()) << '\n';
    }

    const auto scalersPath = root / "scalers.csv";
    std::ofstream scalers(scalersPath);
    requireStream(scalers, scalersPath);
    scalers << "approach,kind,index,method,shape,offset,scale\n" << std::setprecision(17);
    for (const auto& entry : exportResults) {
        const auto writeState = [&](const char* kind, const HydroScalerState& state) {
            for (std::size_t i = 0; i < state.offset.size(); ++i) {
                scalers << entry.first << ',' << kind << ',' << i << ',' << state.method << ",\"";
                for (std::size_t dimension = 0; dimension < state.shape.size(); ++dimension) {
                    if (dimension) scalers << ';';
                    scalers << state.shape[dimension];
                }
                scalers << "\"," << state.offset[i] << ',' << state.scale.at(i) << '\n';
            }
        };
        writeState("input", entry.second.input_scaler);
        writeState("target", entry.second.target_scaler);
    }

    const auto physicsPath = root / "physics_residuals.csv";
    std::ofstream physics(physicsPath);
    requireStream(physics, physicsPath);
    physics << "approach,index,split,x,physics_residual\n" << std::setprecision(17);
    for (const auto& entry : exportResults) {
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
    for (const auto& entry : exportResults) {
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
    history << "approach,epoch,training_loss,validation_loss,selected_checkpoint\n" << std::setprecision(17);
    for (const auto& entry : exportResults) {
        for (size_t epoch = 0; epoch < entry.second.training_loss_history.size(); ++epoch) {
            const double validationLoss = epoch < entry.second.validation_loss_history.size()
                ? entry.second.validation_loss_history[epoch] : std::numeric_limits<double>::quiet_NaN();
            history << entry.first << ',' << (epoch + 1) << ',' << entry.second.training_loss_history[epoch] << ','
                    << validationLoss << ',' << (entry.second.best_epoch == static_cast<int>(epoch + 1) ? 1 : 0) << '\n';
        }
    }
    finalizeStream(configOut, configPath);
    finalizeStream(environment, environmentPath);
    finalizeStream(metrics, metricsPath);
    finalizeStream(modelManifest, modelManifestPath);
    finalizeStream(scalers, scalersPath);
    finalizeStream(physics, physicsPath);
    finalizeStream(predictions, predictionsPath);
    finalizeStream(history, historyPath);
    staged.commit();
}
