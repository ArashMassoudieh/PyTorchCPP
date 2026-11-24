#include "../evaluation/experiment_exporter.h"
#include "../evaluation/experiment_loader.h"
#include "../dataset/gistohq_package_adapter.h"
#include "../models/ffn_wrapper.h"
#include "../models/ffn_pinn_wrapper.h"
#include "../models/ffn_reservoir_pinn_wrapper.h"
#include "../models/pinn_wrapper.h"
#include "../models/lstm_wrapper.h"
#include "../models/lstm_pinn_wrapper.h"
#include "../models/lstm_regional_pretrain.h"
#include "../models/ffn_hydrofusion_wrapper.h"
#include "../models/hydrofusion_forecast_wrapper.h"
#include "../models/hydrofusion_forecast_modes.h"
#include "../experiments/gistohq_sligo/hydrofusion_metadata_sidecar.h"
#include "../models/hydrofusion_context_schema.h"
#include "../models/hydrofusion_preissue_rain_runtime.h"
#include "../dataset/hydro_checksum.h"

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

std::string csvCell(const std::string& value) {
    if (value.find_first_of(",\"\n\r") == std::string::npos) return value;
    std::string escaped;
    escaped.reserve(value.size() + 2);
    escaped.push_back('"');
    for (char c : value) {
        if (c == '"') escaped.push_back('"');
        escaped.push_back(c);
    }
    escaped.push_back('"');
    return escaped;
}

bool isPhysicsMode(const std::string& mode) {
    return mode == "ffn_pinn" || mode == "lstm_pinn" || mode == "pinn";
}

// Dispatch mode (job.mode) and export/manifest approach label are not the
// same thing for multiscale jobs: training can only be dispatched via mode
// "lstm_pinn" (the CLI's fixed whitelist has no separate "lstm_multiscale"
// mode), but experiment_exporter.cpp's checkpointFormatMatchesApproach and
// artifact_loader.cpp's reload-time format check both require the approach
// label "lstm_multiscale" to accept a "torch-module-v1-multiscale"
// checkpoint -- confirmed by a real export failure ("Hydro export
// checkpoint is missing or has an unsupported format for: lstm_pinn") on
// the first real multiscale-screen dispatch. Matched-control jobs
// (multiscale_matched_control=true) train a plain HydroLSTM
// ("torch-module-v1") and are unaffected, so they keep the "lstm_pinn"
// label used everywhere else.
std::string exportApproachFor(const std::string& mode, const HydroRunConfig& config) {
    if (mode == "lstm_pinn" && config.pinn_physics_profile == "lstm_shared_scale_ablation_zero" &&
        config.multiscale_daily_history_days > 0 && !config.multiscale_matched_control) {
        return "lstm_multiscale";
    }
    return mode;
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
    if (!config.regional_training_basin_list_path.empty()) {
        fs::path p(config.regional_training_basin_list_path);
        if (p.is_relative()) p = repository_root / p;
        config.regional_training_basin_list_path = fs::weakly_canonical(p).string();
    }
    if (!config.regional_validation_basin_list_path.empty()) {
        fs::path p(config.regional_validation_basin_list_path);
        if (p.is_relative()) p = repository_root / p;
        config.regional_validation_basin_list_path = fs::weakly_canonical(p).string();
    }
}

void preparePhysicsConfig(const std::string& mode, HydroRunConfig& config) {
    if (!isPhysicsMode(mode)) return;

    const bool gisToOhq = config.use_hydro_package && !config.hydro_package_path.empty() &&
                          isGisToOhqHydroPinnExport(config.hydro_package_path);
    const bool processAwareHybrid =
        (mode == "lstm_pinn" && config.pinn_physics_profile == "two_reservoir_hybrid") ||
        (mode == "lstm_pinn" && config.pinn_physics_profile == "lstm_residual_pinn_hybrid") ||
        (mode == "lstm_pinn" && config.pinn_physics_profile == "lstm_shared_scale_ablation_physics") ||
        (mode == "lstm_pinn" && config.pinn_physics_profile == "lstm_shared_scale_ablation_urban") ||
        (mode == "lstm_pinn" && config.pinn_physics_profile == "lstm_shared_scale_ablation_zero") ||
        (mode == "ffn_pinn" && config.pinn_physics_profile == "ffn_two_reservoir_hybrid") ||
        (mode == "ffn_pinn" && config.pinn_physics_profile == "ffn_residual_pinn_hybrid") ||
        (mode == "pinn" && config.pinn_physics_profile == "pinn_two_reservoir_hybrid");

    if (gisToOhq && processAwareHybrid) {
        // The process-aware FFN/LSTM/PINN hybrids intentionally use the GIStoOHQ
        // forcing-only physical layout, but they must retain their two-reservoir
        // routing profile rather than being rewritten to the legacy single
        // linear-reservoir benchmark.
        config.use_latent_storage_physics = true; // legacy name: forcing-only layout selector
    } else if (gisToOhq) {
        // GIStoOHQ exports do not contain an independently observed storage
        // state, so legacy PINN paths use the reduced forcing-only reservoir.
        config.use_latent_storage_physics = true;
        if (config.storage_coeff > 0.0) config.latent_storage_recession_per_hour = config.storage_coeff;
        config.pinn_physics_profile = "linear_reservoir";
        config.lambda_decay = config.latent_storage_recession_per_hour;
        config.forcing_gain = config.latent_storage_recession_per_hour;
    } else if (config.pinn_physics_profile == "linear_reservoir") {
        config.use_latent_storage_physics = true;
        const double k = config.storage_coeff > 0.0
            ? config.storage_coeff
            : (config.latent_storage_recession_per_hour > 0.0
               ? config.latent_storage_recession_per_hour
               : config.lambda_decay);
        config.latent_storage_recession_per_hour = k;
        config.lambda_decay = k;
        config.forcing_gain = k;
    }

    if (config.pinn_physics_profile == "linear_reservoir" && config.normalization != "none") {
        std::cout << "[batch] reduced-reservoir physics requires physical-unit residuals; overriding normalization="
                  << config.normalization << " -> none\n";
        config.normalization = "none";
    }
    if (mode == "ffn_pinn" && config.pinn_physics_profile == "linear_reservoir") {
        config.use_time_lagged_ffn = false;
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
                                     "; expected: <ffn|ffn_pinn|lstm|lstm_pinn|pinn|ffn_hydrofusion> <config.json>; line='" + stripped + "'");
        }
        std::string extra;
        if (parser >> extra) {
            throw std::runtime_error("Unexpected extra token in Hydro batch entry at line " +
                                     std::to_string(line_number) + "; line='" + stripped + "'");
        }
        if (mode != "ffn" && mode != "ffn_pinn" && mode != "lstm" &&
            mode != "lstm_pinn" && mode != "pinn" && mode != "lstm_regional_pretrain" &&
            mode != "ffn_hydrofusion" && !hydrofusion::isExtendedForecast(mode)) {
            throw std::runtime_error("Unsupported Hydro batch mode at line " + std::to_string(line_number) + ": " + mode);
        }
        fs::path config_path(config);
        if (config_path.is_relative()) config_path = path.parent_path() / config_path;
        config_path = fs::weakly_canonical(config_path);
        if (!fs::exists(config_path)) {
            throw std::runtime_error("Hydro batch config does not exist at line " + std::to_string(line_number) + ": " + config_path.string());
        }
        jobs.push_back({mode, config_path});
    }
    if (jobs.empty()) throw std::runtime_error("Hydro batch file contains no jobs.");
    return jobs;
}

HydroRunResult runJob(const std::string& mode, const HydroRunConfig& config) {
    if(!hydrofusion::isExtendedForecast(mode)&&(config.hydrofusion_preissue_rain_recipe!="disabled"||!config.hydrofusion_preissue_rain_path.empty()||!config.hydrofusion_preissue_rain_sha256.empty()||!config.hydrofusion_preissue_rain_provenance_sha256.empty()||config.hydrofusion_development_only||config.hydrofusion_context_recipe!="legacy20"||config.hydrofusion_history_hours!="0"||config.hydrofusion_objective!="log_mse_v1"||config.hydrofusion_ffn_activation!="tanh"||config.pinn_physics_profile=="causal_two_reservoir_mean_v2"))
        throw std::invalid_argument("Forecast v2 options require an extended forecast approach.");
    if (mode == "ffn") { FFNWrapper runner; return runner.train(config); }
    if (mode == "ffn_pinn") { FFNReservoirPINNWrapper runner; return runner.train(config); }
    if (mode == "lstm") { LSTMWrapper runner; return runner.train(config); }
    if (mode == "lstm_pinn") { LSTMPINNWrapper runner; return runner.train(config); }
    if (mode == "pinn") { PINNWrapper runner; return runner.train(config); }
    if (mode == "lstm_regional_pretrain") { return trainLstmRegionalPretrainFromConfig(config); }
    if (hydrofusion::isExtendedForecast(mode)) { HydroFusionForecastWrapper runner; return runner.train(config, mode); }
    if (mode == "ffn_hydrofusion") { FFNHydroFusionWrapper runner; return runner.train(config); }
    throw std::runtime_error("Unsupported Hydro batch mode: " + mode);
}

const std::string& actualActivation(const std::string& mode,const HydroRunConfig& config){
    return hydrofusion::isExtendedForecast(mode)&&!hydrofusion::isRecurrentForecast(mode)&&mode!="pinn_causal_forecast"?config.hydrofusion_ffn_activation:config.activation;
}
void printHyperparameters(const std::string& mode, const HydroRunConfig& config) {
    if (mode == "lstm" || mode == "lstm_pinn") std::cout << " sequence_length=" << config.lstm_sequence_length;
    else if (mode == "ffn" || mode == "ffn_pinn") std::cout << " input_lags=" << config.input_lags_csv;
    std::cout << " hidden_layers=" << config.hidden_layers_csv
              << " activation=" << actualActivation(mode,config)
              << " learning_rate=" << config.learning_rate
              << " batch_size=" << config.batch_size
              << " seed=" << config.random_seed
              << " normalization=" << config.normalization;
    if (isPhysicsMode(mode)) {
        std::cout << " physics_profile=" << config.pinn_physics_profile
                  << " data_weight=" << config.data_weight
                  << " physics_weight=" << config.physics_weight
                  << " latent_storage=" << (config.use_latent_storage_physics ? "yes" : "no")
                  << " latent_recession_per_hour=" << config.latent_storage_recession_per_hour;
        if (mode == "lstm_pinn" && config.pinn_physics_profile == "two_reservoir_hybrid") {
            std::cout << " fast_k=" << config.storage_coeff
                      << " slow_k=" << config.lambda_decay
                      << " routing_alpha=" << config.runoff_coeff
                      << " predictor_scaling=train_only_standardize"
                      << " routing_units=physical";
        }
    }
}

void printMetrics(const std::string& experiment_id, const std::string& mode,
                  const HydroRunConfig& config, const HydroRunResult& result) {
    std::cout << std::setprecision(9)
              << "[batch] experiment=" << experiment_id
              << " mode=" << mode
              << " success=" << (result.success ? "yes" : "no");
    printHyperparameters(mode, config);
    std::cout << (config.hydrofusion_development_only?" validation_mse=":" test_mse=") << result.mse
              << " rmse=" << result.rmse
              << " mae=" << result.mae
              << " r2=" << result.r2
              << " nse=" << result.nse
              << " kge=" << result.kge
              << " pbias=" << result.pbias
              << " physics_loss=" << result.physics_loss
              << " physics_residual_rmse=" << result.physics_residual_rmse << '\n';
}

const char* summaryHeader() {
    return "experiment_id,mode,lstm_sequence_length,input_lags,hidden_layers,activation,learning_rate,batch_size,random_seed,normalization,physics_profile,storage_coeff,lambda_decay,runoff_coeff,pinn_routing_lag_hours,pinn_flow_exponent,lstm_bias_weight,physical_error_weight,high_flow_loss_weight,high_flow_loss_quantile,lstm_dropout,antecedent_index_halflife_hours,use_mrms_spatial_cv_feature,data_weight,physics_weight,latent_storage,latent_recession_per_hour,success,final_loss,validation_mse,test_mse,rmse,mae,r2,nse,kge,correlation,pbias,volume_error_percent,peak_timing_error,peak_magnitude_error_percent,high_flow_rmse,low_flow_rmse,physics_loss,physics_residual_mean,physics_residual_rmse,cumulative_physics_residual";
}

void prepareSummaryFile(const fs::path& summary_path) {
    if (fs::exists(summary_path) && fs::file_size(summary_path) > 0) {
        std::ifstream in(summary_path);
        std::string header;
        std::getline(in, header);
        const std::string expected = summaryHeader();
        if (header != expected) {
            fs::path backup = summary_path.parent_path() / "batch_summary.pre_physics.csv";
            for (int suffix = 1; fs::exists(backup); ++suffix) {
                backup = summary_path.parent_path() / ("batch_summary.pre_physics." + std::to_string(suffix) + ".csv");
            }
            fs::rename(summary_path, backup);
            std::cout << "[batch] archived legacy summary=" << backup << '\n';
        } else return;
    }
    std::ofstream out(summary_path, std::ios::trunc);
    out << summaryHeader() << '\n';
}

void appendSummary(const fs::path& summary_path, const std::string& experiment_id,
                   const std::string& mode, const HydroRunConfig& config, const HydroRunResult& r) {
    std::ofstream out(summary_path, std::ios::app);
    out << std::setprecision(12)
        << csvCell(experiment_id) << ',' << csvCell(mode) << ','
        << config.lstm_sequence_length << ',' << csvCell(config.input_lags_csv) << ','
        << csvCell(config.hidden_layers_csv) << ',' << csvCell(actualActivation(mode,config)) << ','
        << config.learning_rate << ',' << config.batch_size << ',' << config.random_seed << ','
        << csvCell(config.normalization) << ',' << csvCell(config.pinn_physics_profile) << ','
        << config.storage_coeff << ',' << config.lambda_decay << ',' << config.runoff_coeff << ','
        << config.pinn_routing_lag_hours << ',' << config.pinn_flow_exponent << ','
        << config.lstm_bias_weight << ',' << config.physical_error_weight << ','
        << config.high_flow_loss_weight << ',' << config.high_flow_loss_quantile << ','
        << config.lstm_dropout << ',' << config.antecedent_index_halflife_hours << ','
        << (config.use_mrms_spatial_cv_feature ? 1 : 0) << ','
        << config.data_weight << ',' << config.physics_weight << ','
        << (config.use_latent_storage_physics ? "true" : "false") << ','
        << config.latent_storage_recession_per_hour << ',' << (r.success ? "true" : "false") << ','
        << r.final_loss << ',' << r.validation_mse << ',' << (config.hydrofusion_development_only?std::numeric_limits<double>::quiet_NaN():r.mse) << ',' << r.rmse << ',' << r.mae << ','
        << r.r2 << ',' << r.nse << ',' << r.kge << ',' << r.correlation << ',' << r.pbias << ','
        << r.volume_error_percent << ',' << r.peak_timing_error << ',' << r.peak_magnitude_error_percent << ','
        << r.high_flow_rmse << ',' << r.low_flow_rmse << ',' << r.physics_loss << ','
        << r.physics_residual_mean << ',' << r.physics_residual_rmse << ',' << r.cumulative_physics_residual << '\n';
}

// ffn_hydrofusion only: writes the HydroFusion metadata + origin/target-map
// sidecars documented in hydrofusion_metadata_sidecar.h, read back from the
// real exporter's own just-written models.csv/scalers.csv. Register both
// metadata files in the managed manifest: completeness rejects UNLISTED
// additions, not metadata files as a category.
void exportHydroFusionSidecarForBatchJob(const fs::path& output_root, const std::string& experiment_id,
                                         const std::string& approach, const HydroRunResult& result, const HydroRunConfig& config) {
    const fs::path bundleDir = output_root / experiment_id;
    const std::string modelSha256 = hydrofusion::readModelSha256FromModelsCsv(bundleDir, approach);
    const std::string scalersSha256 = sha256File((bundleDir / "scalers.csv").string());

    hydrofusion::TargetMode mode;
    if (result.hydrofusion_target_mode == "direct_log_q") mode = hydrofusion::TargetMode::DirectLogQ;
    else if (result.hydrofusion_target_mode == "log_residual") mode = hydrofusion::TargetMode::LogResidual;
    else throw std::runtime_error("exportHydroFusionSidecarForBatchJob: invalid hydrofusion_target_mode on result.");

    hydrofusion::HydroFusionMetadata metadata;
    metadata.mode = mode;
    metadata.qRef = result.hydrofusion_q_ref;
    metadata.cutoffs = hydrofusion::SplitCutoffs{result.hydrofusion_train_end_x, result.hydrofusion_validation_end_x,
                                                   result.hydrofusion_test_end_x};
    metadata.modelSha256 = modelSha256;
    metadata.scalersSha256 = scalersSha256;

    const fs::path metadataPath = bundleDir / "hydrofusion_metadata.txt";
    hydrofusion::exportHydroFusionMetadata(metadata, metadataPath.string());

    // result.x already carries target_x (set by FFNHydroFusionWrapper::train);
    // result.hydrofusion_test_origin_x carries the matching origin_x, same
    // row order -- see hydro_run_types.h's doc comment on both fields.
    std::vector<int64_t> targetX;
    targetX.reserve(result.x.size());
    for (const double value : result.x) targetX.push_back(static_cast<int64_t>(value));
    const fs::path mapPath = bundleDir / "hydrofusion_origin_target_map.csv";
    hydrofusion::writeOriginTargetMapCsv(mapPath, result.hydrofusion_test_origin_x, targetX, metadata.horizonHours);

    std::vector<fs::path> managedPaths{metadataPath,mapPath};
    if(hydrofusion::forecastRecipeV4(config)){
        const auto path=bundleDir/"hydrofusion_context_schema.txt";std::ofstream schema(path);
        schema<<hydrofusion::forecastContextSchema(config);schema.close();
        if(!schema)throw std::runtime_error("Could not export context schema.");
        managedPaths.push_back(path);
    }
    if(hydrofusion::forecastRecipeV5(config)){
        (void)hydrofusion::loadBoundPreissueRain(config);
        const fs::path source(config.hydrofusion_preissue_rain_path);
        for(const auto& pair:std::vector<std::pair<fs::path,fs::path>>{{source,bundleDir/"preissue_rain.csv"},{source.parent_path()/"preissue_rain_provenance.json",bundleDir/"preissue_rain_provenance.json"}}){fs::copy_file(pair.first,pair.second,fs::copy_options::none);managedPaths.push_back(pair.second);}
        const auto path=bundleDir/"hydrofusion_preissue_rain_schema.txt";std::ofstream schema(path);schema<<hydrofusion::rainSchema(config);schema.close();if(!schema)throw std::runtime_error("Could not export preissue rain schema.");managedPaths.push_back(path);
    }
    if(!config.hydrofusion_synthetic_verification && !config.hydrofusion_native_manifest_path.empty()){
        const fs::path source(config.hydrofusion_native_manifest_path);
        if(sha256File(source.string())!=config.hydrofusion_native_manifest_sha256)throw std::runtime_error("Native source manifest changed during fit.");
        for(const auto& pair:std::vector<std::pair<fs::path,fs::path>>{{source,bundleDir/"native_input_manifest.json"},{source.parent_path()/"intervals.csv",bundleDir/"intervals.csv"}}){
            fs::copy_file(pair.first,pair.second,fs::copy_options::none);managedPaths.push_back(pair.second);
        }
    }
    // Replace the manifest only after the registered metadata files are closed.
    // A failed/partial write leaves an invalid bundle, never a successful reload.
    const auto manifestPath = bundleDir / "artifact_manifest.csv";
    std::ifstream previous(manifestPath, std::ios::binary);
    if (!previous) throw std::runtime_error("Missing managed bundle manifest.");
    const std::string oldManifest((std::istreambuf_iterator<char>(previous)), std::istreambuf_iterator<char>());
    const auto pending = bundleDir / "artifact_manifest.pending";
    {
        std::ofstream next(pending, std::ios::binary);
        next << oldManifest;
        for (const auto& path : managedPaths) {
            next << path.filename().generic_string() << ',' << fs::file_size(path) << ',' << sha256File(path.string()) << '\n';
        }
        next.close();
        if (!next) throw std::runtime_error("Could not register HydroFusion bundle metadata.");
    }
    fs::rename(pending, manifestPath);

    std::cout << "[batch] wrote HydroFusion sidecar: " << metadataPath << '\n'
              << "[batch] wrote HydroFusion origin/target map: " << mapPath << '\n';
}

// lstm_regional_pretrain produces a reusable checkpoint, not a forecast
// series - its HydroRunResult has no x/y_true/y_pred/split (there is no
// single time axis to report for a basin-pooled run), so it cannot go
// through HydroExperimentExporter::exportRun, which requires exactly that.
// Writes the checkpoint directly plus a small per-epoch loss-history CSV
// for provenance, instead.
void exportRegionalPretrainRun(const fs::path& output_root, const std::string& experiment_id,
                               const HydroRunResult& result) {
    const fs::path dir = output_root / experiment_id;
    fs::create_directories(dir / "models");
    if (!result.success) return;
    {
        std::ofstream model_file(dir / "models" / "lstm_regional_pretrain.pt", std::ios::binary);
        if (!model_file) throw std::runtime_error("Unable to write regional pretrain checkpoint for: " + experiment_id);
        model_file.write(reinterpret_cast<const char*>(result.model_checkpoint.data()),
                          static_cast<std::streamsize>(result.model_checkpoint.size()));
    }
    std::ofstream summary(dir / "regional_pretrain_summary.csv");
    summary << "epoch,training_loss,validation_loss\n" << std::setprecision(12);
    for (std::size_t i = 0; i < result.training_loss_history.size(); ++i) {
        summary << (i + 1) << ',' << result.training_loss_history[i] << ','
                << (i < result.validation_loss_history.size()
                        ? result.validation_loss_history[i]
                        : std::numeric_limits<double>::quiet_NaN())
                << '\n';
    }
}

fs::path archiveExistingExperiment(const fs::path& output_root, const std::string& experiment_id) {
    const fs::path current = output_root / experiment_id;
    if (!fs::exists(current)) return {};
    fs::path archive = output_root / (experiment_id + ".previous");
    for (int suffix = 2; fs::exists(archive); ++suffix) {
        archive = output_root / (experiment_id + ".previous." + std::to_string(suffix));
    }
    fs::rename(current, archive);
    return archive;
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            std::cerr << "Usage: HydroBatch <batch-file> <output-directory>\n"
                      << "Batch format: one '<ffn|ffn_pinn|lstm|lstm_pinn|pinn|lstm_regional_pretrain> <config.json>' entry per line.\n";
            return 2;
        }
        torch::set_num_threads(1);
        torch::set_num_interop_threads(1);
        const fs::path batch_path = fs::weakly_canonical(fs::path(argv[1]));
        const fs::path repository_root = findRepositoryRoot(batch_path);
        const fs::path output_root = fs::absolute(fs::path(argv[2]));
        fs::create_directories(output_root);
        const fs::path summary_path = output_root / "batch_summary.csv";
        prepareSummaryFile(summary_path);
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
                preparePhysicsConfig(job.mode, config);
                std::cout << "[batch] starting experiment=" << loaded.experiment_id << " mode=" << job.mode;
                printHyperparameters(job.mode, config);
                std::cout << '\n';
                if (config.use_hydro_package) std::cout << "[batch] hydro_package_path=" << config.hydro_package_path << '\n';

                HydroRunResult result = runJob(job.mode, config);
                printMetrics(loaded.experiment_id, job.mode, config, result);

                const fs::path archived = archiveExistingExperiment(output_root, loaded.experiment_id);
                if (!archived.empty()) std::cout << "[batch] archived existing experiment=" << archived << '\n';
                if (job.mode == "lstm_regional_pretrain") {
                    // No x/y_true/y_pred/split series to validate - a
                    // basin-pooled run has no single time axis to report.
                    // See exportRegionalPretrainRun's doc comment.
                    exportRegionalPretrainRun(output_root, loaded.experiment_id, result);
                } else {
                    const std::string approach = exportApproachFor(job.mode, config);
                    std::map<std::string, HydroRunResult> results;
                    results.emplace(approach, result);
                    HydroExperimentExporter().exportRun(output_root.string(), loaded.experiment_id, config, results);
                    if (job.mode == "ffn_hydrofusion" || hydrofusion::isExtendedForecast(job.mode)) {
                        exportHydroFusionSidecarForBatchJob(output_root, loaded.experiment_id, approach, result, config);
                    }
                    appendSummary(summary_path, loaded.experiment_id, job.mode, config, result);
                }
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
