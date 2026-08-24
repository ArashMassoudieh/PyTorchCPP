#include "../evaluation/experiment_exporter.h"
#include "../evaluation/experiment_loader.h"
#include "../evaluation/artifact_loader.h"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <iterator>

int main() {
    const std::filesystem::path output = "/tmp/hydro_experiment_export";
    std::filesystem::remove_all(output);
    HydroRunConfig config;
    config.random_seed = 7;
    config.lambda_decay = 0.25;
    config.evaluate_metrics = false;
    config.optimizer = "rmsprop";
    config.shuffle_training = false;
    const auto package = output / "source_package";
    std::filesystem::create_directories(package);
    {
        std::ofstream manifest(package / "manifest.json");
        manifest << R"({"schema_name":"hydro-observations","dataset_id":"fixture"})";
    }
    config.use_hydro_package = true;
    config.hydro_package_path = package.string();
    config.hydro_catchment_id = "watershed_\"a";
    config.use_hydro_forecast_feature = true;
    config.hydro_forecast_variable = "precipitation";
    config.hydro_forecast_lead_hours = 6.0;
    config.hydro_forecast_ensemble_member = "m01";
    HydroRunResult result;
    result.success = true;
    result.mse = 0.25;
    result.rmse = 0.5;
    result.x = {0.0, 1.0};
    result.y_true = {1.0, 2.0};
    result.y_pred = {1.5, 1.5};
    result.split = {"train", "test"};
    result.training_loss_history = {1.0, 0.5};
    result.validation_loss_history = {0.8, 0.4};
    result.best_epoch = 2;
    result.input_scaler = {"minmax", {0.0}, {2.0}, {1, 1}};
    result.target_scaler = {"standardize", {1.0}, {0.5}, {1, 1}};
    result.model_checkpoint_format = "neuralnetworkwrapper-v1";
    result.model_checkpoint = {0x01, 0x02, 0x03};
    result.physics_residual = {0.1, -0.2};
    result.peak_timing_error = 1.0;
    result.peak_magnitude_error_percent = -5.0;
    result.high_flow_rmse = 0.25;
    result.low_flow_rmse = 0.125;
    result.physics_residual_mean = -0.5;
    result.physics_residual_rmse = 0.25;
    result.cumulative_physics_residual = -0.25;
    HydroExperimentExporter().exportRun(output.string(), "run_001", config, {{"ffn", result}});
    const auto root = output / "run_001";
    assert(std::filesystem::is_regular_file(root / "experiment_config.json"));
    assert(std::filesystem::is_regular_file(root / "environment.json"));
    assert(std::filesystem::is_regular_file(root / "dataset_manifest.json"));
    assert(std::filesystem::is_regular_file(root / "provenance.json"));
    assert(std::filesystem::is_regular_file(root / "metrics.csv"));
    assert(std::filesystem::is_regular_file(root / "predictions.csv"));
    assert(std::filesystem::is_regular_file(root / "training_history.csv"));
    assert(std::filesystem::is_regular_file(root / "physics_residuals.csv"));
    assert(std::filesystem::is_regular_file(root / "scalers.csv"));
    assert(std::filesystem::is_regular_file(root / "models.csv"));
    {
        std::ifstream metrics(root / "metrics.csv");
        const std::string metricsText((std::istreambuf_iterator<char>(metrics)), std::istreambuf_iterator<char>());
        assert(metricsText.find("peak_timing_error,peak_magnitude_error_percent,high_flow_rmse,low_flow_rmse") != std::string::npos);
        assert(metricsText.find(",1,-5,0.25,0.125,-0.5,0.25,-0.25,") != std::string::npos);
    }
    assert(std::filesystem::file_size(root / "models" / "ffn.pt") == 3);
    const auto models = HydroArtifactLoader().loadModels(root.string());
    assert(models.at("ffn").bytes == result.model_checkpoint);
    assert(models.at("ffn").format == "neuralnetworkwrapper-v1");
    const auto scalerStates = HydroArtifactLoader().loadScalers(root.string());
    assert(scalerStates.at("ffn").input.method == "minmax");
    assert(scalerStates.at("ffn").input.shape == std::vector<int64_t>({1, 1}));
    assert(scalerStates.at("ffn").input.offset == std::vector<double>({0.0}));
    assert(scalerStates.at("ffn").input.scale == std::vector<double>({2.0}));
    assert(scalerStates.at("ffn").target.method == "standardize");
    assert(scalerStates.at("ffn").target.offset == std::vector<double>({1.0}));
    assert(scalerStates.at("ffn").target.scale == std::vector<double>({0.5}));
    const auto inferenceArtifacts = HydroArtifactLoader().loadForInference(root.string());
    assert(inferenceArtifacts.experiment.experiment_id == "run_001");
    assert(inferenceArtifacts.models.count("ffn") == 1);
    assert(inferenceArtifacts.scalers.count("ffn") == 1);
    const auto storedResults = HydroArtifactLoader().loadPredictions(root.string());
    assert(storedResults.at("ffn").success);
    assert(storedResults.at("ffn").split == std::vector<std::string>({"train", "test"}));
    assert(storedResults.at("ffn").y_pred == std::vector<double>({1.5, 1.5}));
    assert(storedResults.at("ffn").mse == 0.25);
    assert(storedResults.at("ffn").rmse == 0.5);
    const auto storedResiduals = HydroArtifactLoader().loadPhysicsResiduals(root.string());
    assert(storedResiduals.at("ffn") == result.physics_residual);
    std::ifstream predictions(root / "predictions.csv");
    const std::string text((std::istreambuf_iterator<char>(predictions)), std::istreambuf_iterator<char>());
    assert(text.find("ffn,0,train,0,1,1.5,0.5") != std::string::npos);
    assert(text.find("ffn,1,test,1,2,1.5,-0.5") != std::string::npos);
    std::ifstream history(root / "training_history.csv");
    const std::string historyText((std::istreambuf_iterator<char>(history)), std::istreambuf_iterator<char>());
    assert(historyText.find("ffn,2,0.5,") != std::string::npos);
    assert(historyText.find(",1\n") != std::string::npos);
    std::ifstream scalers(root / "scalers.csv");
    const std::string scalerText((std::istreambuf_iterator<char>(scalers)), std::istreambuf_iterator<char>());
    assert(scalerText.find("ffn,input,0,minmax,\"1;1\",0,2") != std::string::npos);
    std::ifstream physics(root / "physics_residuals.csv");
    const std::string physicsText((std::istreambuf_iterator<char>(physics)), std::istreambuf_iterator<char>());
    assert(physicsText.find("ffn,1,test,1,-0.2") != std::string::npos);
    std::ifstream configFile(root / "experiment_config.json");
    const std::string configText((std::istreambuf_iterator<char>(configFile)), std::istreambuf_iterator<char>());
    assert(configText.find("\"optimizer\": \"rmsprop\"") != std::string::npos);
    assert(configText.find("\"shuffle_training\": false") != std::string::npos);
    assert(configText.find("\"hydro_forecast_lead_hours\": 6") != std::string::npos);
    assert(configText.find("\"hydro_forecast_ensemble_member\": \"m01\"") != std::string::npos);
    std::ifstream provenance(root / "provenance.json");
    const std::string provenanceText((std::istreambuf_iterator<char>(provenance)), std::istreambuf_iterator<char>());
    assert(provenanceText.find("\"dataset_manifest_sha256\": \"") != std::string::npos);
    const auto loaded = HydroExperimentLoader().loadConfig((root / "experiment_config.json").string());
    assert(loaded.experiment_id == "run_001");
    assert(loaded.config.random_seed == 7);
    assert(loaded.config.lambda_decay == 0.25);
    assert(!loaded.config.evaluate_metrics);
    assert(loaded.config.optimizer == "rmsprop");
    assert(!loaded.config.shuffle_training);
    assert(loaded.config.use_hydro_forecast_feature);
    assert(loaded.config.hydro_forecast_lead_hours == 6.0);
    assert(loaded.config.hydro_forecast_ensemble_member == "m01");
    assert(loaded.config.hydro_catchment_id == "watershed_\"a");
    {
        std::ofstream incompatibleModels(root / "models.csv", std::ios::trunc);
        incompatibleModels << "approach,file,format,size_bytes,sha256\n"
                           << "ffn,models/ffn.pt,torch-module-v1,3," << models.at("ffn").sha256 << '\n';
    }
    bool rejectedInferenceBundle = false;
    try { (void)HydroArtifactLoader().loadForInference(root.string()); }
    catch (const std::runtime_error&) { rejectedInferenceBundle = true; }
    assert(rejectedInferenceBundle);
    {
        std::ofstream compatibleModels(root / "models.csv", std::ios::trunc);
        compatibleModels << "approach,file,format,size_bytes,sha256\n"
                         << "ffn,models/ffn.pt,neuralnetworkwrapper-v1,3," << models.at("ffn").sha256 << '\n';
    }
    {
        std::ofstream corrupt(root / "models" / "ffn.pt", std::ios::binary | std::ios::app);
        corrupt.put('\x04');
    }
    bool rejectedModel = false;
    try { (void)HydroArtifactLoader().loadModels(root.string()); }
    catch (const std::runtime_error&) { rejectedModel = true; }
    assert(rejectedModel);
    {
        std::ofstream invalidScalers(root / "scalers.csv", std::ios::trunc);
        invalidScalers << "approach,kind,index,method,shape,offset,scale\n"
                       << "ffn,input,1,minmax,\"1;1\",0,2\n";
    }
    bool rejectedScalers = false;
    try { (void)HydroArtifactLoader().loadScalers(root.string()); }
    catch (const std::runtime_error&) { rejectedScalers = true; }
    assert(rejectedScalers);
    {
        std::ofstream invalidPredictions(root / "predictions.csv", std::ios::trunc);
        invalidPredictions << "approach,index,split,x,observed,predicted,residual\n"
                           << "ffn,0,test,0,1,2,99\n";
    }
    bool rejectedPredictions = false;
    try { (void)HydroArtifactLoader().loadPredictions(root.string()); }
    catch (const std::runtime_error&) { rejectedPredictions = true; }
    assert(rejectedPredictions);
    std::filesystem::remove_all(output);
    return 0;
}
