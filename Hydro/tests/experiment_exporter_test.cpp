#include "../evaluation/experiment_exporter.h"
#include "../evaluation/experiment_loader.h"

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
    result.model_checkpoint_format = "fixture-v1";
    result.model_checkpoint = {0x01, 0x02, 0x03};
    result.physics_residual = {0.1, -0.2};
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
    assert(std::filesystem::file_size(root / "models" / "ffn.pt") == 3);
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
    std::filesystem::remove_all(output);
    return 0;
}
