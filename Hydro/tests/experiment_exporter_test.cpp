#include "../evaluation/experiment_exporter.h"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <iterator>

int main() {
    const std::filesystem::path output = "/tmp/hydro_experiment_export";
    std::filesystem::remove_all(output);
    HydroRunConfig config;
    config.random_seed = 7;
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
    config.hydro_catchment_id = "watershed_a";
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
    std::ifstream predictions(root / "predictions.csv");
    const std::string text((std::istreambuf_iterator<char>(predictions)), std::istreambuf_iterator<char>());
    assert(text.find("ffn,0,train,0,1,1.5,0.5") != std::string::npos);
    assert(text.find("ffn,1,test,1,2,1.5,-0.5") != std::string::npos);
    std::ifstream history(root / "training_history.csv");
    const std::string historyText((std::istreambuf_iterator<char>(history)), std::istreambuf_iterator<char>());
    assert(historyText.find("ffn,2,0.5") != std::string::npos);
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
    std::filesystem::remove_all(output);
    return 0;
}
