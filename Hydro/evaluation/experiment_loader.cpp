#include "experiment_loader.h"

#include <fstream>
#include <regex>
#include <stdexcept>

namespace {
std::string readConfig(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Unable to open Hydro experiment configuration: " + path);
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

std::string stringValue(const std::string& json, const std::string& key) {
    std::smatch match;
    const std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*\\\"((?:\\\\.|[^\\\"])*)\\\"");
    if (!std::regex_search(json, match, pattern)) throw std::runtime_error("Experiment configuration is missing string: " + key);
    const std::string encoded = match[1].str();
    std::string decoded;
    for (std::size_t i = 0; i < encoded.size(); ++i) {
        if (encoded[i] != '\\') {
            decoded.push_back(encoded[i]);
            continue;
        }
        if (++i >= encoded.size()) throw std::runtime_error("Experiment configuration has an invalid escape in: " + key);
        if (encoded[i] == 'n') decoded.push_back('\n');
        else if (encoded[i] == '\\' || encoded[i] == '"') decoded.push_back(encoded[i]);
        else throw std::runtime_error("Experiment configuration has an unsupported escape in: " + key);
    }
    return decoded;
}

double numberValue(const std::string& json, const std::string& key) {
    std::smatch match;
    const std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*(-?[0-9]+(?:\\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)");
    if (!std::regex_search(json, match, pattern)) throw std::runtime_error("Experiment configuration is missing number: " + key);
    return std::stod(match[1].str());
}

bool boolValue(const std::string& json, const std::string& key) {
    std::smatch match;
    const std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*(true|false)");
    if (!std::regex_search(json, match, pattern)) throw std::runtime_error("Experiment configuration is missing boolean: " + key);
    return match[1].str() == "true";
}
}

LoadedHydroExperiment HydroExperimentLoader::loadConfig(const std::string& configPath) const {
    const std::string json = readConfig(configPath);
    LoadedHydroExperiment loaded;
    auto& c = loaded.config;
    loaded.experiment_id = stringValue(json, "experiment_id");
    c.epochs = static_cast<int>(numberValue(json, "epochs"));
    c.batch_size = static_cast<int>(numberValue(json, "batch_size"));
    c.learning_rate = numberValue(json, "learning_rate");
    c.optimizer = stringValue(json, "optimizer");
    c.weight_decay = numberValue(json, "weight_decay");
    c.momentum = numberValue(json, "momentum");
    c.random_seed = static_cast<int>(numberValue(json, "random_seed"));
    c.train_split_ratio = numberValue(json, "train_fraction");
    c.validation_split_ratio = numberValue(json, "validation_fraction");
    c.shuffle_training = boolValue(json, "shuffle_training");
    c.normalization = stringValue(json, "normalization");
    c.hidden_layers_csv = stringValue(json, "hidden_layers");
    c.input_lags_csv = stringValue(json, "input_lags");
    c.activation = stringValue(json, "activation");
    c.use_time_lagged_ffn = boolValue(json, "use_time_lagged_ffn");
    c.lstm_sequence_length = static_cast<int>(numberValue(json, "lstm_sequence_length"));
    c.pinn_physics_profile = stringValue(json, "physics_profile");
    c.data_weight = numberValue(json, "data_weight");
    c.physics_weight = numberValue(json, "physics_weight");
    c.physics_dt = numberValue(json, "physics_dt");
    c.forcing_gain = numberValue(json, "forcing_gain");
    c.runoff_coeff = numberValue(json, "runoff_coeff");
    c.storage_coeff = numberValue(json, "storage_coeff");
    c.pinn_collocation_points = static_cast<int>(numberValue(json, "pinn_collocation_points"));
    c.use_hydro_package = boolValue(json, "use_hydro_package");
    c.hydro_package_path = stringValue(json, "hydro_package_path");
    c.hydro_catchment_id = stringValue(json, "hydro_catchment_id");
    c.hydro_package_profile = stringValue(json, "hydro_package_profile");
    c.use_hydro_forecast_feature = boolValue(json, "use_hydro_forecast_feature");
    c.hydro_forecast_variable = stringValue(json, "hydro_forecast_variable");
    c.hydro_forecast_lead_hours = numberValue(json, "hydro_forecast_lead_hours");
    c.hydro_forecast_ensemble_member = stringValue(json, "hydro_forecast_ensemble_member");
    if (loaded.experiment_id.empty() || c.epochs <= 0 || c.batch_size <= 0 || c.learning_rate <= 0.0 ||
        c.train_split_ratio <= 0.0 || c.validation_split_ratio <= 0.0 ||
        c.train_split_ratio + c.validation_split_ratio >= 1.0) {
        throw std::runtime_error("Experiment configuration contains invalid training or split settings.");
    }
    return loaded;
}
