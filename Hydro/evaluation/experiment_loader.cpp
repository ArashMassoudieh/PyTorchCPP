#include "experiment_loader.h"

#include <cstdint>
#include <fstream>
#include <regex>
#include <stdexcept>

namespace {
std::string readConfig(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Unable to open Hydro experiment configuration: " + path);
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

std::uint16_t hexCodeUnit(const std::string& encoded, const std::size_t offset,
                          const std::string& key) {
    if (offset + 4 > encoded.size()) {
        throw std::runtime_error("Experiment configuration has a truncated Unicode escape in: " + key);
    }
    std::uint16_t value = 0;
    for (std::size_t i = offset; i < offset + 4; ++i) {
        const char digit = encoded[i];
        value = static_cast<std::uint16_t>(value << 4);
        if (digit >= '0' && digit <= '9') value |= static_cast<std::uint16_t>(digit - '0');
        else if (digit >= 'a' && digit <= 'f') value |= static_cast<std::uint16_t>(digit - 'a' + 10);
        else if (digit >= 'A' && digit <= 'F') value |= static_cast<std::uint16_t>(digit - 'A' + 10);
        else throw std::runtime_error("Experiment configuration has an invalid Unicode escape in: " + key);
    }
    return value;
}

void appendUtf8(std::string& decoded, const std::uint32_t codePoint, const std::string& key) {
    if (codePoint <= 0x7f) decoded.push_back(static_cast<char>(codePoint));
    else if (codePoint <= 0x7ff) {
        decoded.push_back(static_cast<char>(0xc0 | (codePoint >> 6)));
        decoded.push_back(static_cast<char>(0x80 | (codePoint & 0x3f)));
    } else if (codePoint <= 0xffff) {
        decoded.push_back(static_cast<char>(0xe0 | (codePoint >> 12)));
        decoded.push_back(static_cast<char>(0x80 | ((codePoint >> 6) & 0x3f)));
        decoded.push_back(static_cast<char>(0x80 | (codePoint & 0x3f)));
    } else if (codePoint <= 0x10ffff) {
        decoded.push_back(static_cast<char>(0xf0 | (codePoint >> 18)));
        decoded.push_back(static_cast<char>(0x80 | ((codePoint >> 12) & 0x3f)));
        decoded.push_back(static_cast<char>(0x80 | ((codePoint >> 6) & 0x3f)));
        decoded.push_back(static_cast<char>(0x80 | (codePoint & 0x3f)));
    } else throw std::runtime_error("Experiment configuration has an invalid Unicode code point in: " + key);
}

std::string stringValue(const std::string& json, const std::string& key) {
    std::smatch match;
    const std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*\\\"((?:\\\\.|[^\\\"])*)\\\"");
    if (!std::regex_search(json, match, pattern)) throw std::runtime_error("Experiment configuration is missing string: " + key);
    const std::string encoded = match[1].str();
    std::string decoded;
    for (std::size_t i = 0; i < encoded.size(); ++i) {
        if (encoded[i] != '\\') {
            if (static_cast<unsigned char>(encoded[i]) < 0x20) {
                throw std::runtime_error("Experiment configuration has an unescaped control character in: " + key);
            }
            decoded.push_back(encoded[i]);
            continue;
        }
        if (++i >= encoded.size()) throw std::runtime_error("Experiment configuration has an invalid escape in: " + key);
        switch (encoded[i]) {
        case '"': case '\\': case '/': decoded.push_back(encoded[i]); break;
        case 'b': decoded.push_back('\b'); break;
        case 'f': decoded.push_back('\f'); break;
        case 'n': decoded.push_back('\n'); break;
        case 'r': decoded.push_back('\r'); break;
        case 't': decoded.push_back('\t'); break;
        case 'u': {
            const std::uint16_t first = hexCodeUnit(encoded, i + 1, key);
            i += 4;
            std::uint32_t codePoint = first;
            if (first >= 0xd800 && first <= 0xdbff) {
                if (i + 6 >= encoded.size() || encoded[i + 1] != '\\' || encoded[i + 2] != 'u') {
                    throw std::runtime_error("Experiment configuration has an incomplete Unicode surrogate pair in: " + key);
                }
                const std::uint16_t second = hexCodeUnit(encoded, i + 3, key);
                if (second < 0xdc00 || second > 0xdfff) {
                    throw std::runtime_error("Experiment configuration has an invalid Unicode surrogate pair in: " + key);
                }
                codePoint = 0x10000u + ((static_cast<std::uint32_t>(first) - 0xd800u) << 10) +
                            (static_cast<std::uint32_t>(second) - 0xdc00u);
                i += 6;
            } else if (first >= 0xdc00 && first <= 0xdfff) {
                throw std::runtime_error("Experiment configuration has an unpaired Unicode surrogate in: " + key);
            }
            appendUtf8(decoded, codePoint, key);
            break;
        }
        default: throw std::runtime_error("Experiment configuration has an unsupported escape in: " + key);
        }
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
    c.lambda_decay = numberValue(json, "lambda_decay");
    c.optimizer = stringValue(json, "optimizer");
    c.weight_decay = numberValue(json, "weight_decay");
    c.momentum = numberValue(json, "momentum");
    c.random_seed = static_cast<int>(numberValue(json, "random_seed"));
    c.train_split_ratio = numberValue(json, "train_fraction");
    c.validation_split_ratio = numberValue(json, "validation_fraction");
    c.shuffle_training = boolValue(json, "shuffle_training");
    c.evaluate_metrics = boolValue(json, "evaluate_metrics");
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
    c.use_csv_data = boolValue(json, "use_csv_data");
    c.csv_path = stringValue(json, "csv_path");
    c.csv_x_column = static_cast<int>(numberValue(json, "csv_x_column"));
    c.csv_y_column = static_cast<int>(numberValue(json, "csv_y_column"));
    c.csv_has_header = boolValue(json, "csv_has_header");
    c.synthetic_profile = stringValue(json, "synthetic_profile");
    c.sample_count = static_cast<int>(numberValue(json, "sample_count"));
    c.t_start = numberValue(json, "t_start");
    c.t_end = numberValue(json, "t_end");
    c.hydro_package_path = stringValue(json, "hydro_package_path");
    c.hydro_catchment_id = stringValue(json, "hydro_catchment_id");
    c.hydro_package_profile = stringValue(json, "hydro_package_profile");
    c.use_hydro_forecast_feature = boolValue(json, "use_hydro_forecast_feature");
    c.hydro_forecast_variable = stringValue(json, "hydro_forecast_variable");
    c.hydro_forecast_lead_hours = numberValue(json, "hydro_forecast_lead_hours");
    c.hydro_forecast_ensemble_member = stringValue(json, "hydro_forecast_ensemble_member");
    if (c.use_hydro_package && c.use_csv_data) throw std::runtime_error("Experiment configuration selects multiple data sources.");
    if (loaded.experiment_id.empty() || c.epochs <= 0 || c.batch_size <= 0 || c.learning_rate <= 0.0 || c.sample_count < 3 ||
        c.train_split_ratio <= 0.0 || c.validation_split_ratio <= 0.0 ||
        c.train_split_ratio + c.validation_split_ratio >= 1.0) {
        throw std::runtime_error("Experiment configuration contains invalid training or split settings.");
    }
    return loaded;
}
