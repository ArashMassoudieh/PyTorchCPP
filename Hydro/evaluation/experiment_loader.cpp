#include "experiment_loader.h"
#include "../models/hydrofusion_checkpoint_policy.h"

#include <cstdint>
#include <cmath>
#include <fstream>
#include <iterator>
#include <limits>
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

void rejectDuplicateField(const std::string& json, const std::string& key) {
    const std::regex fieldPattern("\\\"" + key + "\\\"\\s*:");
    const auto begin = std::sregex_iterator(json.begin(), json.end(), fieldPattern);
    const auto count = std::distance(begin, std::sregex_iterator());
    if (count > 1) throw std::runtime_error("Experiment configuration contains duplicate field: " + key);
}

std::string stringValue(const std::string& json, const std::string& key) {
    rejectDuplicateField(json, key);
    std::smatch match;
    const std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*\\\"((?:\\\\.|[^\\\"])*)\\\"\\s*[,}]");
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
    rejectDuplicateField(json, key);
    std::smatch match;
    const std::regex pattern("\\\"" + key +
                             "\\\"\\s*:\\s*(-?(?:0|[1-9][0-9]*)(?:\\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)\\s*[,}]");
    if (!std::regex_search(json, match, pattern)) throw std::runtime_error("Experiment configuration is missing number: " + key);
    try {
        const double value = std::stod(match[1].str());
        if (!std::isfinite(value)) throw std::out_of_range("non-finite");
        return value;
    } catch (...) {
        throw std::runtime_error("Experiment configuration has an invalid number: " + key);
    }
}

double optionalNumberValue(const std::string& json, const std::string& key, const double fallback) {
    rejectDuplicateField(json, key);
    std::smatch match;
    const std::regex pattern("\\\"" + key +
                             "\\\"\\s*:\\s*(-?(?:0|[1-9][0-9]*)(?:\\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)\\s*[,}]");
    if (!std::regex_search(json, match, pattern)) return fallback;
    try {
        const double value = std::stod(match[1].str());
        if (!std::isfinite(value)) throw std::out_of_range("non-finite");
        return value;
    } catch (...) {
        throw std::runtime_error("Experiment configuration has an invalid number: " + key);
    }
}

// Exact split indices must not be silently rounded or converted out of range.
int64_t optionalCutoffValue(const std::string& json, const std::string& key, int64_t fallback = -1) {
    rejectDuplicateField(json, key);
    const std::regex field("\\\"" + key + "\\\"\\s*:");
    if (!std::regex_search(json, field)) return fallback;
    std::smatch match;
    const std::regex number("\\\"" + key +
        "\\\"\\s*:\\s*(-?(?:0|[1-9][0-9]*)(?:\\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)\\s*[,}]");
    try {
        if (!std::regex_search(json, match, number)) throw std::invalid_argument("not numeric");
        const long double value = std::stold(match[1].str());
        if (!std::isfinite(value) || std::floor(value) != value || value < -1 ||
            value >= 0x1p63L) {
            throw std::out_of_range("not an exact supported index");
        }
        return static_cast<int64_t>(value);
    } catch (const std::exception&) {
        throw std::runtime_error("Experiment configuration cutoff must be an exact int64 index >= -1: " + key);
    }
}

int integerValue(const std::string& json, const std::string& key) {
    rejectDuplicateField(json, key);
    std::smatch match;
    const std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*(-?(?:0|[1-9][0-9]*))\\s*[,}]");
    if (!std::regex_search(json, match, pattern)) {
        throw std::runtime_error("Experiment configuration is missing integer: " + key);
    }
    try {
        const long long value = std::stoll(match[1].str());
        if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max()) {
            throw std::out_of_range("integer range");
        }
        return static_cast<int>(value);
    } catch (...) {
        throw std::runtime_error("Experiment configuration has an invalid integer: " + key);
    }
}

bool boolValue(const std::string& json, const std::string& key) {
    rejectDuplicateField(json, key);
    std::smatch match;
    const std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*(true|false)\\s*[,}]");
    if (!std::regex_search(json, match, pattern)) throw std::runtime_error("Experiment configuration is missing boolean: " + key);
    return match[1].str() == "true";
}

bool optionalBoolValue(const std::string& json, const std::string& key, bool fallback) {
    rejectDuplicateField(json, key);
    std::smatch match;
    const std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*(true|false)\\s*[,}]");
    if (!std::regex_search(json, match, pattern)) return fallback;
    return match[1].str() == "true";
}

std::string optionalStringValue(const std::string& json, const std::string& key, const std::string& fallback) {
    const std::regex presence("\\\"" + key + "\\\"\\s*:\\s*\\\"(?:\\\\.|[^\\\"])*\\\"\\s*[,}]");
    if (!std::regex_search(json, presence)) return fallback;
    return stringValue(json, key);
}
}

LoadedHydroExperiment HydroExperimentLoader::loadConfig(const std::string& configPath) const {
    const std::string json = readConfig(configPath);
    LoadedHydroExperiment loaded;
    auto& c = loaded.config;
    loaded.experiment_id = stringValue(json, "experiment_id");
    c.epochs = integerValue(json, "epochs");
    c.batch_size = integerValue(json, "batch_size");
    c.learning_rate = numberValue(json, "learning_rate");
    c.lambda_decay = numberValue(json, "lambda_decay");
    c.optimizer = stringValue(json, "optimizer");
    c.weight_decay = numberValue(json, "weight_decay");
    c.momentum = numberValue(json, "momentum");
    c.random_seed = integerValue(json, "random_seed");
    c.train_split_ratio = numberValue(json, "train_fraction");
    c.validation_split_ratio = numberValue(json, "validation_fraction");
    // Optional - absent in every config predating this field, which must
    // keep working unchanged (-1.0 = unbounded test partition). Prefer
    // test_end_x below when the exact row index is known - see
    // hydro_run_types.h's doc comment on why the ratio form is retained
    // only as a fallback.
    c.test_end_split_ratio = optionalNumberValue(json, "test_end_fraction", -1.0);
    // Optional exact row index, absent in every config predating this
    // field (-1 = unset, takes priority over test_end_split_ratio when
    // both are present - see chronological_split.h's testEndX doc comment).
    c.test_end_x = optionalCutoffValue(json, "test_end_x");
    c.train_end_x = optionalCutoffValue(json, "train_end_x");
    c.validation_end_x = optionalCutoffValue(json, "validation_end_x");
    c.pinn_initial_flow = optionalNumberValue(json, "pinn_initial_flow", -1.0);
    c.shuffle_training = boolValue(json, "shuffle_training");
    c.evaluate_metrics = boolValue(json, "evaluate_metrics");
    c.normalization = stringValue(json, "normalization");
    c.hidden_layers_csv = stringValue(json, "hidden_layers");
    c.input_lags_csv = stringValue(json, "input_lags");
    c.activation = stringValue(json, "activation");
    c.use_time_lagged_ffn = boolValue(json, "use_time_lagged_ffn");
    c.lstm_sequence_length = integerValue(json, "lstm_sequence_length");
    if (c.lstm_sequence_length < 1) throw std::runtime_error("Experiment configuration contains invalid LSTM sequence length.");
    c.pinn_physics_profile = stringValue(json, "physics_profile");
    c.data_weight = numberValue(json, "data_weight");
    c.physics_weight = numberValue(json, "physics_weight");
    c.physics_dt = numberValue(json, "physics_dt");
    c.forcing_gain = numberValue(json, "forcing_gain");
    c.runoff_coeff = numberValue(json, "runoff_coeff");
    c.storage_coeff = numberValue(json, "storage_coeff");
    c.pinn_collocation_points = integerValue(json, "pinn_collocation_points");
    c.pinn_routing_lag_hours = optionalNumberValue(json, "pinn_routing_lag_hours", 0.0);
    c.pinn_flow_exponent = optionalNumberValue(json, "pinn_flow_exponent", 0.0);
    c.lstm_bias_weight = optionalNumberValue(json, "lstm_bias_weight", 0.0);
    c.shared_scale_volume_weight = optionalNumberValue(json, "shared_scale_volume_weight", 0.0);
    if (!std::isfinite(c.shared_scale_volume_weight) || c.shared_scale_volume_weight < 0.0) {
        throw std::runtime_error("shared_scale_volume_weight must be finite and nonnegative");
    }
    c.physical_error_weight = optionalNumberValue(json, "physical_error_weight", 0.0);
    c.high_flow_loss_weight = optionalNumberValue(json, "high_flow_loss_weight", 0.0);
    c.high_flow_loss_quantile = optionalNumberValue(json, "high_flow_loss_quantile", 0.99);
    if (c.high_flow_loss_weight < 0.0) {
        throw std::runtime_error("Experiment configuration contains negative high_flow_loss_weight.");
    }
    if (c.high_flow_loss_quantile < 0.0 || c.high_flow_loss_quantile >= 1.0) {
        throw std::runtime_error("Experiment configuration high_flow_loss_quantile must be in [0, 1).");
    }
    c.lstm_bias_weight_adaptive = optionalBoolValue(json, "lstm_bias_weight_adaptive", false);
    c.lstm_warm_start_checkpoint_path = optionalStringValue(json, "lstm_warm_start_checkpoint_path", "");
    c.lstm_dropout = optionalNumberValue(json, "lstm_dropout", 0.0);
    c.antecedent_index_halflife_hours = optionalNumberValue(json, "antecedent_index_halflife_hours", 0.0);
    c.use_mrms_spatial_cv_feature = optionalBoolValue(json, "use_mrms_spatial_cv_feature", false);
    c.urban_quick_k = optionalNumberValue(json, "urban_quick_k", 1.0);
    c.urban_slow_k = optionalNumberValue(json, "urban_slow_k", 0.04);
    c.urban_detention_k = optionalNumberValue(json, "urban_detention_k", 0.08);
    c.urban_channel_k = optionalNumberValue(json, "urban_channel_k", 1.0);
    c.urban_quick_fraction = optionalNumberValue(json, "urban_quick_fraction", 0.26);
    c.urban_detention_fraction = optionalNumberValue(json, "urban_detention_fraction", 0.183);
    c.urban_runoff_coefficient = optionalNumberValue(json, "urban_runoff_coefficient", 1.0);
    c.urban_intensity_quick_strength = optionalNumberValue(json, "urban_intensity_quick_strength", 0.0);
    c.urban_intensity_reference_mm_h = optionalNumberValue(json, "urban_intensity_reference_mm_h", 1.0);
    if (!std::isfinite(c.urban_intensity_quick_strength) || c.urban_intensity_quick_strength < 0.0 || c.urban_intensity_quick_strength > 1.0 ||
        !std::isfinite(c.urban_intensity_reference_mm_h) || c.urban_intensity_reference_mm_h <= 0.0) {
        throw std::runtime_error("Urban intensity response requires strength in [0,1] and positive finite reference");
    }
    c.multiscale_daily_history_days = optionalCutoffValue(json, "multiscale_daily_history_days", 0);
    c.multiscale_daily_hidden_size = optionalCutoffValue(json, "multiscale_daily_hidden_size", 4);
    if (c.multiscale_daily_history_days < 0 || c.multiscale_daily_hidden_size < 1) {
        throw std::runtime_error("multiscale_daily_history_days must be >= 0 and multiscale_daily_hidden_size must be >= 1.");
    }
    c.multiscale_matched_control = optionalBoolValue(json, "multiscale_matched_control", false);
    c.regional_training_basin_list_path = optionalStringValue(json, "regional_training_basin_list_path", "");
    c.regional_validation_basin_list_path = optionalStringValue(json, "regional_validation_basin_list_path", "");
    c.regional_per_basin_sequence_count = optionalCutoffValue(json, "regional_per_basin_sequence_count", -1);
    c.regional_window_start = optionalStringValue(json, "regional_window_start", "");
    c.regional_window_end = optionalStringValue(json, "regional_window_end", "");
    // Opt-in FFN+HydroFusion mode only ("ffn_hydrofusion") -- absent in
    // every config predating this field, which must keep working
    // unchanged. No other mode reads this.
    c.hydrofusion_synthetic_verification = optionalBoolValue(json, "hydrofusion_synthetic_verification", false);
    c.hydrofusion_native_manifest_path = optionalStringValue(json, "hydrofusion_native_manifest_path", "");
    c.hydrofusion_native_manifest_sha256 = optionalStringValue(json, "hydrofusion_native_manifest_sha256", "");
    // Scope/recipe controls must never silently fall back on a wrong JSON type.
    const auto hasForecastField=[&](const std::string& key){
        rejectDuplicateField(json,key);
        return std::regex_search(json,std::regex("\\\""+key+"\\\"\\s*:"));
    };
    c.hydrofusion_development_only = hasForecastField("hydrofusion_development_only") ? boolValue(json, "hydrofusion_development_only") : false;
    c.hydrofusion_context_recipe = hasForecastField("hydrofusion_context_recipe") ? stringValue(json, "hydrofusion_context_recipe") : "legacy20";
    c.hydrofusion_history_hours = hasForecastField("hydrofusion_history_hours") ? stringValue(json, "hydrofusion_history_hours") : "0";
    c.hydrofusion_preissue_rain_recipe = hasForecastField("hydrofusion_preissue_rain_recipe") ? stringValue(json, "hydrofusion_preissue_rain_recipe") : "disabled";
    c.hydrofusion_preissue_rain_path = hasForecastField("hydrofusion_preissue_rain_path") ? stringValue(json, "hydrofusion_preissue_rain_path") : "";
    c.hydrofusion_preissue_rain_sha256 = hasForecastField("hydrofusion_preissue_rain_sha256") ? stringValue(json, "hydrofusion_preissue_rain_sha256") : "";
    c.hydrofusion_preissue_rain_provenance_sha256 = hasForecastField("hydrofusion_preissue_rain_provenance_sha256") ? stringValue(json, "hydrofusion_preissue_rain_provenance_sha256") : "";
    c.hydrofusion_objective = optionalStringValue(json, "hydrofusion_objective", "log_mse_v1");
    c.hydrofusion_ffn_activation = optionalStringValue(json, "hydrofusion_ffn_activation", "tanh");
    c.hydrofusion_checkpoint_selection = hasForecastField("hydrofusion_checkpoint_selection") ? stringValue(json, "hydrofusion_checkpoint_selection") : "best_validation_v1";
    hydrofusion::validateForecastCheckpointPolicy(c.hydrofusion_checkpoint_selection);
    if(c.hydrofusion_objective!="log_mse_v1" && c.hydrofusion_objective!="log_flow_mse_v2") throw std::runtime_error("Unknown hydrofusion_objective.");
    if(c.hydrofusion_ffn_activation!="tanh" && c.hydrofusion_ffn_activation!="silu") throw std::runtime_error("Unknown hydrofusion_ffn_activation.");
    c.hydrofusion_target_mode = optionalStringValue(json, "hydrofusion_target_mode", "log_residual");
    if (c.hydrofusion_target_mode != "log_residual" && c.hydrofusion_target_mode != "direct_log_q") {
        throw std::runtime_error(
            "Experiment configuration hydrofusion_target_mode must be 'log_residual' or 'direct_log_q'.");
    }
    c.use_hydro_package = boolValue(json, "use_hydro_package");
    c.use_csv_data = boolValue(json, "use_csv_data");
    c.csv_path = stringValue(json, "csv_path");
    c.csv_x_column = integerValue(json, "csv_x_column");
    c.csv_y_column = integerValue(json, "csv_y_column");
    c.csv_has_header = boolValue(json, "csv_has_header");
    c.synthetic_profile = stringValue(json, "synthetic_profile");
    c.sample_count = integerValue(json, "sample_count");
    c.t_start = numberValue(json, "t_start");
    c.t_end = numberValue(json, "t_end");
    c.synthetic_reservoir_truth_k = optionalNumberValue(json, "synthetic_reservoir_truth_k", c.synthetic_reservoir_truth_k);
    c.hydro_package_path = stringValue(json, "hydro_package_path");
    c.hydro_catchment_id = stringValue(json, "hydro_catchment_id");
    c.hydro_package_profile = stringValue(json, "hydro_package_profile");
    c.use_hydro_forecast_feature = boolValue(json, "use_hydro_forecast_feature");
    c.hydro_forecast_variable = stringValue(json, "hydro_forecast_variable");
    c.hydro_forecast_lead_hours = numberValue(json, "hydro_forecast_lead_hours");
    c.hydro_forecast_ensemble_member = stringValue(json, "hydro_forecast_ensemble_member");
    if (c.use_hydro_package && c.use_csv_data) throw std::runtime_error("Experiment configuration selects multiple data sources.");
    if (loaded.experiment_id.empty() || c.epochs <= 0 || c.batch_size <= 0 || c.learning_rate <= 0.0 || c.sample_count < 3 ||
        c.synthetic_reservoir_truth_k <= 0.0 || c.train_split_ratio <= 0.0 || c.validation_split_ratio <= 0.0 ||
        c.train_split_ratio + c.validation_split_ratio >= 1.0) {
        throw std::runtime_error("Experiment configuration contains invalid training, source, or split settings.");
    }
    hydroLstmSequenceLengthRuntimeDefault() = c.lstm_sequence_length;
    return loaded;
}
