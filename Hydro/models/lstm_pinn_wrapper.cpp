#include "lstm_pinn_wrapper.h"

#include "neuralnetworkwrapper.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace {
std::vector<int> parseHiddenLayers(const std::string& csv) {
    std::vector<int> layers;
    std::stringstream ss(csv);
    std::string token;
    while (std::getline(ss, token, ',')) {
        try {
            int v = std::stoi(token);
            if (v > 0) layers.push_back(v);
        } catch (...) {}
    }
    if (layers.empty()) layers = {32, 24};
    return layers;
}

std::vector<std::string> splitCsvRow(const std::string& line) {
    std::vector<std::string> cols;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) {
        cols.push_back(cell);
    }
    return cols;
}

bool loadSeriesFromCsv(const HydroRunConfig& config, torch::Tensor& t, torch::Tensor& y) {
    if (!config.use_csv_data) return false;
    if (config.csv_path.empty()) throw std::runtime_error("CSV data source selected but csv_path is empty.");

    std::ifstream in(config.csv_path);
    if (!in.is_open()) throw std::runtime_error("Unable to open CSV file: " + config.csv_path);

    std::vector<float> xs;
    std::vector<float> ys;
    std::string line;
    bool firstLine = true;
    const int requiredCol = std::max(config.csv_x_column, config.csv_y_column);
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (firstLine && config.csv_has_header) {
            firstLine = false;
            continue;
        }
        firstLine = false;

        const std::vector<std::string> cols = splitCsvRow(line);
        if (static_cast<int>(cols.size()) <= requiredCol) continue;
        try {
            xs.push_back(static_cast<float>(std::stod(cols[config.csv_x_column])));
            ys.push_back(static_cast<float>(std::stod(cols[config.csv_y_column])));
        } catch (...) {
            continue;
        }
    }

    if (xs.size() < 10) throw std::runtime_error("CSV parsing yielded too few numeric samples (<10).");

    t = torch::from_blob(xs.data(), {(long)xs.size(), 1}, torch::kFloat32).clone();
    y = torch::from_blob(ys.data(), {(long)ys.size(), 1}, torch::kFloat32).clone();
    return true;
}

void fillPlotVectors(HydroRunResult& result, const torch::Tensor& x, const torch::Tensor& yTrue, const torch::Tensor& yPred) {
    auto xc = x.squeeze(1).contiguous();
    auto tc = yTrue.squeeze(1).contiguous();
    auto pc = yPred.squeeze(1).contiguous();
    const int64_t n = xc.size(0);
    result.x.reserve(static_cast<size_t>(n));
    result.y_true.reserve(static_cast<size_t>(n));
    result.y_pred.reserve(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        result.x.push_back(xc[i].item<double>());
        result.y_true.push_back(tc[i].item<double>());
        result.y_pred.push_back(pc[i].item<double>());
    }
}
}

HydroRunResult LSTMPINNWrapper::train(const HydroRunConfig& config) {
    HydroRunResult result;

    // Temporary backend: uses NeuralNetworkWrapper FFN core while LSTM-PINN is scaffolded.
    NeuralNetworkWrapper model;
    model.setHiddenLayers(parseHiddenLayers(config.hidden_layers_csv));
    model.setLags({{1}});
    model.initializeNetwork(1, config.activation);

    const double lambda = config.lambda_decay;
    torch::Tensor t;
    torch::Tensor y;
    if (!loadSeriesFromCsv(config, t, y)) {
        const int samples = std::max(32, config.sample_count);
        t = torch::linspace(config.t_start, config.t_end, samples, torch::kFloat32).unsqueeze(1);
        y = torch::exp(-lambda * t);
    }

    const int64_t nTrain = static_cast<int64_t>(t.size(0) * 0.8);
    torch::Tensor tTrain = t.slice(0, 0, nTrain);
    torch::Tensor yTrain = y.slice(0, 0, nTrain);
    torch::Tensor tTest = t.slice(0, nTrain, t.size(0));
    torch::Tensor yTest = y.slice(0, nTrain, y.size(0));

    model.setTensorData(DataType::Train, tTrain, yTrain);
    model.setTensorData(DataType::Test, tTest, yTest);

    std::vector<double> losses = model.trainPINNExponentialDecay(config.epochs,
                                                                 config.batch_size,
                                                                 config.learning_rate,
                                                                 lambda,
                                                                 config.data_weight,
                                                                 config.physics_weight);
    if (losses.empty() || !std::isfinite(losses.back())) {
        throw std::runtime_error("LSTM-PINN training produced empty/non-finite loss history.");
    }
    result.final_loss = losses.back();

    torch::Tensor pred = model.forward(DataType::Test);
    if (!pred.defined() || pred.size(0) != yTest.size(0) || !pred.isfinite().all().item<bool>()) {
        throw std::runtime_error("LSTM-PINN prediction failed or produced non-finite values.");
    }

    fillPlotVectors(result, tTest, yTest, pred);
    result.success = true;
    result.message = config.use_csv_data
                         ? "LSTM-PINN-like run completed with CSV input (temporary FFN backend)."
                         : "LSTM-PINN-like run completed with synthetic input (temporary FFN backend).";
    return result;
}
