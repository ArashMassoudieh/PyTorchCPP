#include "lstm_pinn_wrapper.h"

#include "neuralnetworkwrapper.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <map>
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

torch::Tensor buildTargetFromProfile(const torch::Tensor& t, const HydroRunConfig& config) {
    if (config.synthetic_profile == "neuroforge_inputs_target") {
        std::srand(42);
        auto tc = t.squeeze(1).contiguous();
        const int64_t n = tc.size(0);
        std::vector<float> ys(static_cast<size_t>(n), 0.0f);

        const double tStart = tc[0].item<double>();
        const double tEnd = tc[n - 1].item<double>();
        const double dt = (n > 1) ? (tEnd - tStart) / static_cast<double>(n - 1) : 1.0;
        const double bufferStart = tStart - 1.0;
        const int totalSteps = static_cast<int>(std::floor((tEnd - bufferStart) / dt)) + 1;

        std::vector<double> allT;
        std::vector<double> allTemp;
        std::vector<double> allPress;
        std::vector<double> allConc;
        std::vector<double> allVel;
        allT.reserve(static_cast<size_t>(totalSteps));
        allTemp.reserve(static_cast<size_t>(totalSteps));
        allPress.reserve(static_cast<size_t>(totalSteps));
        allConc.reserve(static_cast<size_t>(totalSteps));
        allVel.reserve(static_cast<size_t>(totalSteps));

        double x0 = 0.0, x1 = 0.0, x3 = 0.0, x4 = 0.0;
        auto noise = []() { return (static_cast<double>(std::rand()) / RAND_MAX - 0.5) * 2.0; };

        for (int i = 0; i < totalSteps; ++i) {
            const double tt = bufferStart + dt * static_cast<double>(i);
            x0 = x0 + 0.5 * (0.0 - x0) * dt + 1.5 * std::sqrt(dt) * noise();
            x1 = x1 + 1.0 * (0.0 - x1) * dt + 1.2 * std::sqrt(dt) * noise();
            x3 = x3 + 0.3 * (0.0 - x3) * dt + 1.0 * std::sqrt(dt) * noise();
            x4 = x4 + 0.8 * (0.0 - x4) * dt + 1.8 * std::sqrt(dt) * noise();
            allT.push_back(tt);
            allTemp.push_back(x0);
            allPress.push_back(x1);
            allConc.push_back(x3);
            allVel.push_back(x4);
        }

        auto interpol = [&](const std::vector<double>& vals, double tq) {
            if (tq <= allT.front()) return vals.front();
            if (tq >= allT.back()) return vals.back();
            const auto it = std::lower_bound(allT.begin(), allT.end(), tq);
            const size_t hi = static_cast<size_t>(it - allT.begin());
            const size_t lo = hi - 1;
            const double t0 = allT[lo];
            const double t1 = allT[hi];
            const double r = (tq - t0) / (t1 - t0);
            return vals[lo] * (1.0 - r) + vals[hi] * r;
        };

        for (int64_t i = 0; i < n; ++i) {
            const double tt = tc[i].item<double>();
            const double target = 0.4 * interpol(allTemp, tt - 0.1) +
                                  0.3 * interpol(allPress, tt - 0.3) +
                                  0.2 * interpol(allConc, tt - 0.2) +
                                  0.1 * interpol(allVel, tt - 0.5) +
                                  0.05 * (static_cast<double>(std::rand()) / RAND_MAX - 0.5);
            ys[static_cast<size_t>(i)] = static_cast<float>(target);
        }
        return torch::from_blob(ys.data(), {n, 1}, torch::kFloat32).clone();
    }

    if (config.synthetic_profile == "mixed_wave") {
        return 0.7 * torch::sin(1.5 * t) + 0.3 * torch::cos(0.5 * t);
    }
    if (config.synthetic_profile == "damped_sine") {
        return torch::sin(t) * torch::exp(-0.15 * t);
    }
    return torch::exp(-config.lambda_decay * t);
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
        y = buildTargetFromProfile(t, config);
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

    if (config.evaluate_metrics) {
        std::map<std::string, double> metrics = model.evaluate();
        auto it = metrics.find("mse");
        if (it != metrics.end()) {
            if (!std::isfinite(it->second)) {
                throw std::runtime_error("LSTM-PINN evaluation produced non-finite MSE.");
            }
            result.mse = it->second;
        }
    }

    fillPlotVectors(result, tTest, yTest, pred);
    result.success = true;
    result.message = config.use_csv_data
                         ? "LSTM-PINN-like run completed with CSV input (temporary FFN backend)."
                         : "LSTM-PINN-like run completed with synthetic input (temporary FFN backend).";
    return result;
}
