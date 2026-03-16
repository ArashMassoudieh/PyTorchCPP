#include "ffn_wrapper.h"

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
    if (layers.empty()) layers = {24, 24};
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

bool loadSeriesFromCsv(const HydroRunConfig& config,
                       torch::Tensor& x,
                       torch::Tensor& y,
                       torch::Tensor& plotX) {
    if (!config.use_csv_data) {
        return false;
    }
    if (config.csv_path.empty()) {
        throw std::runtime_error("CSV data source selected but csv_path is empty.");
    }

    std::ifstream in(config.csv_path);
    if (!in.is_open()) {
        throw std::runtime_error("Unable to open CSV file: " + config.csv_path);
    }

    std::vector<float> flatInputs;
    std::vector<float> ys;
    std::vector<float> plotXs;
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
            if (config.synthetic_profile == "neuroforge_inputs_target") {
                int features = 0;
                for (int c = 0; c < static_cast<int>(cols.size()); ++c) {
                    if (c == config.csv_y_column) continue;
                    flatInputs.push_back(static_cast<float>(std::stod(cols[c])));
                    ++features;
                }
                if (features == 0) continue;
                plotXs.push_back((config.csv_x_column >= 0 && config.csv_x_column < static_cast<int>(cols.size()) &&
                                  config.csv_x_column != config.csv_y_column)
                                     ? static_cast<float>(std::stod(cols[config.csv_x_column]))
                                     : flatInputs[flatInputs.size() - features]);
            } else {
                flatInputs.push_back(static_cast<float>(std::stod(cols[config.csv_x_column])));
                plotXs.push_back(flatInputs.back());
            }
            ys.push_back(static_cast<float>(std::stod(cols[config.csv_y_column])));
        } catch (...) { continue; }
    }

    if (ys.size() < 10) {
        throw std::runtime_error("CSV parsing yielded too few numeric samples (<10).");
    }

    const int64_t samples = static_cast<int64_t>(ys.size());
    const int64_t inputDim = static_cast<int64_t>(flatInputs.size() / ys.size());
    if (inputDim <= 0 || static_cast<size_t>(samples * inputDim) != flatInputs.size()) {
        throw std::runtime_error("CSV parsing yielded inconsistent input feature widths.");
    }

    x = torch::from_blob(flatInputs.data(), {samples, inputDim}, torch::kFloat32).clone();
    y = torch::from_blob(ys.data(), {(long)ys.size(), 1}, torch::kFloat32).clone();
    plotX = torch::from_blob(plotXs.data(), {(long)plotXs.size(), 1}, torch::kFloat32).clone();
    return true;
}

void buildSyntheticSeries(const HydroRunConfig& config, torch::Tensor& x, torch::Tensor& y, torch::Tensor& plotX) {
    const int samples = std::max(32, config.sample_count);
    torch::Tensor t = torch::linspace(config.t_start, config.t_end, samples, torch::kFloat32).unsqueeze(1);
    plotX = t.clone();
    const std::string& profile = config.synthetic_profile;

    if (profile == "neuroforge_inputs_target") {
        std::srand(42);
        auto tc = t.squeeze(1).contiguous();
        const int64_t n = tc.size(0);
        std::vector<float> ys(static_cast<size_t>(n), 0.0f);
        std::vector<float> inTemp(static_cast<size_t>(n), 0.0f);
        std::vector<float> inPress(static_cast<size_t>(n), 0.0f);
        std::vector<float> inFlow(static_cast<size_t>(n), 0.0f);
        std::vector<float> inConc(static_cast<size_t>(n), 0.0f);
        std::vector<float> inVel(static_cast<size_t>(n), 0.0f);

        const double tStart = tc[0].item<double>();
        const double tEnd = tc[n - 1].item<double>();
        const double dt = (n > 1) ? (tEnd - tStart) / static_cast<double>(n - 1) : 1.0;
        const double bufferStart = tStart - 1.0;
        const int totalSteps = static_cast<int>(std::floor((tEnd - bufferStart) / dt)) + 1;

        std::vector<double> allT;
        std::vector<double> allTemp;
        std::vector<double> allPress;
        std::vector<double> allFlow;
        std::vector<double> allConc;
        std::vector<double> allVel;
        allT.reserve(static_cast<size_t>(totalSteps));
        allTemp.reserve(static_cast<size_t>(totalSteps));
        allPress.reserve(static_cast<size_t>(totalSteps));
        allConc.reserve(static_cast<size_t>(totalSteps));
        allVel.reserve(static_cast<size_t>(totalSteps));

        double x0 = 0.0, x1 = 0.0, x2 = 1.0, x3 = 0.0, x4 = 0.0;
        auto noise = []() { return (static_cast<double>(std::rand()) / RAND_MAX - 0.5) * 2.0; };

        for (int i = 0; i < totalSteps; ++i) {
            const double tt = bufferStart + dt * static_cast<double>(i);
            x0 = x0 + 0.5 * (0.0 - x0) * dt + 1.5 * std::sqrt(dt) * noise();
            x1 = x1 + 1.0 * (0.0 - x1) * dt + 1.2 * std::sqrt(dt) * noise();
            x2 = x2 + 2.0 * (1.0 - x2) * dt + 0.8 * std::sqrt(dt) * noise();
            x3 = x3 + 0.3 * (0.0 - x3) * dt + 1.0 * std::sqrt(dt) * noise();
            x4 = x4 + 0.8 * (0.0 - x4) * dt + 1.8 * std::sqrt(dt) * noise();
            allT.push_back(tt);
            allTemp.push_back(x0);
            allPress.push_back(x1);
            allFlow.push_back(x2);
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
            const double flow = interpol(allFlow, tt);
            const double target = 0.4 * interpol(allTemp, tt - 0.1) +
                                  0.3 * interpol(allPress, tt - 0.3) +
                                  0.2 * interpol(allConc, tt - 0.2) +
                                  0.1 * interpol(allVel, tt - 0.5) +
                                  0.05 * (static_cast<double>(std::rand()) / RAND_MAX - 0.5);
            inTemp[static_cast<size_t>(i)] = static_cast<float>(interpol(allTemp, tt));
            inPress[static_cast<size_t>(i)] = static_cast<float>(interpol(allPress, tt));
            inFlow[static_cast<size_t>(i)] = static_cast<float>(flow);
            inConc[static_cast<size_t>(i)] = static_cast<float>(interpol(allConc, tt));
            inVel[static_cast<size_t>(i)] = static_cast<float>(interpol(allVel, tt));
            ys[static_cast<size_t>(i)] = static_cast<float>(target);
        }

        std::vector<float> flatInputs;
        flatInputs.reserve(static_cast<size_t>(n) * 5);
        for (int64_t i = 0; i < n; ++i) {
            const size_t k = static_cast<size_t>(i);
            flatInputs.push_back(inTemp[k]);
            flatInputs.push_back(inPress[k]);
            flatInputs.push_back(inFlow[k]);
            flatInputs.push_back(inConc[k]);
            flatInputs.push_back(inVel[k]);
        }

        x = torch::from_blob(flatInputs.data(), {n, 5}, torch::kFloat32).clone();
        y = torch::from_blob(ys.data(), {n, 1}, torch::kFloat32).clone();
        return;
    }

    x = t;
    if (profile == "damped_sine") y = torch::sin(t) * torch::exp(-0.15 * t);
    else if (profile == "mixed_wave") y = 0.7 * torch::sin(1.5 * t) + 0.3 * torch::cos(0.5 * t);
    else y = torch::exp(-0.8 * t);
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
} // namespace

HydroRunResult FFNWrapper::train(const HydroRunConfig& config) {
    HydroRunResult result;

    torch::manual_seed(static_cast<uint64_t>(std::max(0, config.random_seed)));

    NeuralNetworkWrapper model;
    torch::Tensor x;
    torch::Tensor y;
    torch::Tensor plotX;
    if (!loadSeriesFromCsv(config, x, y, plotX)) {
        buildSyntheticSeries(config, x, y, plotX);
    }

    const int inputDim = static_cast<int>(x.size(1));
    model.setHiddenLayers(parseHiddenLayers(config.hidden_layers_csv));
    model.setLags(std::vector<std::vector<int>>(static_cast<size_t>(inputDim), std::vector<int>{1}));
    model.initializeNetwork(1, config.activation);

    const double split = std::min(0.95, std::max(0.1, config.train_split_ratio));
    const int64_t nTrain = static_cast<int64_t>(x.size(0) * split);
    torch::Tensor xTrain = x.slice(0, 0, nTrain);
    torch::Tensor yTrain = y.slice(0, 0, nTrain);
    torch::Tensor xTest = x.slice(0, nTrain, x.size(0));
    torch::Tensor yTest = y.slice(0, nTrain, y.size(0));

    model.setTensorData(DataType::Train, xTrain, yTrain);
    model.setTensorData(DataType::Test, xTest, yTest);

    std::vector<double> losses = model.train(config.epochs, config.batch_size, config.learning_rate);
    if (losses.empty() || !std::isfinite(losses.back())) {
        throw std::runtime_error("FFN training produced empty/non-finite loss history.");
    }
    result.final_loss = losses.back();

    torch::Tensor predTest = model.forward(DataType::Test);
    if (!predTest.defined() || predTest.size(0) != yTest.size(0) || !predTest.isfinite().all().item<bool>()) {
        throw std::runtime_error("FFN prediction on test set failed or produced non-finite values.");
    }

    if (config.evaluate_metrics) {
        std::map<std::string, double> metrics = model.evaluate();
        auto it = metrics.find("mse");
        if (it != metrics.end()) {
            if (!std::isfinite(it->second)) {
                throw std::runtime_error("FFN evaluation produced non-finite MSE.");
            }
            result.mse = it->second;
        }
    }

    // Keep metrics on held-out test set, but plot full-series predictions for better visual coverage.
    model.setTensorData(DataType::Test, x, y);
    torch::Tensor predFull = model.forward(DataType::Test);
    if (!predFull.defined() || predFull.size(0) != y.size(0) || !predFull.isfinite().all().item<bool>()) {
        throw std::runtime_error("Full-series prediction for plotting failed or produced non-finite values.");
    }
    fillPlotVectors(result, plotX, y, predFull);
    result.success = true;
    result.message = config.use_csv_data ? "FFN run completed with CSV input." : "FFN run completed with synthetic input.";
    return result;
}
