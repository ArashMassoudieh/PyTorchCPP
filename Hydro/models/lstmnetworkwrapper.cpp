#include "lstmnetworkwrapper.h"

#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <vector>

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
    if (layers.empty()) layers = {32};
    return layers;
}

int maxConfiguredLag(const std::string& lagSpec) {
    int maxLag = 1;
    std::stringstream groups(lagSpec);
    std::string group;
    while (std::getline(groups, group, ';')) {
        std::stringstream groupStream(group);
        std::string token;
        while (std::getline(groupStream, token, ',')) {
            try {
                const int lag = std::stoi(token);
                if (lag > maxLag) maxLag = lag;
            } catch (...) {}
        }
    }
    return std::max(1, maxLag);
}

std::vector<std::string> splitCsvRow(const std::string& line) {
    std::vector<std::string> cols;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) cols.push_back(cell);
    return cols;
}

bool loadSeriesFromCsv(const HydroRunConfig& config,
                       torch::Tensor& x,
                       torch::Tensor& y,
                       torch::Tensor& plotX) {
    if (!config.use_csv_data) return false;
    if (config.csv_path.empty()) throw std::runtime_error("CSV data source selected but csv_path is empty.");

    std::ifstream in(config.csv_path);
    if (!in.is_open()) throw std::runtime_error("Unable to open CSV file: " + config.csv_path);

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
                plotXs.push_back((config.csv_x_column >= 0 &&
                                  config.csv_x_column < static_cast<int>(cols.size()) &&
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

    if (ys.size() < 10) throw std::runtime_error("CSV parsing yielded too few numeric samples (<10).");

    const int64_t samples = static_cast<int64_t>(ys.size());
    const int64_t inputDim = static_cast<int64_t>(flatInputs.size() / ys.size());
    if (inputDim <= 0 || static_cast<size_t>(samples * inputDim) != flatInputs.size()) {
        throw std::runtime_error("CSV parsing yielded inconsistent input feature widths.");
    }

    x = torch::from_blob(flatInputs.data(), {samples, inputDim}, torch::kFloat32).clone();
    y = torch::from_blob(ys.data(), {samples, 1}, torch::kFloat32).clone();
    plotX = torch::from_blob(plotXs.data(), {samples, 1}, torch::kFloat32).clone();
    return true;
}

void buildSyntheticSeries(const HydroRunConfig& config,
                          torch::Tensor& x,
                          torch::Tensor& y,
                          torch::Tensor& plotX) {
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

        std::vector<double> allT, allTemp, allPress, allFlow, allConc, allVel;
        allT.reserve(static_cast<size_t>(totalSteps));
        allTemp.reserve(static_cast<size_t>(totalSteps));
        allPress.reserve(static_cast<size_t>(totalSteps));
        allFlow.reserve(static_cast<size_t>(totalSteps));
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
            const double r = (tq - allT[lo]) / (allT[hi] - allT[lo]);
            return vals[lo] * (1.0 - r) + vals[hi] * r;
        };

        for (int64_t i = 0; i < n; ++i) {
            const double tt = tc[i].item<double>();
            const double target = 0.4 * interpol(allTemp, tt - 0.1) +
                                  0.3 * interpol(allPress, tt - 0.3) +
                                  0.2 * interpol(allConc, tt - 0.2) +
                                  0.1 * interpol(allVel, tt - 0.5) +
                                  0.05 * (static_cast<double>(std::rand()) / RAND_MAX - 0.5);
            const size_t k = static_cast<size_t>(i);
            inTemp[k] = static_cast<float>(interpol(allTemp, tt));
            inPress[k] = static_cast<float>(interpol(allPress, tt));
            inFlow[k] = static_cast<float>(interpol(allFlow, tt));
            inConc[k] = static_cast<float>(interpol(allConc, tt));
            inVel[k] = static_cast<float>(interpol(allVel, tt));
            ys[k] = static_cast<float>(target);
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



    if (profile == "watershed_balance") {
        auto tc = t.squeeze(1).contiguous();
        const int64_t n = tc.size(0);
        const double tStart = tc[0].item<double>();
        const double tEnd = tc[n - 1].item<double>();
        const double dt = (n > 1) ? 1.0 / static_cast<double>(n - 1) : 1.0;
        constexpr double kPi = 3.14159265358979323846;

        std::vector<float> flatInputs;
        std::vector<float> ys(static_cast<size_t>(n), 0.0f);
        flatInputs.reserve(static_cast<size_t>(n) * 7);
        double soilStorage = 12.0;
        double groundwaterStorage = 18.0;
        for (int64_t i = 0; i < n; ++i) {
            const double tt = tc[i].item<double>();
            const double r = (n > 1) ? static_cast<double>(i) / static_cast<double>(n - 1) : 0.0;
            const double stormA = 16.0 * std::exp(-0.5 * std::pow((tt - (tStart + 0.18 * (tEnd - tStart))) / std::max(0.05, 0.035 * (tEnd - tStart)), 2.0));
            const double stormB = 10.0 * std::exp(-0.5 * std::pow((tt - (tStart + 0.46 * (tEnd - tStart))) / std::max(0.05, 0.055 * (tEnd - tStart)), 2.0));
            const double stormC = 7.0 * std::exp(-0.5 * std::pow((tt - (tStart + 0.78 * (tEnd - tStart))) / std::max(0.05, 0.08 * (tEnd - tStart)), 2.0));
            const double rainfall = stormA + stormB + stormC + 1.5 * std::max(0.0, std::sin(2.0 * kPi * r * 4.0));
            const double temperature = 4.0 + 16.0 * std::sin(kPi * r - 0.25);
            const double snowpackFactor = std::max(0.0, 1.0 - temperature / 4.0);
            const double snowmelt = std::max(0.0, temperature - 1.0) * (0.12 + 0.18 * snowpackFactor);
            const double evapotranspiration = std::max(0.0, 0.06 * (temperature + 3.0) * (0.6 + 0.4 * std::sin(kPi * r)));
            const double imperviousFraction = 0.12 + 0.10 * std::sin(2.0 * kPi * r + 0.5);
            const double effectivePrecip = rainfall + snowmelt;
            const double infiltration = std::min(effectivePrecip * (0.55 + 0.20 * std::sin(2.0 * kPi * r - 0.3)), std::max(0.0, 30.0 - soilStorage));
            const double quickRunoff = effectivePrecip * std::max(0.0, imperviousFraction) + std::max(0.0, effectivePrecip - infiltration) * 0.45;
            const double recharge = 0.10 * soilStorage;
            const double baseflow = 0.045 * groundwaterStorage;
            const double lateralFlow = 0.035 * soilStorage;
            const double runoff = quickRunoff + lateralFlow + baseflow;
            soilStorage = std::max(0.0, soilStorage + (infiltration - evapotranspiration - recharge - lateralFlow) * dt);
            groundwaterStorage = std::max(0.0, groundwaterStorage + (recharge - baseflow) * dt);

            flatInputs.push_back(static_cast<float>(r));
            flatInputs.push_back(static_cast<float>(effectivePrecip));
            flatInputs.push_back(static_cast<float>(evapotranspiration));
            flatInputs.push_back(static_cast<float>(temperature));
            flatInputs.push_back(static_cast<float>(soilStorage));
            flatInputs.push_back(static_cast<float>(groundwaterStorage));
            flatInputs.push_back(static_cast<float>(imperviousFraction));
            ys[static_cast<size_t>(i)] = static_cast<float>(runoff);
        }

        x = torch::from_blob(flatInputs.data(), {n, 7}, torch::kFloat32).clone();
        y = torch::from_blob(ys.data(), {n, 1}, torch::kFloat32).clone();
        return;
    }

    if (profile == "rainfall_runoff") {
        auto tc = t.squeeze(1).contiguous();
        const int64_t n = tc.size(0);
        const double tStart = tc[0].item<double>();
        const double tEnd = tc[n - 1].item<double>();
        // Use normalized simulation time for storage dynamics and model input so the displayed
        // t-range remains a plotting/export choice instead of changing the synthetic process scale.
        const double dt = (n > 1) ? 1.0 / static_cast<double>(n - 1) : 1.0;
        constexpr double kPi = 3.14159265358979323846;

        std::vector<float> flatInputs;
        std::vector<float> ys(static_cast<size_t>(n), 0.0f);
        flatInputs.reserve(static_cast<size_t>(n) * 5);
        double storage = 8.0;
        for (int64_t i = 0; i < n; ++i) {
            const double tt = tc[i].item<double>();
            const double r = (n > 1) ? static_cast<double>(i) / static_cast<double>(n - 1) : 0.0;
            const double storm1 = 18.0 * std::exp(-0.5 * std::pow((tt - (tStart + 0.22 * (tEnd - tStart))) / std::max(0.05, 0.04 * (tEnd - tStart)), 2.0));
            const double storm2 = 12.0 * std::exp(-0.5 * std::pow((tt - (tStart + 0.58 * (tEnd - tStart))) / std::max(0.05, 0.07 * (tEnd - tStart)), 2.0));
            const double seasonalRain = 2.0 * std::max(0.0, std::sin(2.0 * kPi * r * 3.0));
            const double rain = storm1 + storm2 + seasonalRain;
            const double temp = 12.0 + 10.0 * std::sin(2.0 * kPi * r - 0.4);
            const double et = std::max(0.0, 0.08 * (temp + 5.0));
            const double quickflow = 0.35 * rain;
            const double baseflow = 0.08 * storage;
            const double runoff = quickflow + baseflow;
            storage = std::max(0.0, storage + (rain - et - runoff) * dt);

            flatInputs.push_back(static_cast<float>(r));
            flatInputs.push_back(static_cast<float>(rain));
            flatInputs.push_back(static_cast<float>(et));
            flatInputs.push_back(static_cast<float>(temp));
            flatInputs.push_back(static_cast<float>(storage));
            ys[static_cast<size_t>(i)] = static_cast<float>(runoff);
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

struct SequenceData {
    torch::Tensor xSeq;
    torch::Tensor ySeq;
    torch::Tensor plotSeq;
};

SequenceData makeSequences(const torch::Tensor& x,
                           const torch::Tensor& y,
                           const torch::Tensor& plotX,
                           int sequenceLength) {
    if (!x.defined() || !y.defined() || x.dim() != 2 || y.dim() != 2) {
        throw std::runtime_error("LSTM sequence builder expects 2-D x/y tensors.");
    }
    sequenceLength = std::max(2, sequenceLength);
    const int64_t n = x.size(0);
    const int64_t inputDim = x.size(1);
    const int64_t m = n - sequenceLength + 1;
    if (m < 4) throw std::runtime_error("Too few samples for requested LSTM sequence length.");

    std::vector<torch::Tensor> windows;
    windows.reserve(static_cast<size_t>(m));
    for (int64_t i = 0; i < m; ++i) {
        windows.push_back(x.slice(0, i, i + sequenceLength).unsqueeze(0));
    }

    SequenceData seq;
    seq.xSeq = torch::cat(windows, 0).contiguous().view({m, sequenceLength, inputDim});
    seq.ySeq = y.slice(0, sequenceLength - 1, n).contiguous();
    seq.plotSeq = plotX.slice(0, sequenceLength - 1, plotX.size(0)).contiguous();
    return seq;
}

struct HydroLSTMImpl : torch::nn::Module {
    HydroLSTMImpl(int64_t inputDim, int64_t hiddenDim, int64_t outputDim, int64_t numLayers)
        : lstm(torch::nn::LSTMOptions(inputDim, hiddenDim)
                   .num_layers(numLayers)
                   .batch_first(true)),
          fc(hiddenDim, outputDim) {
        register_module("lstm", lstm);
        register_module("fc", fc);
    }

    torch::Tensor forward(const torch::Tensor& x) {
        auto outTuple = lstm->forward(x);
        torch::Tensor out = std::get<0>(outTuple);
        torch::Tensor last = out.select(1, out.size(1) - 1);
        return fc->forward(last);
    }

    torch::nn::LSTM lstm{nullptr};
    torch::nn::Linear fc{nullptr};
};
TORCH_MODULE(HydroLSTM);

double tensorMSEValue(const torch::Tensor& pred, const torch::Tensor& truth) {
    return torch::mse_loss(pred, truth).item<double>();
}

void fillPlotVectors(HydroRunResult& result,
                     const torch::Tensor& x,
                     const torch::Tensor& yTrue,
                     const torch::Tensor& yPred) {
    auto xc = x.squeeze(1).contiguous();
    auto tc = yTrue.squeeze(1).contiguous();
    auto pc = yPred.squeeze(1).contiguous();
    const int64_t n = std::min({xc.size(0), tc.size(0), pc.size(0)});
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


HydroRunResult LSTMNetworkWrapper::train(const HydroRunConfig& config, bool physicsInformed) {
    HydroRunResult result;
    torch::manual_seed(static_cast<uint64_t>(std::max(0, config.random_seed)));

    torch::Tensor x, y, plotX;
    if (!loadSeriesFromCsv(config, x, y, plotX)) buildSyntheticSeries(config, x, y, plotX);

    const bool needsForcing = physicsInformed &&
        (config.pinn_physics_profile == "linear_reservoir" ||
        config.pinn_physics_profile == "cstr_first_order" ||
        config.pinn_physics_profile == "water_balance");

    if (needsForcing && x.defined() && x.dim() == 2 && x.size(1) < 2) {
        torch::Tensor forcing;
        if (config.pinn_physics_profile == "water_balance") {
            const double c = std::max(1.0e-8, config.runoff_coeff);
            forcing = torch::clamp(y / c, 0.0);
        } else {
            const double g = std::max(1.0e-8, config.forcing_gain);
            forcing = torch::clamp((config.lambda_decay * y) / g, -1.0e6, 1.0e6);
        }
        x = torch::cat({x.slice(1, 0, 1), forcing}, 1).contiguous();
    }

    const std::vector<int> hiddenLayers = parseHiddenLayers(config.hidden_layers_csv);
    const int64_t hiddenDim = static_cast<int64_t>(hiddenLayers.front());
    const int64_t numLayers = static_cast<int64_t>(std::max<size_t>(1, hiddenLayers.size()));
    const int sequenceLength = std::max(2, maxConfiguredLag(config.input_lags_csv) + 1);

    SequenceData seq = makeSequences(x, y, plotX, sequenceLength);
    const int64_t totalSeq = seq.xSeq.size(0);
    const double split = std::min(0.95, std::max(0.1, config.train_split_ratio));
    const int64_t nTrain = std::max<int64_t>(2, std::min<int64_t>(totalSeq - 1, static_cast<int64_t>(totalSeq * split)));

    torch::Tensor xTrain = seq.xSeq.slice(0, 0, nTrain).contiguous();
    torch::Tensor yTrain = seq.ySeq.slice(0, 0, nTrain).contiguous();
    torch::Tensor xTest = seq.xSeq.slice(0, nTrain, totalSeq).contiguous();
    torch::Tensor yTest = seq.ySeq.slice(0, nTrain, totalSeq).contiguous();

    HydroLSTM model(seq.xSeq.size(2), hiddenDim, y.size(1), numLayers);
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(config.learning_rate).weight_decay(config.weight_decay));

    std::vector<double> losses;
    const int64_t trainN = xTrain.size(0);
    const int batchSize = std::max(1, config.batch_size);
    const double lambda = config.lambda_decay;
    const double dt = ((config.synthetic_profile == "watershed_balance" || config.synthetic_profile == "rainfall_runoff"))
                          ? 1.0 / static_cast<double>(std::max<int64_t>(2, x.size(0)) - 1)
                          : std::max(1.0e-8, config.physics_dt);

    auto physicsResidualLoss = [&]() {
        torch::Tensor p = model->forward(xTrain);
        if (p.size(0) < 2) return torch::zeros({}, p.options());
        torch::Tensor dy = (p.slice(0, 1, p.size(0)) - p.slice(0, 0, p.size(0) - 1)) / dt;
        torch::Tensor yMid = p.slice(0, 1, p.size(0));
        torch::Tensor residual;
        if (needsForcing && config.pinn_physics_profile == "water_balance" &&
            (config.synthetic_profile == "watershed_balance" || config.synthetic_profile == "rainfall_runoff") && xTrain.size(2) >= 5) {
            // watershed_balance/rainfall_runoff columns start [normalized_time, effective precipitation, evapotranspiration, temperature, soil_storage].
            torch::Tensor lastStep = xTrain.select(1, xTrain.size(1) - 1);
            torch::Tensor rain = lastStep.slice(1, 1, 2).slice(0, 1, lastStep.size(0));
            torch::Tensor et = lastStep.slice(1, 2, 3).slice(0, 1, lastStep.size(0));
            torch::Tensor storageNow = lastStep.slice(1, 4, 5).slice(0, 1, lastStep.size(0));
            torch::Tensor storagePrev = lastStep.slice(1, 4, 5).slice(0, 0, lastStep.size(0) - 1);
            torch::Tensor dSdt = (storageNow - storagePrev) / dt;
            residual = rain - et - yMid - dSdt;
        } else if (needsForcing) {
            const double effectiveGain =
                (config.pinn_physics_profile == "water_balance") ? config.runoff_coeff : config.forcing_gain;
            torch::Tensor forcing = xTrain.slice(0, 1, xTrain.size(0)).select(1, xTrain.size(1) - 1).slice(1, 1, 2);
            residual = dy + lambda * yMid - effectiveGain * forcing;
        } else {
            residual = dy + lambda * yMid;
        }
        return torch::mean(residual * residual);
    };

    for (int epoch = 0; epoch < std::max(1, config.epochs); ++epoch) {
        model->train();
        torch::Tensor order = config.shuffle_training ? torch::randperm(trainN, torch::kLong) : torch::arange(trainN, torch::kLong);
        double epochDataLoss = 0.0;
        int64_t seen = 0;

        for (int64_t start = 0; start < trainN; start += batchSize) {
            const int64_t end = std::min<int64_t>(start + batchSize, trainN);
            torch::Tensor idx = order.slice(0, start, end);
            torch::Tensor xb = xTrain.index_select(0, idx);
            torch::Tensor yb = yTrain.index_select(0, idx);

            optimizer.zero_grad();
            torch::Tensor pred = model->forward(xb);
            torch::Tensor dataLoss = torch::mse_loss(pred, yb);
            torch::Tensor loss = dataLoss;
            if (physicsInformed) {
                torch::Tensor physLoss = physicsResidualLoss();
                loss = config.data_weight * dataLoss + config.physics_weight * physLoss;
            }
            loss.backward();
            optimizer.step();

            const int64_t count = end - start;
            epochDataLoss += loss.item<double>() * static_cast<double>(count);
            seen += count;
        }
        losses.push_back(epochDataLoss / static_cast<double>(std::max<int64_t>(1, seen)));
    }

    if (losses.empty() || !std::isfinite(losses.back())) {
        throw std::runtime_error(physicsInformed ? "LSTM-PINN training produced empty/non-finite loss history." : "LSTM training produced empty/non-finite loss history.");
    }
    result.final_loss = losses.back();

    model->eval();
    torch::NoGradGuard noGrad;
    torch::Tensor predTest = model->forward(xTest);
    if (!predTest.defined() || predTest.size(0) != yTest.size(0) || !predTest.isfinite().all().item<bool>()) {
        throw std::runtime_error(physicsInformed ? "LSTM-PINN prediction failed or produced non-finite values." : "LSTM prediction failed or produced non-finite values.");
    }
    if (config.evaluate_metrics) result.mse = tensorMSEValue(predTest, yTest);

    torch::Tensor predFull = model->forward(seq.xSeq);
    if (!predFull.defined() || predFull.size(0) != seq.ySeq.size(0) || !predFull.isfinite().all().item<bool>()) {
        throw std::runtime_error(physicsInformed ? "Full-series LSTM-PINN prediction for plotting failed or produced non-finite values." : "Full-series LSTM prediction for plotting failed or produced non-finite values.");
    }
    fillPlotVectors(result, seq.plotSeq, seq.ySeq, predFull);
    result.success = true;
    result.message = physicsInformed
        ? (config.use_csv_data ? "LSTM-PINN run completed with CSV input." : "LSTM-PINN run completed with synthetic input.")
        : (config.use_csv_data ? "LSTM run completed with CSV input." : "LSTM run completed with synthetic input.");
    return result;
}
