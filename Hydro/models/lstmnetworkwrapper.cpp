#include "lstmnetworkwrapper.h"
#include "hydro_lstm_module.h"
#include "../dataset/chronological_split.h"
#include "../dataset/tensor_scaler.h"
#include "../dataset/hydro_tensor_builder.h"
#include "../dataset/csv_tensor_builder.h"
#include "../evaluation/hydro_metrics.h"
#include "../evaluation/model_checkpoint.h"

#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
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
            const double perviousFraction = std::max(0.0, 1.0 - imperviousFraction);
            const double infiltrationCapacity = effectivePrecip * perviousFraction * (0.55 + 0.20 * std::sin(2.0 * kPi * r - 0.3));
            const double infiltration = std::min(infiltrationCapacity, std::max(0.0, 30.0 - soilStorage));
            const double quickRunoff = std::max(0.0, effectivePrecip - infiltration);
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
                           int sequenceLength,
                           const bool enforceRegularTime) {
    if (!x.defined() || !y.defined() || !plotX.defined() || x.dim() != 2 || y.dim() != 2 ||
        plotX.numel() != x.size(0) || y.size(0) != x.size(0)) {
        throw std::runtime_error("LSTM sequence builder expects 2-D x/y tensors.");
    }
    sequenceLength = std::max(2, sequenceLength);
    const int64_t n = x.size(0);
    const int64_t inputDim = x.size(1);
    const int64_t m = n - sequenceLength + 1;
    if (m < 4) throw std::runtime_error("Too few samples for requested LSTM sequence length.");

    std::vector<torch::Tensor> windows, targets, endpoints;
    windows.reserve(static_cast<size_t>(m));
    double expectedDt = 0.0;
    if (enforceRegularTime) {
        const auto flatTime = plotX.reshape({-1});
        auto intervals = flatTime.slice(0, 1, n) - flatTime.slice(0, 0, n - 1);
        const auto positive = intervals.index({intervals > 0});
        if (positive.numel() == 0) throw std::runtime_error("LSTM package timestamps are not increasing.");
        expectedDt = positive.min().item<double>();
    }
    for (int64_t i = 0; i < m; ++i) {
        if (enforceRegularTime) {
            const auto time = plotX.slice(0, i, i + sequenceLength).reshape({-1});
            const auto intervals = time.slice(0, 1, sequenceLength) - time.slice(0, 0, sequenceLength - 1);
            const double tolerance = std::max(1.0e-12, std::abs(expectedDt) * 1.0e-6);
            if ((torch::abs(intervals - expectedDt) > tolerance).any().item<bool>()) continue;
        }
        windows.push_back(x.slice(0, i, i + sequenceLength).unsqueeze(0));
        targets.push_back(y.slice(0, i + sequenceLength - 1, i + sequenceLength));
        endpoints.push_back(plotX.slice(0, i + sequenceLength - 1, i + sequenceLength));
    }
    if (windows.size() < 4) throw std::runtime_error("Too few contiguous samples for requested LSTM sequence length.");

    SequenceData seq;
    seq.xSeq = torch::cat(windows, 0).contiguous().view({static_cast<int64_t>(windows.size()), sequenceLength, inputDim});
    seq.ySeq = torch::cat(targets, 0).contiguous();
    seq.plotSeq = torch::cat(endpoints, 0).contiguous();
    return seq;
}

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

std::vector<double> tensorValues(const torch::Tensor& tensor) {
    auto values = tensor.detach().to(torch::kCPU).reshape({-1}).contiguous();
    std::vector<double> out;
    out.reserve(static_cast<size_t>(values.size(0)));
    for (int64_t i = 0; i < values.size(0); ++i) out.push_back(values[i].item<double>());
    return out;
}

} // namespace


HydroRunResult LSTMNetworkWrapper::train(const HydroRunConfig& config, bool physicsInformed) {
    HydroRunResult result;
    if (physicsInformed && config.normalization != "none") {
        throw std::invalid_argument("LSTM-PINN normalization requires physical-unit inverse transforms inside the residual; use normalization=none until enabled.");
    }
    torch::manual_seed(static_cast<uint64_t>(std::max(0, config.random_seed)));

    torch::Tensor x, y, plotX;
    if (!loadHydroPackageTensors(config, x, y, plotX)) {
        if (config.use_csv_data) loadHydroCsvTensors(config, x, y, plotX);
        else buildSyntheticSeries(config, x, y, plotX);
    }

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
    const int sequenceLength = std::max(2, config.lstm_sequence_length);

    SequenceData seq = makeSequences(x, y, plotX, sequenceLength, config.use_hydro_package);
    const int64_t totalSeq = seq.xSeq.size(0);
    const ChronologicalSplit split = makeChronologicalSplit(totalSeq,
                                                            config.train_split_ratio,
                                                            config.validation_split_ratio);
    const int64_t nTrain = split.train_end;

    torch::Tensor xTrain = seq.xSeq.slice(0, 0, nTrain).contiguous();
    torch::Tensor yTrain = seq.ySeq.slice(0, 0, nTrain).contiguous();
    torch::Tensor xValidation = seq.xSeq.slice(0, nTrain, split.validation_end).contiguous();
    torch::Tensor yValidation = seq.ySeq.slice(0, nTrain, split.validation_end).contiguous();
    torch::Tensor xTest = seq.xSeq.slice(0, split.validation_end, totalSeq).contiguous();
    torch::Tensor yTest = seq.ySeq.slice(0, split.validation_end, totalSeq).contiguous();
    torch::Tensor yValidationPhysical = yValidation.clone();
    torch::Tensor yTestPhysical = yTest.clone();

    TensorScaler inputScaler;
    TensorScaler targetScaler;
    inputScaler.fit(xTrain, config.normalization);
    targetScaler.fit(yTrain, config.normalization);
    xTrain = inputScaler.transform(xTrain);
    yTrain = targetScaler.transform(yTrain);
    xValidation = inputScaler.transform(xValidation);
    yValidation = targetScaler.transform(yValidation);
    xTest = inputScaler.transform(xTest);
    yTest = targetScaler.transform(yTest);

    HydroLSTM model(seq.xSeq.size(2), hiddenDim, y.size(1), numLayers);
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(config.learning_rate).weight_decay(config.weight_decay));

    std::vector<double> losses;
    std::vector<double> validationLosses;
    std::vector<torch::Tensor> bestParameters;
    double bestValidation = std::numeric_limits<double>::infinity();
    int bestEpoch = 0;
    const int64_t trainN = xTrain.size(0);
    const int batchSize = std::max(1, config.batch_size);
    const double lambda = config.lambda_decay;
    const double dt = physicsInformed
        ? (config.use_hydro_package ? regularPhysicalTimeStepFromTime(plotX)
           : ((config.synthetic_profile == "watershed_balance" || config.synthetic_profile == "rainfall_runoff")
              ? 1.0 / static_cast<double>(std::max<int64_t>(2, x.size(0)) - 1)
              : std::max(1.0e-8, config.physics_dt)))
        : std::max(1.0e-8, config.physics_dt);

    auto physicsResidualLoss = [&]() {
        torch::Tensor p = model->forward(xTrain);
        if (p.size(0) < 2) return torch::zeros({}, p.options());
        torch::Tensor dy = (p.slice(0, 1, p.size(0)) - p.slice(0, 0, p.size(0) - 1)) / dt;
        torch::Tensor yMid = p.slice(0, 1, p.size(0));
        torch::Tensor residual;
        if (needsForcing && config.pinn_physics_profile == "water_balance" &&
            (config.use_hydro_package || config.synthetic_profile == "watershed_balance" || config.synthetic_profile == "rainfall_runoff") && xTrain.size(2) >= 5) {
            // watershed_balance/rainfall_runoff columns start [normalized_time, effective precipitation, evapotranspiration, temperature, soil_storage].
            torch::Tensor lastStep = xTrain.select(1, xTrain.size(1) - 1);
            torch::Tensor rain = lastStep.slice(1, 1, 2).slice(0, 1, lastStep.size(0));
            torch::Tensor et = lastStep.slice(1, 2, 3).slice(0, 1, lastStep.size(0));
            torch::Tensor storageSeries = lastStep.slice(1, 4, 5);
            if (config.synthetic_profile == "watershed_balance" && lastStep.size(1) >= 6) {
                storageSeries = storageSeries + lastStep.slice(1, 5, 6);
            }
            torch::Tensor storageNow = storageSeries.slice(0, 1, lastStep.size(0));
            torch::Tensor storagePrev = storageSeries.slice(0, 0, lastStep.size(0) - 1);
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
            const bool dataWarmup = physicsInformed && config.data_weight > 0.0 && epoch < std::max(1, config.epochs / 5);
            const double effectiveDataWeight = dataWarmup ? std::max(1.0, config.data_weight) : config.data_weight;
            torch::Tensor loss = physicsInformed ? effectiveDataWeight * dataLoss : dataLoss;
            loss.backward();
            optimizer.step();

            const int64_t count = end - start;
            epochDataLoss += loss.item<double>() * static_cast<double>(count);
            seen += count;
        }
        // Conservation requires ordered samples. Evaluate its full chronological
        // gradient once per epoch instead of repeating the same full-sequence
        // gradient inside every shuffled supervised mini-batch.
        if (physicsInformed && epoch >= std::max(1, config.epochs / 5) && config.physics_weight > 0.0) {
            optimizer.zero_grad();
            torch::Tensor physLoss = physicsResidualLoss();
            (config.physics_weight * physLoss).backward();
            optimizer.step();
            result.physics_loss = physLoss.item<double>();
        }
        losses.push_back(epochDataLoss / static_cast<double>(std::max<int64_t>(1, seen)));
        model->eval();
        double epochValidation = 0.0;
        {
            torch::NoGradGuard noGrad;
            const torch::Tensor validationPrediction = targetScaler.inverseTransform(model->forward(xValidation));
            epochValidation = tensorMSEValue(validationPrediction, yValidationPhysical);
        }
        if (!std::isfinite(epochValidation)) throw std::runtime_error("LSTM validation produced a non-finite loss.");
        validationLosses.push_back(epochValidation);
        if (epochValidation < bestValidation) {
            bestValidation = epochValidation;
            bestEpoch = epoch + 1;
            bestParameters.clear();
            for (const auto& parameter : model->parameters()) bestParameters.push_back(parameter.detach().clone());
        }
    }

    if (losses.empty() || !std::isfinite(losses.back())) {
        throw std::runtime_error(physicsInformed ? "LSTM-PINN training produced empty/non-finite loss history." : "LSTM training produced empty/non-finite loss history.");
    }
    result.training_loss_history = losses;
    result.validation_loss_history = validationLosses;
    result.best_epoch = bestEpoch;
    result.input_scaler = inputScaler.exportState();
    result.target_scaler = targetScaler.exportState();
    if (bestParameters.empty()) throw std::runtime_error("LSTM training did not produce a validation-selected checkpoint.");
    {
        torch::NoGradGuard noGrad;
        auto parameters = model->parameters();
        if (parameters.size() != bestParameters.size()) throw std::runtime_error("LSTM checkpoint parameter count changed.");
        for (std::size_t i = 0; i < parameters.size(); ++i) parameters[i].copy_(bestParameters[i]);
    }
    result.final_loss = losses.at(static_cast<std::size_t>(bestEpoch - 1));
    {
        const auto checkpoint = temporaryHydroCheckpointPath(physicsInformed ? "hydro_lstm_pinn" : "hydro_lstm");
        torch::serialize::OutputArchive archive;
        model->save(archive);
        archive.save_to(checkpoint.string());
        result.model_checkpoint = readHydroCheckpoint(checkpoint);
        result.model_checkpoint_format = "torch-module-v1";
        std::filesystem::remove(checkpoint);
    }

    model->eval();
    torch::NoGradGuard noGrad;
    torch::Tensor predValidation = targetScaler.inverseTransform(model->forward(xValidation));
    result.validation_mse = tensorMSEValue(predValidation, yValidationPhysical);
    torch::Tensor predTest = targetScaler.inverseTransform(model->forward(xTest));
    if (!predTest.defined() || predTest.size(0) != yTestPhysical.size(0) || !predTest.isfinite().all().item<bool>()) {
        throw std::runtime_error(physicsInformed ? "LSTM-PINN prediction failed or produced non-finite values." : "LSTM prediction failed or produced non-finite values.");
    }
    if (config.evaluate_metrics) {
        populateHydroMetrics(result, tensorValues(yTestPhysical), tensorValues(predTest));
        if (!hydroMetricsAreFinite(result)) throw std::runtime_error(physicsInformed ? "LSTM-PINN evaluation produced non-finite hydrology metrics." : "LSTM evaluation produced non-finite hydrology metrics.");
    }

    torch::Tensor predFull = targetScaler.inverseTransform(model->forward(inputScaler.transform(seq.xSeq)));
    if (!predFull.defined() || predFull.size(0) != seq.ySeq.size(0) || !predFull.isfinite().all().item<bool>()) {
        throw std::runtime_error(physicsInformed ? "Full-series LSTM-PINN prediction for plotting failed or produced non-finite values." : "Full-series LSTM prediction for plotting failed or produced non-finite values.");
    }
    fillPlotVectors(result, seq.plotSeq, seq.ySeq, predFull);
    result.split.resize(result.x.size(), "test");
    for (size_t i = 0; i < result.split.size(); ++i) {
        if (static_cast<int64_t>(i) < split.train_end) result.split[i] = "train";
        else if (static_cast<int64_t>(i) < split.validation_end) result.split[i] = "validation";
    }
    populateHydroPeakMetrics(result);
    if (physicsInformed && config.pinn_physics_profile == "water_balance" && seq.xSeq.size(0) >= 2 && seq.xSeq.size(2) >= 5) {
        torch::Tensor lastStep = seq.xSeq.select(1, seq.xSeq.size(1) - 1);
        torch::Tensor storage = lastStep.slice(1, 4, 5);
        if (!config.use_hydro_package && config.synthetic_profile == "watershed_balance" && lastStep.size(1) >= 6) {
            storage = storage + lastStep.slice(1, 5, 6);
        }
        torch::Tensor residual = lastStep.slice(1, 1, 2).slice(0, 1, lastStep.size(0))
                                 - lastStep.slice(1, 2, 3).slice(0, 1, lastStep.size(0))
                                 - predFull.slice(0, 1, predFull.size(0))
                                 - (storage.slice(0, 1, storage.size(0)) - storage.slice(0, 0, storage.size(0) - 1)) / dt;
        auto values = residual.detach().to(torch::kCPU).reshape({-1}).contiguous();
        result.physics_residual.assign(result.x.size(), std::numeric_limits<double>::quiet_NaN());
        for (int64_t i = 0; i < values.size(0) && static_cast<size_t>(i + 1) < result.physics_residual.size(); ++i) {
            result.physics_residual[static_cast<size_t>(i + 1)] = values[i].item<double>();
        }
        result.physics_loss = torch::mean(residual * residual).item<double>();
    }
    if (physicsInformed && !result.physics_residual.empty()) {
        populateHydroPhysicsResidualMetrics(result);
    }
    result.success = true;
    result.message = physicsInformed
        ? (config.use_hydro_package ? "LSTM-PINN run completed with Hydro package input." : (config.use_csv_data ? "LSTM-PINN run completed with CSV input." : "LSTM-PINN run completed with synthetic input."))
        : (config.use_hydro_package ? "LSTM run completed with Hydro package input." : (config.use_csv_data ? "LSTM run completed with CSV input." : "LSTM run completed with synthetic input."));
    return result;
}
