#include "ffn_wrapper.h"
#include "../dataset/chronological_split.h"
#include "../dataset/tensor_scaler.h"
#include "../dataset/hydro_tensor_builder.h"
#include "../dataset/csv_tensor_builder.h"
#include "../dataset/lagged_tensor_builder.h"
#include "../evaluation/hydro_metrics.h"
#include "../evaluation/model_checkpoint.h"

#include "neuralnetworkwrapper.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
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

std::vector<std::vector<int>> currentInputLags(int inputDim) {
    return std::vector<std::vector<int>>(static_cast<size_t>(std::max(0, inputDim)), std::vector<int>{1});
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

std::vector<double> tensorValues(const torch::Tensor& tensor) {
    auto values = tensor.detach().to(torch::kCPU).reshape({-1}).contiguous();
    std::vector<double> out;
    out.reserve(static_cast<size_t>(values.size(0)));
    for (int64_t i = 0; i < values.size(0); ++i) out.push_back(values[i].item<double>());
    return out;
}
} // namespace

HydroRunResult FFNWrapper::train(const HydroRunConfig& config) {
    HydroRunResult result;

    torch::manual_seed(static_cast<uint64_t>(std::max(0, config.random_seed)));

    NeuralNetworkWrapper model;
    torch::Tensor x;
    torch::Tensor y;
    torch::Tensor plotX;
    if (!loadHydroPackageTensors(config, x, y, plotX)) {
        if (config.use_csv_data) loadHydroCsvTensors(config, x, y, plotX);
        else buildSyntheticSeries(config, x, y, plotX);
    }

    const int inputDim = static_cast<int>(x.size(1));
    if (config.use_time_lagged_ffn) {
        const auto lagged = buildHydroLaggedTensor(x, config.input_lags_csv);
        x = lagged.inputs;
        y = y.slice(0, lagged.leading_rows, y.size(0)).contiguous();
        plotX = plotX.slice(0, lagged.leading_rows, plotX.size(0)).contiguous();
    }
    model.setHiddenLayers(parseHiddenLayers(config.hidden_layers_csv));
    model.setLags(currentInputLags(static_cast<int>(x.size(1))));
    model.initializeNetwork(1, config.activation);

    const ChronologicalSplit split = makeChronologicalSplit(x.size(0), config.train_split_ratio, config.validation_split_ratio);
    const int64_t nTrain = split.train_end;
    torch::Tensor xTrain = x.slice(0, 0, nTrain);
    torch::Tensor yTrain = y.slice(0, 0, nTrain);
    torch::Tensor xValidation = x.slice(0, nTrain, split.validation_end);
    torch::Tensor yValidation = y.slice(0, nTrain, split.validation_end);
    torch::Tensor xTest = x.slice(0, split.validation_end, x.size(0));
    torch::Tensor yTest = y.slice(0, split.validation_end, y.size(0));

    TensorScaler inputScaler;
    TensorScaler targetScaler;
    inputScaler.fit(xTrain, config.normalization);
    targetScaler.fit(yTrain, config.normalization);
    xTrain = inputScaler.transform(xTrain);
    yTrain = targetScaler.transform(yTrain);
    xValidation = inputScaler.transform(xValidation);
    yValidation = targetScaler.transform(yValidation);
    xTest = inputScaler.transform(xTest);
    torch::Tensor yTestScaled = targetScaler.transform(yTest);

    model.setTensorData(DataType::Train, xTrain, yTrain);
    model.setTensorData(DataType::Test, xTest, yTestScaled);

    std::vector<double> validationLossesScaled;
    int bestEpoch = 0;
    std::vector<double> losses = model.train(config.epochs, config.batch_size, config.learning_rate,
                                             xValidation, yValidation, &validationLossesScaled, &bestEpoch);
    if (losses.empty() || !std::isfinite(losses.back())) {
        throw std::runtime_error("FFN training produced empty/non-finite loss history.");
    }
    result.best_epoch = bestEpoch;
    result.final_loss = losses.at(static_cast<std::size_t>(bestEpoch - 1));
    result.training_loss_history = losses;
    result.validation_loss_history.reserve(validationLossesScaled.size());
    for (const double value : validationLossesScaled) result.validation_loss_history.push_back(targetScaler.mseToPhysical(value));
    result.input_scaler = inputScaler.exportState();
    result.target_scaler = targetScaler.exportState();
    {
        const auto checkpoint = temporaryHydroCheckpointPath("hydro_ffn");
        model.saveModel(checkpoint.string());
        result.model_checkpoint = readHydroCheckpoint(checkpoint);
        result.model_checkpoint_format = "neuralnetworkwrapper-v1";
        std::filesystem::remove(checkpoint);
    }

    model.setTensorData(DataType::Test, xValidation, yValidation);
    torch::Tensor predValidation = targetScaler.inverseTransform(model.forward(DataType::Test));
    result.validation_mse = torch::mse_loss(predValidation, y.slice(0, nTrain, split.validation_end)).item<double>();
    model.setTensorData(DataType::Test, xTest, yTestScaled);
    torch::Tensor predTest = targetScaler.inverseTransform(model.forward(DataType::Test));
    if (!predTest.defined() || predTest.size(0) != yTest.size(0) || !predTest.isfinite().all().item<bool>()) {
        throw std::runtime_error("FFN prediction on test set failed or produced non-finite values.");
    }

    if (config.evaluate_metrics) {
        populateHydroMetrics(result, tensorValues(yTest), tensorValues(predTest));
        if (!hydroMetricsAreFinite(result)) throw std::runtime_error("FFN evaluation produced non-finite hydrology metrics.");
    }

    // Keep metrics on held-out test set, but plot full-series predictions for better visual coverage.
    torch::Tensor xFullScaled = inputScaler.transform(x);
    model.setTensorData(DataType::Test, xFullScaled, targetScaler.transform(y));
    torch::Tensor predFull = targetScaler.inverseTransform(model.forward(DataType::Test));
    if (!predFull.defined() || predFull.size(0) != y.size(0) || !predFull.isfinite().all().item<bool>()) {
        throw std::runtime_error("Full-series prediction for plotting failed or produced non-finite values.");
    }
    fillPlotVectors(result, plotX, y, predFull);
    result.split.resize(result.x.size(), "test");
    for (size_t i = 0; i < result.split.size(); ++i) {
        if (static_cast<int64_t>(i) < split.train_end) result.split[i] = "train";
        else if (static_cast<int64_t>(i) < split.validation_end) result.split[i] = "validation";
    }
    populateHydroPeakMetrics(result);
    result.success = true;
    result.message = config.use_hydro_package ? "FFN run completed with Hydro package input." : (config.use_csv_data ? "FFN run completed with CSV input." : "FFN run completed with synthetic input.");
    return result;
}
