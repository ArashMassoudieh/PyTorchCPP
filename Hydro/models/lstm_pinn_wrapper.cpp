#include "lstm_pinn_wrapper.h"
#include "lstmnetworkwrapper.h"
#include "hydro_lstm_module.h"

#include "../dataset/chronological_split.h"
#include "../dataset/reservoir_physics_tensor_builder.h"
#include "../evaluation/hydro_metrics.h"
#include "../evaluation/model_checkpoint.h"

#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
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
            const int value = std::stoi(token);
            if (value > 0) layers.push_back(value);
        } catch (...) {}
    }
    if (layers.empty()) layers = {32};
    return layers;
}

std::vector<double> tensorValues(const torch::Tensor& tensor) {
    auto values = tensor.detach().to(torch::kCPU).reshape({-1}).contiguous();
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(values.size(0)));
    for (int64_t i = 0; i < values.size(0); ++i) out.push_back(values[i].item<double>());
    return out;
}

struct SequenceData {
    torch::Tensor x;
    torch::Tensor y;
    torch::Tensor time;
};

SequenceData makeSequences(const torch::Tensor& x,
                           const torch::Tensor& y,
                           const torch::Tensor& time,
                           int sequenceLength) {
    if (!x.defined() || !y.defined() || !time.defined() || x.dim() != 2 || y.dim() != 2 ||
        x.size(0) != y.size(0) || time.numel() != x.size(0)) {
        throw std::runtime_error("LSTM-PINN sequence builder expects aligned 2-D x/y tensors and physical time.");
    }
    sequenceLength = std::max(2, sequenceLength);
    if (x.size(0) < sequenceLength + 3) {
        throw std::runtime_error("Too few samples for requested LSTM-PINN sequence length.");
    }

    regularPhysicalTimeStepFromTime(time);
    std::vector<torch::Tensor> sequences;
    sequences.reserve(static_cast<std::size_t>(x.size(0) - sequenceLength + 1));
    for (int64_t end = sequenceLength - 1; end < x.size(0); ++end) {
        sequences.push_back(x.slice(0, end - sequenceLength + 1, end + 1));
    }

    SequenceData result;
    result.x = torch::stack(sequences, 0).contiguous();
    result.y = y.slice(0, sequenceLength - 1, y.size(0)).contiguous();
    result.time = time.reshape({-1, 1}).slice(0, sequenceLength - 1, time.numel()).contiguous();
    return result;
}

void fillPlotVectors(HydroRunResult& result,
                     const torch::Tensor& time,
                     const torch::Tensor& truth,
                     const torch::Tensor& prediction) {
    auto tc = time.reshape({-1}).to(torch::kCPU).contiguous();
    auto yc = truth.reshape({-1}).to(torch::kCPU).contiguous();
    auto pc = prediction.reshape({-1}).to(torch::kCPU).contiguous();
    result.x.reserve(static_cast<std::size_t>(tc.size(0)));
    result.y_true.reserve(static_cast<std::size_t>(tc.size(0)));
    result.y_pred.reserve(static_cast<std::size_t>(tc.size(0)));
    for (int64_t i = 0; i < tc.size(0); ++i) {
        result.x.push_back(tc[i].item<double>());
        result.y_true.push_back(yc[i].item<double>());
        result.y_pred.push_back(pc[i].item<double>());
    }
}

} // namespace

HydroRunResult LSTMPINNWrapper::train(const HydroRunConfig& config) {
    if (config.pinn_physics_profile != "linear_reservoir") {
        LSTMNetworkWrapper backend;
        return backend.train(config, true);
    }
    if (config.normalization != "none") {
        throw std::invalid_argument("LSTM-PINN runoff-reservoir physics must be trained in physical units (normalization=none).");
    }

    HydroRunResult result;
    torch::manual_seed(static_cast<uint64_t>(std::max(0, config.random_seed)));

    torch::Tensor x, y, plotX;
    if (!loadReservoirPhysicsTensors(config, x, y, plotX)) {
        throw std::runtime_error("Unable to construct reduced-reservoir LSTM-PINN tensors.");
    }
    if (x.dim() != 2 || x.size(1) < 2) {
        throw std::runtime_error("LSTM-PINN reservoir physics requires [time, Peff, ...] input features.");
    }

    SequenceData seq = makeSequences(x, y, plotX, config.lstm_sequence_length);
    const ChronologicalSplit split = makeChronologicalSplit(seq.x.size(0),
                                                            config.train_split_ratio,
                                                            config.validation_split_ratio);
    const int64_t nTrain = split.train_end;
    torch::Tensor xTrain = seq.x.slice(0, 0, nTrain).contiguous();
    torch::Tensor yTrain = seq.y.slice(0, 0, nTrain).contiguous();
    torch::Tensor xValidation = seq.x.slice(0, nTrain, split.validation_end).contiguous();
    torch::Tensor yValidation = seq.y.slice(0, nTrain, split.validation_end).contiguous();
    torch::Tensor xTest = seq.x.slice(0, split.validation_end, seq.x.size(0)).contiguous();
    torch::Tensor yTest = seq.y.slice(0, split.validation_end, seq.y.size(0)).contiguous();

    const std::vector<int> hiddenLayers = parseHiddenLayers(config.hidden_layers_csv);
    const int64_t hiddenDim = static_cast<int64_t>(hiddenLayers.front());
    const int64_t numLayers = static_cast<int64_t>(std::max<std::size_t>(1, hiddenLayers.size()));
    HydroLSTM model(seq.x.size(2), hiddenDim, 1, numLayers);
    torch::optim::Adam optimizer(model->parameters(),
                                 torch::optim::AdamOptions(config.learning_rate).weight_decay(config.weight_decay));

    const double dt = regularPhysicalTimeStepFromTime(seq.time);
    const double k = std::max(1.0e-8, config.latent_storage_recession_per_hour > 0.0
                                       ? config.latent_storage_recession_per_hour
                                       : config.lambda_decay);
    const int64_t trainN = xTrain.size(0);
    const int batchSize = std::max(2, config.batch_size);
    const int warmupEpochs = config.data_weight > 0.0 ? std::max(1, config.epochs / 5) : 0;

    std::vector<torch::Tensor> bestParameters;
    std::vector<double> losses;
    std::vector<double> validationLosses;
    double bestValidationObjective = std::numeric_limits<double>::infinity();
    double bestValidationMse = std::numeric_limits<double>::infinity();
    int bestEpoch = 0;

    for (int epoch = 0; epoch < std::max(1, config.epochs); ++epoch) {
        model->train();
        double epochLoss = 0.0;
        int64_t seen = 0;

        for (int64_t start = 0; start < trainN; start += batchSize) {
            const int64_t end = std::min<int64_t>(start + batchSize, trainN);
            if (end - start < 2) continue;
            torch::Tensor xb = xTrain.slice(0, start, end);
            torch::Tensor yb = yTrain.slice(0, start, end);

            optimizer.zero_grad();
            torch::Tensor pred = model->forward(xb);
            torch::Tensor dataLoss = torch::mse_loss(pred, yb);
            torch::Tensor lastStep = xb.select(1, xb.size(1) - 1);
            torch::Tensor effectiveRain = lastStep.slice(1, 1, 2);
            torch::Tensor dQdt = (pred.slice(0, 1, pred.size(0)) - pred.slice(0, 0, pred.size(0) - 1)) / dt;
            torch::Tensor qNow = pred.slice(0, 1, pred.size(0));
            torch::Tensor peffNow = effectiveRain.slice(0, 1, effectiveRain.size(0));
            torch::Tensor residual = dQdt - k * (peffNow - qNow);
            torch::Tensor physicsLoss = torch::mean(residual * residual);
            torch::Tensor negative = torch::relu(-pred);
            torch::Tensor nonnegativeLoss = torch::mean(negative * negative);

            const double physicsWeight = epoch < warmupEpochs ? 0.0 : config.physics_weight;
            torch::Tensor totalLoss = config.data_weight * dataLoss +
                                      physicsWeight * (physicsLoss + 0.05 * nonnegativeLoss);
            totalLoss.backward();
            optimizer.step();

            const int64_t count = end - start;
            epochLoss += totalLoss.item<double>() * static_cast<double>(count);
            seen += count;
        }
        losses.push_back(epochLoss / static_cast<double>(std::max<int64_t>(1, seen)));

        model->eval();
        double validationMse = 0.0;
        double validationObjective = 0.0;
        {
            torch::NoGradGuard noGrad;
            torch::Tensor predValidation = model->forward(xValidation);
            torch::Tensor dataLoss = torch::mse_loss(predValidation, yValidation);
            validationMse = dataLoss.item<double>();
            torch::Tensor lastStep = xValidation.select(1, xValidation.size(1) - 1);
            torch::Tensor peff = lastStep.slice(1, 1, 2);
            torch::Tensor dQdt = (predValidation.slice(0, 1, predValidation.size(0)) -
                                   predValidation.slice(0, 0, predValidation.size(0) - 1)) / dt;
            torch::Tensor qNow = predValidation.slice(0, 1, predValidation.size(0));
            torch::Tensor residual = dQdt - k * (peff.slice(0, 1, peff.size(0)) - qNow);
            torch::Tensor physicsLoss = torch::mean(residual * residual);
            torch::Tensor negative = torch::relu(-predValidation);
            torch::Tensor nonnegativeLoss = torch::mean(negative * negative);
            validationObjective = (config.data_weight * dataLoss +
                                   config.physics_weight * (physicsLoss + 0.05 * nonnegativeLoss)).item<double>();
        }
        if (!std::isfinite(validationMse) || !std::isfinite(validationObjective)) {
            throw std::runtime_error("LSTM-PINN validation produced a non-finite objective.");
        }
        validationLosses.push_back(validationObjective);

        // For hybrid runs, do not allow a data-only warm-up epoch to become the
        // restored final checkpoint. Select only after physics is active, using
        // the same joint data+physics tradeoff used for training.
        const bool checkpointEligible = (config.physics_weight <= 0.0) || (epoch >= warmupEpochs);
        if (checkpointEligible && validationObjective < bestValidationObjective) {
            bestValidationObjective = validationObjective;
            bestValidationMse = validationMse;
            bestEpoch = epoch + 1;
            bestParameters.clear();
            for (const auto& parameter : model->parameters()) bestParameters.push_back(parameter.detach().clone());
        }
    }

    if (bestParameters.empty()) throw std::runtime_error("LSTM-PINN did not produce a validation-selected checkpoint.");
    {
        torch::NoGradGuard noGrad;
        auto parameters = model->parameters();
        for (std::size_t i = 0; i < parameters.size(); ++i) parameters[i].copy_(bestParameters[i]);
    }

    result.training_loss_history = losses;
    result.validation_loss_history = validationLosses;
    result.best_epoch = bestEpoch;
    result.final_loss = losses.at(static_cast<std::size_t>(bestEpoch - 1));
    result.validation_mse = bestValidationMse;
    result.input_scaler.method = "none";
    result.target_scaler.method = "none";

    {
        const auto checkpoint = temporaryHydroCheckpointPath("hydro_lstm_pinn_reservoir");
        torch::serialize::OutputArchive archive;
        model->save(archive);
        archive.save_to(checkpoint.string());
        result.model_checkpoint = readHydroCheckpoint(checkpoint);
        result.model_checkpoint_format = "torch-module-v1";
        std::filesystem::remove(checkpoint);
    }

    model->eval();
    torch::NoGradGuard noGrad;
    torch::Tensor predTest = model->forward(xTest);
    if (!predTest.defined() || !predTest.isfinite().all().item<bool>()) {
        throw std::runtime_error("LSTM-PINN prediction produced non-finite values.");
    }
    if (config.evaluate_metrics) {
        populateHydroMetrics(result, tensorValues(yTest), tensorValues(predTest));
        if (!hydroMetricsAreFinite(result)) throw std::runtime_error("LSTM-PINN evaluation produced invalid core hydrology metrics.");
    }

    torch::Tensor predFull = model->forward(seq.x);
    fillPlotVectors(result, seq.time, seq.y, predFull);
    result.split.resize(result.x.size(), "test");
    for (std::size_t i = 0; i < result.split.size(); ++i) {
        if (static_cast<int64_t>(i) < split.train_end) result.split[i] = "train";
        else if (static_cast<int64_t>(i) < split.validation_end) result.split[i] = "validation";
    }
    populateHydroPeakMetrics(result);

    if (predFull.size(0) >= 2) {
        torch::Tensor lastStep = seq.x.select(1, seq.x.size(1) - 1);
        torch::Tensor peff = lastStep.slice(1, 1, 2);
        torch::Tensor dQdt = (predFull.slice(0, 1, predFull.size(0)) - predFull.slice(0, 0, predFull.size(0) - 1)) / dt;
        torch::Tensor qNow = predFull.slice(0, 1, predFull.size(0));
        torch::Tensor residual = dQdt - k * (peff.slice(0, 1, peff.size(0)) - qNow);
        result.physics_loss = torch::mean(residual * residual).item<double>();
        auto values = residual.detach().to(torch::kCPU).reshape({-1}).contiguous();
        result.physics_residual.assign(result.x.size(), std::numeric_limits<double>::quiet_NaN());
        for (int64_t i = 0; i < values.size(0); ++i) result.physics_residual[static_cast<std::size_t>(i + 1)] = values[i].item<double>();
        populateHydroPhysicsResidualMetrics(result);
    }

    result.success = true;
    result.message = config.use_hydro_package
        ? "LSTM-PINN completed on Hydro package input with joint reduced-reservoir physics."
        : (config.use_csv_data
           ? "LSTM-PINN completed on CSV input with joint reduced-reservoir physics."
           : "LSTM-PINN completed on synthetic input with joint reduced-reservoir physics.");
    return result;
}