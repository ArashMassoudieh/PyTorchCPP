#include "lstm_pinn_wrapper.h"
#include "lstmnetworkwrapper.h"
#include "hydro_lstm_module.h"

#include "../dataset/chronological_split.h"
#include "../dataset/reservoir_physics_tensor_builder.h"
#include "../dataset/tensor_scaler.h"
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

// Keep the recurrent predictor on the same meteorological forcing contract as the
// supervised LSTM whenever the GIStoOHQ eight-column physics tensor is present.
// GIStoOHQ physics layout: [time, I*, P, PET, T, RH, wind, solar].
torch::Tensor predictorFeatures(const torch::Tensor& physicsX) {
    if (physicsX.dim() != 2 || physicsX.size(1) < 3) {
        throw std::runtime_error("LSTM-PINN predictor feature builder requires [time, forcing, ...].");
    }
    if (physicsX.size(1) >= 8) {
        return torch::cat({
            physicsX.slice(1, 2, 3), // P
            physicsX.slice(1, 4, 5), // T
            physicsX.slice(1, 5, 6), // RH
            physicsX.slice(1, 6, 7), // wind
            physicsX.slice(1, 7, 8), // solar
            physicsX.slice(1, 3, 4)  // PET
        }, 1).contiguous();
    }
    // Controlled reduced-reservoir synthetic/CSV contract: [time, I*, P, PET].
    if (physicsX.size(1) >= 4) return physicsX.slice(1, 2, 4).contiguous();
    return physicsX.slice(1, 1, physicsX.size(1)).contiguous();
}

struct SequenceData {
    torch::Tensor x;
    torch::Tensor y;
    torch::Tensor time;
    torch::Tensor peff;
    torch::Tensor precipitation;
};

SequenceData makeSequences(const torch::Tensor& modelX,
                           const torch::Tensor& y,
                           const torch::Tensor& time,
                           const torch::Tensor& peff,
                           const torch::Tensor& precipitation,
                           int sequenceLength) {
    if (!modelX.defined() || !y.defined() || !time.defined() || !peff.defined() ||
        !precipitation.defined() || modelX.dim() != 2 || y.dim() != 2 ||
        modelX.size(0) != y.size(0) || time.numel() != modelX.size(0) ||
        peff.numel() != modelX.size(0) || precipitation.numel() != modelX.size(0)) {
        throw std::runtime_error("LSTM-PINN sequence builder expects aligned model inputs, target, time, I*, and precipitation.");
    }
    sequenceLength = std::max(2, sequenceLength);
    if (modelX.size(0) < sequenceLength + 3) {
        throw std::runtime_error("Too few samples for requested LSTM-PINN sequence length.");
    }

    regularPhysicalTimeStepFromTime(time);
    std::vector<torch::Tensor> sequences;
    sequences.reserve(static_cast<std::size_t>(modelX.size(0) - sequenceLength + 1));
    for (int64_t end = sequenceLength - 1; end < modelX.size(0); ++end) {
        sequences.push_back(modelX.slice(0, end - sequenceLength + 1, end + 1));
    }

    SequenceData result;
    result.x = torch::stack(sequences, 0).contiguous();
    result.y = y.slice(0, sequenceLength - 1, y.size(0)).contiguous();
    result.time = time.reshape({-1, 1}).slice(0, sequenceLength - 1, time.numel()).contiguous();
    result.peff = peff.reshape({-1, 1}).slice(0, sequenceLength - 1, peff.numel()).contiguous();
    result.precipitation = precipitation.reshape({-1, 1})
                               .slice(0, sequenceLength - 1, precipitation.numel()).contiguous();
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

torch::Tensor physicalResidual(const torch::Tensor& predPhysical,
                               const torch::Tensor& peff,
                               const double dt,
                               const double k) {
    if (predPhysical.size(0) < 2) return torch::zeros({}, predPhysical.options());
    torch::Tensor dQdt = (predPhysical.slice(0, 1, predPhysical.size(0)) -
                           predPhysical.slice(0, 0, predPhysical.size(0) - 1)) / dt;
    torch::Tensor qNow = predPhysical.slice(0, 1, predPhysical.size(0));
    torch::Tensor pNow = peff.slice(0, 1, peff.size(0));
    return dQdt - k * (pNow - qNow);
}

void copyParameters(const std::vector<torch::Tensor>& source,
                    const std::vector<torch::Tensor>& destination) {
    if (source.size() != destination.size()) {
        throw std::runtime_error("Cannot warm-start recurrent backbone: parameter counts differ.");
    }
    torch::NoGradGuard noGrad;
    for (std::size_t i = 0; i < source.size(); ++i) {
        if (source[i].sizes() != destination[i].sizes()) {
            throw std::runtime_error("Cannot warm-start recurrent backbone: parameter shapes differ.");
        }
        destination[i].copy_(source[i]);
    }
}

HydroRunResult trainTwoReservoirHybrid(const HydroRunConfig& config) {
    HydroRunResult result;
    torch::manual_seed(static_cast<uint64_t>(std::max(0, config.random_seed)));

    torch::Tensor physicsX, y, plotX;
    if (!loadReservoirPhysicsTensors(config, physicsX, y, plotX)) {
        throw std::runtime_error("Unable to construct process-aware LSTM-PINN tensors.");
    }
    if (physicsX.dim() != 2 || physicsX.size(1) < 4) {
        throw std::runtime_error("Two-reservoir LSTM-PINN requires [time, I*, P, PET, ...] forcing layout.");
    }

    const torch::Tensor modelX = predictorFeatures(physicsX);
    const torch::Tensor peff = physicsX.slice(1, 1, 2).contiguous();
    const torch::Tensor precipitation = physicsX.slice(1, 2, 3).contiguous();
    SequenceData seq = makeSequences(modelX, y, plotX, peff, precipitation,
                                     config.lstm_sequence_length);
    const ChronologicalSplit split = makeChronologicalSplit(seq.x.size(0),
                                                            config.train_split_ratio,
                                                            config.validation_split_ratio);
    const int64_t nTrain = split.train_end;
    const int64_t nValidationEnd = split.validation_end;

    torch::Tensor xTrainPhysical = seq.x.slice(0, 0, nTrain).contiguous();
    torch::Tensor yTrainPhysical = seq.y.slice(0, 0, nTrain).contiguous();
    torch::Tensor xValidationPhysical = seq.x.slice(0, nTrain, nValidationEnd).contiguous();
    torch::Tensor yValidationPhysical = seq.y.slice(0, nTrain, nValidationEnd).contiguous();

    TensorScaler inputScaler;
    TensorScaler targetScaler;
    inputScaler.fit(xTrainPhysical, "standardize");
    targetScaler.fit(yTrainPhysical, "standardize");
    const torch::Tensor xFull = inputScaler.transform(seq.x);
    const torch::Tensor xTrain = xFull.slice(0, 0, nTrain).contiguous();
    const torch::Tensor yTrain = targetScaler.transform(yTrainPhysical);
    const torch::Tensor xValidation = inputScaler.transform(xValidationPhysical);
    const torch::Tensor yValidation = targetScaler.transform(yValidationPhysical);

    const std::vector<int> hiddenLayers = parseHiddenLayers(config.hidden_layers_csv);
    const int64_t hiddenDim = static_cast<int64_t>(hiddenLayers.front());
    const int64_t numLayers = static_cast<int64_t>(std::max<std::size_t>(1, hiddenLayers.size()));
    const double dt = regularPhysicalTimeStepFromTime(seq.time);
    const double fastK = config.storage_coeff;
    const double slowK = config.lambda_decay;
    const double alpha = config.runoff_coeff;
    if (!(fastK > 0.0 && slowK > 0.0 && fastK > slowK && alpha > 0.0 && alpha < 1.0)) {
        throw std::runtime_error("Two-reservoir LSTM-PINN requires fast_k>slow_k>0 and 0<routing_alpha<1.");
    }
    if (dt * fastK > 1.0 || dt * slowK > 1.0) {
        throw std::runtime_error("Two-reservoir LSTM-PINN explicit routing requires dt*k <= 1 for both stores.");
    }

    const int totalEpochs = std::max(1, config.epochs);
    const int pretrainEpochs = std::min(totalEpochs - 1, std::max(10, (2 * totalEpochs) / 5));

    // Stage A: fit a conventional supervised LSTM parent on the same standardized
    // meteorological sequences and copy its recurrent backbone into the hybrid.
    HydroLSTM parent(seq.x.size(2), hiddenDim, 1, numLayers);
    torch::optim::Adam parentOptimizer(parent->parameters(),
        torch::optim::AdamOptions(config.learning_rate).weight_decay(config.weight_decay));
    std::vector<torch::Tensor> bestParentParameters;
    double bestParentValidation = std::numeric_limits<double>::infinity();
    const int batchSize = std::max(2, config.batch_size);
    for (int epoch = 0; epoch < pretrainEpochs; ++epoch) {
        parent->train();
        for (int64_t start = 0; start < nTrain; start += batchSize) {
            const int64_t end = std::min<int64_t>(start + batchSize, nTrain);
            parentOptimizer.zero_grad();
            const torch::Tensor pred = parent->forward(xTrain.slice(0, start, end));
            const torch::Tensor loss = torch::mse_loss(pred, yTrain.slice(0, start, end));
            loss.backward();
            parentOptimizer.step();
        }
        parent->eval();
        torch::NoGradGuard noGrad;
        const double vmse = torch::mse_loss(parent->forward(xValidation), yValidation).item<double>();
        if (vmse < bestParentValidation) {
            bestParentValidation = vmse;
            bestParentParameters.clear();
            for (const auto& p : parent->parameters()) bestParentParameters.push_back(p.detach().clone());
        }
    }
    if (bestParentParameters.empty()) {
        throw std::runtime_error("Supervised LSTM parent did not produce a warm-start checkpoint.");
    }
    copyParameters(bestParentParameters, parent->parameters());

    HydroTwoReservoirLSTM model(seq.x.size(2), hiddenDim, numLayers,
                                dt, fastK, slowK, alpha);
    copyParameters(parent->lstm->parameters(), model->lstm->parameters());
    torch::optim::Adam optimizer(model->parameters(),
        torch::optim::AdamOptions(config.learning_rate).weight_decay(config.weight_decay));

    const double targetVariance = std::max(1.0e-10, yTrainPhysical.var(false).item<double>());
    const double precipitationScale = std::max(1.0e-8,
        seq.precipitation.slice(0, 0, nTrain).pow(2).mean().item<double>());
    const int fineTuneEpochs = std::max(1, totalEpochs - pretrainEpochs);
    std::vector<torch::Tensor> bestParameters;
    std::vector<double> losses;
    std::vector<double> validationLosses;
    double bestValidationMse = std::numeric_limits<double>::infinity();
    int bestEpoch = 0;

    // Stage B: differentiable process-aware fine tuning.  The LSTM learns R_nn,
    // while fast/slow reservoirs enforce routing structurally.  Physics weight is
    // used only for the weak water-availability prior R_nn <= precipitation;
    // it does not force raw P-PET to equal effective runoff.
    for (int epoch = 0; epoch < fineTuneEpochs; ++epoch) {
        model->train();
        optimizer.zero_grad();
        const torch::Tensor runoff = model->runoffGeneration(xTrain);
        const auto routed = model->routeRunoff(runoff);
        const torch::Tensor qPhysical = std::get<0>(routed);
        const torch::Tensor qScaled = targetScaler.transform(qPhysical);
        const torch::Tensor dataLoss = torch::mse_loss(qScaled, yTrain);
        const torch::Tensor pTrain = seq.precipitation.slice(0, 0, nTrain);
        const torch::Tensor excess = torch::relu(runoff - pTrain);
        const torch::Tensor availabilityLoss = torch::mean(excess * excess) / precipitationScale;
        const double ramp = static_cast<double>(epoch + 1) / static_cast<double>(fineTuneEpochs);
        const double effectivePhysicsWeight = config.physics_weight * ramp * ramp;
        const torch::Tensor totalLoss = config.data_weight * dataLoss +
                                        effectivePhysicsWeight * availabilityLoss;
        totalLoss.backward();
        torch::nn::utils::clip_grad_norm_(model->parameters(), 5.0);
        optimizer.step();
        losses.push_back(totalLoss.item<double>());

        model->eval();
        double validationMsePhysical = 0.0;
        {
            torch::NoGradGuard noGrad;
            // Route from the beginning of the record so validation receives the
            // physically carried storage state from the training period.
            const torch::Tensor qThroughValidation = model->forward(xFull.slice(0, 0, nValidationEnd));
            const torch::Tensor qValidation = qThroughValidation.slice(0, nTrain, nValidationEnd);
            validationMsePhysical = torch::mse_loss(qValidation, yValidationPhysical).item<double>();
        }
        validationLosses.push_back(validationMsePhysical);
        if (validationMsePhysical < bestValidationMse) {
            bestValidationMse = validationMsePhysical;
            bestEpoch = pretrainEpochs + epoch + 1;
            bestParameters.clear();
            for (const auto& p : model->parameters()) bestParameters.push_back(p.detach().clone());
        }
    }

    if (bestParameters.empty()) {
        throw std::runtime_error("Two-reservoir LSTM-PINN did not produce a validation checkpoint.");
    }
    copyParameters(bestParameters, model->parameters());

    result.training_loss_history = losses;
    result.validation_loss_history = validationLosses;
    result.best_epoch = bestEpoch;
    result.final_loss = losses.empty() ? bestValidationMse : losses.back();
    result.validation_mse = bestValidationMse;
    result.input_scaler = inputScaler.exportState();
    result.target_scaler = targetScaler.exportState();

    {
        const auto checkpoint = temporaryHydroCheckpointPath("hydro_lstm_pinn_two_reservoir");
        torch::serialize::OutputArchive archive;
        model->save(archive);
        archive.save_to(checkpoint.string());
        result.model_checkpoint = readHydroCheckpoint(checkpoint);
        result.model_checkpoint_format = "torch-module-v1";
        std::filesystem::remove(checkpoint);
    }

    model->eval();
    torch::NoGradGuard noGrad;
    const torch::Tensor predFullPhysical = model->forward(xFull);
    const torch::Tensor yTestPhysical = seq.y.slice(0, nValidationEnd, seq.y.size(0)).contiguous();
    const torch::Tensor predTestPhysical = predFullPhysical.slice(0, nValidationEnd, predFullPhysical.size(0));
    if (!predFullPhysical.defined() || !predFullPhysical.isfinite().all().item<bool>()) {
        throw std::runtime_error("Two-reservoir LSTM-PINN produced non-finite runoff.");
    }
    if (config.evaluate_metrics) {
        populateHydroMetrics(result, tensorValues(yTestPhysical), tensorValues(predTestPhysical));
        if (!hydroMetricsAreFinite(result)) {
            throw std::runtime_error("Two-reservoir LSTM-PINN evaluation produced invalid core hydrology metrics.");
        }
    }

    fillPlotVectors(result, seq.time, seq.y, predFullPhysical);
    result.split.resize(result.x.size(), "test");
    for (std::size_t i = 0; i < result.split.size(); ++i) {
        if (static_cast<int64_t>(i) < split.train_end) result.split[i] = "train";
        else if (static_cast<int64_t>(i) < split.validation_end) result.split[i] = "validation";
    }
    populateHydroPeakMetrics(result);

    const torch::Tensor runoffFull = model->runoffGeneration(xFull);
    const torch::Tensor availabilityResidual = torch::relu(runoffFull - seq.precipitation);
    result.physics_loss = torch::mean(availabilityResidual * availabilityResidual).item<double>();
    result.physics_residual = tensorValues(availabilityResidual);
    populateHydroPhysicsResidualMetrics(result);

    result.success = true;
    result.message = "LSTM-PINN completed with supervised-LSTM backbone warm start, train-only standardization, learned effective runoff generation, and differentiable fast/slow reservoir routing.";
    return result;
}

HydroRunResult trainLinearReservoir(const HydroRunConfig& config) {
    HydroRunResult result;
    torch::manual_seed(static_cast<uint64_t>(std::max(0, config.random_seed)));

    torch::Tensor physicsX, y, plotX;
    if (!loadReservoirPhysicsTensors(config, physicsX, y, plotX)) {
        throw std::runtime_error("Unable to construct reduced-reservoir LSTM-PINN tensors.");
    }
    if (physicsX.dim() != 2 || physicsX.size(1) < 2) {
        throw std::runtime_error("LSTM-PINN reservoir physics requires [time, I*, ...] input features.");
    }

    const torch::Tensor modelX = predictorFeatures(physicsX);
    const torch::Tensor peff = physicsX.slice(1, 1, 2).contiguous();
    const torch::Tensor precipitation = physicsX.size(1) >= 3
        ? physicsX.slice(1, 2, 3).contiguous()
        : peff;
    SequenceData seq = makeSequences(modelX, y, plotX, peff, precipitation,
                                     config.lstm_sequence_length);
    const ChronologicalSplit split = makeChronologicalSplit(seq.x.size(0),
                                                            config.train_split_ratio,
                                                            config.validation_split_ratio);
    const int64_t nTrain = split.train_end;
    torch::Tensor xTrainPhysical = seq.x.slice(0, 0, nTrain).contiguous();
    torch::Tensor yTrainPhysical = seq.y.slice(0, 0, nTrain).contiguous();
    torch::Tensor pTrain = seq.peff.slice(0, 0, nTrain).contiguous();
    torch::Tensor xValidationPhysical = seq.x.slice(0, nTrain, split.validation_end).contiguous();
    torch::Tensor yValidationPhysical = seq.y.slice(0, nTrain, split.validation_end).contiguous();
    torch::Tensor pValidation = seq.peff.slice(0, nTrain, split.validation_end).contiguous();
    torch::Tensor xTestPhysical = seq.x.slice(0, split.validation_end, seq.x.size(0)).contiguous();
    torch::Tensor yTestPhysical = seq.y.slice(0, split.validation_end, seq.y.size(0)).contiguous();

    TensorScaler inputScaler;
    TensorScaler targetScaler;
    inputScaler.fit(xTrainPhysical, "standardize");
    targetScaler.fit(yTrainPhysical, "standardize");
    torch::Tensor xTrain = inputScaler.transform(xTrainPhysical);
    torch::Tensor yTrain = targetScaler.transform(yTrainPhysical);
    torch::Tensor xValidation = inputScaler.transform(xValidationPhysical);
    torch::Tensor yValidation = targetScaler.transform(yValidationPhysical);
    torch::Tensor xTest = inputScaler.transform(xTestPhysical);

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
    const int totalEpochs = std::max(1, config.epochs);
    const int pretrainEpochs = config.data_weight > 0.0
        ? std::min(totalEpochs - 1, std::max(10, (2 * totalEpochs) / 5))
        : 0;
    const int rampEpochs = std::max(1, totalEpochs - pretrainEpochs);
    const double targetVariance = std::max(1.0e-10,
        yTrainPhysical.var(false).item<double>());

    std::vector<torch::Tensor> bestParameters;
    std::vector<double> losses;
    std::vector<double> validationLosses;
    double bestValidationObjective = std::numeric_limits<double>::infinity();
    double bestValidationMse = std::numeric_limits<double>::infinity();
    double physicsReference = std::numeric_limits<double>::quiet_NaN();
    int bestEpoch = 0;

    for (int epoch = 0; epoch < totalEpochs; ++epoch) {
        model->train();
        double epochLoss = 0.0;
        int64_t seen = 0;
        const bool physicsActive = epoch >= pretrainEpochs && config.physics_weight > 0.0;
        const double rampFraction = physicsActive
            ? std::min(1.0, static_cast<double>(epoch - pretrainEpochs + 1) / static_cast<double>(rampEpochs))
            : 0.0;
        const double effectivePhysicsWeight = config.physics_weight * rampFraction * rampFraction;

        for (int64_t start = 0; start < trainN; start += batchSize) {
            const int64_t end = std::min<int64_t>(start + batchSize, trainN);
            if (end - start < 2) continue;
            torch::Tensor xb = xTrain.slice(0, start, end);
            torch::Tensor yb = yTrain.slice(0, start, end);
            torch::Tensor pb = pTrain.slice(0, start, end);

            optimizer.zero_grad();
            torch::Tensor predScaled = model->forward(xb);
            torch::Tensor dataLoss = torch::mse_loss(predScaled, yb);
            torch::Tensor predPhysical = targetScaler.inverseTransform(predScaled);
            torch::Tensor residual = physicalResidual(predPhysical, pb, dt, k);
            torch::Tensor physicsLoss = residual.numel() > 0
                ? torch::mean(residual * residual)
                : torch::zeros({}, predScaled.options());
            if (physicsActive && !std::isfinite(physicsReference)) {
                physicsReference = std::max(1.0e-10, physicsLoss.detach().item<double>());
            }
            const double reference = std::isfinite(physicsReference) ? physicsReference : 1.0;
            torch::Tensor normalizedPhysicsLoss = physicsLoss / reference;
            torch::Tensor negative = torch::relu(-predPhysical);
            torch::Tensor normalizedNonnegativeLoss = torch::mean(negative * negative) / targetVariance;
            torch::Tensor totalLoss = config.data_weight * dataLoss +
                                      effectivePhysicsWeight * (normalizedPhysicsLoss +
                                                                0.05 * normalizedNonnegativeLoss);
            totalLoss.backward();
            optimizer.step();

            const int64_t count = end - start;
            epochLoss += totalLoss.item<double>() * static_cast<double>(count);
            seen += count;
        }
        losses.push_back(epochLoss / static_cast<double>(std::max<int64_t>(1, seen)));

        model->eval();
        double validationMsePhysical = 0.0;
        double validationObjective = 0.0;
        {
            torch::NoGradGuard noGrad;
            torch::Tensor predValidationScaled = model->forward(xValidation);
            torch::Tensor dataLossScaled = torch::mse_loss(predValidationScaled, yValidation);
            validationMsePhysical = targetScaler.mseToPhysical(dataLossScaled.item<double>());
            torch::Tensor predValidationPhysical = targetScaler.inverseTransform(predValidationScaled);
            torch::Tensor residual = physicalResidual(predValidationPhysical, pValidation, dt, k);
            torch::Tensor physicsLoss = residual.numel() > 0
                ? torch::mean(residual * residual)
                : torch::zeros({}, predValidationScaled.options());
            const double reference = std::isfinite(physicsReference) ? physicsReference : 1.0;
            torch::Tensor normalizedPhysicsLoss = physicsLoss / reference;
            torch::Tensor negative = torch::relu(-predValidationPhysical);
            torch::Tensor normalizedNonnegativeLoss = torch::mean(negative * negative) / targetVariance;
            validationObjective = (config.data_weight * dataLossScaled +
                                   effectivePhysicsWeight * (normalizedPhysicsLoss +
                                                             0.05 * normalizedNonnegativeLoss)).item<double>();
        }
        if (!std::isfinite(validationMsePhysical) || !std::isfinite(validationObjective)) {
            throw std::runtime_error("LSTM-PINN validation produced a non-finite objective.");
        }
        validationLosses.push_back(validationObjective);

        const bool checkpointEligible = (config.physics_weight <= 0.0) || (epoch >= pretrainEpochs);
        if (checkpointEligible && validationObjective < bestValidationObjective) {
            bestValidationObjective = validationObjective;
            bestValidationMse = validationMsePhysical;
            bestEpoch = epoch + 1;
            bestParameters.clear();
            for (const auto& parameter : model->parameters()) bestParameters.push_back(parameter.detach().clone());
        }
    }

    if (bestParameters.empty()) throw std::runtime_error("LSTM-PINN did not produce a validation-selected checkpoint.");
    copyParameters(bestParameters, model->parameters());

    result.training_loss_history = losses;
    result.validation_loss_history = validationLosses;
    result.best_epoch = bestEpoch;
    result.final_loss = losses.at(static_cast<std::size_t>(bestEpoch - 1));
    result.validation_mse = bestValidationMse;
    result.input_scaler = inputScaler.exportState();
    result.target_scaler = targetScaler.exportState();

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
    torch::Tensor predTestPhysical = targetScaler.inverseTransform(model->forward(xTest));
    if (!predTestPhysical.defined() || !predTestPhysical.isfinite().all().item<bool>()) {
        throw std::runtime_error("LSTM-PINN prediction produced non-finite values.");
    }
    if (config.evaluate_metrics) {
        populateHydroMetrics(result, tensorValues(yTestPhysical), tensorValues(predTestPhysical));
        if (!hydroMetricsAreFinite(result)) throw std::runtime_error("LSTM-PINN evaluation produced invalid core hydrology metrics.");
    }

    torch::Tensor xFull = inputScaler.transform(seq.x);
    torch::Tensor predFullPhysical = targetScaler.inverseTransform(model->forward(xFull));
    fillPlotVectors(result, seq.time, seq.y, predFullPhysical);
    result.split.resize(result.x.size(), "test");
    for (std::size_t i = 0; i < result.split.size(); ++i) {
        if (static_cast<int64_t>(i) < split.train_end) result.split[i] = "train";
        else if (static_cast<int64_t>(i) < split.validation_end) result.split[i] = "validation";
    }
    populateHydroPeakMetrics(result);

    if (predFullPhysical.size(0) >= 2) {
        torch::Tensor residual = physicalResidual(predFullPhysical, seq.peff, dt, k);
        result.physics_loss = torch::mean(residual * residual).item<double>();
        auto values = residual.detach().to(torch::kCPU).reshape({-1}).contiguous();
        result.physics_residual.assign(result.x.size(), std::numeric_limits<double>::quiet_NaN());
        for (int64_t i = 0; i < values.size(0); ++i) {
            result.physics_residual[static_cast<std::size_t>(i + 1)] = values[i].item<double>();
        }
        populateHydroPhysicsResidualMetrics(result);
    }

    result.success = true;
    result.message = config.use_hydro_package
        ? "LSTM-PINN completed on Hydro package input with supervised-parent warm start, scaled predictor inputs, and ramped normalized reduced-reservoir physics."
        : (config.use_csv_data
           ? "LSTM-PINN completed on CSV input with supervised-parent warm start and ramped normalized reduced-reservoir physics."
           : "LSTM-PINN completed on synthetic input with supervised-parent warm start and ramped normalized reduced-reservoir physics.");
    return result;
}

} // namespace

HydroRunResult LSTMPINNWrapper::train(const HydroRunConfig& config) {
    if (config.pinn_physics_profile == "two_reservoir_hybrid") {
        return trainTwoReservoirHybrid(config);
    }
    if (config.pinn_physics_profile == "linear_reservoir") {
        return trainLinearReservoir(config);
    }
    LSTMNetworkWrapper backend;
    return backend.train(config, true);
}
