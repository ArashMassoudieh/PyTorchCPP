#include "pinn_wrapper.h"

#include "ffn_pinn_wrapper.h"
#include "../dataset/chronological_split.h"
#include "../dataset/hydro_tensor_builder.h"
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
    if (layers.empty()) layers = {24, 24};
    return layers;
}

torch::nn::Sequential makeNetwork(int64_t inputDim,
                                  const std::vector<int>& hidden,
                                  const std::string& activation) {
    torch::nn::Sequential model;
    int64_t in = inputDim;
    for (const int width : hidden) {
        model->push_back(torch::nn::Linear(in, width));
        if (activation == "relu") model->push_back(torch::nn::ReLU());
        else if (activation == "sigmoid") model->push_back(torch::nn::Sigmoid());
        else model->push_back(torch::nn::Tanh());
        in = width;
    }
    model->push_back(torch::nn::Linear(in, 1));
    return model;
}

std::vector<double> tensorValues(const torch::Tensor& tensor) {
    auto values = tensor.detach().to(torch::kCPU).reshape({-1}).contiguous();
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(values.size(0)));
    for (int64_t i = 0; i < values.size(0); ++i) out.push_back(values[i].item<double>());
    return out;
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

HydroRunResult PINNWrapper::train(const HydroRunConfig& config) {
    // Preserve the generic legacy PINN for non-GIStoOHQ inputs.  The real
    // GIStoOHQ formulation below is physics-only except for the single initial
    // condition required to identify a first-order forced ODE solution.
    if (!config.use_hydro_package || !config.use_latent_storage_physics) {
        HydroRunConfig physicsOnly = config;
        physicsOnly.use_time_lagged_ffn = false;
        physicsOnly.data_weight = 0.0;
        physicsOnly.physics_weight = std::max(1.0, physicsOnly.physics_weight);
        FFNPINNWrapper backend;
        HydroRunResult result = backend.train(physicsOnly);
        if (result.success) result.message = "Physics-only PINN run completed (legacy non-GIStoOHQ backend).";
        return result;
    }
    if (config.normalization != "none") {
        throw std::invalid_argument("Standalone runoff-reservoir PINN requires normalization=none.");
    }

    HydroRunResult result;
    torch::manual_seed(static_cast<uint64_t>(std::max(0, config.random_seed)));

    torch::Tensor x, y, plotX;
    if (!loadHydroPackageTensors(config, x, y, plotX)) {
        throw std::runtime_error("Standalone GIStoOHQ PINN requires a Hydro package input.");
    }
    if (x.dim() != 2 || x.size(1) < 2 || y.dim() != 2 || y.size(1) != 1) {
        throw std::runtime_error("Standalone PINN expects [time, Peff, ...] inputs and scalar runoff targets.");
    }

    const ChronologicalSplit split = makeChronologicalSplit(x.size(0),
                                                            config.train_split_ratio,
                                                            config.validation_split_ratio);
    torch::Tensor xTrain = x.slice(0, 0, split.train_end).contiguous();
    torch::Tensor yTrain = y.slice(0, 0, split.train_end).contiguous();
    torch::Tensor xValidation = x.slice(0, split.train_end, split.validation_end).contiguous();
    torch::Tensor yValidation = y.slice(0, split.train_end, split.validation_end).contiguous();
    torch::Tensor xTest = x.slice(0, split.validation_end, x.size(0)).contiguous();
    torch::Tensor yTest = y.slice(0, split.validation_end, y.size(0)).contiguous();

    torch::nn::Sequential model = makeNetwork(x.size(1), parseHiddenLayers(config.hidden_layers_csv), config.activation);
    torch::optim::Adam optimizer(model->parameters(),
                                 torch::optim::AdamOptions(config.learning_rate).weight_decay(config.weight_decay));

    const double dt = regularPhysicalTimeStepFromTime(plotX);
    const double k = std::max(1.0e-8, config.latent_storage_recession_per_hour);
    const double physicsWeight = std::max(1.0e-12, config.physics_weight);
    constexpr double initialConditionWeight = 1.0;
    constexpr double nonnegativeWeight = 0.05;

    std::vector<torch::Tensor> bestParameters;
    std::vector<double> losses;
    double bestObjective = std::numeric_limits<double>::infinity();
    int bestEpoch = 0;

    // A first-order forced ODE needs one boundary/initial condition.  Only the
    // first observed runoff value is used for that purpose; the remaining
    // training targets do not appear in the optimization objective.
    const torch::Tensor q0Observed = yTrain.slice(0, 0, 1).detach();

    for (int epoch = 0; epoch < std::max(1, config.epochs); ++epoch) {
        model->train();
        optimizer.zero_grad();
        torch::Tensor pred = model->forward(xTrain);
        if (pred.size(0) < 2) throw std::runtime_error("Standalone PINN training segment is too short.");

        torch::Tensor peff = xTrain.slice(1, 1, 2);
        torch::Tensor dQdt = (pred.slice(0, 1, pred.size(0)) - pred.slice(0, 0, pred.size(0) - 1)) / dt;
        torch::Tensor qNow = pred.slice(0, 1, pred.size(0));
        torch::Tensor residual = dQdt - k * (peff.slice(0, 1, peff.size(0)) - qNow);
        torch::Tensor physicsLoss = torch::mean(residual * residual);
        torch::Tensor initialConditionLoss = torch::mse_loss(pred.slice(0, 0, 1), q0Observed);
        torch::Tensor negative = torch::relu(-pred);
        torch::Tensor nonnegativeLoss = torch::mean(negative * negative);
        torch::Tensor totalLoss = physicsWeight * physicsLoss +
                                  initialConditionWeight * initialConditionLoss +
                                  nonnegativeWeight * nonnegativeLoss;
        totalLoss.backward();
        optimizer.step();

        const double objective = totalLoss.item<double>();
        if (!std::isfinite(objective)) throw std::runtime_error("Standalone PINN produced a non-finite objective.");
        losses.push_back(objective);
        if (objective < bestObjective) {
            bestObjective = objective;
            bestEpoch = epoch + 1;
            bestParameters.clear();
            for (const auto& parameter : model->parameters()) bestParameters.push_back(parameter.detach().clone());
        }
    }

    if (bestParameters.empty()) throw std::runtime_error("Standalone PINN did not produce a valid checkpoint.");
    {
        torch::NoGradGuard noGrad;
        auto parameters = model->parameters();
        for (std::size_t i = 0; i < parameters.size(); ++i) parameters[i].copy_(bestParameters[i]);
    }

    result.training_loss_history = losses;
    result.best_epoch = bestEpoch;
    result.final_loss = bestObjective;
    result.input_scaler.method = "none";
    result.target_scaler.method = "none";

    {
        const auto checkpoint = temporaryHydroCheckpointPath("hydro_pinn_reservoir");
        torch::serialize::OutputArchive archive;
        model->save(archive);
        archive.save_to(checkpoint.string());
        result.model_checkpoint = readHydroCheckpoint(checkpoint);
        result.model_checkpoint_format = "torch-sequential-v1";
        std::filesystem::remove(checkpoint);
    }

    model->eval();
    torch::NoGradGuard noGrad;
    torch::Tensor predValidation = model->forward(xValidation);
    result.validation_mse = torch::mse_loss(predValidation, yValidation).item<double>();
    torch::Tensor predTest = model->forward(xTest);
    if (!predTest.defined() || !predTest.isfinite().all().item<bool>()) {
        throw std::runtime_error("Standalone PINN prediction produced non-finite values.");
    }
    if (config.evaluate_metrics) {
        populateHydroMetrics(result, tensorValues(yTest), tensorValues(predTest));
        if (!hydroMetricsAreFinite(result)) {
            throw std::runtime_error("Standalone PINN evaluation produced invalid core hydrology metrics.");
        }
    }

    torch::Tensor predFull = model->forward(x);
    fillPlotVectors(result, plotX, y, predFull);
    result.split.resize(result.x.size(), "test");
    for (std::size_t i = 0; i < result.split.size(); ++i) {
        if (static_cast<int64_t>(i) < split.train_end) result.split[i] = "train";
        else if (static_cast<int64_t>(i) < split.validation_end) result.split[i] = "validation";
    }
    populateHydroPeakMetrics(result);

    torch::Tensor peff = x.slice(1, 1, 2);
    torch::Tensor dQdt = (predFull.slice(0, 1, predFull.size(0)) - predFull.slice(0, 0, predFull.size(0) - 1)) / dt;
    torch::Tensor qNow = predFull.slice(0, 1, predFull.size(0));
    torch::Tensor residual = dQdt - k * (peff.slice(0, 1, peff.size(0)) - qNow);
    result.physics_loss = torch::mean(residual * residual).item<double>();
    auto residualValues = residual.detach().to(torch::kCPU).reshape({-1}).contiguous();
    result.physics_residual.assign(result.x.size(), std::numeric_limits<double>::quiet_NaN());
    for (int64_t i = 0; i < residualValues.size(0); ++i) {
        result.physics_residual[static_cast<std::size_t>(i + 1)] = residualValues[i].item<double>();
    }
    populateHydroPhysicsResidualMetrics(result);

    result.success = true;
    result.message = "Standalone PINN completed with runoff-reservoir physics and one initial-condition observation; no full-series supervised loss was used.";
    return result;
}
