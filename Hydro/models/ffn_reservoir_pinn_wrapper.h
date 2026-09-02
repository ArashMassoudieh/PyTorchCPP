#pragma once

#include "ffn_pinn_wrapper.h"
#include "hydro_run_types.h"
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

namespace hydro_ffn_reservoir_detail {

inline std::vector<int> parseHiddenLayers(const std::string& csv) {
    std::vector<int> layers;
    std::stringstream ss(csv);
    std::string token;
    while (std::getline(ss, token, ',')) {
        try {
            const int value = std::stoi(token);
            if (value > 0) layers.push_back(value);
        } catch (...) {}
    }
    if (layers.empty()) layers = {16, 16};
    return layers;
}

inline torch::nn::Sequential makeNetwork(int64_t inputDim,
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

inline std::vector<double> tensorValues(const torch::Tensor& tensor) {
    auto values = tensor.detach().to(torch::kCPU).reshape({-1}).contiguous();
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(values.size(0)));
    for (int64_t i = 0; i < values.size(0); ++i) out.push_back(values[i].item<double>());
    return out;
}

inline void fillPlotVectors(HydroRunResult& result,
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

} // namespace hydro_ffn_reservoir_detail

/**
 * Reduced-reservoir FFN+PINN implementation shared by Synthetic, CSV, and Hydro
 * package inputs.  The forcing contract is [time, Peff, ...], with
 * Peff=max(P-PET,0), and the residual is
 *
 *     dQ/dt = k (Peff - Q).
 *
 * Known-state synthetic water-balance experiments remain available through the
 * legacy FFNPINNWrapper by selecting physics_profile="water_balance".  This
 * wrapper is used for physics_profile="linear_reservoir".
 */
class FFNReservoirPINNWrapper {
public:
    HydroRunResult train(const HydroRunConfig& config) {
        if (config.pinn_physics_profile != "linear_reservoir") {
            FFNPINNWrapper legacy;
            return legacy.train(config);
        }
        if (config.normalization != "none") {
            throw std::invalid_argument("FFN-PINN runoff-reservoir physics requires normalization=none.");
        }
        if (config.use_time_lagged_ffn) {
            throw std::invalid_argument("FFN-PINN runoff-reservoir physics currently uses current forcing states; disable time-lagged FFN inputs.");
        }

        using namespace hydro_ffn_reservoir_detail;
        HydroRunResult result;
        torch::manual_seed(static_cast<uint64_t>(std::max(0, config.random_seed)));

        torch::Tensor x, y, plotX;
        if (!loadReservoirPhysicsTensors(config, x, y, plotX)) {
            throw std::runtime_error("Unable to construct reduced-reservoir physics tensors.");
        }
        if (x.dim() != 2 || x.size(1) < 2 || y.dim() != 2 || y.size(1) != 1) {
            throw std::runtime_error("FFN-PINN reservoir physics expects [time, Peff, ...] inputs and scalar runoff targets.");
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
        const double k = std::max(1.0e-8, config.latent_storage_recession_per_hour > 0.0
                                           ? config.latent_storage_recession_per_hour
                                           : config.lambda_decay);
        const int64_t trainN = xTrain.size(0);
        const int batchSize = std::max(2, config.batch_size);
        const int warmupEpochs = config.data_weight > 0.0 ? std::max(1, config.epochs / 5) : 0;

        std::vector<torch::Tensor> bestParameters;
        std::vector<double> losses;
        std::vector<double> validationLosses;
        double bestValidation = std::numeric_limits<double>::infinity();
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
                torch::Tensor peff = xb.slice(1, 1, 2);
                torch::Tensor dQdt = (pred.slice(0, 1, pred.size(0)) - pred.slice(0, 0, pred.size(0) - 1)) / dt;
                torch::Tensor qNow = pred.slice(0, 1, pred.size(0));
                torch::Tensor residual = dQdt - k * (peff.slice(0, 1, peff.size(0)) - qNow);
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
            {
                torch::NoGradGuard noGrad;
                validationMse = torch::mse_loss(model->forward(xValidation), yValidation).item<double>();
            }
            if (!std::isfinite(validationMse)) throw std::runtime_error("FFN-PINN validation produced a non-finite loss.");
            validationLosses.push_back(validationMse);
            if (validationMse < bestValidation) {
                bestValidation = validationMse;
                bestEpoch = epoch + 1;
                bestParameters.clear();
                for (const auto& parameter : model->parameters()) bestParameters.push_back(parameter.detach().clone());
            }
        }

        if (bestParameters.empty()) throw std::runtime_error("FFN-PINN did not produce a validation-selected checkpoint.");
        {
            torch::NoGradGuard noGrad;
            auto parameters = model->parameters();
            for (std::size_t i = 0; i < parameters.size(); ++i) parameters[i].copy_(bestParameters[i]);
        }

        result.training_loss_history = losses;
        result.validation_loss_history = validationLosses;
        result.best_epoch = bestEpoch;
        result.final_loss = losses.at(static_cast<std::size_t>(bestEpoch - 1));
        result.validation_mse = bestValidation;
        result.input_scaler.method = "none";
        result.target_scaler.method = "none";

        {
            const auto checkpoint = temporaryHydroCheckpointPath("hydro_ffn_pinn_reservoir");
            torch::serialize::OutputArchive archive;
            model->save(archive);
            archive.save_to(checkpoint.string());
            result.model_checkpoint = readHydroCheckpoint(checkpoint);
            result.model_checkpoint_format = "torch-sequential-v1";
            std::filesystem::remove(checkpoint);
        }

        model->eval();
        torch::NoGradGuard noGrad;
        torch::Tensor predTest = model->forward(xTest);
        if (!predTest.defined() || !predTest.isfinite().all().item<bool>()) {
            throw std::runtime_error("FFN-PINN prediction produced non-finite values.");
        }
        if (config.evaluate_metrics) {
            populateHydroMetrics(result, tensorValues(yTest), tensorValues(predTest));
            if (!hydroMetricsAreFinite(result)) {
                throw std::runtime_error("FFN-PINN evaluation produced invalid core hydrology metrics.");
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
        result.message = config.use_hydro_package
            ? "FFN-PINN completed on Hydro package input with joint reduced-reservoir physics."
            : (config.use_csv_data
               ? "FFN-PINN completed on CSV input with joint reduced-reservoir physics."
               : "FFN-PINN completed on synthetic input with joint reduced-reservoir physics.");
        return result;
    }
};