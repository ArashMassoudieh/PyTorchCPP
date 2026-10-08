#pragma once

#include "ffn_reservoir_pinn_wrapper.h"
#include "hydro_run_types.h"
#include "two_reservoir_routing.h"
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
#include <stdexcept>
#include <vector>

// Residual-correction hybrid: a physics baseline (the same fast/slow
// two-reservoir routing used by the standalone PINN) supplies the bulk of the
// hydrograph shape, and a small FFN learns only the residual
// (observed - baseline) from the available forcing features. This differs
// from ffn_two_reservoir_hybrid, where the network generates the *input* to
// the routing (a joint-loss architecture that has consistently underperformed
// its parent FFN/LSTM on real Sligo Creek data - see
// paper_run_20260925_130303's genuine_hydrologic_improvement=False finding).
// Here the physics term is exact and untrainable; the network only has to
// learn what the physics leaves on the table, which is a much easier target.
class FFNResidualPINNWrapper {
public:
    HydroRunResult train(const HydroRunConfig& config) {
        using namespace hydro_ffn_reservoir_detail;
        // "none": raw everything, the original behavior (kept byte-identical
        // for back-compat/control-group use). "standardize_input": scale the
        // network's own input features only (fixes an input-scale bug - the
        // raw [time, Peff, ...] tensor's time column grows unbounded with
        // record length and otherwise dominates small meteorological
        // features). "standardize": also z-score the residual *target*
        // linearly (fit on train only). "standardize_input_asinh_residual":
        // input scaling as above, but the residual target is compressed with
        // asinh first, then z-scored - unlike a plain linear standardize,
        // asinh is a genuine nonlinear compression of extreme values and
        // handles signed residuals natively (log_standardize cannot: its
        // log1p has domain x>=-1, and residuals go negative whenever the
        // physics baseline overshoots). The physics baseline itself always
        // runs on raw physical peff/q0, regardless of this setting - only
        // what the network itself sees/predicts is affected.
        const bool scaleInput = config.normalization == "standardize_input" ||
                                 config.normalization == "standardize" ||
                                 config.normalization == "standardize_input_asinh_residual";
        const bool scaleResidualLinear = config.normalization == "standardize";
        const bool scaleResidualAsinh = config.normalization == "standardize_input_asinh_residual";
        const bool scaleResidual = scaleResidualLinear || scaleResidualAsinh;
        if (config.normalization != "none" && !scaleInput) {
            throw std::invalid_argument(
                "Residual-correction PINN hybrid normalization must be one of: none, standardize_input, standardize, "
                "standardize_input_asinh_residual (log_standardize is not valid here - residuals can be negative, "
                "outside log1p's domain).");
        }
        HydroRunResult result;
        torch::manual_seed(static_cast<uint64_t>(std::max(0, config.random_seed)));

        torch::Tensor x, y, plotX;
        if (!loadReservoirPhysicsTensors(config, x, y, plotX)) {
            throw std::runtime_error("Unable to construct reservoir physics tensors for residual-correction PINN hybrid.");
        }
        if (x.dim() != 2 || x.size(1) < 2 || y.dim() != 2 || y.size(1) != 1) {
            throw std::runtime_error("Residual-correction PINN hybrid expects [time, Peff, ...] inputs and scalar runoff targets.");
        }

        const ChronologicalSplit split = makeChronologicalSplit(x.size(0), config.train_split_ratio, config.validation_split_ratio);
        const double dt = regularPhysicalTimeStepFromTime(plotX);
        const double fastK = config.storage_coeff;
        const double slowK = config.lambda_decay;
        const double alpha = config.runoff_coeff;
        if (!(fastK > 0.0 && slowK > 0.0 && fastK > slowK && alpha > 0.0 && alpha < 1.0)) {
            throw std::runtime_error("Residual-correction PINN hybrid requires fast_k>slow_k>0 and 0<routing_alpha<1.");
        }

        const auto peff = x.slice(1, 1, 2).reshape({-1}).to(torch::kCPU).contiguous();
        const auto observed = y.reshape({-1}).to(torch::kCPU).contiguous();
        const int64_t n = peff.size(0);
        const double q0 = observed[0].item<double>();
        const double runoffCoefficient = std::max(1.0e-6, config.forcing_gain);
        const int64_t lagSteps = dt > 0.0
            ? static_cast<int64_t>(std::llround(std::max(0.0, config.pinn_routing_lag_hours) / dt))
            : 0;

        std::vector<double> peffVec(static_cast<std::size_t>(n));
        for (int64_t i = 0; i < n; ++i) peffVec[static_cast<std::size_t>(i)] = peff[i].item<double>();
        TwoReservoirRoutingParams routingParams;
        routingParams.fastK = fastK;
        routingParams.slowK = slowK;
        routingParams.alpha = alpha;
        routingParams.runoffCoefficient = runoffCoefficient;
        routingParams.dt = dt;
        routingParams.lagSteps = lagSteps;
        routingParams.flowExponent = config.pinn_flow_exponent;
        const std::vector<double> baseline = simulateTwoReservoirBaseline(peffVec, q0, routingParams);
        if (std::any_of(baseline.begin(), baseline.end(), [](double v) { return !std::isfinite(v); })) {
            throw std::runtime_error("Residual-correction PINN hybrid physics baseline produced non-finite values.");
        }
        torch::Tensor baselineTensor = torch::from_blob(const_cast<double*>(baseline.data()), {n}, torch::kDouble)
                                            .clone()
                                            .to(torch::kFloat32)
                                            .reshape({n, 1});
        // The residual is what the exact physics baseline leaves unexplained
        // (model structural error, unmodeled storage, etc.); the network only
        // has to learn that remainder, not the whole hydrograph from scratch.
        torch::Tensor residualTarget = y - baselineTensor;

        torch::Tensor xTrainRaw = x.slice(0, 0, split.train_end).contiguous();
        torch::Tensor rTrainRaw = residualTarget.slice(0, 0, split.train_end).contiguous();
        torch::Tensor xValidationRaw = x.slice(0, split.train_end, split.validation_end).contiguous();
        torch::Tensor yValidation = y.slice(0, split.train_end, split.validation_end).contiguous();
        torch::Tensor baselineValidation = baselineTensor.slice(0, split.train_end, split.validation_end).contiguous();
        torch::Tensor xTestRaw = x.slice(0, split.validation_end, x.size(0)).contiguous();
        torch::Tensor yTest = y.slice(0, split.validation_end, y.size(0)).contiguous();
        torch::Tensor baselineTest = baselineTensor.slice(0, split.validation_end, x.size(0)).contiguous();

        // Fit scalers on the training split only, exactly like every other
        // wrapper in this project. inputScaler covers the *whole* feature
        // tensor (including the time column) when scaleInput is set - that is
        // the point, since raw elapsed time is otherwise fed to the network
        // unscaled. targetScaler covers only the residual, never used to
        // touch the physics baseline.
        TensorScaler inputScaler;
        TensorScaler targetScaler;
        if (scaleInput) inputScaler.fit(xTrainRaw, "standardize");
        if (scaleResidual) targetScaler.fit(rTrainRaw, scaleResidualAsinh ? "asinh_standardize" : "standardize");
        torch::Tensor xTrain = scaleInput ? inputScaler.transform(xTrainRaw) : xTrainRaw;
        torch::Tensor rTrain = scaleResidual ? targetScaler.transform(rTrainRaw) : rTrainRaw;
        torch::Tensor xValidation = scaleInput ? inputScaler.transform(xValidationRaw) : xValidationRaw;
        torch::Tensor xTest = scaleInput ? inputScaler.transform(xTestRaw) : xTestRaw;

        torch::nn::Sequential model = makeNetwork(x.size(1), parseHiddenLayers(config.hidden_layers_csv), config.activation);
        torch::optim::Adam optimizer(model->parameters(),
                                     torch::optim::AdamOptions(config.learning_rate).weight_decay(config.weight_decay));

        const int64_t trainN = xTrain.size(0);
        const int batchSize = std::max(2, config.batch_size);

        std::vector<torch::Tensor> bestParameters;
        std::vector<double> losses;
        std::vector<double> validationLosses;
        double bestValidationMse = std::numeric_limits<double>::infinity();
        int bestEpoch = 0;

        for (int epoch = 0; epoch < std::max(1, config.epochs); ++epoch) {
            model->train();
            double epochLoss = 0.0;
            int64_t seen = 0;
            for (int64_t start = 0; start < trainN; start += batchSize) {
                const int64_t end = std::min<int64_t>(start + batchSize, trainN);
                if (end - start < 2) continue;
                optimizer.zero_grad();
                torch::Tensor predResidual = model->forward(xTrain.slice(0, start, end));
                torch::Tensor loss = torch::mse_loss(predResidual, rTrain.slice(0, start, end));
                loss.backward();
                optimizer.step();
                const int64_t count = end - start;
                epochLoss += loss.item<double>() * static_cast<double>(count);
                seen += count;
            }
            losses.push_back(epochLoss / static_cast<double>(std::max<int64_t>(1, seen)));

            model->eval();
            double validationMse = 0.0;
            {
                torch::NoGradGuard noGrad;
                // Select on the fully-reconstructed (physics + residual) discharge
                // against observed, not on residual MSE alone - that is what the
                // model is actually evaluated on, and keeps this comparable to
                // every other method's validation-selection criterion.
                torch::Tensor predictedResidualValidation = model->forward(xValidation);
                if (scaleResidual) predictedResidualValidation = targetScaler.inverseTransform(predictedResidualValidation);
                torch::Tensor predValidation = (baselineValidation + predictedResidualValidation).clamp_min(0.0);
                validationMse = torch::mse_loss(predValidation, yValidation).item<double>();
            }
            if (!std::isfinite(validationMse)) {
                throw std::runtime_error("Residual-correction PINN hybrid validation produced a non-finite loss.");
            }
            validationLosses.push_back(validationMse);
            if (validationMse < bestValidationMse) {
                bestValidationMse = validationMse;
                bestEpoch = epoch + 1;
                bestParameters.clear();
                for (const auto& parameter : model->parameters()) bestParameters.push_back(parameter.detach().clone());
            }
        }

        if (bestParameters.empty()) throw std::runtime_error("Residual-correction PINN hybrid did not produce a validation-selected checkpoint.");
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
        result.input_scaler = scaleInput ? inputScaler.exportState() : HydroScalerState{};
        result.target_scaler = scaleResidual ? targetScaler.exportState() : HydroScalerState{};

        {
            const auto checkpoint = temporaryHydroCheckpointPath("hydro_ffn_residual_pinn");
            torch::serialize::OutputArchive archive;
            model->save(archive);
            archive.save_to(checkpoint.string());
            result.model_checkpoint = readHydroCheckpoint(checkpoint);
            result.model_checkpoint_format = "torch-sequential-v1";
            std::filesystem::remove(checkpoint);
        }

        model->eval();
        torch::NoGradGuard noGrad;
        torch::Tensor predictedResidualTest = model->forward(xTest);
        if (scaleResidual) predictedResidualTest = targetScaler.inverseTransform(predictedResidualTest);
        torch::Tensor predTest = (baselineTest + predictedResidualTest).clamp_min(0.0);
        if (!predTest.defined() || !predTest.isfinite().all().item<bool>()) {
            throw std::runtime_error("Residual-correction PINN hybrid prediction produced non-finite values.");
        }
        if (config.evaluate_metrics) {
            populateHydroMetrics(result, tensorValues(yTest), tensorValues(predTest));
            if (!hydroMetricsAreFinite(result)) {
                throw std::runtime_error("Residual-correction PINN hybrid evaluation produced invalid core hydrology metrics.");
            }
        }

        torch::Tensor xFull = scaleInput ? inputScaler.transform(x) : x;
        torch::Tensor predictedResidualFull = model->forward(xFull);
        if (scaleResidual) predictedResidualFull = targetScaler.inverseTransform(predictedResidualFull);
        torch::Tensor predFull = (baselineTensor + predictedResidualFull).clamp_min(0.0);
        fillPlotVectors(result, plotX, y, predFull);
        result.split.resize(result.x.size(), "test");
        for (std::size_t i = 0; i < result.split.size(); ++i) {
            if (static_cast<int64_t>(i) < split.train_end) result.split[i] = "train";
            else if (static_cast<int64_t>(i) < split.validation_end) result.split[i] = "validation";
        }
        result.success = true;
        return result;
    }
};
