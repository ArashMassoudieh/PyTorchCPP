#include "shared_scale_inference.h"

#include "experiment_loader.h"
#include "../dataset/hydro_checksum.h"
#include "../dataset/reservoir_physics_tensor_builder.h"
#include "../dataset/configured_chronological_split.h"
#include "../dataset/tensor_scaler.h"
#include "../models/two_reservoir_routing.h"
#include "../models/urban_managed_routing.h"
#include "../models/hydro_lstm_module.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <sstream>
#include <stdexcept>

namespace {

// ---------------------------------------------------------------------------
// Verbatim copies from Hydro/models/lstm_pinn_wrapper.cpp (anonymous
// namespace there; lines as read 2026-10-28, unchanged since the original
// extraction into the CLI driver -- see improvement_loop_20261023/47). Not
// reimplemented or approximated.
// ---------------------------------------------------------------------------

struct SequenceData {
    torch::Tensor x;
    torch::Tensor y;
    torch::Tensor time;
    torch::Tensor peff;
    torch::Tensor precipitation;
};

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
    if (physicsX.size(1) >= 4) return physicsX.slice(1, 2, 4).contiguous();
    return physicsX.slice(1, 1, physicsX.size(1)).contiguous();
}

torch::Tensor sequenceEndpointIndices(const torch::Tensor& segmentId, int64_t sequenceLength) {
    const auto segments = segmentId.reshape({-1}).contiguous();
    const int64_t n = segments.size(0);
    std::vector<int64_t> endpoints;
    int64_t begin = 0;
    while (begin < n) {
        const int64_t segment = segments[begin].item<int64_t>();
        int64_t end = begin + 1;
        while (end < n && segments[end].item<int64_t>() == segment) ++end;
        if (end - begin >= sequenceLength) {
            for (int64_t windowEnd = begin + sequenceLength - 1; windowEnd < end; ++windowEnd) {
                endpoints.push_back(windowEnd);
            }
        }
        begin = end;
    }
    return torch::tensor(endpoints, torch::kInt64);
}

SequenceData makeSequences(const torch::Tensor& modelX,
                           const torch::Tensor& y,
                           const torch::Tensor& time,
                           const torch::Tensor& peff,
                           const torch::Tensor& precipitation,
                           int sequenceLength,
                           const torch::Tensor& segmentId) {
    if (!modelX.defined() || !y.defined() || !time.defined() || !peff.defined() ||
        !precipitation.defined() || !segmentId.defined() || modelX.dim() != 2 || y.dim() != 2 ||
        modelX.size(0) != y.size(0) || time.numel() != modelX.size(0) ||
        peff.numel() != modelX.size(0) || precipitation.numel() != modelX.size(0) ||
        segmentId.numel() != modelX.size(0)) {
        throw std::runtime_error("LSTM-PINN sequence builder expects aligned model inputs, target, time, I*, precipitation, and segment id.");
    }
    sequenceLength = std::max(2, sequenceLength);
    if (modelX.size(0) < sequenceLength + 3) {
        throw std::runtime_error("Too few samples for requested LSTM-PINN sequence length.");
    }

    const auto segments = segmentId.reshape({-1}).contiguous();
    const int64_t n = modelX.size(0);
    {
        int64_t begin = 0;
        while (begin < n) {
            const int64_t segment = segments[begin].item<int64_t>();
            int64_t end = begin + 1;
            while (end < n && segments[end].item<int64_t>() == segment) ++end;
            if (end - begin >= sequenceLength) regularPhysicalTimeStepFromTime(time.slice(0, begin, end));
            begin = end;
        }
    }

    const torch::Tensor endpoints = sequenceEndpointIndices(segmentId, sequenceLength);
    if (endpoints.numel() == 0) {
        throw std::runtime_error("No GIStoOHQ segment is long enough for the requested LSTM-PINN sequence length.");
    }
    std::vector<torch::Tensor> blockSequences;
    blockSequences.reserve(static_cast<std::size_t>(endpoints.numel()));
    for (int64_t i = 0; i < endpoints.numel(); ++i) {
        const int64_t windowEnd = endpoints[i].item<int64_t>();
        blockSequences.push_back(modelX.slice(0, windowEnd - sequenceLength + 1, windowEnd + 1));
    }

    SequenceData result;
    result.x = torch::stack(blockSequences, 0).contiguous();
    result.y = y.index_select(0, endpoints).contiguous();
    result.time = time.reshape({-1, 1}).index_select(0, endpoints).contiguous();
    result.peff = peff.reshape({-1, 1}).index_select(0, endpoints).contiguous();
    result.precipitation = precipitation.reshape({-1, 1}).index_select(0, endpoints).contiguous();
    return result;
}

// ---------------------------------------------------------------------------
// End verbatim copies.
// ---------------------------------------------------------------------------

constexpr int64_t kSealStartX = 77016; // 2023-10-15T00:00:00Z, hours since 2015-01-01
constexpr int64_t kHoldoutSanityCeilingX = 120000;

void requireDevelopmentOnly(const torch::Tensor& plotX, const char* stage, bool holdoutMode) {
    const double maxX = plotX.max().item<double>();
    const double ceiling = holdoutMode ? static_cast<double>(kHoldoutSanityCeilingX)
                                       : static_cast<double>(kSealStartX);
    if (maxX >= ceiling) {
        throw std::runtime_error(
            std::string("Development cutoff guard tripped at stage '") + stage +
            "': max x=" + std::to_string(maxX) + " reaches or exceeds the " +
            (holdoutMode ? "holdout sanity ceiling" : "reserved-period seal") +
            " (x>=" + std::to_string(static_cast<int64_t>(ceiling)) + "). Refusing.");
    }
}

std::vector<int> parseHiddenLayersCsv(const std::string& csv) {
    std::vector<int> layers;
    std::stringstream stream(csv);
    std::string token;
    while (std::getline(stream, token, ',')) layers.push_back(std::stoi(token));
    if (layers.empty()) throw std::runtime_error("Config has no hidden layers.");
    return layers;
}

} // namespace

bool parseSharedScaleInferenceProfile(const std::string& profile, SharedScaleInferenceProfile& out) {
    if (profile == "lstm_shared_scale_ablation_zero") {
        out = SharedScaleInferenceProfile::Zero;
        return true;
    }
    if (profile == "lstm_shared_scale_ablation_urban") {
        out = SharedScaleInferenceProfile::UrbanManaged;
        return true;
    }
    return false;
}

SharedScaleInferenceResult runSharedScaleInference(
    const HydroRunConfig& configIn,
    SharedScaleInferenceProfile profile,
    const std::string& frozenExperimentDir,
    const std::string& approach,
    bool holdoutMode) {
    torch::NoGradGuard noGrad; // belt-and-suspenders: no gradient tape anywhere in this function
    HydroRunConfig config = configIn;

    if (!config.use_hydro_package) {
        throw std::runtime_error("runSharedScaleInference only supports use_hydro_package configs.");
    }
    if (!config.hydro_package_path.empty() && !std::filesystem::path(config.hydro_package_path).is_absolute()) {
        throw std::runtime_error("Expected an absolute hydro_package_path.");
    }
    // Replicates hydro_batch_runner.cpp's preparePhysicsConfig for exactly the
    // gisToOhq && processAwareHybrid branch that applies to both
    // lstm_shared_scale_ablation_{zero,urban}: the only effect is this field.
    config.use_latent_storage_physics = true;

    torch::Tensor physicsX, y, plotX, segmentId;
    if (!loadReservoirPhysicsTensors(config, physicsX, y, plotX, segmentId)) {
        throw std::runtime_error("Unable to load reservoir physics tensors from the configured package.");
    }
    requireDevelopmentOnly(plotX, "raw-tensors", holdoutMode);

    const double dt = firstSegmentDt(plotX, segmentId);
    const torch::Tensor peffFull = physicsX.slice(1, 1, 2).reshape({-1}).contiguous();
    const torch::Tensor observedFull = y.reshape({-1}).contiguous();
    const int64_t n = peffFull.size(0);

    torch::Tensor baselineFull;
    if (profile == SharedScaleInferenceProfile::Zero) {
        baselineFull = torch::zeros({n, 1}, torch::kFloat32);
    } else {
        std::vector<double> peffVec(static_cast<std::size_t>(n));
        std::vector<double> observedVec(static_cast<std::size_t>(n));
        std::vector<std::int64_t> segmentVec(static_cast<std::size_t>(n));
        const auto segmentCpu = segmentId.reshape({-1}).to(torch::kCPU).contiguous();
        for (int64_t i = 0; i < n; ++i) {
            peffVec[static_cast<std::size_t>(i)] = peffFull[i].item<double>();
            observedVec[static_cast<std::size_t>(i)] = observedFull[i].item<double>();
            segmentVec[static_cast<std::size_t>(i)] = segmentCpu[i].item<std::int64_t>();
        }
        UrbanManagedRoutingParams routingParams;
        routingParams.intensityQuickStrength = config.urban_intensity_quick_strength;
        routingParams.intensityReferenceMmH = config.urban_intensity_reference_mm_h;
        routingParams.quickK = config.urban_quick_k;
        routingParams.slowK = config.urban_slow_k;
        routingParams.detentionK = config.urban_detention_k;
        routingParams.channelK = config.urban_channel_k;
        routingParams.quickFraction = config.urban_quick_fraction;
        routingParams.detentionCatchmentFraction = config.urban_detention_fraction;
        routingParams.runoffCoefficient = config.urban_runoff_coefficient;
        routingParams.dtHours = dt;
        routingParams.lagSteps = dt > 0.0
            ? static_cast<int64_t>(std::llround(std::max(0.0, config.pinn_routing_lag_hours) / dt))
            : 0;
        const std::vector<double> baselineVec = simulateUrbanManagedBaselinePerSegment(
            peffVec, observedVec, segmentVec, routingParams, config.pinn_initial_flow);
        if (std::any_of(baselineVec.begin(), baselineVec.end(), [](double v) { return !std::isfinite(v); })) {
            throw std::runtime_error("Urban managed baseline produced non-finite values.");
        }
        baselineFull = torch::from_blob(const_cast<double*>(baselineVec.data()), {n}, torch::kDouble)
                           .clone().to(torch::kFloat32).reshape({n, 1});
    }

    const torch::Tensor targetFull = y - baselineFull;
    const torch::Tensor modelX = predictorFeatures(physicsX);
    const torch::Tensor precipitation = physicsX.size(1) >= 3
        ? physicsX.slice(1, 2, 3).contiguous()
        : peffFull.reshape({-1, 1});

    SequenceData seq = makeSequences(modelX, targetFull, plotX, peffFull.reshape({-1, 1}), precipitation,
                                     config.lstm_sequence_length, segmentId);
    const int sequenceLength = std::max(2, config.lstm_sequence_length);
    const torch::Tensor sequenceEndpoints = sequenceEndpointIndices(segmentId, sequenceLength);
    const torch::Tensor qObservedSeq = observedFull.reshape({-1, 1}).index_select(0, sequenceEndpoints).contiguous();
    const torch::Tensor baselineSeq = baselineFull.index_select(0, sequenceEndpoints).contiguous();
    requireDevelopmentOnly(seq.time, "sequence-endpoints", holdoutMode);

    const ChronologicalSplit split = configuredChronologicalSplit(seq.time, config);

    const auto scalers = HydroArtifactLoader().loadScalers(frozenExperimentDir);
    const auto scalerIt = scalers.find(approach);
    if (scalerIt == scalers.end()) {
        throw std::runtime_error("No scaler artifact found for approach '" + approach + "' in " + frozenExperimentDir);
    }
    TensorScaler inputScaler;
    inputScaler.importState(scalerIt->second.input);
    TensorScaler targetScaler;
    targetScaler.importState(scalerIt->second.target);

    const auto models = HydroArtifactLoader().loadModels(frozenExperimentDir);
    const auto modelIt = models.find(approach);
    if (modelIt == models.end()) {
        throw std::runtime_error("No model artifact found for approach '" + approach + "' in " + frozenExperimentDir);
    }
    const std::string checkpointPath = frozenExperimentDir + "/" + modelIt->second.relative_path;
    const std::string checkpointSha256Before = sha256File(checkpointPath);
    if (modelIt->second.format != "torch-module-v1") {
        throw std::runtime_error("Expected torch-module-v1 checkpoint format, found: " + modelIt->second.format);
    }
    if (!modelIt->second.sha256.empty() && modelIt->second.sha256 != checkpointSha256Before) {
        throw std::runtime_error("Checkpoint file hash does not match the artifact bundle's own recorded hash.");
    }

    const auto hidden = parseHiddenLayersCsv(config.hidden_layers_csv);
    HydroLSTM model(seq.x.size(2), hidden.front(), 1, static_cast<int64_t>(std::max<std::size_t>(1, hidden.size())));
    {
        torch::serialize::InputArchive archive;
        archive.load_from(checkpointPath);
        model->load(archive);
    }
    model->eval();

    torch::Tensor xFullScaled = inputScaler.transform(seq.x);
    torch::Tensor predFull = (baselineSeq + targetScaler.inverseTransform(model->forward(xFullScaled))).clamp_min(0.0);
    if (!predFull.defined() || !predFull.isfinite().all().item<bool>()) {
        throw std::runtime_error("Shared-scale inference produced non-finite values.");
    }

    const std::string checkpointSha256After = sha256File(checkpointPath);
    if (checkpointSha256After != checkpointSha256Before) {
        throw std::runtime_error("Checkpoint file changed on disk during inference (hash mismatch before/after). This must never happen.");
    }

    SharedScaleInferenceResult result;
    result.checkpointSha256 = checkpointSha256After;
    const auto timeCpu = seq.time.reshape({-1}).to(torch::kCPU).contiguous();
    const auto obsCpu = qObservedSeq.reshape({-1}).to(torch::kCPU).contiguous();
    const auto predCpu = predFull.reshape({-1}).to(torch::kCPU).contiguous();
    result.x.reserve(static_cast<std::size_t>(timeCpu.size(0)));
    result.observed.reserve(result.x.capacity());
    result.predicted.reserve(result.x.capacity());
    result.split.reserve(result.x.capacity());
    for (int64_t i = 0; i < timeCpu.size(0); ++i) {
        result.x.push_back(timeCpu[i].item<double>());
        result.observed.push_back(obsCpu[i].item<double>());
        result.predicted.push_back(predCpu[i].item<double>());
        std::string label = "beyond_test_end";
        if (i < split.train_end) label = "train";
        else if (i < split.validation_end) label = "validation";
        else if (i < split.test_end) label = "test";
        result.split.push_back(std::move(label));
    }
    return result;
}
