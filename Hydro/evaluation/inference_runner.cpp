#include "inference_runner.h"

#include "../dataset/tensor_scaler.h"
#include "../models/hydro_lstm_module.h"
#include "../../neuralnetworkwrapper.h"

#include <algorithm>
#include <istream>
#include <memory>
#include <sstream>
#include <streambuf>
#include <stdexcept>
#include <vector>

namespace {
std::vector<int> parseHiddenLayers(const std::string& csv) {
    std::vector<int> layers;
    std::stringstream stream(csv);
    std::string token;
    while (std::getline(stream, token, ',')) {
        try {
            std::size_t consumed = 0;
            const int size = std::stoi(token, &consumed);
            if (consumed != token.size() || size <= 0) throw std::invalid_argument("invalid layer");
            layers.push_back(size);
        } catch (...) {
            throw std::runtime_error("Inference configuration has an invalid hidden layer: " + token);
        }
    }
    if (layers.empty()) throw std::runtime_error("Inference configuration has no hidden layers.");
    return layers;
}

torch::nn::Sequential makeSequential(int64_t inputDim,
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

// Pre-existing bug, found and fixed here (improvement_loop_20261023/50 and its
// follow-on): this class only ever called setg() and never overrode
// seekoff/seekpos. std::streambuf's default seekoff/seekpos unconditionally
// fail (return pos_type(off_type(-1))), so tellg()/seekg() on a stream backed
// by this buffer always failed -- exactly the
// "istream reader failed: getting the current position" abort LibTorch's
// zip-format checkpoint reader hits, since it needs random access to parse a
// zip archive. This affects the real HydroInferenceSession checkpoint-loading
// path in production, not only the test that first surfaced it: every
// HydroInferenceSession construction for approach in {ffn_pinn, pinn, lstm,
// lstm_pinn} (the torch-module-v1/sequential paths) loads through this same
// buffer. Confirmed pre-existing, not introduced by any change in this
// session: reproduced identically against the unmodified file checked out
// from git HEAD before this fix was written.
class CheckpointMemoryBuffer : public std::streambuf {
public:
    explicit CheckpointMemoryBuffer(const std::vector<std::uint8_t>& bytes) {
        if (bytes.empty()) throw std::runtime_error("Inference checkpoint is empty.");
        begin_ = const_cast<char*>(reinterpret_cast<const char*>(bytes.data()));
        end_ = begin_ + bytes.size();
        setg(begin_, begin_, end_);
    }

protected:
    pos_type seekoff(off_type off, std::ios_base::seekdir dir, std::ios_base::openmode which) override {
        if (!(which & std::ios_base::in)) return pos_type(off_type(-1));
        const off_type size = static_cast<off_type>(end_ - begin_);
        off_type baseOffset;
        switch (dir) {
            case std::ios_base::beg: baseOffset = 0; break;
            case std::ios_base::cur: baseOffset = static_cast<off_type>(gptr() - begin_); break;
            case std::ios_base::end: baseOffset = size; break;
            default: return pos_type(off_type(-1));
        }
        // Bounds-check in integer space BEFORE forming any pointer, AND
        // before computing `baseOffset + off` itself. `off` is parser-
        // controlled input (read out of the checkpoint byte stream by
        // LibTorch's zip reader) and can be any value `off_type` can hold,
        // including near its min/max -- so `baseOffset + off` can overflow a
        // signed integer (UB) even though `baseOffset` itself is always
        // small (bounded by `size`). Checked against the equivalent
        // non-overflowing bound `off in [-baseOffset, size - baseOffset]`
        // instead of performing the addition speculatively; both ends of
        // that range are safe to compute since `baseOffset` and `size` are
        // always within [0, size], far from off_type's limits.
        if (off < -baseOffset || off > size - baseOffset) return pos_type(off_type(-1));
        const off_type requested = baseOffset + off;
        char* newPos = begin_ + requested;
        setg(begin_, newPos, end_);
        return pos_type(requested);
    }

    pos_type seekpos(pos_type pos, std::ios_base::openmode which) override {
        return seekoff(off_type(pos), std::ios_base::beg, which);
    }

private:
    char* begin_ = nullptr;
    char* end_ = nullptr;
};

class CheckpointMemoryStream : public std::istream {
public:
    explicit CheckpointMemoryStream(const std::vector<std::uint8_t>& bytes)
        : std::istream(nullptr), buffer_(bytes) { rdbuf(&buffer_); }
private:
    CheckpointMemoryBuffer buffer_;
};
}

struct HydroInferenceSession::Impl {
    int64_t featureCount = 0;
    int sequenceLength = 0;
    bool recurrentOutputIsPhysical = false;
    TensorScaler inputScaler;
    TensorScaler targetScaler;
    std::unique_ptr<NeuralNetworkWrapper> feedForward;
    torch::nn::Sequential sequential{nullptr};
    HydroLSTM recurrent{nullptr};
    HydroTwoReservoirLSTM processRecurrent{nullptr};
};

HydroInferenceSession::HydroInferenceSession(const HydroInferenceArtifacts& artifacts,
                                             const std::string& approach)
    : impl_(std::make_unique<Impl>()) {
    const auto modelArtifact = artifacts.models.find(approach);
    const auto scalerArtifact = artifacts.scalers.find(approach);
    if (modelArtifact == artifacts.models.end() || scalerArtifact == artifacts.scalers.end()) {
        throw std::runtime_error("Inference artifacts are incomplete for approach: " + approach);
    }
    impl_->inputScaler.importState(scalerArtifact->second.input);
    impl_->targetScaler.importState(scalerArtifact->second.target);
    impl_->featureCount = static_cast<int64_t>(scalerArtifact->second.input.offset.size());
    if (scalerArtifact->second.input.method == "none" && impl_->featureCount == 1) {
        if (artifacts.experiment.config.pinn_physics_profile == "linear_reservoir" &&
            (approach == "ffn_pinn" || approach == "pinn")) {
            impl_->featureCount = artifacts.experiment.config.use_hydro_package ? 8 : 4;
        }
    }
    if (scalerArtifact->second.target.offset.size() != 1) {
        throw std::runtime_error("Inference requires a scalar target scaler.");
    }

    const auto hiddenLayers = parseHiddenLayers(artifacts.experiment.config.hidden_layers_csv);
    if (approach == "ffn" || approach == "ffn_pinn" || approach == "pinn") {
        if (modelArtifact->second.format == "neuralnetworkwrapper-v1") {
            impl_->feedForward = std::make_unique<NeuralNetworkWrapper>();
            impl_->feedForward->setHiddenLayers(hiddenLayers);
            impl_->feedForward->setLags(std::vector<std::vector<int>>(
                static_cast<std::size_t>(impl_->featureCount), {1}));
            impl_->feedForward->initializeNetwork(1, artifacts.experiment.config.activation);
            CheckpointMemoryStream archiveStream(modelArtifact->second.bytes);
            impl_->feedForward->loadModel(archiveStream);
        } else if (modelArtifact->second.format == "torch-sequential-v1") {
            impl_->sequential = makeSequential(impl_->featureCount, hiddenLayers,
                                               artifacts.experiment.config.activation);
            CheckpointMemoryStream archiveStream(modelArtifact->second.bytes);
            torch::serialize::InputArchive archive;
            archive.load_from(archiveStream);
            impl_->sequential->load(archive);
            impl_->sequential->eval();
        } else {
            throw std::runtime_error("Feed-forward inference requires neuralnetworkwrapper-v1 or torch-sequential-v1.");
        }
    } else if (approach == "lstm" || approach == "lstm_pinn") {
        if (modelArtifact->second.format != "torch-module-v1") {
            throw std::runtime_error("Recurrent inference requires torch-module-v1.");
        }
        // Hydro/models/lstm_pinn_wrapper.cpp's LSTMPINNWrapper::train() dispatches
        // on pinn_physics_profile to one of six training functions. Four of them
        // (trainResidualHybrid for lstm_residual_pinn_hybrid; trainSharedScaleAblation
        // for the three lstm_shared_scale_ablation_* profiles) compute predFull as
        // clamp_min(0, baseline + inverseTransform(model->forward(...))) -- the clip
        // applies to the SUM, not to the residual before adding the baseline; those
        // are different numbers -- over segment-aware, warm-up-excluding sequences.
        // This bare forward-pass engine implements none of that (confirmed by direct
        // source comparison, improvement_loop_20261023/47). Checked BEFORE loading
        // the checkpoint at all, so an unsupported profile fails with this message
        // instead of whatever the loader happens to do first (archive loading has its
        // own, unrelated failure modes -- see improvement_loop_20261023/50 -- and must
        // never be allowed to preempt this check).
        static const std::vector<std::string> kKnownUnsupportedLstmPinnProfiles = {
            "lstm_residual_pinn_hybrid",
            "lstm_shared_scale_ablation_zero",
            "lstm_shared_scale_ablation_urban",
            "lstm_shared_scale_ablation_physics",
        };
        if (approach == "lstm_pinn" &&
            std::find(kKnownUnsupportedLstmPinnProfiles.begin(), kKnownUnsupportedLstmPinnProfiles.end(),
                      artifacts.experiment.config.pinn_physics_profile) != kKnownUnsupportedLstmPinnProfiles.end()) {
            throw std::runtime_error(
                "HydroInferenceSession does not support lstm_pinn profile '" +
                artifacts.experiment.config.pinn_physics_profile + "'. This profile requires a physics "
                "baseline, output clipping, and segment-aware sequence construction this bare forward-pass "
                "engine does not perform, and would silently produce a wrong (not merely approximate) "
                "prediction. See Hydro/experiments/gistohq_sligo/run_frozen_shared_scale_inference.cpp for "
                "the validated implementation, and "
                "improvement_loop_20261023/47_frozen_inference_validation.md for how this gap was found.");
        }
        // NOTE: rejecting the four profiles above is NOT the same as establishing that
        // every remaining profile (two_reservoir_hybrid aside) produces a prediction
        // matching its training pipeline. The "water_balance"/default path's own
        // backend (lstmnetworkwrapper.cpp) also applies clamp_min(0, ...) to its
        // output, which this engine's bare fallback below still does not apply at
        // all. Matching model family is not the same claim as matching predictions --
        // that parity is unverified for every profile this engine does not explicitly
        // reject, not just the four it does. Tracked for the shared forward-inference
        // extraction (Task 6 part 2), not resolved by this explicit-rejection pass.
        impl_->sequenceLength = std::max(2, artifacts.experiment.config.lstm_sequence_length);
        CheckpointMemoryStream archiveStream(modelArtifact->second.bytes);
        torch::serialize::InputArchive archive;
        archive.load_from(archiveStream);

        const bool processAware = approach == "lstm_pinn" &&
            artifacts.experiment.config.pinn_physics_profile == "two_reservoir_hybrid";
        if (processAware) {
            impl_->processRecurrent = HydroTwoReservoirLSTM(
                impl_->featureCount,
                hiddenLayers.front(),
                static_cast<int64_t>(hiddenLayers.size()),
                artifacts.experiment.config.physics_dt,
                artifacts.experiment.config.storage_coeff,
                artifacts.experiment.config.lambda_decay,
                artifacts.experiment.config.runoff_coeff);
            impl_->processRecurrent->load(archive);
            impl_->processRecurrent->eval();
            impl_->recurrentOutputIsPhysical = true;
        } else {
            impl_->recurrent = HydroLSTM(impl_->featureCount, hiddenLayers.front(), 1,
                                         static_cast<int64_t>(hiddenLayers.size()));
            impl_->recurrent->load(archive);
            impl_->recurrent->eval();
        }
    } else {
        throw std::invalid_argument("Unsupported inference approach: " + approach);
    }
}

HydroInferenceSession::~HydroInferenceSession() = default;
HydroInferenceSession::HydroInferenceSession(HydroInferenceSession&&) noexcept = default;
HydroInferenceSession& HydroInferenceSession::operator=(HydroInferenceSession&&) noexcept = default;

torch::Tensor HydroInferenceSession::predict(const torch::Tensor& physicalInputs) const {
    if (!physicalInputs.defined() || physicalInputs.size(0) == 0) {
        throw std::invalid_argument("Inference inputs must be non-empty.");
    }
    torch::NoGradGuard noGrad;
    torch::Tensor prediction;
    if (impl_->feedForward || impl_->sequential) {
        if (physicalInputs.dim() != 2 || physicalInputs.size(1) != impl_->featureCount) {
            throw std::invalid_argument("Feed-forward inference input shape does not match the exported model feature count.");
        }
        const auto scaled = impl_->inputScaler.transform(physicalInputs);
        prediction = impl_->feedForward ? impl_->feedForward->forwardTensor(scaled)
                                        : impl_->sequential->forward(scaled);
    } else {
        if (physicalInputs.dim() != 3 || physicalInputs.size(1) != impl_->sequenceLength ||
            physicalInputs.size(2) != impl_->featureCount) {
            throw std::invalid_argument("Recurrent inference input shape does not match the exported configuration.");
        }
        const auto scaled = impl_->inputScaler.transform(physicalInputs);
        prediction = impl_->processRecurrent
            ? impl_->processRecurrent->forward(scaled)
            : impl_->recurrent->forward(scaled);
    }
    if (!impl_->recurrentOutputIsPhysical) {
        prediction = impl_->targetScaler.inverseTransform(prediction);
    }
    if (!prediction.defined() || prediction.dim() != 2 || prediction.size(0) != physicalInputs.size(0) ||
        prediction.size(1) != 1 || !prediction.isfinite().all().item<bool>()) {
        throw std::runtime_error("Checkpoint produced invalid predictions.");
    }
    return prediction;
}

torch::Tensor HydroInferenceSession::predictSeries(const torch::Tensor& physicalSeries) const {
    if (!physicalSeries.defined() || physicalSeries.dim() != 2 ||
        physicalSeries.size(0) == 0 || physicalSeries.size(1) != impl_->featureCount) {
        throw std::invalid_argument("Inference series must have shape [samples, configured features].");
    }
    if (impl_->feedForward || impl_->sequential) return predict(physicalSeries);
    if (physicalSeries.size(0) < impl_->sequenceLength) {
        throw std::invalid_argument("Inference series is shorter than the exported LSTM sequence length.");
    }
    const auto sequences = physicalSeries.unfold(0, impl_->sequenceLength, 1)
                               .transpose(1, 2)
                               .contiguous();
    return predict(sequences);
}

torch::Tensor HydroInferenceRunner::predictFeedForward(
    const HydroInferenceArtifacts& artifacts,
    const std::string& approach,
    const torch::Tensor& physicalInputs) const {
    if (approach != "ffn" && approach != "ffn_pinn" && approach != "pinn") {
        throw std::invalid_argument("Feed-forward inference does not support approach: " + approach);
    }
    return HydroInferenceSession(artifacts, approach).predict(physicalInputs);
}

torch::Tensor HydroInferenceRunner::predictRecurrent(
    const HydroInferenceArtifacts& artifacts,
    const std::string& approach,
    const torch::Tensor& physicalSequences) const {
    if (approach != "lstm" && approach != "lstm_pinn") {
        throw std::invalid_argument("Recurrent inference does not support approach: " + approach);
    }
    return HydroInferenceSession(artifacts, approach).predict(physicalSequences);
}

SharedScaleInferenceResult HydroInferenceRunner::runSharedScaleAblation(
    const HydroRunConfig& config,
    const std::string& frozenExperimentDir,
    const std::string& approach,
    bool holdoutMode) const {
    SharedScaleInferenceProfile profile;
    if (!parseSharedScaleInferenceProfile(config.pinn_physics_profile, profile)) {
        throw std::runtime_error(
            "runSharedScaleAblation does not support pinn_physics_profile '" + config.pinn_physics_profile +
            "'. Call parseSharedScaleInferenceProfile first to check before routing here.");
    }
    return runSharedScaleInference(config, profile, frozenExperimentDir, approach, holdoutMode);
}
