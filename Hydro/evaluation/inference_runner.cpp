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

class CheckpointMemoryBuffer : public std::streambuf {
public:
    explicit CheckpointMemoryBuffer(const std::vector<std::uint8_t>& bytes) {
        if (bytes.empty()) throw std::runtime_error("Inference checkpoint is empty.");
        auto* begin = const_cast<char*>(reinterpret_cast<const char*>(bytes.data()));
        setg(begin, begin, begin + bytes.size());
    }
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
    TensorScaler inputScaler;
    TensorScaler targetScaler;
    std::unique_ptr<NeuralNetworkWrapper> feedForward;
    torch::nn::Sequential sequential{nullptr};
    HydroLSTM recurrent{nullptr};
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
    // Identity normalization is persisted as a scalar state and intentionally
    // broadcasts over all physical input features.
    if (scalerArtifact->second.input.method == "none" && impl_->featureCount == 1) {
        if (artifacts.experiment.config.pinn_physics_profile == "linear_reservoir" &&
            (approach == "ffn_pinn" || approach == "pinn")) {
            // Reduced-reservoir exported models use [time, Peff, P, PET, ...].
            // The exact width is encoded by the first linear layer, but the
            // current artifact schema has no dedicated feature-count field.
            // For the canonical reduced-reservoir contract use four features
            // for synthetic/CSV and eight for GIStoOHQ Hydro packages.
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
        impl_->sequenceLength = std::max(2, artifacts.experiment.config.lstm_sequence_length);
        impl_->recurrent = HydroLSTM(impl_->featureCount, hiddenLayers.front(), 1,
                                     static_cast<int64_t>(hiddenLayers.size()));
        CheckpointMemoryStream archiveStream(modelArtifact->second.bytes);
        torch::serialize::InputArchive archive;
        archive.load_from(archiveStream);
        impl_->recurrent->load(archive);
        impl_->recurrent->eval();
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
        prediction = impl_->recurrent->forward(impl_->inputScaler.transform(physicalInputs));
    }
    prediction = impl_->targetScaler.inverseTransform(prediction);
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
