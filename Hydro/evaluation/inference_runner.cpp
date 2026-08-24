#include "inference_runner.h"

#include "../dataset/tensor_scaler.h"
#include "../models/hydro_lstm_module.h"
#include "../../neuralnetworkwrapper.h"

#include <istream>
#include <streambuf>
#include <sstream>
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

torch::Tensor HydroInferenceRunner::predictFeedForward(
    const HydroInferenceArtifacts& artifacts,
    const std::string& approach,
    const torch::Tensor& physicalInputs) const {
    if (approach != "ffn" && approach != "ffn_pinn" && approach != "pinn") {
        throw std::invalid_argument("Feed-forward inference does not support approach: " + approach);
    }
    const auto modelArtifact = artifacts.models.find(approach);
    const auto scalerArtifact = artifacts.scalers.find(approach);
    if (modelArtifact == artifacts.models.end() || scalerArtifact == artifacts.scalers.end()) {
        throw std::runtime_error("Inference artifacts are incomplete for approach: " + approach);
    }
    if (modelArtifact->second.format != "neuralnetworkwrapper-v1") {
        throw std::runtime_error("Feed-forward inference requires neuralnetworkwrapper-v1.");
    }
    if (!physicalInputs.defined() || physicalInputs.dim() != 2 || physicalInputs.size(0) == 0) {
        throw std::invalid_argument("Feed-forward inference inputs must be a non-empty 2D tensor.");
    }
    if (scalerArtifact->second.input.offset.size() != static_cast<std::size_t>(physicalInputs.size(1))) {
        throw std::invalid_argument("Inference input feature count does not match the exported scaler state.");
    }
    if (scalerArtifact->second.target.offset.size() != 1) {
        throw std::runtime_error("Feed-forward inference requires a scalar target scaler.");
    }

    TensorScaler inputScaler;
    TensorScaler targetScaler;
    inputScaler.importState(scalerArtifact->second.input);
    targetScaler.importState(scalerArtifact->second.target);
    const auto scaledInputs = inputScaler.transform(physicalInputs);

    NeuralNetworkWrapper model;
    model.setHiddenLayers(parseHiddenLayers(artifacts.experiment.config.hidden_layers_csv));
    model.setLags(std::vector<std::vector<int>>(static_cast<std::size_t>(scaledInputs.size(1)), {1}));
    model.initializeNetwork(1, artifacts.experiment.config.activation);
    CheckpointMemoryStream archiveStream(modelArtifact->second.bytes);
    model.loadModel(archiveStream);
    model.setTensorData(DataType::Test, scaledInputs,
                        torch::zeros({scaledInputs.size(0), 1}, scaledInputs.options()));
    const auto prediction = targetScaler.inverseTransform(model.forward(DataType::Test));
    if (!prediction.defined() || prediction.dim() != 2 || prediction.size(0) != physicalInputs.size(0) ||
        prediction.size(1) != 1 || !prediction.isfinite().all().item<bool>()) {
        throw std::runtime_error("Feed-forward checkpoint produced invalid predictions.");
    }
    return prediction;
}

torch::Tensor HydroInferenceRunner::predictRecurrent(
    const HydroInferenceArtifacts& artifacts,
    const std::string& approach,
    const torch::Tensor& physicalSequences) const {
    if (approach != "lstm" && approach != "lstm_pinn") {
        throw std::invalid_argument("Recurrent inference does not support approach: " + approach);
    }
    const auto modelArtifact = artifacts.models.find(approach);
    const auto scalerArtifact = artifacts.scalers.find(approach);
    if (modelArtifact == artifacts.models.end() || scalerArtifact == artifacts.scalers.end()) {
        throw std::runtime_error("Inference artifacts are incomplete for approach: " + approach);
    }
    if (modelArtifact->second.format != "torch-module-v1") {
        throw std::runtime_error("Recurrent inference requires torch-module-v1.");
    }
    if (!physicalSequences.defined() || physicalSequences.dim() != 3 ||
        physicalSequences.size(0) == 0 || physicalSequences.size(1) == 0) {
        throw std::invalid_argument("Recurrent inference inputs must be a non-empty [batch, sequence, feature] tensor.");
    }
    if (physicalSequences.size(1) != artifacts.experiment.config.lstm_sequence_length) {
        throw std::invalid_argument("Inference sequence length does not match the exported configuration.");
    }
    if (scalerArtifact->second.input.offset.size() != static_cast<std::size_t>(physicalSequences.size(2))) {
        throw std::invalid_argument("Inference input feature count does not match the exported scaler state.");
    }
    if (scalerArtifact->second.target.offset.size() != 1) {
        throw std::runtime_error("Recurrent inference requires a scalar target scaler.");
    }

    TensorScaler inputScaler;
    TensorScaler targetScaler;
    inputScaler.importState(scalerArtifact->second.input);
    targetScaler.importState(scalerArtifact->second.target);
    const auto scaledSequences = inputScaler.transform(physicalSequences);
    const auto hiddenLayers = parseHiddenLayers(artifacts.experiment.config.hidden_layers_csv);
    HydroLSTM model(physicalSequences.size(2), hiddenLayers.front(), 1,
                    static_cast<int64_t>(hiddenLayers.size()));
    CheckpointMemoryStream archiveStream(modelArtifact->second.bytes);
    torch::serialize::InputArchive archive;
    archive.load_from(archiveStream);
    model->load(archive);
    model->eval();
    torch::NoGradGuard noGrad;
    const auto prediction = targetScaler.inverseTransform(model->forward(scaledSequences));
    if (!prediction.defined() || prediction.dim() != 2 || prediction.size(0) != physicalSequences.size(0) ||
        prediction.size(1) != 1 || !prediction.isfinite().all().item<bool>()) {
        throw std::runtime_error("Recurrent checkpoint produced invalid predictions.");
    }
    return prediction;
}
