/**
 * @file DataPreprocessor.cpp
 * @brief Implementation of data preprocessing functionality
 */

#include "DataPreprocessor.h"
#include <QFile>
#include <QTextStream>
#include <QStringList>
#include <QDebug>
#include <algorithm>
#include <random>

namespace lstm_predictor {

DataPreprocessor::DataPreprocessor(int seqLength, double testSize)
    : seqLength_(seqLength), testSize_(testSize) {
    TORCH_CHECK(seqLength > 0, "Sequence length must be positive");
    TORCH_CHECK(testSize > 0.0 && testSize < 1.0, "Test size must be between 0 and 1");
}

ProcessedData DataPreprocessor::loadAndPreprocess(const QString& filePath, bool randomSplit) {
    qDebug() << "Loading data from:" << filePath;
    
    // Read CSV file
    torch::Tensor data = readCSV(filePath);
    qDebug() << "Data shape:" << data.sizes();
    
    // Separate features (all columns except last) and target (last column)
    auto X = data.index({torch::indexing::Slice(), 
                         torch::indexing::Slice(torch::indexing::None, -1)});
    auto y = data.index({torch::indexing::Slice(), -1}).unsqueeze(1);
    
    // Normalize features and target separately
    auto [XNormalized, meanX, stdX] = normalize(X);
    auto [yNormalized, meanY, stdY] = normalize(y);
    
    // Combine normalized data for sequence creation
    auto dataNormalized = torch::cat({XNormalized, yNormalized}, 1);
    
    // Create sequences
    auto [XSeq, ySeq] = createSequences(dataNormalized);
    qDebug() << "Sequence shapes - X:" << XSeq.sizes() << "y:" << ySeq.sizes();
    
    // Split into train and test
    int64_t numSamples = XSeq.size(0);
    int64_t numTest = static_cast<int64_t>(numSamples * testSize_);
    int64_t numTrain = numSamples - numTest;
    
    torch::Tensor XTrain, XTest, yTrain, yTest;
    std::vector<int> trainTestMask(numSamples);
    
    if (randomSplit) {
        qDebug() << "Using RANDOM split";
        
        // Create indices and shuffle them
        std::vector<int64_t> indices(numSamples);
        std::iota(indices.begin(), indices.end(), 0);
        
        std::random_device rd;
        std::mt19937 gen(42);  // Fixed seed for reproducibility
        std::shuffle(indices.begin(), indices.end(), gen);
        
        // Split indices
        std::vector<int64_t> trainIndices(indices.begin(), indices.begin() + numTrain);
        std::vector<int64_t> testIndices(indices.begin() + numTrain, indices.end());
        
        // Create tensors from indices
        auto trainIdxTensor = torch::tensor(trainIndices);
        auto testIdxTensor = torch::tensor(testIndices);
        
        // Index into data
        XTrain = XSeq.index_select(0, trainIdxTensor);
        yTrain = ySeq.index_select(0, trainIdxTensor);
        XTest = XSeq.index_select(0, testIdxTensor);
        yTest = ySeq.index_select(0, testIdxTensor);
        
        // Create mask
        for (int64_t idx : trainIndices) {
            trainTestMask[idx] = 0;  // Training
        }
        for (int64_t idx : testIndices) {
            trainTestMask[idx] = 1;  // Testing
        }
    } else {
        qDebug() << "Using SEQUENTIAL split";
        
        // Sequential split: first numTrain for training, rest for testing
        XTrain = XSeq.narrow(0, 0, numTrain);
        yTrain = ySeq.narrow(0, 0, numTrain);
        XTest = XSeq.narrow(0, numTrain, numTest);
        yTest = ySeq.narrow(0, numTrain, numTest);
        
        // Create mask
        for (int64_t i = 0; i < numTrain; ++i) {
            trainTestMask[i] = 0;  // Training
        }
        for (int64_t i = numTrain; i < numSamples; ++i) {
            trainTestMask[i] = 1;  // Testing
        }
    }
    
    qDebug() << "Training samples:" << numTrain;
    qDebug() << "Testing samples:" << numTest;
    
    // Create ProcessedData structure
    ProcessedData processed;
    processed.XTrain = XTrain;
    processed.yTrain = yTrain;
    processed.XTest = XTest;
    processed.yTest = yTest;
    processed.XAll = XSeq;
    processed.yAll = ySeq;
    processed.trainTestMask = trainTestMask;
    processed.meanX = meanX;
    processed.stdX = stdX;
    processed.meanY = meanY;
    processed.stdY = stdY;
    
    return processed;
}

torch::Tensor DataPreprocessor::readCSV(const QString& filePath) {
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        throw std::runtime_error("Cannot open file: " + filePath.toStdString());
    }
    
    QTextStream in(&file);
    std::vector<std::vector<double>> dataVec;
    
    // Skip header line
    if (!in.atEnd()) {
        in.readLine();
    }
    
    // Read data lines
    while (!in.atEnd()) {
        QString line = in.readLine();
        QStringList fields = line.split(',');
        
        std::vector<double> row;
        for (const QString& field : fields) {
            bool ok;
            double value = field.trimmed().toDouble(&ok);
            if (ok) {
                row.push_back(value);
            }
        }
        
        if (!row.empty()) {
            dataVec.push_back(row);
        }
    }
    
    file.close();
    
    // Convert to tensor
    int64_t numRows = dataVec.size();
    int64_t numCols = dataVec[0].size();
    
    std::vector<double> flatData;
    flatData.reserve(numRows * numCols);
    for (const auto& row : dataVec) {
        flatData.insert(flatData.end(), row.begin(), row.end());
    }
    
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    return torch::from_blob(flatData.data(), {numRows, numCols}, options).clone();
}

std::pair<torch::Tensor, torch::Tensor> 
DataPreprocessor::createSequences(const torch::Tensor& data) {
    int64_t numSamples = data.size(0) - seqLength_;
    int64_t numFeatures = data.size(1) - 1;  // Exclude target column
    
    // Pre-allocate tensors
    auto X = torch::zeros({numSamples, seqLength_, numFeatures});
    auto y = torch::zeros({numSamples});
    
    for (int64_t i = 0; i < numSamples; ++i) {
        // Input: sequence of features (all columns except last)
        X[i] = data.index({torch::indexing::Slice(i, i + seqLength_),
                           torch::indexing::Slice(torch::indexing::None, -1)});
        
        // Target: last column at position i + seqLength
        y[i] = data[i + seqLength_][-1];
    }
    
    return {X, y};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> 
DataPreprocessor::normalize(const torch::Tensor& data) {
    // Compute mean and std along dimension 0 (across samples)
    auto mean = data.mean(0, true);
    auto std = data.std(0, true);
    
    // Prevent division by zero
    std = torch::where(std < 1e-8, torch::ones_like(std), std);
    
    // Normalize: (x - mean) / std
    auto normalized = (data - mean) / std;
    
    return {normalized, mean, std};
}

torch::Tensor DataPreprocessor::denormalize(const torch::Tensor& normalizedY,
                                            const torch::Tensor& meanY,
                                            const torch::Tensor& stdY) {
    // Denormalize: x = normalized * std + mean
    return normalizedY * stdY + meanY;
}

} // namespace lstm_predictor
