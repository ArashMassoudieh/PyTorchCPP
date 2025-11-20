/**
 * @file ModelTrainer.cpp
 * @brief Implementation of model training logic
 */

#include "ModelTrainer.h"
#include <QDebug>

namespace lstm_predictor {

ModelTrainer::ModelTrainer(LSTMModel model, torch::Device device, QObject* parent)
    : QObject(parent), model_(model), device_(device) {
    model_->to(device_);
}

TrainingHistory ModelTrainer::train(TimeSeriesDataset& trainDataset,
                                    TimeSeriesDataset& valDataset,
                                    int numEpochs,
                                    int batchSize,
                                    double learningRate,
                                    double weightDecay,
                                    int patience) {
    
    qDebug() << "Starting training...";
    qDebug() << "Num epochs:" << numEpochs;
    qDebug() << "Batch size:" << batchSize;
    qDebug() << "Learning rate:" << learningRate;
    
    // Create data loaders
    auto trainLoader = torch::data::make_data_loader(
        trainDataset,
        torch::data::DataLoaderOptions().batch_size(batchSize).drop_last(false)
    );
    
    auto valLoader = torch::data::make_data_loader(
        valDataset,
        torch::data::DataLoaderOptions().batch_size(batchSize).drop_last(false)
    );
    
    // Setup optimizer
    torch::optim::Adam optimizer(model_->parameters(), 
                                 torch::optim::AdamOptions(learningRate)
                                    .weight_decay(weightDecay));
    
    // Loss criterion
    torch::nn::MSELoss criterion;
    
    // Early stopping
    EarlyStopping earlyStopping(patience, 0.0001, true);
    
    // Training history
    TrainingHistory history;
    
    // Training loop
    for (int epoch = 0; epoch < numEpochs; ++epoch) {
        // ============ Training Phase ============
        model_->train();
        double epochTrainLoss = 0.0;
        std::vector<double> trainPreds, trainActuals;
        
        for (auto& batch : *trainLoader) {
            auto data = batch.data.to(device_);
            auto targets = batch.target.to(device_);
            
            // Forward pass
            optimizer.zero_grad();
            auto outputs = model_->forward(data);
            auto loss = criterion(outputs.squeeze(), targets);
            
            // Backward pass
            loss.backward();
            optimizer.step();
            
            // Accumulate metrics
            epochTrainLoss += loss.item<double>();
            
            auto outputsVec = outputs.squeeze().detach().cpu();
            auto targetsVec = targets.cpu();
            for (int i = 0; i < outputsVec.size(0); ++i) {
                trainPreds.push_back(outputsVec[i].item<double>());
                trainActuals.push_back(targetsVec[i].item<double>());
            }
        }
        
        double avgTrainLoss = epochTrainLoss / trainLoader->size().value();
        
        // Calculate training R²
        auto trainPredsTensor = torch::tensor(trainPreds);
        auto trainActualsTensor = torch::tensor(trainActuals);
        double trainR2 = calculateR2(trainActualsTensor, trainPredsTensor);
        
        // ============ Validation Phase ============
        model_->eval();
        double epochValLoss = 0.0;
        std::vector<double> valPreds, valActuals;
        
        torch::NoGradGuard noGrad;
        for (auto& batch : *valLoader) {
            auto data = batch.data.to(device_);
            auto targets = batch.target.to(device_);
            
            auto outputs = model_->forward(data);
            auto loss = criterion(outputs.squeeze(), targets);
            
            epochValLoss += loss.item<double>();
            
            auto outputsVec = outputs.squeeze().cpu();
            auto targetsVec = targets.cpu();
            for (int i = 0; i < outputsVec.size(0); ++i) {
                valPreds.push_back(outputsVec[i].item<double>());
                valActuals.push_back(targetsVec[i].item<double>());
            }
        }
        
        double avgValLoss = epochValLoss / valLoader->size().value();
        
        // Calculate validation R²
        auto valPredsTensor = torch::tensor(valPreds);
        auto valActualsTensor = torch::tensor(valActuals);
        double valR2 = calculateR2(valActualsTensor, valPredsTensor);
        
        // Store history
        history.trainLosses.push_back(avgTrainLoss);
        history.valLosses.push_back(avgValLoss);
        history.trainR2Scores.push_back(trainR2);
        history.valR2Scores.push_back(valR2);
        
        // Print progress every 20 epochs
        if ((epoch + 1) % 20 == 0) {
            qDebug() << QString("Epoch [%1/%2]").arg(epoch + 1).arg(numEpochs);
            qDebug() << QString("  Train - Loss: %1, R²: %2")
                        .arg(avgTrainLoss, 0, 'f', 6)
                        .arg(trainR2, 0, 'f', 6);
            qDebug() << QString("  Val   - Loss: %1, R²: %2")
                        .arg(avgValLoss, 0, 'f', 6)
                        .arg(valR2, 0, 'f', 6);
        }
        
        // Emit signal for GUI update
        emit epochCompleted(epoch + 1, avgTrainLoss, avgValLoss, trainR2, valR2);
        
        // Check early stopping
        if (earlyStopping(avgValLoss, model_)) {
            qDebug() << "Early stopping triggered at epoch" << (epoch + 1);
            
            // Note: In the Python version we restore the best model
            // In C++, this would require loading the saved state
            // For now, we'll continue with the current model
            
            break;
        }
    }
    
    emit trainingCompleted(true, "Training completed successfully");
    
    return history;
}

torch::Tensor ModelTrainer::predict(const torch::Tensor& X) {
    model_->eval();
    torch::NoGradGuard noGrad;
    
    // Move to device and predict
    auto XDevice = X.to(device_);
    
    std::vector<torch::Tensor> predictions;
    int64_t batchSize = 64;
    int64_t numSamples = X.size(0);
    
    for (int64_t i = 0; i < numSamples; i += batchSize) {
        int64_t endIdx = std::min(i + batchSize, numSamples);
        auto batch = XDevice.slice(0, i, endIdx);
        auto output = model_->forward(batch);
        predictions.push_back(output.cpu());
    }
    
    return torch::cat(predictions, 0);
}

double ModelTrainer::calculateR2(const torch::Tensor& yTrue, const torch::Tensor& yPred) {
    // R² = 1 - (SS_res / SS_tot)
    // SS_res = sum of squared residuals
    // SS_tot = total sum of squares
    
    auto mean = yTrue.mean();
    auto ssTot = ((yTrue - mean) * (yTrue - mean)).sum();
    auto ssRes = ((yTrue - yPred) * (yTrue - yPred)).sum();
    
    double r2 = 1.0 - (ssRes.item<double>() / ssTot.item<double>());
    return r2;
}

} // namespace lstm_predictor
