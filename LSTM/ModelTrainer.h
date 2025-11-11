/**
 * @file ModelTrainer.h
 * @brief Training logic for LSTM model
 * 
 * Handles the complete training process including:
 * - Training loop with validation
 * - Early stopping integration
 * - Loss and R² tracking
 * - Progress reporting via Qt signals
 * 
 * @author Generated for Time Series Prediction
 * @date 2025
 */

#ifndef MODELTRAINER_H
#define MODELTRAINER_H

#include "LSTMModel.h"
#include "TimeSeriesDataset.h"
#include "EarlyStopping.h"
#include <QObject>
#include <torch/torch.h>
#include <vector>
#include <memory>

namespace lstm_predictor {

/**
 * @struct TrainingHistory
 * @brief Container for training metrics history
 */
struct TrainingHistory {
    std::vector<double> trainLosses;    ///< Training loss per epoch
    std::vector<double> valLosses;      ///< Validation loss per epoch
    std::vector<double> trainR2Scores;  ///< Training R² per epoch
    std::vector<double> valR2Scores;    ///< Validation R² per epoch
};

/**
 * @class ModelTrainer
 * @brief Manages the training process for LSTM model
 * 
 * This class inherits from QObject to enable Qt signals for progress updates
 */
class ModelTrainer : public QObject {
    Q_OBJECT

public:
    /**
     * @brief Constructor
     * 
     * @param model LSTM model to train
     * @param device Torch device (CPU or CUDA)
     * @param parent Qt parent object
     */
    explicit ModelTrainer(LSTMModel model, torch::Device device, QObject* parent = nullptr);

    /**
     * @brief Train the model
     * 
     * @param trainDataset Training dataset
     * @param valDataset Validation/test dataset
     * @param numEpochs Maximum number of epochs
     * @param batchSize Batch size for training
     * @param learningRate Learning rate for optimizer
     * @param weightDecay L2 regularization parameter
     * @param patience Early stopping patience
     * @return TrainingHistory containing all metrics
     */
    TrainingHistory train(TimeSeriesDataset& trainDataset,
                         TimeSeriesDataset& valDataset,
                         int numEpochs,
                         int batchSize,
                         double learningRate,
                         double weightDecay,
                         int patience);

    /**
     * @brief Make predictions on dataset
     * 
     * @param X Input sequences
     * @return Predictions tensor
     */
    torch::Tensor predict(const torch::Tensor& X);

    /**
     * @brief Calculate R² score
     * 
     * @param yTrue True values
     * @param yPred Predicted values
     * @return R² score value
     */
    static double calculateR2(const torch::Tensor& yTrue, const torch::Tensor& yPred);

signals:
    /**
     * @brief Signal emitted when epoch completes
     * 
     * @param epoch Current epoch number
     * @param trainLoss Training loss
     * @param valLoss Validation loss
     * @param trainR2 Training R²
     * @param valR2 Validation R²
     */
    void epochCompleted(int epoch, double trainLoss, double valLoss, 
                       double trainR2, double valR2);

    /**
     * @brief Signal emitted when training completes
     * 
     * @param success Whether training completed successfully
     * @param message Status message
     */
    void trainingCompleted(bool success, QString message);

private:
    LSTMModel model_;       ///< LSTM model
    torch::Device device_;  ///< Computation device
};

} // namespace lstm_predictor

#endif // MODELTRAINER_H
