/**
 * @file EarlyStopping.h
 * @brief Early stopping mechanism to prevent overfitting
 * 
 * Monitors validation loss and stops training when it stops improving.
 * Saves the best model state automatically.
 * 
 * @author Generated for Time Series Prediction
 * @date 2025
 */

#ifndef EARLYSTOPPING_H
#define EARLYSTOPPING_H

#include <torch/torch.h>
#include <memory>

namespace lstm_predictor {

/**
 * @class EarlyStopping
 * @brief Implements early stopping based on validation loss
 * 
 * This class monitors validation loss and triggers early stopping
 * when the loss stops improving for a specified number of epochs.
 */
class EarlyStopping {
public:
    /**
     * @brief Constructor
     * 
     * @param patience Number of epochs to wait before stopping
     * @param minDelta Minimum change to qualify as improvement
     * @param verbose If true, print messages when triggered
     */
    EarlyStopping(int patience = 30, double minDelta = 0.0001, bool verbose = true);

    /**
     * @brief Check if training should stop
     * 
     * @param valLoss Current validation loss
     * @param model Pointer to the model (to save best weights)
     * @return true if training should stop, false otherwise
     */
    bool operator()(double valLoss, std::shared_ptr<torch::nn::Module> model);

    /**
     * @brief Get the best model state dict
     * @return OrderedDict containing best model parameters
     */
    torch::serialize::OutputArchive getBestModel() const;

    /**
     * @brief Get the best validation loss achieved
     * @return Best validation loss value
     */
    double getBestLoss() const { return bestLoss_; }

    /**
     * @brief Check if early stopping has been triggered
     * @return true if early stopping condition is met
     */
    bool shouldStop() const { return earlyStop_; }

    /**
     * @brief Get current counter value
     * @return Number of epochs without improvement
     */
    int getCounter() const { return counter_; }

private:
    int patience_;          ///< Number of epochs to wait before stopping
    double minDelta_;       ///< Minimum change to qualify as improvement
    bool verbose_;          ///< Whether to print messages
    int counter_;           ///< Counter for epochs without improvement
    double bestLoss_;       ///< Best validation loss achieved
    bool earlyStop_;        ///< Flag indicating if should stop
    std::shared_ptr<torch::serialize::OutputArchive> bestModelState_;  ///< Best model state
};

} // namespace lstm_predictor

#endif // EARLYSTOPPING_H
