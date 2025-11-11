/**
 * @file EarlyStopping.cpp
 * @brief Implementation of early stopping mechanism
 */

#include "EarlyStopping.h"
#include <QDebug>
#include <limits>

namespace lstm_predictor {

EarlyStopping::EarlyStopping(int patience, double minDelta, bool verbose)
    : patience_(patience),
      minDelta_(minDelta),
      verbose_(verbose),
      counter_(0),
      bestLoss_(std::numeric_limits<double>::infinity()),
      earlyStop_(false),
      bestModelState_(nullptr) {
}

bool EarlyStopping::operator()(double valLoss, std::shared_ptr<torch::nn::Module> model) {
    // First call - initialize best loss
    if (bestLoss_ == std::numeric_limits<double>::infinity()) {
        bestLoss_ = valLoss;
        
        // Save model state
        bestModelState_ = std::make_shared<torch::serialize::OutputArchive>();
        model->save(*bestModelState_);
        
        return false;
    }
    
    // Check if validation loss improved
    if (valLoss < bestLoss_ - minDelta_) {
        // Improvement detected
        bestLoss_ = valLoss;
        counter_ = 0;
        
        // Save new best model
        bestModelState_ = std::make_shared<torch::serialize::OutputArchive>();
        model->save(*bestModelState_);
        
    } else {
        // No improvement
        counter_++;
        
        if (verbose_ && counter_ % 5 == 0) {
            qDebug() << "EarlyStopping counter:" << counter_ << "out of" << patience_;
        }
        
        if (counter_ >= patience_) {
            earlyStop_ = true;
            if (verbose_) {
                qDebug() << "Early stopping triggered!";
            }
        }
    }
    
    return earlyStop_;
}

torch::serialize::OutputArchive EarlyStopping::getBestModel() const {
    if (bestModelState_) {
        return *bestModelState_;
    }
    throw std::runtime_error("No best model state available");
}

} // namespace lstm_predictor
