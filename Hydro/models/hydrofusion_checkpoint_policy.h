#pragma once
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
namespace hydrofusion {
inline void validateForecastCheckpointPolicy(const std::string& policy){
    if(policy!="best_validation_v1"&&policy!="fixed_last_epoch_v1")
        throw std::invalid_argument("Unknown HydroFusion checkpoint selection policy.");
}
inline bool selectForecastCheckpoint(const std::string& policy,int epoch,int epochs,double validation,double best){
    validateForecastCheckpointPolicy(policy);
    if(epoch<1||epochs<1||epoch>epochs||!std::isfinite(validation)||
       (!std::isfinite(best)&&best!=std::numeric_limits<double>::infinity()))
        throw std::invalid_argument("Invalid HydroFusion checkpoint selection state.");
    // Fixed epochs are selected prospectively outside this training run. Loss
    // remains a diagnostic; its value cannot choose the saved weights.
    return policy=="fixed_last_epoch_v1"?epoch==epochs:validation<best;
}
}
