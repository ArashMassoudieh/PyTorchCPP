#include "lstm_wrapper.h"
#include "lstmnetworkwrapper.h"

HydroRunResult LSTMWrapper::train(const HydroRunConfig& config) {
    LSTMNetworkWrapper backend;
    return backend.train(config, false);
}
