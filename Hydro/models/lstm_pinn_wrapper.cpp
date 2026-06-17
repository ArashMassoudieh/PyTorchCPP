#include "lstm_pinn_wrapper.h"
#include "lstmnetworkwrapper.h"

HydroRunResult LSTMPINNWrapper::train(const HydroRunConfig& config) {
    LSTMNetworkWrapper backend;
    return backend.train(config, true);
}
