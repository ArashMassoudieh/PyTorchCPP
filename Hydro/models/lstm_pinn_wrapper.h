#pragma once

/**
 * @file lstm_pinn_wrapper.h
 * @brief PINN-enabled LSTM wrapper interface.
 */

/**
 * @brief LSTM + physics-informed training wrapper.
 */
class LSTMPINNWrapper {
public:
    /**
     * @brief Run LSTM training with physics constraints.
     * @return True on successful completion.
     */
    bool train();
};
