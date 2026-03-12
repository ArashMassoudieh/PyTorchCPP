#pragma once

/**
 * @file lstm_wrapper.h
 * @brief Baseline LSTM wrapper interface.
 */

/**
 * @brief Lightweight LSTM baseline wrapper for HydroPINN experiments.
 */
class LSTMWrapper {
public:
    /**
     * @brief Run baseline LSTM training.
     * @return True on successful completion.
     */
    bool train();
};
