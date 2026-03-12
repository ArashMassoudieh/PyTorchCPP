#pragma once

/**
 * @file ffn_wrapper.h
 * @brief Baseline feed-forward network wrapper interface.
 */

/**
 * @brief Lightweight FFN baseline wrapper for HydroPINN experiments.
 */
class FFNWrapper {
public:
    /**
     * @brief Run training for the FFN baseline.
     * @return True on successful completion.
     */
    bool train();
};
