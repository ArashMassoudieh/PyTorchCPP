#pragma once

/**
 * @file ffn_pinn_wrapper.h
 * @brief PINN-enabled feed-forward wrapper interface.
 */

/**
 * @brief FFN + physics-informed training wrapper.
 */
class FFNPINNWrapper {
public:
    /**
     * @brief Run training for FFN with physics residual terms.
     * @return True on successful completion.
     */
    bool train();
};
