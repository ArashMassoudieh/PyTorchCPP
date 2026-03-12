#pragma once

#include <torch/torch.h>
#include "physics_config.h"

/**
 * @file rr_physics.h
 * @brief Physics residual utilities for rainfall-runoff style constraints.
 */

/**
 * @brief Residual helpers for PINN physics terms.
 */
class RRPhysics {
public:
    /**
     * @brief Compute exponential decay residual: dy/dt + lambda * y.
     * @param dy_dt Time derivative term.
     * @param y State/prediction tensor.
     * @param cfg Physics configuration.
     * @return Residual tensor to be minimized by PINN training.
     */
    torch::Tensor exponentialResidual(const torch::Tensor& dy_dt,
                                      const torch::Tensor& y,
                                      const PhysicsConfig& cfg) const;
};
