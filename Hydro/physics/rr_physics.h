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

    /**
     * @brief Compute a rainfall-runoff water-balance residual.
     *
     * Residual form:
     *     P - ET - Q - (S_t - S_{t-1}) / dt = 0
     *
     * Tensors are expected to have matching leading dimensions. The first
     * sample is dropped internally because dS/dt is computed by backward
     * finite difference.
     */
    torch::Tensor waterBalanceResidual(const torch::Tensor& rainfall,
                                       const torch::Tensor& evapotranspiration,
                                       const torch::Tensor& runoff,
                                       const torch::Tensor& storage,
                                       const PhysicsConfig& cfg) const;

    /**
     * @brief Penalty for negative runoff values: max(0, -Q).
     */
    torch::Tensor nonNegativeRunoffResidual(const torch::Tensor& runoff) const;
};
