#pragma once

/**
 * @file physics_config.h
 * @brief Shared configuration values for HydroPINN physics constraints.
 */

/**
 * @brief Physics weights and toggles used by residual-based losses.
 */
struct PhysicsConfig {
    double lambda_decay = 1.0;          ///< Exponential-decay coefficient lambda.
    double data_weight = 1.0;           ///< Weight for supervised data loss.
    double physics_weight = 1.0;        ///< Weight for physics residual loss.
    bool use_boundary_conditions = false; ///< Enable boundary condition penalties.
    bool use_initial_conditions = true;   ///< Enable initial condition penalties.
};
