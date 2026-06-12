#pragma once

/**
 * @file physics_config.h
 * @brief Shared configuration values for HydroPINN physics constraints.
 */

/**
 * @brief Physics weights and toggles used by residual-based losses.
 */
struct PhysicsConfig {
    double lambda_decay = 1.0;          ///< Exponential-decay / reservoir coefficient lambda.
    double data_weight = 1.0;           ///< Weight for supervised data loss.
    double physics_weight = 1.0;        ///< Weight for physics residual loss.

    // Rainfall-runoff / water-balance options. These are additive and do not
    // change the older exp_decay, linear_reservoir, or cstr_first_order paths.
    double runoff_coeff = 0.7;          ///< Optional runoff coefficient for simple runoff constraints.
    double storage_coeff = 1.0;         ///< Optional lumped storage coefficient.
    double dt = 1.0;                    ///< Time step used by finite-difference storage terms.

    bool use_boundary_conditions = false; ///< Enable boundary condition penalties.
    bool use_initial_conditions = true;   ///< Enable initial condition penalties.
};
