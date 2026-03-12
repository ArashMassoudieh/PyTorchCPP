#pragma once

struct PhysicsConfig {
    double lambda_decay = 1.0;
    double data_weight = 1.0;
    double physics_weight = 1.0;
    bool use_boundary_conditions = false;
    bool use_initial_conditions = true;
};
