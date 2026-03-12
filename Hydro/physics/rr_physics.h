#pragma once

#include <torch/torch.h>
#include "physics_config.h"

class RRPhysics {
public:
    torch::Tensor exponentialResidual(const torch::Tensor& dy_dt,
                                      const torch::Tensor& y,
                                      const PhysicsConfig& cfg) const;
};
