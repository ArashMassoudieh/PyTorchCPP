#include "rr_physics.h"

torch::Tensor RRPhysics::exponentialResidual(const torch::Tensor& dy_dt,
                                             const torch::Tensor& y,
                                             const PhysicsConfig& cfg) const {
    return dy_dt + cfg.lambda_decay * y;
}
