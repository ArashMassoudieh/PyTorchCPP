#include "rr_physics.h"

#include <algorithm>

torch::Tensor RRPhysics::exponentialResidual(const torch::Tensor& dy_dt,
                                             const torch::Tensor& y,
                                             const PhysicsConfig& cfg) const {
    return dy_dt + cfg.lambda_decay * y;
}


torch::Tensor RRPhysics::waterBalanceResidual(const torch::Tensor& rainfall,
                                              const torch::Tensor& evapotranspiration,
                                              const torch::Tensor& runoff,
                                              const torch::Tensor& storage,
                                              const PhysicsConfig& cfg) const {
    if (!rainfall.defined() || !evapotranspiration.defined() || !runoff.defined() || !storage.defined()) {
        return torch::zeros({1}, torch::kFloat32);
    }

    const auto n = std::min({rainfall.size(0), evapotranspiration.size(0), runoff.size(0), storage.size(0)});
    if (n < 2) {
        return torch::zeros({1}, runoff.options());
    }

    const double dt = (cfg.dt > 0.0) ? cfg.dt : 1.0;

    auto P = rainfall.slice(0, 1, n);
    auto ET = evapotranspiration.slice(0, 1, n);
    auto Q = runoff.slice(0, 1, n);
    auto S_now = storage.slice(0, 1, n);
    auto S_prev = storage.slice(0, 0, n - 1);
    auto dSdt = (S_now - S_prev) / dt;

    return P - ET - Q - dSdt;
}

torch::Tensor RRPhysics::nonNegativeRunoffResidual(const torch::Tensor& runoff) const {
    if (!runoff.defined()) {
        return torch::zeros({1}, torch::kFloat32);
    }
    return torch::relu(-runoff);
}
