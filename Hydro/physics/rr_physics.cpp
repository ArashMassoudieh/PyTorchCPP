#include "rr_physics.h"

#include <algorithm>
#include <stdexcept>

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

torch::Tensor RRPhysics::waterBalanceResidualAtTimes(const torch::Tensor& rainfall,
                                                     const torch::Tensor& evapotranspiration,
                                                     const torch::Tensor& runoff,
                                                     const torch::Tensor& storage,
                                                     const torch::Tensor& timestamps) const {
    if (!rainfall.defined() || !evapotranspiration.defined() || !runoff.defined() ||
        !storage.defined() || !timestamps.defined()) {
        return torch::zeros({1}, torch::kFloat32);
    }
    auto rain = rainfall.reshape({-1});
    auto et = evapotranspiration.reshape({-1});
    auto q = runoff.reshape({-1});
    auto s = storage.reshape({-1});
    auto time = timestamps.reshape({-1});
    const auto n = std::min({rain.size(0), et.size(0), q.size(0), s.size(0), time.size(0)});
    if (n < 2) return torch::zeros({1}, runoff.options());
    auto dt = time.slice(0, 1, n) - time.slice(0, 0, n - 1);
    if ((dt <= 0).any().item<bool>()) {
        throw std::invalid_argument("Water-balance timestamps must be strictly increasing.");
    }
    auto dSdt = (s.slice(0, 1, n) - s.slice(0, 0, n - 1)) / dt;
    return rain.slice(0, 1, n) - et.slice(0, 1, n) - q.slice(0, 1, n) - dSdt;
}

torch::Tensor RRPhysics::runoffReservoirResidual(const torch::Tensor& rainfall,
                                                 const torch::Tensor& evapotranspiration,
                                                 const torch::Tensor& runoff,
                                                 double recession_per_time,
                                                 const PhysicsConfig& cfg) const {
    if (!rainfall.defined() || !evapotranspiration.defined() || !runoff.defined()) {
        return torch::zeros({1}, torch::kFloat32);
    }
    const auto n = std::min({rainfall.size(0), evapotranspiration.size(0), runoff.size(0)});
    if (n < 2) return torch::zeros({1}, runoff.options());

    const double dt = (cfg.dt > 0.0) ? cfg.dt : 1.0;
    const double k = std::max(1.0e-12, recession_per_time);
    auto q_now = runoff.slice(0, 1, n);
    auto q_prev = runoff.slice(0, 0, n - 1);
    auto dqdt = (q_now - q_prev) / dt;
    auto net_input = rainfall.slice(0, 1, n) - evapotranspiration.slice(0, 1, n);
    return dqdt - k * (net_input - q_now);
}

torch::Tensor RRPhysics::runoffReservoirResidualAtTimes(const torch::Tensor& rainfall,
                                                        const torch::Tensor& evapotranspiration,
                                                        const torch::Tensor& runoff,
                                                        double recession_per_time,
                                                        const torch::Tensor& timestamps) const {
    if (!rainfall.defined() || !evapotranspiration.defined() || !runoff.defined() || !timestamps.defined()) {
        return torch::zeros({1}, torch::kFloat32);
    }
    auto rain = rainfall.reshape({-1});
    auto et = evapotranspiration.reshape({-1});
    auto q = runoff.reshape({-1});
    auto time = timestamps.reshape({-1});
    const auto n = std::min({rain.size(0), et.size(0), q.size(0), time.size(0)});
    if (n < 2) return torch::zeros({1}, runoff.options());

    auto dt = time.slice(0, 1, n) - time.slice(0, 0, n - 1);
    if ((dt <= 0).any().item<bool>()) {
        throw std::invalid_argument("Runoff-reservoir timestamps must be strictly increasing.");
    }
    const double k = std::max(1.0e-12, recession_per_time);
    auto q_now = q.slice(0, 1, n);
    auto q_prev = q.slice(0, 0, n - 1);
    auto dqdt = (q_now - q_prev) / dt;
    auto net_input = rain.slice(0, 1, n) - et.slice(0, 1, n);
    return dqdt - k * (net_input - q_now);
}

torch::Tensor RRPhysics::nonNegativeRunoffResidual(const torch::Tensor& runoff) const {
    if (!runoff.defined()) {
        return torch::zeros({1}, torch::kFloat32);
    }
    return torch::relu(-runoff);
}
