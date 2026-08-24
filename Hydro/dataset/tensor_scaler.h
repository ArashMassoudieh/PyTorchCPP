#pragma once

#include <torch/torch.h>
#include "../models/hydro_run_types.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

class TensorScaler {
public:
    void fit(const torch::Tensor& training, const std::string& method) {
        if (!training.defined() || training.numel() == 0) throw std::invalid_argument("Cannot fit scaler on empty training data.");
        method_ = method;
        const auto reduceDimensions = training.dim() == 3 ? std::vector<int64_t>{0, 1}
                                                           : std::vector<int64_t>{0};
        if (method_ == "standardize") {
            offset_ = training.mean(reduceDimensions, true);
            scale_ = training.std(reduceDimensions, false, true);
        } else if (method_ == "minmax") {
            offset_ = training.amin(reduceDimensions, true);
            scale_ = training.amax(reduceDimensions, true) - offset_;
        } else if (method_ == "none") {
            offset_ = torch::zeros_like(training.mean(reduceDimensions, true));
            scale_ = torch::ones_like(offset_);
        } else {
            throw std::invalid_argument("Unknown normalization method: " + method_);
        }
        scale_ = torch::where(torch::abs(scale_) < 1.0e-12, torch::ones_like(scale_), scale_);
    }

    torch::Tensor transform(const torch::Tensor& values) const {
        ensureFitted();
        return (values - offset_) / scale_;
    }

    torch::Tensor inverseTransform(const torch::Tensor& values) const {
        ensureFitted();
        return values * scale_ + offset_;
    }

    double mseToPhysical(double scaledMse) const {
        ensureFitted();
        if (scale_.numel() != 1) throw std::logic_error("MSE conversion requires a scalar target scale.");
        const double scale = scale_.item<double>();
        return scaledMse * scale * scale;
    }

    HydroScalerState exportState() const {
        ensureFitted();
        HydroScalerState state;
        state.method = method_;
        const auto offset = offset_.to(torch::kCPU).to(torch::kFloat64).contiguous();
        const auto scale = scale_.to(torch::kCPU).to(torch::kFloat64).contiguous();
        state.shape.assign(offset.sizes().begin(), offset.sizes().end());
        state.offset.assign(offset.data_ptr<double>(), offset.data_ptr<double>() + offset.numel());
        state.scale.assign(scale.data_ptr<double>(), scale.data_ptr<double>() + scale.numel());
        return state;
    }

    void importState(const HydroScalerState& state) {
        if (state.offset.empty() || state.offset.size() != state.scale.size() || state.shape.empty()) {
            throw std::invalid_argument("Scaler state is incomplete.");
        }
        if (state.method != "none" && state.method != "standardize" && state.method != "minmax") {
            throw std::invalid_argument("Scaler state has an unsupported method.");
        }
        int64_t expected = 1;
        for (const int64_t extent : state.shape) {
            if (extent <= 0) throw std::invalid_argument("Scaler state has an invalid shape.");
            if (expected > std::numeric_limits<int64_t>::max() / extent) {
                throw std::invalid_argument("Scaler state shape is too large.");
            }
            expected *= extent;
        }
        if (expected != static_cast<int64_t>(state.offset.size())) throw std::invalid_argument("Scaler state shape does not match its values.");
        for (std::size_t i = 0; i < state.offset.size(); ++i) {
            if (!std::isfinite(state.offset[i]) || !std::isfinite(state.scale[i]) || state.scale[i] == 0.0) {
                throw std::invalid_argument("Scaler state contains invalid numeric values.");
            }
        }
        auto offset = torch::tensor(state.offset, torch::kFloat64).to(torch::kFloat32).reshape(state.shape);
        auto scale = torch::tensor(state.scale, torch::kFloat64).to(torch::kFloat32).reshape(state.shape);
        if (!offset.isfinite().all().item<bool>() || !scale.isfinite().all().item<bool>() ||
            torch::eq(scale, 0).any().item<bool>()) {
            throw std::invalid_argument("Scaler state cannot be represented safely as float tensors.");
        }
        method_ = state.method;
        offset_ = std::move(offset);
        scale_ = std::move(scale);
    }

private:
    void ensureFitted() const {
        if (!offset_.defined() || !scale_.defined()) throw std::logic_error("Scaler must be fitted before use.");
    }

    std::string method_ = "none";
    torch::Tensor offset_;
    torch::Tensor scale_;
};
