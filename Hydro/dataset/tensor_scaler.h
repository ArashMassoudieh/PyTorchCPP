#pragma once

#include <torch/torch.h>

#include <stdexcept>
#include <string>
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

private:
    void ensureFitted() const {
        if (!offset_.defined() || !scale_.defined()) throw std::logic_error("Scaler must be fitted before use.");
    }

    std::string method_ = "none";
    torch::Tensor offset_;
    torch::Tensor scale_;
};
