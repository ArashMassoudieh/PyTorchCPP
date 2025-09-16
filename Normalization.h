#pragma once
#include <limits>
#include <algorithm>
#include <cmath>
#include "TimeSeries.h"
#include "TimeSeriesSet.h"

enum class NormType {
    None,
    Standardize,  // z-score
    MinMax        // scale to [0, 1]
};

template<typename T>
class Normalizer {
public:
    Normalizer(NormType type = NormType::None)
        : type_(type), mean_(0), std_(1), min_(0), max_(1) {}

    // --- Fit ---
    void fit(const TimeSeriesSet<T>& data) {
        if (type_ == NormType::Standardize) {
            double sum = 0.0, sum_sq = 0.0;
            size_t count = 0;
            for (const auto& ts : data) {
                for (const auto& p : ts) {
                    sum += p.c;
                    sum_sq += p.c * p.c;
                    ++count;
                }
            }
            mean_ = (count > 0) ? sum / count : 0.0;
            std_ = (count > 0) ? std::sqrt(sum_sq / count - mean_ * mean_) : 1.0;
            if (std_ == 0.0) std_ = 1.0;
        } else if (type_ == NormType::MinMax) {
            min_ = std::numeric_limits<double>::max();
            max_ = std::numeric_limits<double>::lowest();
            for (const auto& ts : data) {
                for (const auto& p : ts) {
                    min_ = std::min(min_, static_cast<double>(p.c));
                    max_ = std::max(max_, static_cast<double>(p.c));
                }
            }
        }
    }

    void fit(const TimeSeries<T>& data) {
        if (type_ == NormType::Standardize) {
            double sum = 0.0, sum_sq = 0.0;
            size_t count = data.size();
            for (const auto& p : data) {
                sum += p.c;
                sum_sq += p.c * p.c;
            }
            mean_ = (count > 0) ? sum / count : 0.0;
            std_ = (count > 0) ? std::sqrt(sum_sq / count - mean_ * mean_) : 1.0;
            if (std_ == 0.0) std_ = 1.0;
        } else if (type_ == NormType::MinMax) {
            min_ = std::numeric_limits<double>::max();
            max_ = std::numeric_limits<double>::lowest();
            for (const auto& p : data) {
                min_ = std::min(min_, static_cast<double>(p.c));
                max_ = std::max(max_, static_cast<double>(p.c));
            }
        }
    }

    // --- Transform ---
    void transform(TimeSeriesSet<T>& data) const {
        if (type_ == NormType::Standardize) {
            for (auto& ts : data) {
                for (auto& p : ts) {
                    p.c = (p.c - mean_) / std_;
                }
            }
        } else if (type_ == NormType::MinMax) {
            double range = (max_ - min_ == 0.0) ? 1.0 : (max_ - min_);
            for (auto& ts : data) {
                for (auto& p : ts) {
                    p.c = (p.c - min_) / range;
                }
            }
        }
    }

    void transform(TimeSeries<T>& data) const {
        if (type_ == NormType::Standardize) {
            for (auto& p : data) {
                p.c = (p.c - mean_) / std_;
            }
        } else if (type_ == NormType::MinMax) {
            double range = (max_ - min_ == 0.0) ? 1.0 : (max_ - min_);
            for (auto& p : data) {
                p.c = (p.c - min_) / range;
            }
        }
    }

    // --- Inverse Transform ---
    void inverseTransform(TimeSeriesSet<T>& data) const {
        if (type_ == NormType::Standardize) {
            for (auto& ts : data) {
                for (auto& p : ts) {
                    p.c = p.c * std_ + mean_;
                }
            }
        } else if (type_ == NormType::MinMax) {
            double range = (max_ - min_ == 0.0) ? 1.0 : (max_ - min_);
            for (auto& ts : data) {
                for (auto& p : ts) {
                    p.c = p.c * range + min_;
                }
            }
        }
    }

    void inverseTransform(TimeSeries<T>& data) const {
        if (type_ == NormType::Standardize) {
            for (auto& p : data) {
                p.c = p.c * std_ + mean_;
            }
        } else if (type_ == NormType::MinMax) {
            double range = (max_ - min_ == 0.0) ? 1.0 : (max_ - min_);
            for (auto& p : data) {
                p.c = p.c * range + min_;
            }
        }
    }

private:
    NormType type_;
    double mean_, std_, min_, max_;
};
