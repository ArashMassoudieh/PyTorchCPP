// Synthetic-only test of run_frozen_shared_scale_inference.cpp's
// requireDevelopmentOnly guard (constants + function copied verbatim,
// lines 170-190 of that file as last read, 2026-10-28). No real file is
// opened, no package is read, no reserved observation of any kind is
// touched -- every x value below is a fabricated torch::tensor literal.
// This exists specifically so the holdout-mode branch can be exercised
// before the real holdout run, per the explicit instruction to test it
// with synthetic timestamps first.

#include <torch/torch.h>
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

// --- verbatim copy from run_frozen_shared_scale_inference.cpp ---
constexpr int64_t kSealStartX = 77016;
constexpr int64_t kHoldoutSanityCeilingX = 120000;

void requireDevelopmentOnly(const torch::Tensor& plotX, const char* stage, bool holdoutMode) {
    const double maxX = plotX.max().item<double>();
    const double ceiling = holdoutMode ? static_cast<double>(kHoldoutSanityCeilingX)
                                       : static_cast<double>(kSealStartX);
    if (maxX >= ceiling) {
        throw std::runtime_error(
            std::string("Development cutoff guard tripped at stage '") + stage +
            "': max x=" + std::to_string(maxX) + " reaches or exceeds the " +
            (holdoutMode ? "holdout sanity ceiling" : "reserved-period seal") +
            " (x>=" + std::to_string(static_cast<int64_t>(ceiling)) + "). Refusing.");
    }
}
// --- end verbatim copy ---

bool throws(const torch::Tensor& x, bool holdoutMode) {
    try {
        requireDevelopmentOnly(x, "synthetic-test", holdoutMode);
        return false;
    } catch (const std::runtime_error&) {
        return true;
    }
}

} // namespace

int main() {
    // Default (non-holdout) path: identical to every one of the 12 real
    // development runs so far (max x always <= 77015, all passed).
    assert(!throws(torch::tensor({100.0, 50000.0, 77015.0}), /*holdoutMode=*/false));
    // At the seal boundary itself: must refuse, non-holdout.
    assert(throws(torch::tensor({77016.0}), /*holdoutMode=*/false));
    // Past the seal: must refuse, non-holdout.
    assert(throws(torch::tensor({90000.0}), /*holdoutMode=*/false));

    // Holdout mode: the seal boundary itself must now be ALLOWED.
    assert(!throws(torch::tensor({77016.0}), /*holdoutMode=*/true));
    // Holdout mode: a value in the known full-package range (~87672) must be allowed.
    assert(!throws(torch::tensor({87672.0}), /*holdoutMode=*/true));
    // Holdout mode: right at the sanity ceiling must still refuse (the ceiling
    // itself is not a safe value, only values strictly below it are).
    assert(throws(torch::tensor({120000.0}), /*holdoutMode=*/true));
    // Holdout mode: comfortably below the ceiling must be allowed.
    assert(!throws(torch::tensor({119999.0}), /*holdoutMode=*/true));
    // Holdout mode does NOT disable the guard entirely -- a grossly wrong
    // value far past any real package must still refuse even in holdout mode.
    assert(throws(torch::tensor({500000.0}), /*holdoutMode=*/true));

    std::cout << "ALL SYNTHETIC GUARD TESTS PASSED (no real file opened; no reserved observation touched)\n";
    return 0;
}
