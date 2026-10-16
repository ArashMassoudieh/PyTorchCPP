#include "../dataset/precip_accumulation.h"
#include <cassert>
#include <cmath>
#include <iostream>

int main() {
    torch::set_num_threads(1);

    // Single segment, window=3: a plain trailing sum, partial near the start.
    {
        auto precip = torch::tensor({1., 2., 3., 4., 5., 6.}, torch::kFloat64);
        auto segment = torch::zeros({6}, torch::kInt64);
        auto acc = trailingPrecipAccumulation(precip, segment, 3);
        const std::vector<double> expected = {1., 3., 6., 9., 12., 15.}; // 3-window trailing sums
        for (int i = 0; i < 6; ++i) {
            assert(std::abs(acc[i].item<double>() - expected[static_cast<std::size_t>(i)]) < 1e-9);
        }
    }

    // Two segments: the accumulation at the start of segment 1 must not
    // include any value from segment 0, even though they sit adjacent in
    // the tensor - this is the entire point of the function.
    {
        auto precip = torch::tensor({10., 10., 10., 1., 2., 3., 4.}, torch::kFloat64);
        auto segment = torch::tensor({0, 0, 0, 1, 1, 1, 1}, torch::kInt64);
        auto acc = trailingPrecipAccumulation(precip, segment, 3);
        // Segment 0 (indices 0-2): plain trailing sums of {10,10,10} -> 10,20,30
        assert(std::abs(acc[0].item<double>() - 10.0) < 1e-9);
        assert(std::abs(acc[1].item<double>() - 20.0) < 1e-9);
        assert(std::abs(acc[2].item<double>() - 30.0) < 1e-9);
        // Segment 1 (indices 3-6): restarts cleanly from {1,2,3,4}, window=3.
        // index 3 (first of segment): just {1} -> 1 (NOT 10+10+1=21)
        assert(std::abs(acc[3].item<double>() - 1.0) < 1e-9);
        // index 4: {1,2} -> 3
        assert(std::abs(acc[4].item<double>() - 3.0) < 1e-9);
        // index 5: {1,2,3} -> 6 (full window, still within segment 1 only)
        assert(std::abs(acc[5].item<double>() - 6.0) < 1e-9);
        // index 6: {2,3,4} -> 9 (window slides, still never reaching index<3)
        assert(std::abs(acc[6].item<double>() - 9.0) < 1e-9);
    }

    // computeStandardPrecipAccumulations exposes exactly the 6/24/72/168h set.
    {
        auto precip = torch::ones({200}, torch::kFloat64);
        auto segment = torch::zeros({200}, torch::kInt64);
        auto diag = computeStandardPrecipAccumulations(precip, segment);
        assert(std::abs(diag.accum_6h[199].item<double>() - 6.0) < 1e-6);
        assert(std::abs(diag.accum_24h[199].item<double>() - 24.0) < 1e-6);
        assert(std::abs(diag.accum_72h[199].item<double>() - 72.0) < 1e-6);
        assert(std::abs(diag.accum_168h[199].item<double>() - 168.0) < 1e-6);
    }

    // Mismatched lengths and a non-positive window must be rejected.
    {
        bool threw = false;
        try {
            trailingPrecipAccumulation(torch::zeros({5}), torch::zeros({4}, torch::kInt64), 3);
        } catch (const std::invalid_argument&) { threw = true; }
        assert(threw);

        threw = false;
        try {
            trailingPrecipAccumulation(torch::zeros({5}), torch::zeros({5}, torch::kInt64), 0);
        } catch (const std::invalid_argument&) { threw = true; }
        assert(threw);
    }

    std::cout << "PASS: trailing precip accumulation respects segment boundaries and matches hand-computed sums" << std::endl;
    return 0;
}
