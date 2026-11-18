#include "../models/hydro_lstm_module.h"
#include "../models/two_reservoir_routing.h"
#include "../models/urban_managed_routing.h"
#include "../dataset/configured_chronological_split.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <sstream>

int main() {
    torch::manual_seed(42);
    torch::set_num_threads(1);
    {
        HydroRunConfig config;
        config.train_end_x = 3; config.validation_end_x = 6; config.test_end_x = 9;
        auto split = configuredChronologicalSplit(torch::tensor({1.,2.,5.,6.,7.,9.}), config);
        assert(split.train_end == 2 && split.validation_end == 4 && split.test_end == 6);
        config.validation_end_x = -1;
        bool rejected = false;
        try { (void)configuredChronologicalSplit(torch::tensor({1.,2.,5.}), config); }
        catch (const std::invalid_argument&) { rejected = true; }
        assert(rejected);
    }
    {
        UrbanManagedRoutingParams p;
        p.quickK = 1.2; p.slowK = .03; p.detentionK = .08; p.channelK = .7;
        p.quickFraction = .26; p.detentionCatchmentFraction = .183;
        const auto constant = simulateUrbanManagedBaseline(std::vector<double>(2000, 2.0), 0.0, p);
        // Every branch conserves its assigned fraction, so steady outflow
        // converges to the common forcing magnitude.
        assert(std::abs(constant.back() - 2.0) < 1e-8);

        const std::vector<double> pulse{0., 1., 0., 0., 0., 0., 0., 0.};
        const auto managed = simulateUrbanManagedBaseline(pulse, 0.0, p);
        auto noDetention = p;
        noDetention.detentionCatchmentFraction = 0.0;
        const auto unmanaged = simulateUrbanManagedBaseline(pulse, 0.0, noDetention);
        assert(managed[2] < unmanaged[2]);
        assert(managed[7] > unmanaged[7]);

        const std::vector<std::int64_t> segments{0,0,0,1,1,1,1,1};
        const auto fixedA = simulateUrbanManagedBaselinePerSegment(
            pulse, std::vector<double>(pulse.size(), 4.0), segments, p, 0.0);
        const auto fixedB = simulateUrbanManagedBaselinePerSegment(
            pulse, std::vector<double>(pulse.size(), 99.0), segments, p, 0.0);
        assert(fixedA == fixedB);
        assert(fixedA[0] == 0.0 && fixedA[3] == 0.0);

        bool rejected = false;
        try {
            auto invalid = p; invalid.detentionCatchmentFraction = 1.1;
            (void)simulateUrbanManagedBaseline(pulse, 0.0, invalid);
        } catch (const std::invalid_argument&) { rejected = true; }
        assert(rejected);
    }
    {
        std::vector<double> rain{0.,1.,0.,0.,1.,0.};
        auto segments = torch::tensor({0,0,0,1,1,1}, torch::kInt64);
        TwoReservoirRoutingParams params;
        auto a = simulateTwoReservoirBaselinePerSegment(rain, {2.,2.,2.,3.,3.,3.}, segments, params, 0.);
        auto b = simulateTwoReservoirBaselinePerSegment(rain, {99.,99.,99.,88.,88.,88.}, segments, params, 0.);
        assert(a == b); // Fixed initialization must not consult target flows.
        assert(a[0] == 0. && a[3] == 0.);
        auto legacy = simulateTwoReservoirBaselinePerSegment(rain, {2.,2.,2.,3.,3.,3.}, segments, params);
        assert(legacy[0] == 2. && legacy[3] == 3.);
    }
    // Constant inflow has an analytic step response, including k*dt > 1.
    for (double step : {1e-12, 0.04, 0.8, 3.0, 100.0}) {
        auto input = torch::ones({19, 1}, torch::kFloat64).set_requires_grad(true);
        auto routed = routeReservoirTensor(input, step, 0.7);
        for (int i = 0; i < 19; ++i) {
            const double expected = 0.7 * -std::expm1(-(i + 1) * step);
            assert(std::abs(routed[i].item<double>() - expected) < 1e-12);
        }
        routed.sum().backward();
        for (int i = 0; i < 19; ++i) {
            const double derivative = 0.7 * -std::expm1(-(19 - i) * step);
            assert(std::abs(input.grad()[i].item<double>() - derivative) < 1e-12);
        }
    }
    // Scalar and differentiable routing agree for arbitrary forcing, zero q0.
    std::vector<double> forcing{0., .3, 1.1, .7, .2, 0., .5};
    TwoReservoirRoutingParams p;
    p.fastK = 3.; p.slowK = .04; p.alpha = .7;
    auto expected = simulateTwoReservoirBaseline(forcing, 0., p);
    auto x = torch::tensor(forcing, torch::kFloat64).reshape({-1, 1});
    auto actual = routeReservoirTensor(x, p.fastK, p.alpha) +
                  routeReservoirTensor(x, p.slowK, 1. - p.alpha);
    for (size_t i = 0; i < forcing.size(); ++i)
        assert(std::abs(expected[i] - actual[i].item<double>()) < 1e-12);
    // Recession is exact and independent of the sampling step for constant k.
    p.dt = .5;
    auto recession = simulateTwoReservoirBaseline(std::vector<double>(9, 0.), 2., p);
    for (int i = 0; i < 9; ++i) {
        double q = 2. * (p.alpha * std::exp(-p.fastK * i * p.dt) +
                         (1. - p.alpha) * std::exp(-p.slowK * i * p.dt));
        assert(std::abs(recession[i] - q) < 1e-12);
    }
    // New archives retain exponential routing; pre-marker archives load Euler.
    HydroTwoReservoirLSTM model(2, 3, 1, 1., .8, .04, .7);
    auto sequence = torch::randn({7, 3, 2});
    model->eval();
    for (bool legacy : {false, true}) {
        torch::serialize::OutputArchive out;
        if (legacy) model->torch::nn::Module::save(out); else model->save(out);
        std::stringstream stream;
        out.save_to(stream);
        torch::serialize::InputArchive in;
        in.load_from(stream);
        HydroTwoReservoirLSTM restored(2, 3, 1, 1., .8, .04, .7);
        restored->load(in); restored->eval();
        assert(restored->exponential_routing == !legacy);
        model->exponential_routing = !legacy;
        assert(torch::allclose(restored->forward(sequence), model->forward(sequence)));
        auto r = model->runoffGeneration(sequence);
        auto q = routeReservoirTensor(r, .8, .7, !legacy) +
                 routeReservoirTensor(r, .04, .3, !legacy);
        assert(torch::allclose(q, restored->forward(sequence)));
    }
    std::cout << "PASS: analytic routing, gradients, scalar agreement, recession, and old/new checkpoint compatibility\n";
}
