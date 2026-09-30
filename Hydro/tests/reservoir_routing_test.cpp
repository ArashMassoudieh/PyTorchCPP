#include "../models/hydro_lstm_module.h"
#include "../models/two_reservoir_routing.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <sstream>

int main() {
    torch::manual_seed(42);
    torch::set_num_threads(1);
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
