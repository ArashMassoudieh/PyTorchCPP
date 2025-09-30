#include "neuralnetworkfactory.h"
#include <random>

NeuralNetworkWrapper NeuralNetworkFactory::createForGA(
    const std::vector<unsigned long int>& parameters,
    const TimeSeriesSet<double>& input_data,
    const TimeSeries<double>& target_data,
    double t_start, double t_end, double dt,
    double split_ratio,
    int available_series_count) {

    NeuralNetworkWrapper model;

    // Configure GA interface data
    model.setTimeSeriesData(input_data, target_data);
    model.setTimeRange(t_start, t_end, dt, split_ratio);
    model.setAvailableSeriesCount(available_series_count);

    // Set parameters and create model
    model.AssignParameters(parameters);
    model.CreateModel();

    return model; // Uses move semantics
}

NeuralNetworkWrapper NeuralNetworkFactory::createFromHyperParams(
    const HyperParameters& hyperparams,
    const TimeSeriesSet<double>& input_data,
    const TimeSeries<double>& target_data,
    double t_start, double t_end, double dt,
    double split_ratio) {

    NeuralNetworkWrapper model;

    // Store a copy of hyperparams (they're copyable by default)
    HyperParameters params_copy = hyperparams;

    // Initialize network
    model.initializeNetwork(&params_copy, 1); // Assuming single output

    // Calculate split time
    double split_time = t_start + split_ratio * (t_end - t_start);

    // Set training data
    model.setInputDataFromHyperParams(DataType::Train, input_data, t_start, split_time, dt);
    model.setTargetData(DataType::Train, target_data, t_start, split_time, dt);

    // Set test data
    model.setInputDataFromHyperParams(DataType::Test, input_data, split_time, t_end, dt);
    model.setTargetData(DataType::Test, target_data, split_time, t_end, dt);

    return model;
}

NeuralNetworkWrapper NeuralNetworkFactory::cloneArchitecture(const NeuralNetworkWrapper& base_model) {
    if (!base_model.isInitialized()) {
        throw std::runtime_error("Cannot clone uninitialized network");
    }

    NeuralNetworkWrapper clone;

    // Copy configuration
    clone.setLags(base_model.getLags());
    clone.setHiddenLayers(base_model.getHiddenLayers());

    // Initialize with same architecture but fresh weights
    clone.initializeNetwork(1, "relu"); // Will be overridden by hyperparams if available

    return clone;
}

std::vector<NeuralNetworkWrapper> NeuralNetworkFactory::createPopulation(
    int population_size,
    const TimeSeriesSet<double>& input_data,
    const TimeSeries<double>& target_data,
    double t_start, double t_end, double dt,
    double split_ratio,
    int available_series_count) {

    if (population_size <= 0) {
        throw std::runtime_error("Population size must be positive");
    }

    std::vector<NeuralNetworkWrapper> population;
    population.reserve(population_size);

    // Get parameter bounds for random generation
    auto bounds = HyperParameters::getOptimizationBounds(available_series_count);

    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());

    for (int i = 0; i < population_size; i++) {
        // Generate random parameters
        std::vector<unsigned long int> random_params;
        random_params.reserve(bounds.size());

        for (size_t j = 0; j < bounds.size(); j++) {
            std::uniform_int_distribution<unsigned long int> dist(1, bounds[j]);
            random_params.push_back(dist(gen));
        }

        // Create network with random parameters
        NeuralNetworkWrapper individual = createForGA(
            random_params, input_data, target_data,
            t_start, t_end, dt, split_ratio, available_series_count);

        population.push_back(std::move(individual));
    }

    return population;
}
