#pragma once

#include "neuralnetworkwrapper.h"
#include "hyperparameters.h"
#include "TimeSeriesSet.h"

class NeuralNetworkFactory {
public:
    /**
     * @brief Create a neural network configured for GA optimization.
     * @param parameters GA chromosome parameters
     * @param input_data Input time series data
     * @param target_data Target time series data
     * @param t_start Start time for data
     * @param t_end End time for data
     * @param dt Time step
     * @param split_ratio Train/test split ratio
     * @param available_series_count Number of available time series
     * @return Configured NeuralNetworkWrapper ready for training
     */
    static NeuralNetworkWrapper createForGA(
        const std::vector<unsigned long int>& parameters,
        const TimeSeriesSet<double>& input_data,
        const TimeSeries<double>& target_data,
        double t_start, double t_end, double dt,
        double split_ratio,
        int available_series_count);

    /**
     * @brief Create a neural network from existing hyperparameters.
     * @param hyperparams Configured HyperParameters object
     * @param input_data Input time series data
     * @param target_data Target time series data
     * @param t_start Start time for data
     * @param t_end End time for data
     * @param dt Time step
     * @param split_ratio Train/test split ratio
     * @return Configured NeuralNetworkWrapper ready for training
     */
    static NeuralNetworkWrapper createFromHyperParams(
        const HyperParameters& hyperparams,
        const TimeSeriesSet<double>& input_data,
        const TimeSeries<double>& target_data,
        double t_start, double t_end, double dt,
        double split_ratio);

    /**
     * @brief Clone a neural network's architecture but with fresh weights.
     * @param base_model Network to clone architecture from
     * @return New network with same architecture but random weights
     */
    static NeuralNetworkWrapper cloneArchitecture(const NeuralNetworkWrapper& base_model);

    /**
     * @brief Create population for genetic algorithm.
     * @param population_size Number of individuals to create
     * @param input_data Input time series data
     * @param target_data Target time series data
     * @param t_start Start time for data
     * @param t_end End time for data
     * @param dt Time step
     * @param split_ratio Train/test split ratio
     * @param available_series_count Number of available time series
     * @return Vector of configured networks ready for GA optimization
     */
    static std::vector<NeuralNetworkWrapper> createPopulation(
        int population_size,
        const TimeSeriesSet<double>& input_data,
        const TimeSeries<double>& target_data,
        double t_start, double t_end, double dt,
        double split_ratio,
        int available_series_count);
};
