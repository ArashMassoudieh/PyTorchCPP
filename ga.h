#ifndef GeneticAlgorithm_H
#define GeneticAlgorithm_H

#include <vector>
#include <string>
#include <fstream>

#include "Binary.h"
#include "individual.h"

#ifdef QT_GUI_SUPPORT
#include "ProgressWindow.h"
#endif


/**
 * @brief Settings structure for configuring the genetic algorithm.
 *
 * This struct defines all tunable parameters of the genetic algorithm,
 * such as population size, number of generations, mutation probability,
 * and optimization target.
 */
struct GeneticAlgorithmsettings
{
    /**
     * @brief Total number of individuals in the population.
     *
     * Default: 40
     */
    unsigned int totalpopulation = 40;

    /**
     * @brief Number of generations for which the algorithm will run.
     *
     * Default: 100
     */
    unsigned int generations = 100;

    /**
     * @brief Probability of mutation applied to individuals.
     *
     * Must be between 0.0 and 1.0. Default: 0.05
     */
    double mutation_probability = 0.05;

    /**
     * @brief Output path where results will be written.
     *
     * Default: empty string (no output).
     */
    std::string outputpath = "";

    /**
     * @brief Fitness optimization mode.
     *
     * - true  → minimize MSE_Test only
     * - false → minimize (MSE_Test + MSE_Train)
     *
     * Default: true
     */
    bool MSE_optimization = true;
};

using namespace std;

/**
 * @brief Template class implementing a Genetic Algorithm.
 *
 * This class provides a general-purpose implementation of a
 * genetic algorithm for model optimization. The type parameter @p T
 * typically represents the model type being optimized.
 *
 * @tparam T Type of the model or object being optimized.
 */
template<class T>
class GeneticAlgorithm
{
public:
    /**
     * @brief Construct a new GeneticAlgorithm object with default settings.
     */
    GeneticAlgorithm();

    /**
     * @brief Run the optimization process.
     *
     * Executes the genetic algorithm using the configured
     * settings and returns the best model found.
     *
     * @return T Best optimized model.
     */
    T& Optimize();

    /**
     * @brief Assign fitness values to all individuals.
     *
     * This function evaluates each individual in the population
     * according to the chosen fitness criterion.
     */
    void AssignFitnesses();

    /**
     * @brief Initialize the genetic algorithm.
     *
     * Prepares the initial population of individuals and sets up
     * necessary data structures before optimization begins.
     */
    void Initialize();

    /**
     * @brief Write current population or results to file.
     *
     * Uses the configured output path to save individuals’
     * fitnesses, models, or other relevant information.
     */
    void WriteToFile();

    /**
     * @brief Vector of individuals in the current population.
     */
    vector<Individual> Individuals;

    /**
     * @brief The current best model.
     */
    T model;

    /**
     * @brief Collection of all models considered during optimization.
     */
    vector<T> models;

    /**
     * @brief Settings used to configure this genetic algorithm instance.
     */
    GeneticAlgorithmsettings Settings;

    /**
     * @brief Retrieve the rank of each individual.
     *
     * @return std::vector<int> Vector of ranks for all individuals.
     */
    std::vector<int> getRanks();

    /**
     * @brief Perform crossover between selected individuals.
     *
     * Produces offspring by recombining selected parents,
     * introducing diversity into the population.
     */
    void CrossOver();

    /**
     * @brief Select an individual based on rank.
     *
     * Individuals are chosen probabilistically using rank-based
     * selection, favoring fitter individuals.
     *
     * @return const Individual& Reference to the selected individual.
     */
    const Individual& selectIndividualByRank();

    /**
     * @brief Perform mutation.
     *
     * Produces offspring by mutation,
     * introducing diversity into the population.
     */
    void mutatePopulation();

    void setVerbose(bool verbose) {
        verbose_ = verbose;
        // Also set verbose on the model
        model.setVerbose(verbose);
    }
    bool getVerbose() const { return verbose_; }



#ifdef QT_GUI_SUPPORT
        /**
     * @brief Set progress window for visualization
     * @param progressWindow Pointer to ProgressWindow
     */
        void setProgressWindow(ProgressWindow* progressWindow) {
            progressWindow_ = progressWindow;
        }

        /**
     * @brief Get progress window pointer
     * @return Pointer to ProgressWindow or nullptr
     */
        ProgressWindow* getProgressWindow() const {
            return progressWindow_;
        }
#endif

private:
    /**
     * @brief Maximum rank assigned within the population.
     */
    unsigned int max_rank = 0;

    /**
     * @brief Output file stream for writing results.
     */
    std::ofstream file;

    /**
     * @brief Current generation counter.
     */
    unsigned int current_generation = 0;

    bool verbose_ = true;

#ifdef QT_GUI_SUPPORT
    ProgressWindow* progressWindow_ = nullptr;
#endif
};

#include "ga.hpp"

#endif // GeneticAlgorithm_H
