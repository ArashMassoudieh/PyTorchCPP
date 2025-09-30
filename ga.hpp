#include "ga.h"
#include <iostream>
#include <fstream>
#include <omp.h>
#include "Utilities.h"
#include "neuralnetworkwrapper.h"

template<class T>
GeneticAlgorithm<T>::GeneticAlgorithm()
{
    omp_set_num_threads(8);
}


template<class T>
T& GeneticAlgorithm<T>::Optimize()
{
    for (current_generation = 0; current_generation < Settings.generations; current_generation++) {
        if (verbose_) {
            std::cout << "\n=== Generation " << current_generation << " ===" << std::endl;
        }

        // Evaluate fitness
        AssignFitnesses();

        // Rank individuals
        vector<int> ranks = getRanks();
        max_rank = ranks[0];

        if (verbose_) {
            std::cout << "Best fitness in generation " << current_generation
                      << ": " << Individuals[max_rank].fitness << std::endl;
        }

        if (current_generation < Settings.generations - 1) {
            // Apply GA operators for the next generation
            AssignFitnesses();
            WriteToFile();
            CrossOver();



        }
    }

    if (verbose_)
    {
        std::cout << "Returning individual " << max_rank
                  << " with fitness " << Individuals[max_rank].fitness << std::endl;
    }

    // ENHANCED DEBUG: Check model state before fixing
    if (verbose_) {
        std::cout << "DEBUG: Best model state before return:" << std::endl;
        std::cout << "  Is initialized: " << models[max_rank].isInitialized() << std::endl;
        std::cout << "  Has hyperparams: " << models[max_rank].getHyperParameters().isValid() << std::endl;
        std::cout << "  Parameter count: " << (models[max_rank].isInitialized() ? models[max_rank].getTotalParameters() : 0) << std::endl;
    }

    // FORCE REINITIALIZATION - regardless of current state
    if (verbose_) {
        std::cout << "DEBUG: Force reinitializing best model..." << std::endl;
    }

    try {
        // Get the parameters for the best individual
        vector<unsigned long int> best_params;
        for (unsigned int j = 0; j < models[max_rank].ParametersSize(); j++) {
            best_params.push_back(Individuals[max_rank][j].toDecimal());
        }

        if (verbose_) {
            std::cout << "DEBUG: Best params: [";
            for (size_t k = 0; k < best_params.size() && k < 5; k++) { // Show first 5
                std::cout << best_params[k];
                if (k < std::min(best_params.size() - 1, size_t(4))) std::cout << ", ";
            }
            if (best_params.size() > 5) std::cout << "...";
            std::cout << "]" << std::endl;
        }

        // FORCE clear and reinitialize
        if (verbose_) std::cout << "DEBUG: Calling clear()..." << std::endl;
        models[max_rank].clear();

        if (verbose_) std::cout << "DEBUG: Calling AssignParameters..." << std::endl;
        models[max_rank].AssignParameters(best_params);

        if (verbose_) std::cout << "DEBUG: Calling CreateModel..." << std::endl;
        models[max_rank].CreateModel();

        if (verbose_) std::cout << "DEBUG: After CreateModel, initialized: " << models[max_rank].isInitialized() << std::endl;

        if (!models[max_rank].isInitialized()) {
            throw std::runtime_error("CreateModel() completed but model still not initialized");
        }

        if (verbose_) std::cout << "DEBUG: Model reinitialized successfully" << std::endl;

    } catch (const std::exception& e) {
        std::cout << "ERROR: Failed to reinitialize best model: " << e.what() << std::endl;

        // If reinitialization fails, create a completely new model
        std::cout << "DEBUG: Attempting to create fresh model..." << std::endl;
        T fresh_model = model; // Copy from base model

        try {
            vector<unsigned long int> best_params;
            for (unsigned int j = 0; j < fresh_model.ParametersSize(); j++) {
                best_params.push_back(Individuals[max_rank][j].toDecimal());
            }

            fresh_model.AssignParameters(best_params);
            fresh_model.CreateModel();

            if (fresh_model.isInitialized()) {
                models[max_rank] = fresh_model;
                std::cout << "DEBUG: Fresh model created successfully" << std::endl;
            } else {
                throw std::runtime_error("Fresh model creation also failed");
            }

        } catch (const std::exception& e2) {
            throw std::runtime_error("Cannot return initialized model. Original error: " + std::string(e.what()) +
                                     ". Fresh model error: " + std::string(e2.what()));
        }
    }

    // FINAL CHECK
    if (verbose_) {
        std::cout << "DEBUG: FINAL model state:" << std::endl;
        std::cout << "  Is initialized: " << models[max_rank].isInitialized() << std::endl;
        std::cout << "  Parameter count: " << (models[max_rank].isInitialized() ? models[max_rank].getTotalParameters() : 0) << std::endl;
        std::cout << "  Has train data: " << models[max_rank].hasInputData(DataType::Train) << std::endl;
        std::cout << "  Has test data: " << models[max_rank].hasInputData(DataType::Test) << std::endl;
    }

    if (!models[max_rank].isInitialized()) {
        throw std::runtime_error("CRITICAL: Cannot return uninitialized model after all attempts");
    }

    return models[max_rank];
}


template<class T>
void GeneticAlgorithm<T>::WriteToFile()
{

    file.open(Settings.outputpath+"/GA_Output.txt", std::ios::app);
    file<<"Generation: "<< current_generation << endl;
    if (file.is_open())
    {
        for (unsigned int i=0; i<Individuals.size(); i++)
        {   file<<i<<":"<<Individuals[i].toBinary().getBinary()<<","<<models[i].ParametersToString();
            for (int constituent = 0; constituent<models[i].getOutputSize(); constituent++)
            {
                file<< ","<<Individuals[i].toAssignmentText("MSE_Train",constituent)<<","<<Individuals[i].toAssignmentText("R2_Train",constituent);
                file<< ","<<Individuals[i].toAssignmentText("MSE_Test",constituent)<<","<<Individuals[i].toAssignmentText("R2_Test",constituent);
            }
            file<<endl;
        }
    }
    file.close();
    for (unsigned int i=0; i<Individuals.size(); i++)
    {
        for (int constituent = 0; constituent<models[i].getOutputSize(); constituent++)
            cout<< ","<<Individuals[i].toAssignmentText("MSE_Train",constituent)<<","<<Individuals[i].toAssignmentText("R2_Train",constituent);
        for (int constituent = 0; constituent<models[i].getOutputSize(); constituent++)
            cout<< ","<<Individuals[i].toAssignmentText("MSE_Test",constituent)<<","<<Individuals[i].toAssignmentText("R2_Test",constituent);
        cout<<std::endl;
    }
}

template<class T>
void GeneticAlgorithm<T>::Initialize()
{
    file.open(Settings.outputpath+"/GA_Output.txt", std::ios::out);
    file.close();
    Individuals.resize(Settings.totalpopulation);
    models.resize(Settings.totalpopulation);
    for (int i=0; i<Individuals.size(); i++)
    {
        models[i] = model;
        Individuals[i].resize(model.ParametersSize());
        vector<int> splitlocations;
        for (int j=0; j<model.ParametersSize(); j++)
        {
            BinaryNumber B = BinaryNumber::randomBinary(model.MaxParameter(j));
            B.fixSize(BinaryNumber::decimalToBinary(model.MaxParameter(j)).numDigits());
            Individuals[i][j] = B;
            Individuals[i].splitlocations.push_back(BinaryNumber::decimalToBinary(model.MaxParameter(j)).numDigits());
        }
        Individuals[i].display();

    }
    AssignFitnesses();
    WriteToFile();
}

template<class T>
void GeneticAlgorithm<T>::AssignFitnesses()
{
    if (verbose_)
    {   cout << "\n=== Starting AssignFitnesses Debug ===" << endl;
        cout << "Population size: " << models.size() << endl;
        cout << "Parameters per model: " << (models.empty() ? 0 : models[0].ParametersSize()) << endl;
    }

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < models.size(); i++)
    {
        if (verbose_)
            cout << "\n--- Processing Individual " << i << " ---" << endl;

        vector<unsigned long int> parameterset;

        // Debug parameter generation
        if (verbose_)
            cout << "Generating parameters:" << endl;
        for (unsigned int j = 0; j < models[i].ParametersSize(); j++)
        {
            unsigned long int param_value = Individuals[i][j].toDecimal();
            long int max_param = models[i].MaxParameter(j);

            parameterset.push_back(param_value);

            if (verbose_)
                cout << "  Param[" << j << "]: " << param_value
                     << " (max: " << max_param
                     << ", binary: " << Individuals[i][j].getBinary() << ")" << endl;
        }

        if (verbose_)
        {   cout << "Complete parameter set for Individual " << i << ": [";
            for (size_t k = 0; k < parameterset.size(); k++) {
                cout << parameterset[k];
                if (k < parameterset.size() - 1) cout << ", ";
            }
            cout << "]" << endl;
        }

        // Clear any existing fitness measures first
        Individuals[i].fitness_measures.clear();
        Individuals[i].fitness = 0;

        // Assign parameters
        if (verbose_)
            cout << "Assigning parameters..." << endl;
        models[i].AssignParameters(parameterset);

        if (verbose_)
            cout << "Creating model..." << endl;
        models[i].CreateModel();

        cout << "Pre-Train: " << i << ":" << models[i].ParametersToString() << endl;

        if (models[i].ValidLags())
        {
            if (verbose_)
                cout << "Valid lags detected - starting fitness evaluation..." << endl;

            try {
                Individuals[i].fitness_measures = models[i].Fitness();

                if (Individuals[i].fitness_measures.empty()) {
                    std::cout << "[DEBUG] GA: Fitness map is empty for individual " << i << std::endl;
                }

                // Ensure ALL required keys are present - add missing ones with penalty values
                for (int constituent = 0; constituent < models[i].getOutputSize(); constituent++) {
                    std::string mse_test_key = "MSE_Test_" + aquiutils::numbertostring(constituent);
                    std::string r2_test_key = "R2_Test_" + aquiutils::numbertostring(constituent);
                    std::string mse_train_key = "MSE_Train_" + aquiutils::numbertostring(constituent);
                    std::string r2_train_key = "R2_Train_" + aquiutils::numbertostring(constituent);

                    // Add missing keys with penalty values
                    if (Individuals[i].fitness_measures.find(mse_test_key) == Individuals[i].fitness_measures.end()) {
                        Individuals[i].fitness_measures[mse_test_key] = 1e12;
                    }
                    if (Individuals[i].fitness_measures.find(r2_test_key) == Individuals[i].fitness_measures.end()) {
                        Individuals[i].fitness_measures[r2_test_key] = -1e12;
                    }
                    if (Individuals[i].fitness_measures.find(mse_train_key) == Individuals[i].fitness_measures.end()) {
                        Individuals[i].fitness_measures[mse_train_key] = 1e12;
                    }
                    if (Individuals[i].fitness_measures.find(r2_train_key) == Individuals[i].fitness_measures.end()) {
                        Individuals[i].fitness_measures[r2_train_key] = -1e12;
                    }
                }

                // Calculate total fitness from measures
                for (int constituent = 0; constituent < models[i].getOutputSize(); constituent++) {
                    if (Settings.MSE_optimization) // true for MSE_Test and false for (MSE_Test + MSE_Train)
                        Individuals[i].fitness += Individuals[i].fitness_measures["MSE_Test_" + aquiutils::numbertostring(constituent)]; // MSE_Test
                    else
                        Individuals[i].fitness += max(Individuals[i].fitness_measures["MSE_Test_" + aquiutils::numbertostring(constituent)],
                                                      Individuals[i].fitness_measures["MSE_Train_" + aquiutils::numbertostring(constituent)]); // MSE_Test and MSE_Train
                }

                if (verbose_)
                    cout << "Fitness evaluation completed. Total fitness: " << Individuals[i].fitness << endl;

            } catch (const std::exception& e) {
                if (verbose_)
                    cout << "Fitness evaluation failed: " << e.what() << " - assigning penalty fitness..." << endl;

                // Assign penalty fitness measures
                for (int constituent = 0; constituent < models[i].getOutputSize(); constituent++)
                {
                    Individuals[i].fitness_measures["MSE_Test_" + aquiutils::numbertostring(constituent)] = 1e12;
                    Individuals[i].fitness_measures["R2_Test_" + aquiutils::numbertostring(constituent)] = -1e12;
                    Individuals[i].fitness_measures["MSE_Train_" + aquiutils::numbertostring(constituent)] = 1e12;
                    Individuals[i].fitness_measures["R2_Train_" + aquiutils::numbertostring(constituent)] = -1e12;
                    Individuals[i].fitness += Individuals[i].fitness_measures["MSE_Test_" + aquiutils::numbertostring(constituent)];
                }
            }
        }
        else
        {
            if (verbose_)
                cout << "Invalid lags - assigning penalty fitness..." << endl;
            for (int constituent = 0; constituent < models[i].getOutputSize(); constituent++)
            {
                Individuals[i].fitness_measures["MSE_Test_" + aquiutils::numbertostring(constituent)] = 1e12;
                Individuals[i].fitness_measures["R2_Test_" + aquiutils::numbertostring(constituent)] = -1e12;
                Individuals[i].fitness_measures["MSE_Train_" + aquiutils::numbertostring(constituent)] = 1e12;
                Individuals[i].fitness_measures["R2_Train_" + aquiutils::numbertostring(constituent)] = -1e12;
                Individuals[i].fitness += Individuals[i].fitness_measures["MSE_Test_" + aquiutils::numbertostring(constituent)];
            }
            if (verbose_)
            {   cout << "Penalty fitness assigned: " << Individuals[i].fitness << endl;
                cout << "Individual " << i << " object address: " << &Individuals[i] << endl;
                cout << "Individual " << i << " fitness_measures address: " << &Individuals[i].fitness_measures << endl;
            }
        }

        cout << i << ":" << models[i].ParametersToString();
        for (int constituent = 0; constituent < models[i].getOutputSize(); constituent++)
            cout << "," << Individuals[i].toAssignmentText("MSE_Train", constituent) << "," << Individuals[i].toAssignmentText("R2_Train", constituent);
        for (int constituent = 0; constituent < models[i].getOutputSize(); constituent++)
            cout << "," << Individuals[i].toAssignmentText("MSE_Test", constituent) << "," << Individuals[i].toAssignmentText("R2_Test", constituent);
        cout << endl;
    }

    cout << "\nCalculating ranks..." << endl;
    vector<int> ranks = getRanks();

    if (verbose_)
        std::cout << "Best individual: " << max_rank
                  << ", Fitness: " << Individuals[max_rank].fitness << std::endl;

    for (unsigned int i = 0; i < Individuals.size(); i++)
    {
        Individuals[i].rank = ranks[i];
        cout << "Individual " << i << " rank: " << ranks[i] << endl;
    }

    if (verbose_)
        cout << "=== AssignFitnesses Debug Complete ===" << endl;
}

template<class T>
void GeneticAlgorithm<T>::CrossOver()
{
    vector<Individual> newIndividuals = Individuals;

    // Elite preservation - just keep the best individual
    newIndividuals[0] = Individuals[max_rank];
    // models[0] stays as is with its trained weights

    for (unsigned int i = 1; i < Individuals.size(); i++)
    {
        // Generate new individual through crossover/mutation
        Individual Ind1 = selectIndividualByRank();
        Individual Ind2 = selectIndividualByRank();
        BinaryNumber FullBinary = Ind1.toBinary();
        FullBinary.mutate(Settings.mutation_probability);
        newIndividuals[i] = FullBinary.split(Individuals[i].splitlocations);

        // Clear existing model instead of creating new one
        // This avoids any copy constructor calls
        models[i].clear();
    }

    Individuals = newIndividuals;
    // models vector stays the same size, just individual models are cleared/reset
}

// Function to randomly select an Individual based on inverse rank probability
template<class T>
const Individual& GeneticAlgorithm<T>::selectIndividualByRank() {
    // Calculate weights as the inverse of rank
    std::vector<double> weights(Individuals.size());
    for (size_t i = 0; i < Individuals.size(); ++i) {
        weights[i] = 1.0 / std::pow(Individuals[i].rank, 2.0);  // Quadratic decay
    }

    // Create a cumulative probability distribution
    std::vector<double> cumulative(weights.size());
    std::partial_sum(weights.begin(), weights.end(), cumulative.begin());

    // Normalize the cumulative probabilities
    double totalWeight = cumulative.back();
    for (double& value : cumulative) {
        value /= totalWeight;
    }

    // Generate a random number between 0 and 1
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    double randomValue = dis(gen);

    // Find the corresponding individual
    for (size_t i = 0; i < cumulative.size(); ++i) {
        if (randomValue <= cumulative[i]) {
            return Individuals[i];
        }
    }

    // Fallback (shouldn't be reached)
    return Individuals.back();
}

void SortIndices(const std::vector<Individual>& individuals, std::vector<int>& indices) {
    size_t n = indices.size();

    // Perform Bubble Sort on indices based on fitness values
    for (size_t i = 0; i < n - 1; ++i) {
        for (size_t j = 0; j < n - i - 1; ++j) {
            // Compare fitness values of the indices
            if (individuals[indices[j]].fitness > individuals[indices[j + 1]].fitness) {
                // Swap indices
                std::swap(indices[j], indices[j + 1]);
            }
        }
    }
}

template<class T>
std::vector<int> GeneticAlgorithm<T>::getRanks() {
    size_t n = Individuals.size();

    // Create a vector of indices from 0 to n-1
    std::vector<int> indices(n);
    for (size_t i = 0; i < n; ++i) {
        indices[i] = i;
    }

    // Sort indices based on fitness using our custom bubble sort
    SortIndices(Individuals, indices);

    // Create a vector to store ranks
    std::vector<int> ranks(n);
    for (size_t i = 0; i < n; ++i) {
        ranks[indices[i]] = i + 1; // Rank starts from 1
    }

    if (verbose_) {
        std::cout << "\n=== Ranking Debug ===" << std::endl;
        std::cout << "After sorting (best to worst):" << std::endl;
        for (size_t i = 0; i < std::min(size_t(5), indices.size()); i++) {
            int idx = indices[i];
            if (Individuals[idx].fitness_measures.count("MSE_Test_0") == 0)
            {
                std::cout << "XXX: Missing MSE_Test_0 for Individual " << idx << std::endl;

                // Optional extra debug info
                std::cout << "  fitness = " << Individuals[idx].fitness << std::endl;
                std::cout << "  fitness_measures size = "
                          << Individuals[idx].fitness_measures.size() << std::endl;

                // Print available keys, if any
                for (const auto& kv : Individuals[idx].fitness_measures) {
                    std::cout << "  has key: " << kv.first
                              << " = " << kv.second << std::endl;
                }
            }
            std::cout << "Rank " << (i+1) << ": Individual " << idx
                      << ", Fitness: " << Individuals[idx].fitness
                      << ", MSE_Test: " << Individuals[idx].fitness_measures.at("MSE_Test_0")
                      << std::endl;
        }
        std::cout << "Best individual (max_rank): " << indices[0] << std::endl;
    }

    max_rank = indices[0];
    return ranks;
}

template<class T>
void GeneticAlgorithm<T>::mutatePopulation() {
    for (unsigned int i = 0; i < Individuals.size(); ++i) {
        if (i == max_rank) continue;  // skip elite
        Individuals[i].mutate(Settings.mutation_probability);
    }
}
