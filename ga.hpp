#include "ga.h"
#include <iostream>
#include <fstream>
#include <omp.h>
#include "Utilities.h"
#include "neuralnetworkwrapper.h"
#ifdef QT_GUI_SUPPORT
#include <QThread>
#include <QApplication>
#include <QString>
#endif
#include <random>

template<class T>
GeneticAlgorithm<T>::GeneticAlgorithm()
{
    omp_set_num_threads(8);
}


template<class T>
T& GeneticAlgorithm<T>::Optimize()
{
#ifdef QT_GUI_SUPPORT
    if (progressWindow_) {
        progressWindow_->SetStatus("Starting GA Optimization...");
        progressWindow_->SetProgress(0.0);
        progressWindow_->ClearPrimaryChartData();
        progressWindow_->ClearSecondaryChartData();
        progressWindow_->SetPrimaryChartTitle("Test R² vs Generation");
        progressWindow_->SetPrimaryChartXAxisTitle("Generation");
        progressWindow_->SetPrimaryChartYAxisTitle("R²");
        progressWindow_->SetSecondaryChartTitle("Test MSE vs Generation");
        progressWindow_->SetSecondaryChartXAxisTitle("Generation");
        progressWindow_->SetSecondaryChartYAxisTitle("MSE");

        // Set X-axis range to [0, total_generations] so user sees full range
        progressWindow_->SetPrimaryChartXRange(0, Settings.generations);
        progressWindow_->SetSecondaryChartXRange(0, Settings.generations);

        // Enable auto-scaling for Y-axis (but not X-axis)
        progressWindow_->SetPrimaryChartAutoScale(true);   // Only Y-axis
        progressWindow_->SetSecondaryChartAutoScale(true); // Only Y-axis

        progressWindow_->AppendLog("GA Optimization Started");
        QApplication::processEvents();
    }
#endif

    for (unsigned int generation = 0; generation < Settings.generations; generation++)
    {
        current_generation = generation;

#ifdef QT_GUI_SUPPORT
        // Process events to keep UI responsive
        if (progressWindow_) {
            QApplication::processEvents();
        }

        // Check for cancel request
        if (progressWindow_ && progressWindow_->IsCancelRequested()) {
            progressWindow_->AppendLog("Optimization cancelled by user");
            progressWindow_->SetStatus("Cancelled");
            QApplication::processEvents();
            break;
        }

        // Update progress bar
        if (progressWindow_) {
            double progress = static_cast<double>(generation) / Settings.generations;
            progressWindow_->SetProgress(progress);
            progressWindow_->SetStatus(QString("Generation %1 / %2")
                                           .arg(generation + 1)
                                           .arg(Settings.generations));
            QApplication::processEvents();
        }
#endif

        // Assign fitness to all individuals
        AssignFitnesses();

#ifdef QT_GUI_SUPPORT
        // Process events after fitness calculation
        if (progressWindow_) {
            QApplication::processEvents();
        }
#endif

        // Rank individuals
        vector<int> ranks = getRanks();
        Individual& bestIndividual = Individuals[ranks[0]];

#ifdef QT_GUI_SUPPORT
        // Update charts with best individual's metrics
        if (progressWindow_ && bestIndividual.fitness_measures.size() > 0) {
            // Extract R² and MSE from fitness measures using CORRECT key names
            double r2_test = 0.0;
            double mse_test = 0.0;
            double r2_train = 0.0;
            double mse_train = 0.0;

            // Use the actual key names from fitness_measures map
            if (bestIndividual.fitness_measures.count("R2_Test_0")) {
                r2_test = bestIndividual.fitness_measures["R2_Test_0"];
            }
            if (bestIndividual.fitness_measures.count("MSE_Test_0")) {
                mse_test = bestIndividual.fitness_measures["MSE_Test_0"];
            }
            if (bestIndividual.fitness_measures.count("R2_Train_0")) {
                r2_train = bestIndividual.fitness_measures["R2_Train_0"];
            }
            if (bestIndividual.fitness_measures.count("MSE_Train_0")) {
                mse_train = bestIndividual.fitness_measures["MSE_Train_0"];
            }

            // Add points to charts
            progressWindow_->AddPrimaryChartPoint(generation, r2_test);
            progressWindow_->AddSecondaryChartPoint(generation, mse_test);

            // Log every 10 generations with complete information
            if (generation % 10 == 0) {
                progressWindow_->AppendLog(
                    QString("Gen %1: R²_test=%2, MSE_test=%3, R²_train=%4, MSE_train=%5, Fitness=%6")
                        .arg(generation)
                        .arg(r2_test, 0, 'f', 4)
                        .arg(mse_test, 0, 'f', 6)
                        .arg(r2_train, 0, 'f', 4)
                        .arg(mse_train, 0, 'f', 6)
                        .arg(bestIndividual.fitness, 0, 'f', 6));
            }

            QApplication::processEvents();
        }

        // Handle pause request
        if (progressWindow_ && progressWindow_->IsPauseRequested()) {
            progressWindow_->SetStatus("Paused");
            progressWindow_->AppendLog("Optimization paused");
            QApplication::processEvents();

            // Wait until resumed or cancelled
            while (progressWindow_->IsPauseRequested() &&
                   !progressWindow_->IsCancelRequested()) {
                QThread::msleep(100);
                QApplication::processEvents();
            }

            if (progressWindow_->IsCancelRequested()) {
                progressWindow_->AppendLog("Optimization cancelled during pause");
                QApplication::processEvents();
                break;
            }

            progressWindow_->ResetPauseRequest();
            progressWindow_->SetStatus(QString("Resumed - Generation %1 / %2")
                                           .arg(generation + 1)
                                           .arg(Settings.generations));
            progressWindow_->AppendLog("Optimization resumed");
            QApplication::processEvents();
        }
#endif

#ifdef QT_GUI_SUPPORT
        if (progressWindow_ && generation % 10 == 0) {
            vector<int> ranks = getRanks();
            Individual& best = Individuals[ranks[0]];

            progressWindow_->AppendLog(QString("\n=== Generation %1 Best Individual ===").arg(generation));
            progressWindow_->AppendLog(QString("  FITNESS = %1 (lower is better)").arg(best.fitness, 0, 'f', 6));
            progressWindow_->AppendLog(QString("  MSE_Train = %1").arg(best.fitness_measures["MSE_Train_0"], 0, 'f', 6));
            progressWindow_->AppendLog(QString("  MSE_Test  = %1 %2")
                                           .arg(best.fitness_measures["MSE_Test_0"], 0, 'f', 6)
                                           .arg(Settings.MSE_optimization ? "<-- FITNESS TARGET" : ""));
            progressWindow_->AppendLog(QString("  R2_Train  = %1").arg(best.fitness_measures["R2_Train_0"], 0, 'f', 4));
            progressWindow_->AppendLog(QString("  R2_Test   = %1").arg(best.fitness_measures["R2_Test_0"], 0, 'f', 4));

            // Show the individual with best MSE_Train for comparison
            double best_train_mse = std::numeric_limits<double>::max();
            int best_train_idx = 0;
            for (int i = 0; i < Individuals.size(); i++) {
                if (Individuals[i].fitness_measures["MSE_Train_0"] < best_train_mse) {
                    best_train_mse = Individuals[i].fitness_measures["MSE_Train_0"];
                    best_train_idx = i;
                }
            }

            if (best_train_idx != ranks[0]) {
                progressWindow_->AppendLog(QString("\n  Note: Individual with best MSE_Train (%1) has:").arg(best_train_mse, 0, 'f', 6));
                progressWindow_->AppendLog(QString("    MSE_Test = %1 (worse than best)").arg(Individuals[best_train_idx].fitness_measures["MSE_Test_0"], 0, 'f', 6));
                progressWindow_->AppendLog(QString("    Fitness  = %1 (rejected as overfitted)").arg(Individuals[best_train_idx].fitness, 0, 'f', 6));
            }
        }
#endif

        max_rank = ranks[0];

        if (verbose_) {
            std::cout << "Best fitness in generation " << current_generation
                      << ": " << Individuals[max_rank].fitness << std::endl;
        }

        if (current_generation < Settings.generations - 1) {
            // Apply GA operators for the next generation
            WriteToFile();
            CrossOver();
            mutatePopulation();
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

#ifdef QT_GUI_SUPPORT
    if (progressWindow_) {
        progressWindow_->SetProgress(1.0);
        progressWindow_->SetComplete("GA Optimization Complete!");

        auto ranks = getRanks();
        Individual& bestIndividual = Individuals[ranks[0]];

        progressWindow_->AppendLog(QString("\n=== Final Results ==="));
        progressWindow_->AppendLog(QString("Best Fitness: %1").arg(bestIndividual.fitness));

        // Use correct key names for final results
        if (bestIndividual.fitness_measures.count("R2_Test_0")) {
            progressWindow_->AppendLog(QString("Best Test R²: %1")
                                           .arg(bestIndividual.fitness_measures["R2_Test_0"], 0, 'f', 4));
        }
        if (bestIndividual.fitness_measures.count("MSE_Test_0")) {
            progressWindow_->AppendLog(QString("Best Test MSE: %1")
                                           .arg(bestIndividual.fitness_measures["MSE_Test_0"], 0, 'f', 6));
        }
        if (bestIndividual.fitness_measures.count("R2_Train_0")) {
            progressWindow_->AppendLog(QString("Best Train R²: %1")
                                           .arg(bestIndividual.fitness_measures["R2_Train_0"], 0, 'f', 4));
        }
        if (bestIndividual.fitness_measures.count("MSE_Train_0")) {
            progressWindow_->AppendLog(QString("Best Train MSE: %1")
                                           .arg(bestIndividual.fitness_measures["MSE_Train_0"], 0, 'f', 6));
        }

        QApplication::processEvents();
    }
#endif

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

                Individuals[i].fitness = 0;

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
    vector<Individual> newIndividuals(Individuals.size());
    vector<int> sorted_indices = getRanks();
    int best_idx = sorted_indices[0];

    // Elite preservation
    newIndividuals[0] = Individuals[best_idx];

    // Check if all individuals have same binary length
    if (verbose_) {
        std::cout << "\n=== Checking Individual Binary Lengths ===" << std::endl;
        for (size_t i = 0; i < std::min(size_t(5), Individuals.size()); i++) {
            BinaryNumber bin = Individuals[i].toBinary();
            std::cout << "Individual " << i << " binary length: " << bin.numDigits() << std::endl;
        }
    }

    for (unsigned int i = 1; i < Individuals.size(); i++)
    {
        const Individual& parent1 = selectIndividualByRank();
        const Individual& parent2 = selectIndividualByRank();

        BinaryNumber binary1 = parent1.toBinary();
        BinaryNumber binary2 = parent2.toBinary();

        if (verbose_ && i == 1) {
            std::cout << "Parent1 length: " << binary1.numDigits() << std::endl;
            std::cout << "Parent2 length: " << binary2.numDigits() << std::endl;
        }

        BinaryNumber offspring_binary = BinaryNumber::crossover(binary1, binary2);

        if (verbose_ && i == 1) {
            std::cout << "Offspring length: " << offspring_binary.numDigits() << std::endl;
            std::cout << "Expected from splitlocations: ";
            int expected = 0;
            for (auto split : parent1.splitlocations) {
                expected += split;
            }
            std::cout << expected << std::endl;
        }

        offspring_binary.mutate(Settings.mutation_probability);

        // Use parent1's splitlocations
        newIndividuals[i] = offspring_binary.split(parent1.splitlocations);
        newIndividuals[i].splitlocations = parent1.splitlocations;
        models[i].clear();
    }

    models[0].clear();
    Individuals = newIndividuals;
}

inline void SortIndices(const std::vector<Individual>& individuals, std::vector<int>& indices) {
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

    // Sort indices based on fitness (best first)
    SortIndices(Individuals, indices);

    // Set max_rank to the index of the best individual
    max_rank = indices[0];

    if (verbose_) {
        std::cout << "\n=== Ranking Debug ===" << std::endl;
        std::cout << "After sorting (best to worst):" << std::endl;
        for (size_t i = 0; i < std::min(size_t(5), indices.size()); i++) {
            int idx = indices[i];
            if (Individuals[idx].fitness_measures.count("MSE_Test_0") == 0) {
                std::cout << "XXX: Missing MSE_Test_0 for Individual " << idx << std::endl;
                std::cout << "  fitness = " << Individuals[idx].fitness << std::endl;
                std::cout << "  fitness_measures size = "
                          << Individuals[idx].fitness_measures.size() << std::endl;
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
        std::cout << "Best individual index: " << indices[0] << std::endl;
    }

    // Return the sorted indices directly (0-indexed)
    // indices[0] = index of best individual
    // indices[1] = index of 2nd best individual, etc.
    return indices;
}


template<class T>
void GeneticAlgorithm<T>::mutatePopulation() {
    for (unsigned int i = 0; i < Individuals.size(); ++i) {
        if (i == max_rank) continue;  // skip elite
        Individuals[i].mutate(Settings.mutation_probability);
    }
}


template<class T>
const Individual& GeneticAlgorithm<T>::selectIndividualByRank()
{
    static std::random_device rd;
    static std::mt19937 gen(rd());

    const int tournament_size = 3;
    std::uniform_int_distribution<int> dist(0, Individuals.size() - 1);

    int best_idx = dist(gen);
    double best_fitness = Individuals[best_idx].fitness;

    for (int i = 1; i < tournament_size; i++) {
        int candidate_idx = dist(gen);

        if (Individuals[candidate_idx].fitness < best_fitness) {
            best_idx = candidate_idx;
            best_fitness = Individuals[candidate_idx].fitness;
        }
    }

    return Individuals[best_idx];
}
