#include "ga.h"
#include <iostream>
#include <fstream>
#include <omp.h>
#include "Utilities.h"

template<class T>
GeneticAlgorithm<T>::GeneticAlgorithm()
{
    omp_set_num_threads(8);
}


template<class T>
T GeneticAlgorithm<T>::Optimize()
{
    Initialize();
    WriteToFile();
    file.open(Settings.outputpath+"/GA_Output.txt", std::ios::out);
    file.close();
    for (current_generation=0; current_generation<Settings.generations; current_generation++)
    {
        cout<<"Generation: "<<current_generation<<endl;
        CrossOver();
        AssignFitnesses();
        WriteToFile();
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
        {   file<<i<<":"<<Individuals[i].toBinary().getBinary()<<","<<models[i].FFN.ModelStructure.ParametersToString().toStdString();
            for (int constituent = 0; constituent<models[i].FFN.ModelStructure.outputcolumns.size(); constituent++)
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
        for (int constituent = 0; constituent<models[i].FFN.ModelStructure.outputcolumns.size(); constituent++)
            cout<< ","<<Individuals[i].toAssignmentText("MSE_Train",constituent)<<","<<Individuals[i].toAssignmentText("R2_Train",constituent);
        for (int constituent = 0; constituent<models[i].FFN.ModelStructure.outputcolumns.size(); constituent++)
            cout<< ","<<Individuals[i].toAssignmentText("MSE_Test",constituent)<<","<<Individuals[i].toAssignmentText("R2_Test",constituent);
        cout<<std::endl;
    }
}

template<class T>
void GeneticAlgorithm<T>::Initialize()
{
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
    //#pragma omp parallel for
    for (unsigned int i=0; i<models.size(); i++)
    {
        vector<unsigned long int> parameterset;
        for (unsigned int j=0; j<models[i].ParametersSize(); j++)
        {
            parameterset.push_back(Individuals[i][j].toDecimal());

        }
        models[i].AssignParameters(parameterset);
        models[i].CreateModel();
        cout<<"Pre-Train: "<<i<<":"<<models[i].FFN.ModelStructure.ParametersToString().toStdString()<<endl; // Debugger
        if (models[i].FFN.ModelStructure.ValidLags())
        {
            Individuals[i].fitness_measures = models[i].Fitness();
            Individuals[i].fitness=0;
            for (int constituent = 0; constituent<models[i].FFN.ModelStructure.outputcolumns.size(); constituent++)
                if (Settings.MSE_optimization) // true for MSE_Test and false for (MSE_Test + MSE_Train)
                Individuals[i].fitness += Individuals[i].fitness_measures["MSE_Test_" + aquiutils::numbertostring(constituent)]; // MSE_Test
                else
                Individuals[i].fitness += max(Individuals[i].fitness_measures["MSE_Test_" + aquiutils::numbertostring(constituent)],Individuals[i].fitness_measures["MSE_Train_" + aquiutils::numbertostring(constituent)]); // MSE_Test and MSE_Train
        }
        else
        {
            for (int constituent = 0; constituent<models[i].FFN.ModelStructure.outputcolumns.size(); constituent++)
            {   Individuals[i].fitness_measures["MSE_Test_"+ aquiutils::numbertostring(constituent)]=1e12;
                Individuals[i].fitness_measures["R2_Test_"+ aquiutils::numbertostring(constituent)]=0;
                Individuals[i].fitness += Individuals[i].fitness_measures["MSE_Test_" + aquiutils::numbertostring(constituent)];
            }
        }
        cout<<i<<":"<<models[i].FFN.ModelStructure.ParametersToString().toStdString();

        for (int constituent = 0; constituent<models[i].FFN.ModelStructure.outputcolumns.size(); constituent++)
            cout<< ","<<Individuals[i].toAssignmentText("MSE_Train",constituent)<<","<<Individuals[i].toAssignmentText("R2_Train",constituent);

        for (int constituent = 0; constituent<models[i].FFN.ModelStructure.outputcolumns.size(); constituent++)
            cout<< ","<<Individuals[i].toAssignmentText("MSE_Test",constituent)<<","<<Individuals[i].toAssignmentText("R2_Test",constituent);
        cout<< endl;
    }



    vector<int> ranks = getRanks();
    for (unsigned int i=0; i<Individuals.size(); i++)
    {
        Individuals[i].rank = ranks[i];
    }
}

template<class T>
void GeneticAlgorithm<T>::CrossOver()
{
    vector<Individual> newIndividuals = Individuals;
    newIndividuals[0] = Individuals[max_rank];
    for (unsigned int i=1; i<Individuals.size(); i++)
    {
        Individual Ind1 = selectIndividualByRank();
        Individual Ind2 = selectIndividualByRank();
        BinaryNumber FullBinary = Ind1.toBinary();
        FullBinary.mutate(Settings.mutation_probability);
        newIndividuals[i] = FullBinary.split(Individuals[i].splitlocations);
    }
    Individuals = newIndividuals;
}


// Function to randomly select an Individual based on inverse rank probability
template<class T>
const Individual& GeneticAlgorithm<T>::selectIndividualByRank() {
    // Calculate weights as the inverse of rank
    std::vector<double> weights(Individuals.size());
    for (size_t i = 0; i < Individuals.size(); ++i) {
        weights[i] = 1.0 / Individuals[i].rank;
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
    max_rank = indices[0];
    return ranks;
}

