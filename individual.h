#ifndef INDIVIDUAL_H
#define INDIVIDUAL_H

#include "Binary.h"
#include "Utilities.h"
#include <map>
#include <vector>
#include <iostream>

/**
 * @brief Represents an individual in a genetic algorithm population.
 *
 * An Individual is essentially a vector of BinaryNumber objects,
 * representing a candidate solution. It stores its fitness, rank,
 * split locations, and other auxiliary fitness measures.
 */
class Individual : public std::vector<BinaryNumber> {
public:
    /// Default constructor.
    Individual() = default;

    /// Copy constructor.
    Individual(const Individual &other) {
        this->clear();
        for (const auto &binary : other) this->push_back(binary);
        fitness = other.fitness;
        fitness_measures = other.fitness_measures;
        splitlocations = other.splitlocations;
        rank = other.rank;
    }

    /// Assignment operator (copy from Individual).
    Individual &operator=(const Individual &other) {
        if (this != &other) {
            this->clear();
            for (const auto &binary : other) this->push_back(binary);
        }
        fitness = other.fitness;
        fitness_measures = other.fitness_measures;
        splitlocations = other.splitlocations;
        rank = other.rank;
        return *this;
    }

    /// Assignment operator (from vector of BinaryNumber).
    Individual &operator=(const std::vector<BinaryNumber> &other) {
        if (this != &other) {
            this->clear();
            for (const auto &binary : other) this->push_back(binary);
        }
        return *this;
    }

    /// Fitness value of the individual.
    double fitness = 0;

    /// Auxiliary fitness measures (map of metric name to value).
    std::map<std::string,double> fitness_measures;

    /// Print the individual’s genome in decimal representation.
    void display() const {
        std::cout << "Individual: ";
        for (const auto &binary : *this) {
            std::cout << binary.toDecimal() << " ";
        }
        std::cout << std::endl;
    }

    /// Locations of splits in genome.
    std::vector<unsigned int> splitlocations;

    /// Rank of the individual (for selection).
    unsigned int rank = 0;

    /// Greater-than operator (based on fitness).
    bool operator>(const Individual &I) { return fitness > I.fitness; }

    /// Less-than operator (based on fitness).
    bool operator<(const Individual &I) { return fitness < I.fitness; }

    /**
     * @brief Combine genome into one BinaryNumber.
     * @return Concatenated BinaryNumber.
     */
    BinaryNumber toBinary() const {
        BinaryNumber B = at(0);
        for (unsigned int i = 1; i < size(); i++) B += at(i);
        return B;
    }

    /**
     * @brief Return fitness measure as assignment string.
     * @param name Base name of the metric.
     * @param iterator Index appended to name.
     * @return String of form "name_i=value".
     */
    std::string toAssignmentText(const std::string &name, int iterator) {
        std::string key = name + "_" + aquiutils::numbertostring(iterator);
        std::string out = key + "=" + aquiutils::numbertostring(fitness_measures[key]);
        return out;
    }

    /**
     * @brief Mutate the individual's genome.
     * Applies mutation to each BinaryNumber gene.
     * @param mutationProbability Probability of flipping each bit.
     */
    void mutate(const double &mutationProbability) {
        for (auto &gene : *this) {
            gene.mutate(mutationProbability);
        }
    }

};

#endif // INDIVIDUAL_H
