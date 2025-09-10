#ifndef INDIVIDUAL_H
#define INDIVIDUAL_H

#include <Binary.h>
#include "Utilities.h"

using namespace std;

class Individual : public std::vector<BinaryNumber> {
public:
    // Default constructor
    Individual() = default;

    // Copy constructor
    Individual(const Individual &other) {
        //std::cout << "Individual Copy Constructor Called\n";
        this->clear(); // Clear current content
        for (const auto &binary : other) {
            this->push_back(binary); // Copy each BinaryNumber object
        }
        fitness = other.fitness;
        fitness_measures = other.fitness_measures;
        splitlocations = other.splitlocations;
        rank = other.rank;
    }

    // Assignment operator
    Individual &operator=(const Individual &other) {
        if (this != &other) {
            //std::cout << "Individual Assignment Operator Called\n";
            this->clear(); // Clear current content
            for (const auto &binary : other) {
                this->push_back(binary); // Copy each BinaryNumber object
            }
        }
        fitness = other.fitness;
        fitness_measures = other.fitness_measures;
        splitlocations = other.splitlocations;
        rank = other.rank;
        return *this;
    }

    Individual &operator=(const vector<BinaryNumber> &other) {
        if (this != &other) {
            //std::cout << "Individual Assignment Operator Called\n";
            this->clear(); // Clear current content
            for (const auto &binary : other) {
                this->push_back(binary); // Copy each BinaryNumber object
            }
        }
        return *this;
    }

    double fitness = 0;
    map<string,double> fitness_measures;

    // Display the entire individual
    void display() const {
        std::cout << "Individual: ";
        for (const auto &binary : *this) {
            // Display the decimal equivalent of each BinaryNumber
            std::cout << binary.toDecimal() << " ";
        }
        std::cout << std::endl;
    }

    vector<unsigned int> splitlocations;
    unsigned int rank = 0;
    bool operator>(const Individual &I)
    {
        return (fitness>I.fitness?true:false);
    }
    bool operator<(const Individual &I)
    {
        return (fitness<I.fitness?true:false);
    }
    BinaryNumber toBinary() const
    {
        BinaryNumber B = at(0);
        for (unsigned int i=1; i<size(); i++)
            B += at(i);
        return B;
    }

    string toAssignmentText(const string &name, int iterator)
    {
        string out = name + "_" + aquiutils::numbertostring(iterator) + "=" + aquiutils::numbertostring(fitness_measures[name + "_" + aquiutils::numbertostring(iterator)]);
        return out;
    }



};


#endif // INDIVIDUAL_H
