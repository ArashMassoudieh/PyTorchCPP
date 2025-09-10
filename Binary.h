#ifndef BINARY_H
#define BINARY_H

#include <iostream>
#include <string>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <vector>

class BinaryNumber {
private:
    std::string binary; // To store binary number as a string

public:
    // Default constructor
    static inline int call_counter=0;
    BinaryNumber(const std::string &bin = "") : binary(bin) {}

    // Copy constructor
    BinaryNumber(const BinaryNumber &other) {
        binary = other.binary;
        //std::cout << "Copy constructor called\n";
    }

    // Assignment operator
    BinaryNumber &operator=(const BinaryNumber &other) {
        if (this != &other) { // Prevent self-assignment
            binary = other.binary;
        }
        return *this;
    }

    // Function to convert a decimal number to binary
    static BinaryNumber decimalToBinary(int decimal) {
        std::string result;
        while (decimal > 0) {
            result += (decimal % 2 == 0 ? "0" : "1");
            decimal /= 2;
        }
        std::reverse(result.begin(), result.end());
        return result.empty() ? "0" : result;
    }

    // Function to convert binary to decimal
    static int binaryToDecimal(const std::string &bin) {
        int decimal = 0;
        for (char digit : bin) {
            decimal = (decimal << 1) + (digit - '0');
        }
        return decimal;
    }

    // Function to set binary value
    void setBinary(const std::string &bin) {
        binary = bin;
    }

    // Function to get binary value
    std::string getBinary() const {
        return binary;
    }

    // Function to display decimal equivalent of the stored binary
    int toDecimal() const {
        return binaryToDecimal(binary);
    }

    // Function to perform crossover between two BinaryNumbers at a random point
    static BinaryNumber crossover(const BinaryNumber &bn1, const BinaryNumber &bn2) {
        // Initialize random seed
        std::srand(static_cast<unsigned>(std::time(nullptr)+call_counter));
        call_counter++;
        // Determine the minimum length of the two binary strings
        size_t minLength = std::min(bn1.binary.size(), bn2.binary.size());

        // Generate a random crossover point
        size_t crossoverPoint = std::rand() % minLength;

        std::cout << "Random Crossover Point: " << crossoverPoint << std::endl;

        // Perform the crossover
        std::string newBinary = bn1.binary.substr(0, crossoverPoint) + bn2.binary.substr(crossoverPoint);
        return BinaryNumber(newBinary);
    }

    // Display the binary number
    void display() const {
        std::cout << "Binary: " << binary << std::endl;
    }

    BinaryNumber operator+(const BinaryNumber &other) const {
            return BinaryNumber(binary + other.binary);
    }

    BinaryNumber &operator+=(const BinaryNumber &other) {
        if (this->binary.empty())
            this->binary = other.binary;
        else
            this->binary += other.binary; // Append the binary string of `other` to this object
        return *this; // Return the current object for chaining
    }

    static BinaryNumber randomBinary(int maxDecimal) {
        call_counter++;
        if (maxDecimal < 0) {
            throw std::invalid_argument("Maximum decimal value must be non-negative.");
        }

        // Seed the random number generator
        std::srand(static_cast<unsigned>(std::time(nullptr)+call_counter));

        // Generate a random decimal number in the range [0, maxDecimal]
        int randomDecimal = std::rand() % (maxDecimal + 1);

        // Convert the random decimal to binary
        BinaryNumber binaryString = decimalToBinary(randomDecimal);

        // Return the resulting BinaryNumber
        return binaryString;
    }
    std::vector<BinaryNumber> split(const std::vector<unsigned int> &segmentLengths) const {
        std::vector<BinaryNumber> segments;
        size_t currentIndex = 0;

        for (unsigned int length : segmentLengths) {
            if (currentIndex + length > binary.size()) {
                throw std::out_of_range("Segment length exceeds binary string length.");
            }

            // Extract the segment
            std::string segment = binary.substr(currentIndex, length);
            segments.emplace_back(segment);

            // Move to the next segment
            currentIndex += length;
        }

        if (currentIndex < binary.size()) {
            throw std::invalid_argument("Unused portion of the binary string remains after splitting.");
        }

        return segments;
    }
    static int maxDecimalForBinarySize(unsigned int size) {
        if (size == 0) {
            throw std::invalid_argument("Binary size must be greater than 0.");
        }
        return (1 << size) - 1; // 2^size - 1
    }

    static unsigned int digitsForMaxDecimal(int maxDecimal) {
        if (maxDecimal < 0) {
            throw std::invalid_argument("Maximum decimal number must be non-negative.");
        }
        unsigned int digits = 0;
        while (maxDecimal > 0) {
            maxDecimal >>= 1; // Equivalent to dividing by 2
            ++digits;
        }
        return digits == 0 ? 1 : digits; // Ensure at least one digit for 0
    }
    void fixSize(unsigned int maxDigits) {
        if (binary.size() > maxDigits) {
            throw std::invalid_argument("Binary string exceeds the maximum specified digits.");
        }

        // Add leading zeros if the current size is less than maxDigits
        while (binary.size() < maxDigits) {
            binary = '0' + binary;
        }
    }
    unsigned int numDigits() const {
        return binary.size();
    }
    void mutate(unsigned int numMutations = 1) {
        if (binary.empty()) {
            throw std::logic_error("Binary string is empty. Cannot perform mutation.");
        }

        // Seed the random number generator
        std::srand(static_cast<unsigned>(std::time(nullptr)));

        // Perform the specified number of mutations
        for (unsigned int i = 0; i < numMutations; ++i) {
            // Randomly select a position in the binary string
            size_t position = std::rand() % binary.size();

            // Flip the bit at the selected position
            binary[position] = (binary[position] == '0') ? '1' : '0';
        }
    }
    void mutate(const double &mutationProbability) {
        if (mutationProbability < 0.0 || mutationProbability > 1.0) {
            throw std::invalid_argument("Mutation probability must be between 0 and 1.");
        }

        if (binary.empty()) {
            throw std::logic_error("Binary string is empty. Cannot perform mutation.");
        }

        // Seed the random number generator
        std::srand(static_cast<unsigned>(std::time(nullptr)+call_counter));
        call_counter++;
        // Iterate through each digit in the binary string
        for (size_t i = 0; i < binary.size(); ++i) {
            double randomValue = static_cast<double>(std::rand()) / RAND_MAX; // Generate a random value between 0 and 1

            // Flip the bit with the given mutation probability
            if (randomValue < mutationProbability) {
                binary[i] = (binary[i] == '0') ? '1' : '0';
            }
        }
    }
};


#endif // BINARY_H
