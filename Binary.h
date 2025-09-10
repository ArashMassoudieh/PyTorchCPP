#ifndef BINARY_H
#define BINARY_H

#include <iostream>
#include <string>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <vector>

/**
 * @brief Represents a binary number and provides utilities for
 *        conversion, mutation, crossover, and manipulation.
 *
 * A BinaryNumber stores its value as a string of '0' and '1' characters,
 * and provides helper functions to convert between decimal and binary,
 * perform crossover operations, mutate bits, and split into segments.
 */
class BinaryNumber {
private:
    /**
     * @brief Binary string representation of the number.
     */
    std::string binary;

public:
    /**
     * @brief Counter used to vary random seeds in static methods.
     */
    static inline int call_counter = 0;

    /**
     * @brief Construct a BinaryNumber from a binary string.
     * @param bin Binary string (e.g., "1011"). Defaults to empty string.
     */
    BinaryNumber(const std::string &bin = "") : binary(bin) {}

    /**
     * @brief Copy constructor.
     * @param other BinaryNumber to copy.
     */
    BinaryNumber(const BinaryNumber &other) { binary = other.binary; }

    /**
     * @brief Assignment operator.
     * @param other BinaryNumber to copy.
     * @return Reference to this BinaryNumber.
     */
    BinaryNumber &operator=(const BinaryNumber &other) {
        if (this != &other) {
            binary = other.binary;
        }
        return *this;
    }

    /**
     * @brief Convert a decimal integer to a BinaryNumber.
     * @param decimal Decimal integer.
     * @return BinaryNumber representation.
     */
    static BinaryNumber decimalToBinary(int decimal) {
        std::string result;
        while (decimal > 0) {
            result += (decimal % 2 == 0 ? "0" : "1");
            decimal /= 2;
        }
        std::reverse(result.begin(), result.end());
        return result.empty() ? "0" : result;
    }

    /**
     * @brief Convert a binary string to a decimal integer.
     * @param bin Binary string.
     * @return Decimal integer value.
     */
    static int binaryToDecimal(const std::string &bin) {
        int decimal = 0;
        for (char digit : bin) {
            decimal = (decimal << 1) + (digit - '0');
        }
        return decimal;
    }

    /// Set the binary string.
    void setBinary(const std::string &bin) { binary = bin; }

    /// Get the binary string.
    std::string getBinary() const { return binary; }

    /// Convert stored binary to its decimal equivalent.
    int toDecimal() const { return binaryToDecimal(binary); }

    /**
     * @brief Perform crossover between two BinaryNumbers.
     * @param bn1 First parent BinaryNumber.
     * @param bn2 Second parent BinaryNumber.
     * @return Offspring BinaryNumber.
     */
    static BinaryNumber crossover(const BinaryNumber &bn1, const BinaryNumber &bn2) {
        std::srand(static_cast<unsigned>(std::time(nullptr) + call_counter));
        call_counter++;
        size_t minLength = std::min(bn1.binary.size(), bn2.binary.size());
        size_t crossoverPoint = std::rand() % minLength;

        std::cout << "Random Crossover Point: " << crossoverPoint << std::endl;
        std::string newBinary = bn1.binary.substr(0, crossoverPoint) + bn2.binary.substr(crossoverPoint);
        return BinaryNumber(newBinary);
    }

    /// Print the binary string to standard output.
    void display() const { std::cout << "Binary: " << binary << std::endl; }

    /// Concatenate two BinaryNumbers.
    BinaryNumber operator+(const BinaryNumber &other) const { return BinaryNumber(binary + other.binary); }

    /// Append another BinaryNumber to this one.
    BinaryNumber &operator+=(const BinaryNumber &other) {
        if (this->binary.empty())
            this->binary = other.binary;
        else
            this->binary += other.binary;
        return *this;
    }

    /**
     * @brief Generate a random BinaryNumber.
     * @param maxDecimal Maximum decimal value to represent.
     * @return Random BinaryNumber within [0, maxDecimal].
     */
    static BinaryNumber randomBinary(int maxDecimal) {
        call_counter++;
        if (maxDecimal < 0) throw std::invalid_argument("Maximum decimal value must be non-negative.");
        std::srand(static_cast<unsigned>(std::time(nullptr) + call_counter));
        int randomDecimal = std::rand() % (maxDecimal + 1);
        return decimalToBinary(randomDecimal);
    }

    /**
     * @brief Split the binary string into segments.
     * @param segmentLengths Vector of lengths for each segment.
     * @return Vector of BinaryNumbers, each representing a segment.
     */
    std::vector<BinaryNumber> split(const std::vector<unsigned int> &segmentLengths) const {
        std::vector<BinaryNumber> segments;
        size_t currentIndex = 0;

        for (unsigned int length : segmentLengths) {
            if (currentIndex + length > binary.size()) {
                throw std::out_of_range("Segment length exceeds binary string length.");
            }
            std::string segment = binary.substr(currentIndex, length);
            segments.emplace_back(segment);
            currentIndex += length;
        }
        if (currentIndex < binary.size()) {
            throw std::invalid_argument("Unused portion of the binary string remains after splitting.");
        }
        return segments;
    }

    /// Compute maximum decimal value representable by `size` bits.
    static int maxDecimalForBinarySize(unsigned int size) {
        if (size == 0) throw std::invalid_argument("Binary size must be greater than 0.");
        return (1 << size) - 1;
    }

    /// Determine number of digits required to represent maxDecimal.
    static unsigned int digitsForMaxDecimal(int maxDecimal) {
        if (maxDecimal < 0) throw std::invalid_argument("Maximum decimal number must be non-negative.");
        unsigned int digits = 0;
        while (maxDecimal > 0) {
            maxDecimal >>= 1;
            ++digits;
        }
        return digits == 0 ? 1 : digits;
    }

    /// Pad with leading zeros to reach fixed size.
    void fixSize(unsigned int maxDigits) {
        if (binary.size() > maxDigits) throw std::invalid_argument("Binary string exceeds max digits.");
        while (binary.size() < maxDigits) {
            binary = '0' + binary;
        }
    }

    /// Return number of digits.
    unsigned int numDigits() const { return binary.size(); }

    /// Perform random mutations at specified positions.
    void mutate(unsigned int numMutations = 1) {
        if (binary.empty()) throw std::logic_error("Binary string is empty. Cannot perform mutation.");
        std::srand(static_cast<unsigned>(std::time(nullptr)));
        for (unsigned int i = 0; i < numMutations; ++i) {
            size_t position = std::rand() % binary.size();
            binary[position] = (binary[position] == '0') ? '1' : '0';
        }
    }

    /// Mutate each bit with given probability.
    void mutate(const double &mutationProbability) {
        if (mutationProbability < 0.0 || mutationProbability > 1.0)
            throw std::invalid_argument("Mutation probability must be between 0 and 1.");
        if (binary.empty()) throw std::logic_error("Binary string is empty. Cannot perform mutation.");
        std::srand(static_cast<unsigned>(std::time(nullptr) + call_counter));
        call_counter++;
        for (size_t i = 0; i < binary.size(); ++i) {
            double randomValue = static_cast<double>(std::rand()) / RAND_MAX;
            if (randomValue < mutationProbability) {
                binary[i] = (binary[i] == '0') ? '1' : '0';
            }
        }
    }
};

#endif // BINARY_H
