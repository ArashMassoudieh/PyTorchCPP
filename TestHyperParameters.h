#ifndef TESTHYPERPARAMETERS_H
#define TESTHYPERPARAMETERS_H

#include "hyperparameters.h"
#include <iostream>
#include <cassert>
#include <stdexcept>

void testConstructorAndDefaults() {
    std::cout << "Testing constructor and default values..." << std::endl;

    HyperParameters params;

    // Test default values
    assert(params.getSelectedSeriesIds().empty());
    assert(params.getHiddenLayers().size() == 2);
    assert(params.getHiddenLayers()[0] == 64);
    assert(params.getHiddenLayers()[1] == 32);
    assert(params.getInputActivation() == "sigmoid");
    assert(params.getHiddenActivation() == "sigmoid");
    assert(params.getOutputActivation() == "");
    assert(params.getLags().empty());
    assert(params.getLagMultiplier().size() == 1);
    assert(params.getLagMultiplier()[0] == 1);
    assert(params.getMaxLagMultiplier() == 10);
    assert(params.getMaxLags() == 10);
    assert(params.getLagSelectionOdd() == 3);
    assert(params.getNumEpochs() == 100);
    assert(params.getBatchSize() == 32);
    assert(params.getLearningRate() == 0.001);
    assert(params.getTrainTestSplit() == 0.8);

    std::cout << "✓ Constructor and defaults test passed" << std::endl;
}

void testTimeSeriesSelection() {
    std::cout << "Testing time series selection..." << std::endl;

    HyperParameters params;

    // Test direct selection
    std::vector<int> selected_ids = {0, 2, 3};
    params.setSelectedSeriesIds(selected_ids);
    assert(params.getSelectedSeriesIds() == selected_ids);

    // Test binary selection
    // selection_code = 5 (binary: 101) with 3 series should select series 0 and 2
    params.setSelectedSeriesFromBinary(5L, 3);
    auto binary_selected = params.getSelectedSeriesIds();
    assert(binary_selected.size() == 2);
    assert(binary_selected[0] == 0);
    assert(binary_selected[1] == 2);

    // Test max selection code
    long int max_code = HyperParameters::getMaxSelectionCode(3);
    assert(max_code == 7L); // 2^3 - 1

    max_code = HyperParameters::getMaxSelectionCode(4);
    assert(max_code == 15L); // 2^4 - 1

    std::cout << "✓ Time series selection test passed" << std::endl;
}

void testNetworkArchitecture() {
    std::cout << "Testing network architecture..." << std::endl;

    HyperParameters params;

    // Test hidden layers
    std::vector<int> hidden_layers = {128, 64, 32};
    params.setHiddenLayers(hidden_layers);
    assert(params.getHiddenLayers() == hidden_layers);

    // Test activation function
    params.setInputActivation("tanh");
    assert(params.getInputActivation() == "tanh");
    params.setHiddenActivation("tanh");
    assert(params.getHiddenActivation() == "tanh");
    params.setOutputActivation("tanh");
    assert(params.getOutputActivation() == "tanh");

    std::cout << "✓ Network architecture test passed" << std::endl;
}

void testLagConfiguration() {
    std::cout << "Testing lag configuration..." << std::endl;

    HyperParameters params;

    // Test max lags
    params.setMaxLags(5);
    assert(params.getMaxLags() == 5);

    // Test lag selection odd
    params.setLagSelectionOdd(2);
    assert(params.getLagSelectionOdd() == 2);

    // Test lag multipliers
    std::vector<int> multipliers = {1, 2, 3};
    params.setLagMultiplier(multipliers);
    assert(params.getLagMultiplier() == multipliers);

    // Test max lag multiplier
    params.setMaxLagMultiplier(5);
    assert(params.getMaxLagMultiplier() == 5);

    // Test lag codes conversion
    std::vector<long int> lag_codes = {0L, 2L, 4L}; // 3 time series
    params.setLagsFromVector(lag_codes);
    auto lags = params.getLags();
    assert(lags.size() == 3);

    // For lag_codes[0] = 0 with base 2:
    // lag 0: 0 % 2 == 0 ✓, lag 1: 0 % 2 == 0 ✓, etc. (all selected since 0/2 = 0 for all)
    // For lag_codes[1] = 2 with base 2:
    // lag 0: 2 % 2 == 0 ✓, lag 1: 1 % 2 != 0 ✗, etc.

    // Test max lag code
    long int max_lag_code = params.getMaxLagCode();
    assert(max_lag_code == 31L); // 2^5 - 1

    std::cout << "✓ Lag configuration test passed" << std::endl;
}

void testTrainingParameters() {
    std::cout << "Testing training parameters..." << std::endl;

    HyperParameters params;

    // Test epochs
    params.setNumEpochs(200);
    assert(params.getNumEpochs() == 200);

    // Test batch size
    params.setBatchSize(64);
    assert(params.getBatchSize() == 64);

    // Test learning rate
    params.setLearningRate(0.0001);
    assert(params.getLearningRate() == 0.0001);

    // Test train/test split
    params.setTrainTestSplit(0.7);
    assert(params.getTrainTestSplit() == 0.7);

    std::cout << "✓ Training parameters test passed" << std::endl;
}

void testValidation() {
    std::cout << "Testing validation..." << std::endl;

    HyperParameters params;

    // Set valid configuration
    params.setSelectedSeriesIds({0, 1});
    params.setHiddenLayers({64, 32});
    params.setInputActivation("relu");
    params.setHiddenActivation("relu");
    params.setOutputActivation("relu");
    params.setNumEpochs(100);
    params.setBatchSize(32);
    params.setLearningRate(0.001);
    params.setTrainTestSplit(0.8);
    params.setMaxLagMultiplier(5);
    params.setLagMultiplier({1, 2});

    assert(params.isValid() == true);

    // Test invalid configuration (empty selected series)
    params.clearSelectedSeriesIds();
    assert(params.isValid() == false);

    std::cout << "✓ Validation test passed" << std::endl;
}

void testErrorHandling() {
    std::cout << "Testing error handling..." << std::endl;

    HyperParameters params;

    // Test negative selection code
    try {
        params.setSelectedSeriesFromBinary(-1L, 3);
        assert(false && "Expected exception for negative selection code");
    } catch (const std::runtime_error&) {
        // Expected
    }

    // Test empty selected series
    try {
        params.setSelectedSeriesIds({});
        assert(false && "Expected exception for empty selected series");
    } catch (const std::runtime_error&) {
        // Expected
    }

    // Test invalid input activation
    try {
        params.setInputActivation("invalid");
        assert(false && "Expected exception for invalid input activation");
    } catch (const std::runtime_error&) {
        // Expected
    }

    // Test invalid hidden activation
    try {
        params.setHiddenActivation("invalid");
        assert(false && "Expected exception for invalid hidden activation");
    } catch (const std::runtime_error&) {
        // Expected
    }

    // Test invalid output activation
    try {
        params.setOutputActivation("invalid");
        assert(false && "Expected exception for invalid output activation");
    } catch (const std::runtime_error&) {
        // Expected
    }

    // Test negative epochs
    try {
        params.setNumEpochs(-1);
        assert(false && "Expected exception for negative epochs");
    } catch (const std::runtime_error&) {
        // Expected
    }

    // Test invalid train/test split
    try {
        params.setTrainTestSplit(1.5);
        assert(false && "Expected exception for invalid train/test split");
    } catch (const std::runtime_error&) {
        // Expected
    }

    // Test lag multiplier exceeding maximum
    params.setMaxLagMultiplier(3);
    try {
        params.setLagMultiplier({1, 5}); // 5 > 3
        assert(false && "Expected exception for exceeding lag multiplier");
    } catch (const std::runtime_error&) {
        // Expected
    }

    std::cout << "✓ Error handling test passed" << std::endl;
}

void testStringRepresentation() {
    std::cout << "Testing string representation..." << std::endl;

    HyperParameters params;
    params.setSelectedSeriesIds({0, 2});
    params.setMaxNumberOfHiddenNodes(64);
    params.setMaxNumberOfHiddenLayers(3);
    params.setHiddenLayersFromCode(50L, 16);  // Generate architecture from code
    params.setInputActivation("tanh");
    params.setHiddenActivation("tanh");
    params.setOutputActivation("tanh");
    params.setMaxLags(3);
    params.setLagSelectionOdd(2);
    params.setLagMultiplier({1, 2, 1});
    params.setMaxLagMultiplier(10);

    // Generate lags from codes
    std::vector<long int> lag_codes = {0L, 4L, 2L}; // 3 time series
    params.setLagsFromVector(lag_codes);

    std::string str = params.toString();

    // Check that key information is present in string
    assert(str.find("Selected series: [0,2]") != std::string::npos);
    assert(str.find("Hidden layers:") != std::string::npos);
    assert(str.find("Number of hidden layers:") != std::string::npos);
    assert(str.find("Total hidden nodes:") != std::string::npos);
    assert(str.find("Max hidden nodes: 64") != std::string::npos);
    assert(str.find("Max hidden layers: 3") != std::string::npos);
    assert(str.find("Activation: tanh") != std::string::npos);
    assert(str.find("Max lags: 3") != std::string::npos);
    assert(str.find("Lag selection odd: 2") != std::string::npos);

    std::cout << "String representation: " << str << std::endl;
    std::cout << "✓ String representation test passed" << std::endl;
}


void testReset() {
    std::cout << "Testing reset functionality..." << std::endl;

    HyperParameters params;

    // Modify parameters
    params.setSelectedSeriesIds({0, 1, 2});
    params.setHiddenLayers({256, 128, 64});
    params.setInputActivation("sigmoid");
    params.setHiddenActivation("sigmoid");
    params.setOutputActivation("");
    params.setNumEpochs(500);

    // Reset to defaults
    params.reset();

    // Verify defaults are restored
    assert(params.getSelectedSeriesIds().empty());
    assert(params.getHiddenLayers().size() == 2);
    assert(params.getHiddenLayers()[0] == 64);
    assert(params.getInputActivation() == "sigmoid");
    assert(params.getHiddenActivation() == "sigmoid");
    assert(params.getOutputActivation() == "");

    assert(params.getNumEpochs() == 100);

    std::cout << "✓ Reset test passed" << std::endl;
}

void testLagGeneration() {
    std::cout << "Testing lag generation from vector..." << std::endl;

    HyperParameters params;

    // Set up configuration
    params.setMaxLags(4);           // Lags will be 0, 1, 2, 3
    params.setLagSelectionOdd(2);   // Base 2 conversion

    // Test case 1: lag_codes = {0, 3, 6}
    // For code 0 (base 2): 0%2=0 (lag 0), 0/2=0, 0%2=0 (lag 1), 0/2=0, 0%2=0 (lag 2), 0/2=0, 0%2=0 (lag 3)
    // All lags selected: [0, 1, 2, 3]
    //
    // For code 3 (base 2): 3%2=1 (lag 0 NOT selected), 3/2=1, 1%2=1 (lag 1 NOT selected), 1/2=0, 0%2=0 (lag 2 selected), 0/2=0, 0%2=0 (lag 3 selected)
    // Selected lags: [2, 3]
    //
    // For code 6 (base 2): 6%2=0 (lag 0 selected), 6/2=3, 3%2=1 (lag 1 NOT selected), 3/2=1, 1%2=1 (lag 2 NOT selected), 1/2=0, 0%2=0 (lag 3 selected)
    // Selected lags: [0, 3]

    std::vector<long int> lag_codes = {0L, 3L, 6L};
    params.setLagsFromVector(lag_codes);

    auto lags = params.getLags();
    assert(lags.size() == 3);

    // Verify first time series (code 0): should have all lags [0, 1, 2, 3]
    assert(lags[0].size() == 4);
    assert(lags[0][0] == 0);
    assert(lags[0][1] == 1);
    assert(lags[0][2] == 2);
    assert(lags[0][3] == 3);

    // Verify second time series (code 3): should have lags [2, 3]
    assert(lags[1].size() == 2);
    assert(lags[1][0] == 2);
    assert(lags[1][1] == 3);

    // Verify third time series (code 6): should have lags [0, 3]
    assert(lags[2].size() == 2);
    assert(lags[2][0] == 0);
    assert(lags[2][1] == 3);

    // Test case 2: Different base and max_lags
    params.setMaxLags(3);           // Lags: 0, 1, 2
    params.setLagSelectionOdd(3);   // Base 3 conversion

    // For code 9 (base 3): 9%3=0 (lag 0 selected), 9/3=3, 3%3=0 (lag 1 selected), 3/3=1, 1%3=1 (lag 2 NOT selected)
    // Selected lags: [0, 1]
    std::vector<long int> lag_codes2 = {9L};
    params.setLagsFromVector(lag_codes2);

    auto lags2 = params.getLags();
    assert(lags2.size() == 1);
    assert(lags2[0].size() == 2);
    assert(lags2[0][0] == 0);
    assert(lags2[0][1] == 1);

    std::cout << "Generated lags for code 0: [";
    for (size_t i = 0; i < lags[0].size(); i++) {
        std::cout << lags[0][i];
        if (i < lags[0].size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "Generated lags for code 3: [";
    for (size_t i = 0; i < lags[1].size(); i++) {
        std::cout << lags[1][i];
        if (i < lags[1].size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "Generated lags for code 6: [";
    for (size_t i = 0; i < lags[2].size(); i++) {
        std::cout << lags[2][i];
        if (i < lags[2].size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "✓ Lag generation test passed" << std::endl;
}

void testHiddenLayerArchitecture() {
    std::cout << "Testing hidden layer architecture generation..." << std::endl;

    HyperParameters params;

    // Set up configuration
    params.setMaxNumberOfHiddenNodes(4);     // Nodes can be 0-4
    params.setMaxNumberOfHiddenLayers(3);    // Up to 3 layers

    // Base will be 5 (4 + 1)
    // Test case 1: architecture_code = 0
    // Should result in default single layer with min_nodes_per_layer (1)
    params.setHiddenLayersFromCode(0L, 1);
    auto layers1 = params.getHiddenLayers();
    assert(layers1.size() == 1);
    assert(layers1[0] == 1);

    // Test case 2: architecture_code = 6 (base 5: 1*5 + 1 = [1,1])
    // Two layers, each with 1 node (min_nodes_per_layer = 1, so 1-1+1 = 1)
    params.setHiddenLayersFromCode(6L, 1);
    auto layers2 = params.getHiddenLayers();
    assert(layers2.size() == 2);
    assert(layers2[0] == 1);  // First layer: 1 node
    assert(layers2[1] == 1);  // Second layer: 1 node

    // Test case 3: architecture_code = 29 (base 5: 1*25 + 0*5 + 4 = [1,0,4])
    // Should skip the 0 layer, result in 2 layers: [1, 4]
    // Node mapping: 1 -> min_nodes_per_layer + (1-1) = 8 + 0 = 8
    //              4 -> min_nodes_per_layer + (4-1) = 8 + 3 = 11
    params.setHiddenLayersFromCode(29L, 8);
    auto layers3 = params.getHiddenLayers();
    assert(layers3.size() == 2);
    assert(layers3[0] == 8);   // First layer: 8 nodes
    assert(layers3[1] == 11);  // Second layer: 11 nodes (8 + (4-1))

    // Test case 4: architecture_code = 25 (base 5: 1*25 + 0*5 + 0 = [1,0,0])
    // Should skip both 0 layers, result in 1 layer: [16]
    params.setHiddenLayersFromCode(25L, 16);
    auto layers4 = params.getHiddenLayers();
    assert(layers4.size() == 1);
    assert(layers4[0] == 16);  // Only layer: 16 nodes

    // Test max architecture code
    long int max_code = params.getMaxArchitectureCode();
    assert(max_code == 124L);  // 5^3 - 1 = 125 - 1 = 124

    // Test with different max values
    params.setMaxNumberOfHiddenNodes(2);
    params.setMaxNumberOfHiddenLayers(2);
    long int max_code2 = params.getMaxArchitectureCode();
    assert(max_code2 == 8L);   // 3^2 - 1 = 9 - 1 = 8

    std::cout << "Architecture for code 6: [";
    for (size_t i = 0; i < layers2.size(); i++) {
        std::cout << layers2[i];
        if (i < layers2.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "Architecture for code 29 (skipping 0): [";
    for (size_t i = 0; i < layers3.size(); i++) {
        std::cout << layers3[i];
        if (i < layers3.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "Max architecture code with 4 max nodes, 3 max layers: " << max_code << std::endl;

    std::cout << "✓ Hidden layer architecture test passed" << std::endl;
}
#endif // TESTHYPERPARAMETERS_H
