#include <torch/script.h>
#include <iostream>
#include <vector>

int main() {
    // Load the model
    torch::jit::script::Module model = torch::jit::load("/home/arash/PycharmProjects/PyTorchTest/ann_lagged_model.pt");
    model.eval();

    // Initial known values (length = 20)
    std::vector<float> history = {
        0.1, 0.2, 0.0, -0.1, -0.2, -0.3, -0.1, 0.0, 0.2, 0.3,
        0.5, 0.6, 0.4, 0.3, 0.2, 0.1, -0.1, -0.2, -0.1, 0.0
    };

    const int horizon = 30; // how many steps ahead to predict
    std::vector<float> predictions;

    for (int i = 0; i < horizon; ++i) {
        // Convert last 20 values into a tensor
        torch::Tensor input_tensor = torch::from_blob(history.data(), {1, 20}, torch::kFloat32).clone();

        // Predict the next value
        at::Tensor output = model.forward({input_tensor}).toTensor();
        float next_value = output.item<float>();
        predictions.push_back(next_value);

        // Slide the window: drop the first, append the prediction
        history.erase(history.begin());
        history.push_back(next_value);
    }

    // Print results
    std::cout << "Predicted " << horizon << " future steps:\n";
    for (size_t i = 0; i < predictions.size(); ++i) {
        std::cout << "Step " << i + 1 << ": " << predictions[i] << std::endl;
    }

    return 0;
}
