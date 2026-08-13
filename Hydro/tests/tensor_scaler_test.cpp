#include "../dataset/tensor_scaler.h"

#include <cassert>

int main() {
    auto train = torch::tensor({{0.0f}, {2.0f}});
    auto heldOut = torch::tensor({{100.0f}});
    TensorScaler scaler;
    scaler.fit(train, "minmax");
    auto transformed = scaler.transform(heldOut);
    // A scaler leaked from held-out data would map this value to one.
    assert(transformed.item<float>() == 50.0f);
    assert(torch::allclose(scaler.inverseTransform(transformed), heldOut));

    auto constant = torch::ones({3, 2});
    scaler.fit(constant, "standardize");
    assert(torch::isfinite(scaler.transform(constant)).all().item<bool>());
    return 0;
}
