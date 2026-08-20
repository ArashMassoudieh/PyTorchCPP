#include "../dataset/tensor_scaler.h"
#include "../dataset/hydro_tensor_builder.h"

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
    assert(scaler.mseToPhysical(4.0) == 16.0);
    const HydroScalerState saved = scaler.exportState();
    TensorScaler restored;
    restored.importState(saved);
    assert(torch::allclose(restored.transform(heldOut), transformed));

    auto constant = torch::ones({3, 2});
    scaler.fit(constant, "standardize");
    assert(torch::isfinite(scaler.transform(constant)).all().item<bool>());

    auto regular = torch::tensor({{0.0f, 1.0f}, {0.5f, 2.0f}, {1.0f, 3.0f}});
    assert(regularPhysicalTimeStep(regular) == 0.5);
    bool irregularRejected = false;
    try {
        (void)regularPhysicalTimeStep(torch::tensor({{0.0f}, {0.5f}, {1.1f}}));
    } catch (const std::runtime_error&) {
        irregularRejected = true;
    }
    assert(irregularRejected);
    return 0;
}
