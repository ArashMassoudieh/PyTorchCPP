#include "../dataset/tensor_scaler.h"
#include "../dataset/hydro_tensor_builder.h"

#include <cassert>
#include <limits>

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
    bool nonFiniteTrainingRejected = false;
    try { scaler.fit(torch::tensor({{0.0f}, {std::numeric_limits<float>::infinity()}}), "minmax"); }
    catch (const std::invalid_argument&) { nonFiniteTrainingRejected = true; }
    assert(nonFiniteTrainingRejected);
    assert(scaler.exportState().method == saved.method);
    bool invalidFitMethodRejected = false;
    try { scaler.fit(train, "custom"); }
    catch (const std::invalid_argument&) { invalidFitMethodRejected = true; }
    assert(invalidFitMethodRejected);
    assert(scaler.exportState().method == saved.method);
    TensorScaler restored;
    restored.importState(saved);
    assert(torch::allclose(restored.transform(heldOut), transformed));
    bool zeroScaleRejected = false;
    try { restored.importState({"minmax", {0.0}, {0.0}, {1, 1}}); }
    catch (const std::invalid_argument&) { zeroScaleRejected = true; }
    assert(zeroScaleRejected);
    bool unsupportedMethodRejected = false;
    try { restored.importState({"custom", {0.0}, {1.0}, {1, 1}}); }
    catch (const std::invalid_argument&) { unsupportedMethodRejected = true; }
    assert(unsupportedMethodRejected);
    bool overflowingShapeRejected = false;
    try { restored.importState({"none", {0.0}, {1.0}, {std::numeric_limits<int64_t>::max(), 2}}); }
    catch (const std::invalid_argument&) { overflowingShapeRejected = true; }
    assert(overflowingShapeRejected);

    auto constant = torch::ones({3, 2});
    scaler.fit(constant, "standardize");
    assert(torch::isfinite(scaler.transform(constant)).all().item<bool>());

    // log_standardize: round-trips through log1p/expm1 and preserves ordering.
    TensorScaler logScaler;
    auto flowTrain = torch::tensor({{0.0f}, {0.02f}, {0.5f}, {4.15f}});
    logScaler.fit(flowTrain, "log_standardize");
    auto flowHeldOut = torch::tensor({{0.45f}});
    auto logTransformed = logScaler.transform(flowHeldOut);
    assert(torch::allclose(logScaler.inverseTransform(logTransformed), flowHeldOut, 1e-4, 1e-4));
    // A big training flood (4.15) must not distort a small held-out value the
    // way plain standardize does: log-space transform of a near-zero value
    // stays finite and its inverse recovers the exact physical value.
    auto smallFlow = torch::tensor({{0.001f}});
    assert(torch::allclose(logScaler.inverseTransform(logScaler.transform(smallFlow)), smallFlow, 1e-4, 1e-4));
    // A tensor outside the log1p domain (e.g. a signed input feature such as
    // P-PET, sharing the same `normalization` config as a nonnegative
    // discharge target) must not fail the run: fall back to plain
    // standardize for that tensor instead of throwing.
    TensorScaler fallbackScaler;
    auto signedFeature = torch::tensor({{-2.0f}, {1.0f}, {3.0f}});
    fallbackScaler.fit(signedFeature, "log_standardize");
    assert(fallbackScaler.exportState().method == "standardize");
    auto plainStandardizeScaler = TensorScaler();
    plainStandardizeScaler.fit(signedFeature, "standardize");
    assert(torch::allclose(
        fallbackScaler.transform(signedFeature),
        plainStandardizeScaler.transform(signedFeature)));

    auto regular = torch::tensor({{0.0f, 1.0f}, {0.5f, 2.0f}, {1.0f, 3.0f}});
    assert(regularPhysicalTimeStep(regular) == 0.5);
    auto physicalTime = torch::tensor({{0.0f}, {0.5f}, {1.0f}});
    assert(regularPhysicalTimeStepFromTime(physicalTime) == 0.5);
    bool irregularRejected = false;
    try {
        (void)regularPhysicalTimeStep(torch::tensor({{0.0f}, {0.5f}, {1.1f}}));
    } catch (const std::runtime_error&) {
        irregularRejected = true;
    }
    assert(irregularRejected);
    return 0;
}
