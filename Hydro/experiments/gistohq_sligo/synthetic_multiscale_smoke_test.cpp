// Synthetic Hydro Package end-to-end smoke test (improvement_loop_20261023
// report 71 / tasks #141, #143, #144). Explicitly authorized as
// "implementation and synthetic smoke tests only -- not the 27-job research
// batch": this builds its own small synthetic on-disk GIStoOHQ package (not
// read from any real basin), trains the matched-control and multiscale arms
// against it through the real LSTMPINNWrapper::train dispatch -- exercising
// runMultiscaleSharedScaleInference's actual package-loading scoring logic,
// not just its rejection path, which is all earlier smoke tests exercised --
// exports both to a temporary on-disk experiment directory via the existing,
// unmodified HydroExperimentExporter, reloads each checkpoint via the
// existing, unmodified runSharedScaleInference/runMultiscaleSharedScaleInference,
// and asserts:
//   1. Eligibility parity: both arms score the identical window set (task
//      #143's core requirement).
//   2. Checkpoint-reload prediction parity: predictions recomputed from the
//      reloaded checkpoint exactly match the in-memory training-time
//      predictions (task #144).
// No frozen candidate, manifest, or real-basin data is read or written.
#include "../../models/lstm_pinn_wrapper.h"
#include "../../evaluation/experiment_exporter.h"
#include "../../evaluation/inference_runner.h"
#include "../../evaluation/shared_scale_inference.h"

#include <cassert>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>

namespace {

constexpr std::int64_t kHourSeconds = 3600;

std::string isoUtc(std::int64_t epochSeconds) {
    const std::time_t t = static_cast<std::time_t>(epochSeconds);
    std::tm tmv{};
    gmtime_r(&t, &tmv);
    std::ostringstream out;
    out << std::put_time(&tmv, "%Y-%m-%dT%H:%M:%SZ");
    return out.str();
}

// Builds a self-contained synthetic GIStoOHQ HydroPINNExport package:
// `totalDays` of complete hourly weather, one EVPTRNS value per UTC
// midnight, and 5-minute discharge covering every hour at 100% -- so the
// package has a single contiguous model-row segment the full span, with no
// forcing or discharge gaps to reason about. Precipitation is a 6-hour storm
// every 10th day driving a simple one-reservoir discharge recession, so the
// observed hydrograph has real variance (finite NSE is possible), not a
// constant.
void buildSyntheticPackage(const std::filesystem::path& root, int totalDays) {
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root / "observations");

    const std::int64_t startEpoch = 1704067200; // 2024-01-01T00:00:00Z
    const int totalHours = totalDays * 24;
    const std::int64_t endEpoch = startEpoch + static_cast<std::int64_t>(totalHours) * kHourSeconds;

    {
        std::ofstream manifest(root / "manifest.json");
        manifest << "{\"schema_name\":\"HydroPINNExport\",\"schema_version\":\"1.2\","
                 << "\"profile\":\"water-balance-v1\",\"site_id\":\"synthetic_multiscale_smoke\","
                 << "\"study_start\":\"" << isoUtc(startEpoch) << "\","
                 << "\"study_end\":\"" << isoUtc(endEpoch - kHourSeconds) << "\","
                 << "\"catchment_area_m2\":10000000}";
    }
    {
        std::ofstream variables(root / "variables.json");
        variables << "[{\"name\":\"PRECTOTCORR\",\"units\":[\"mm/hour\"]},"
                  << "{\"name\":\"T2M\",\"units\":[\"degC\"]},"
                  << "{\"name\":\"RH2M\",\"units\":[\"%\"]},"
                  << "{\"name\":\"WS2M\",\"units\":[\"m/s\"]},"
                  << "{\"name\":\"ALLSKY_SFC_SW_DWN\",\"units\":[\"MJ/m2/h\"]},"
                  << "{\"name\":\"EVPTRNS\",\"units\":[\"MJ/m2/day\"]},"
                  << "{\"name\":\"00060\",\"units\":[\"ft3/s\"]}]";
    }

    std::ofstream weather(root / "observations/temporal_1.csv");
    weather << "timestamp,PRECTOTCORR,T2M,RH2M,WS2M,ALLSKY_SFC_SW_DWN\n";
    std::ofstream evaptrans(root / "observations/temporal_2.csv");
    evaptrans << "timestamp,EVPTRNS\n";
    std::ofstream discharge(root / "observations/temporal_3.csv");
    discharge << "timestamp,00060\n";

    double storage = 5.0;
    constexpr double k = 0.08; // linear-reservoir recession per hour
    constexpr double pi = 3.14159265358979323846;
    for (int hour = 0; hour < totalHours; ++hour) {
        const std::int64_t epoch = startEpoch + static_cast<std::int64_t>(hour) * kHourSeconds;
        const int dayOfCycle = (hour / 24) % 10;
        const int hourOfDay = hour % 24;
        const double precipitation = (dayOfCycle == 0 && hourOfDay < 6) ? 3.0 : 0.0;
        const double temperature = 15.0 + 8.0 * std::sin(2.0 * pi * hourOfDay / 24.0);
        const double humidity = 60.0 + 10.0 * std::cos(2.0 * pi * hourOfDay / 24.0);
        const double wind = 2.0;
        const double solar = (hourOfDay >= 6 && hourOfDay <= 18)
            ? 1.5 * std::sin(pi * (hourOfDay - 6) / 12.0) : 0.0;
        weather << isoUtc(epoch) << ',' << precipitation << ',' << temperature << ','
                << humidity << ',' << wind << ',' << solar << '\n';

        storage = std::max(0.0, storage + precipitation - k * storage);
        const double dischargeCfs = 20.0 + k * storage * 50.0;
        for (int minute = 0; minute < 60; minute += 5) {
            discharge << isoUtc(epoch + static_cast<std::int64_t>(minute) * 60) << ',' << dischargeCfs << '\n';
        }

        if (hourOfDay == 0) {
            evaptrans << isoUtc(epoch) << ",4.9\n"; // ~0.0833 mm/h PET at latent_heat=2.45 MJ/kg
        }
    }
}

bool allclose(const std::vector<double>& a, const std::vector<double>& b, double tol) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i]) || std::abs(a[i] - b[i]) > tol) return false;
    }
    return true;
}

} // namespace

int main() {
    const std::filesystem::path packageRoot = "/tmp/synthetic_multiscale_smoke_package";
    buildSyntheticPackage(packageRoot, /*totalDays=*/40);
    std::cout << "Built synthetic package at " << packageRoot << "\n";

    HydroRunConfig base;
    base.use_hydro_package = true;
    base.hydro_package_path = packageRoot.string();
    base.pinn_physics_profile = "lstm_shared_scale_ablation_zero";
    base.use_latent_storage_physics = true;
    base.epochs = 5;
    base.batch_size = 16;
    base.lstm_sequence_length = 6;
    base.hidden_layers_csv = "8";
    base.multiscale_daily_history_days = 5;
    base.multiscale_daily_hidden_size = 4;
    base.evaluate_metrics = true;
    base.random_seed = 42;

    HydroRunConfig controlConfig = base;
    controlConfig.multiscale_matched_control = true;
    HydroRunConfig multiscaleConfig = base;

    LSTMPINNWrapper wrapper;

    std::cout << "Training matched-control arm...\n";
    HydroRunResult controlResult = wrapper.train(controlConfig);
    assert(controlResult.success);
    std::cout << "  " << controlResult.message << "\n";

    std::cout << "Training multiscale arm...\n";
    HydroRunResult multiscaleResult = wrapper.train(multiscaleConfig);
    assert(multiscaleResult.success);
    std::cout << "  " << multiscaleResult.message << "\n";

    // (1) Eligibility parity: identical window SET, in identical order.
    if (controlResult.x.size() != multiscaleResult.x.size()) {
        std::cerr << "FAIL: eligibility parity -- control scored " << controlResult.x.size()
                  << " windows, multiscale scored " << multiscaleResult.x.size() << "\n";
        return 1;
    }
    for (std::size_t i = 0; i < controlResult.x.size(); ++i) {
        assert(std::abs(controlResult.x[i] - multiscaleResult.x[i]) < 1.0e-6);
    }
    std::cout << "Eligibility parity OK: both arms scored " << controlResult.x.size()
              << " of the same windows.\n";

    // Export both arms into one on-disk experiment directory via the
    // existing, unmodified exporter -- exactly the artifact layout real
    // training jobs produce.
    const std::filesystem::path experimentRoot = "/tmp/synthetic_multiscale_smoke_experiment";
    std::filesystem::remove_all(experimentRoot);
    std::map<std::string, HydroRunResult> results = {
        {"lstm_pinn", controlResult},
        {"lstm_multiscale", multiscaleResult},
    };
    HydroExperimentExporter().exportRun(experimentRoot.string(), "synthetic_multiscale_smoke", multiscaleConfig, results);
    // exportRun writes into <outputDirectory>/<experimentId>/, not directly
    // into outputDirectory itself -- confirmed by inspecting the actual
    // on-disk layout it produced (models.csv, scalers.csv, etc. all live one
    // level deeper).
    const std::filesystem::path experimentDir = experimentRoot / "synthetic_multiscale_smoke";
    std::cout << "Exported both arms to " << experimentDir << "\n";

    // (2) Reload through the PUBLIC CLI/UI entry point -- HydroInferenceRunner::
    // runSharedScaleAblation -- not the lower-level runSharedScaleInference/
    // runMultiscaleSharedScaleInference functions directly. The first version
    // of this test called the lower-level functions itself, which is exactly
    // what let the dispatcher's matched-control routing bug (report 72) go
    // unnoticed: the dispatcher used here is the same one the real CLI driver
    // and Qt UI call, so a bug in it shows up here too.
    HydroInferenceRunner runner;
    const auto controlReload = runner.runSharedScaleAblation(controlConfig, experimentDir.string(), "lstm_pinn", false);
    const auto multiscaleReload = runner.runSharedScaleAblation(multiscaleConfig, experimentDir.string(), "lstm_multiscale", false);

    // (3) Cross-arm comparison through the public path: timestamps, splits,
    // and eligibility must all agree between the two reloaded arms -- not
    // merely each arm against its own training-time result.
    bool crossArmOk = true;
    if (controlReload.x.size() != multiscaleReload.x.size()) {
        std::cerr << "FAIL: public-dispatcher eligibility parity -- control reload has "
                  << controlReload.x.size() << " windows, multiscale reload has "
                  << multiscaleReload.x.size() << "\n";
        crossArmOk = false;
    } else {
        for (std::size_t i = 0; i < controlReload.x.size(); ++i) {
            if (std::abs(controlReload.x[i] - multiscaleReload.x[i]) > 1.0e-6) {
                std::cerr << "FAIL: public-dispatcher timestamp mismatch at row " << i << ": control x="
                          << controlReload.x[i] << " multiscale x=" << multiscaleReload.x[i] << "\n";
                crossArmOk = false;
                break;
            }
            if (controlReload.split[i] != multiscaleReload.split[i]) {
                std::cerr << "FAIL: public-dispatcher split mismatch at row " << i << " (x=" << controlReload.x[i]
                          << "): control split=" << controlReload.split[i]
                          << " multiscale split=" << multiscaleReload.split[i] << "\n";
                crossArmOk = false;
                break;
            }
        }
    }
    std::cout << "Public-dispatcher cross-arm timestamp/split/eligibility parity: "
              << (crossArmOk ? "PASS" : "FAIL") << " (" << controlReload.x.size() << " rows each)\n";
    if (!crossArmOk) return 1;

    // (4) Checkpoint-reload prediction parity, each arm against its own
    // training-time result -- both reloads are now eligibility-restricted
    // and index-aligned with their training-time counterparts (per (3)
    // above), so direct comparison needs no x-value lookup.
    const bool controlParity = allclose(controlResult.y_pred, controlReload.predicted, 1.0e-6);
    std::cout << "Matched-control checkpoint-reload prediction parity (via public dispatcher): "
              << (controlParity ? "PASS" : "FAIL") << "\n";
    if (!controlParity) return 1;

    const bool multiscaleParity = allclose(multiscaleResult.y_pred, multiscaleReload.predicted, 1.0e-6);
    std::cout << "Multiscale checkpoint-reload prediction parity (via public dispatcher): "
              << (multiscaleParity ? "PASS" : "FAIL") << "\n";
    if (!multiscaleParity) return 1;

    std::cout << "ALL SYNTHETIC SMOKE TESTS PASSED (via public HydroInferenceRunner dispatcher)\n";
    return 0;
}
