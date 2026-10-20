#include "../models/lstm_regional_pretrain.h"
#include "../models/hydro_lstm_module.h"

#include <torch/torch.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

// ---------- Part A: tensor-level unit tests (no file I/O) ----------

RegionalBasinInput makeSyntheticBasin(const std::string& staid, int64_t n, int64_t seqLen, int64_t features,
                                      double targetScale, double targetOffset, uint64_t seed) {
    torch::manual_seed(seed);
    RegionalBasinInput basin;
    basin.staid = staid;
    basin.xSeq = torch::rand({n, seqLen, features}, torch::kFloat32);
    // Target deliberately on a very different physical scale per basin
    // (targetScale/targetOffset) - exercises that per-basin normalization
    // actually removes each basin's own scale before pooling, rather than
    // pooling raw physical-unit values across basins of different size.
    basin.ySeq = torch::rand({n, 1}, torch::kFloat32) * static_cast<float>(targetScale) +
                static_cast<float>(targetOffset);
    return basin;
}

void testPooledTrainingTensorLevel() {
    const int64_t seqLen = 4;
    const int64_t features = 3;

    // Three training basins: deliberately different lengths (60, 20, 200)
    // to exercise balancedShuffledEpochIndices's without-replacement path
    // (basin >= perBasinCount) and with-replacement cycling path (basin <
    // perBasinCount) inside a real training loop, not just the isolated
    // sampler test in scientific_core_test.cpp. Deliberately different
    // target scales (1x, 100x, 200x + offset) to exercise per-basin scaling.
    std::vector<RegionalBasinInput> trainingBasins = {
        makeSyntheticBasin("train_small_scale", 60, seqLen, features, 1.0, 0.0, 1),
        makeSyntheticBasin("train_short_record", 20, seqLen, features, 100.0, 50.0, 2),
        makeSyntheticBasin("train_long_record", 200, seqLen, features, 200.0, 0.0, 3),
    };
    std::vector<RegionalBasinInput> validationBasins = {
        makeSyntheticBasin("val_basin_1", 30, seqLen, features, 5.0, 1.0, 4),
        // Non-negative, matching real basin-normalized runoff depth (never
        // physically negative) - a basin whose own target range sits
        // entirely outside the training pool's domain is covered
        // separately by testPooledScalerRejectsOutOfDomainValidationBasin,
        // not by making this basin itself invalid data.
        makeSyntheticBasin("val_basin_2", 40, seqLen, features, 500.0, 2.0, 5),
    };

    RegionalPretrainConfig config;
    config.epochs = 3;
    config.batch_size = 8;
    config.learning_rate = 0.01;
    config.hidden_layers = {6};
    config.random_seed = 42;

    const RegionalPretrainOutput result = trainLstmRegionalPool(trainingBasins, validationBasins, config);

    assert(result.success);
    assert(!result.model_checkpoint.empty());
    assert(result.model_checkpoint_format == "torch-module-v1");
    assert(result.best_epoch >= 1 && result.best_epoch <= config.epochs);
    assert(static_cast<int>(result.training_loss_history.size()) == config.epochs);
    assert(static_cast<int>(result.validation_loss_history.size()) == config.epochs);
    for (const double loss : result.training_loss_history) assert(std::isfinite(loss));
    for (const double loss : result.validation_loss_history) assert(std::isfinite(loss));
    assert(std::isfinite(result.best_validation_loss));

    // Determinism: identical config + identical seed must reproduce the
    // identical loss histories exactly (same sampling draws, same weight
    // init, same optimizer trajectory) - the pilot's own reproducibility
    // requirement, checked here at the unit level before trusting it at
    // basin-pool scale.
    const RegionalPretrainOutput repeat = trainLstmRegionalPool(trainingBasins, validationBasins, config);
    assert(repeat.training_loss_history == result.training_loss_history);
    assert(repeat.validation_loss_history == result.validation_loss_history);
    assert(repeat.best_epoch == result.best_epoch);

    // A different seed must NOT reproduce the identical history (confirms
    // the seed is actually wired through, not silently ignored).
    RegionalPretrainConfig differentSeed = config;
    differentSeed.random_seed = 43;
    const RegionalPretrainOutput varied = trainLstmRegionalPool(trainingBasins, validationBasins, differentSeed);
    assert(varied.training_loss_history != result.training_loss_history);

    // The returned checkpoint must be a genuinely loadable torch module
    // state - the exact property lstm_warm_start_checkpoint_path depends on
    // for fine-tuning (see lstmnetworkwrapper.cpp). Write it to a temp file,
    // load it into a FRESH, architecturally-matching HydroLSTM, and confirm
    // a forward pass runs without throwing.
    {
        const fs::path checkpointPath = fs::temp_directory_path() / "lstm_regional_pretrain_test_checkpoint.pt";
        {
            std::ofstream out(checkpointPath, std::ios::binary);
            out.write(reinterpret_cast<const char*>(result.model_checkpoint.data()),
                      static_cast<std::streamsize>(result.model_checkpoint.size()));
        }
        HydroLSTM reloaded(features, config.hidden_layers.front(), 1, 1, 0.0);
        torch::serialize::InputArchive archive;
        archive.load_from(checkpointPath.string());
        reloaded->load(archive);
        torch::NoGradGuard noGrad;
        const auto probe = torch::rand({2, seqLen, features});
        const auto output = reloaded->forward(probe);
        assert(output.size(0) == 2 && output.size(1) == 1);
        fs::remove(checkpointPath);
    }
}

// A validation basin whose target data falls outside the training pool's
// log_standardize domain (<=-1) must be REJECTED with a clear error, not
// silently produce NaN validation loss (the exact bug this module's own
// test construction caught once already - see scaleBasins's doc comment
// in lstm_regional_pretrain.cpp).
void testPooledScalerRejectsOutOfDomainValidationBasin() {
    const int64_t seqLen = 4;
    const int64_t features = 2;
    std::vector<RegionalBasinInput> trainingBasins = {
        // All non-negative - a realistic training pool.
        makeSyntheticBasin("train1", 30, seqLen, features, 10.0, 0.0, 10),
        makeSyntheticBasin("train2", 30, seqLen, features, 20.0, 5.0, 11),
    };
    std::vector<RegionalBasinInput> outOfDomainValidation = {
        // Deliberately dips below -1 - physically unrealistic for runoff
        // depth, but must be caught explicitly rather than silently
        // producing NaN.
        makeSyntheticBasin("val_bad", 20, seqLen, features, 50.0, -20.0, 12),
    };
    RegionalPretrainConfig config;
    config.epochs = 1;
    config.hidden_layers = {4};

    bool rejected = false;
    std::string message;
    try {
        (void)trainLstmRegionalPool(trainingBasins, outOfDomainValidation, config);
    } catch (const std::runtime_error& error) {
        rejected = true;
        message = error.what();
    }
    assert(rejected);
    assert(message.find("outside the domain") != std::string::npos);
}

// The pooled target scaler must be a pure function of the TRAINING pool -
// changing a validation basin's own target data must not change the
// scaler's parameters at all. This is the leakage-safety property the
// 2026-10-20 fix exists to guarantee: a validation basin's own complete
// target series must never inform the normalization used to score it.
void testPooledTargetScalerIndependentOfValidationData() {
    const int64_t seqLen = 4;
    const int64_t features = 2;
    std::vector<RegionalBasinInput> trainingBasins = {
        makeSyntheticBasin("train1", 40, seqLen, features, 10.0, 1.0, 20),
        makeSyntheticBasin("train2", 25, seqLen, features, 30.0, 2.0, 21),
    };
    RegionalPretrainConfig config;
    config.epochs = 2;
    config.hidden_layers = {4};
    config.random_seed = 99;

    // Two very different validation sets, same training pool.
    std::vector<RegionalBasinInput> validationA = {
        makeSyntheticBasin("valA", 15, seqLen, features, 1.0, 0.5, 30),
    };
    std::vector<RegionalBasinInput> validationB = {
        // Wildly different scale/offset from validationA, on the same
        // training pool - if the target scaler leaked validation
        // information, training would diverge between these two runs.
        makeSyntheticBasin("valB", 60, seqLen, features, 900.0, 300.0, 31),
    };

    torch::manual_seed(99);
    const RegionalPretrainOutput resultA = trainLstmRegionalPool(trainingBasins, validationA, config);
    torch::manual_seed(99);
    const RegionalPretrainOutput resultB = trainLstmRegionalPool(trainingBasins, validationB, config);

    // Training loss history depends only on the training pool + the
    // model's own seeded init/sampling - identical regardless of which
    // validation set rides along, which is only true if the target
    // scaler applied to the TRAINING data (same pooled scaler used for
    // both) is unaffected by which validation set is present.
    assert(resultA.training_loss_history == resultB.training_loss_history);
}

RegionalBasinInput makeBasinWithExactTargets(const std::string& staid, const std::vector<double>& targets,
                                             int64_t seqLen, int64_t features) {
    const int64_t n = static_cast<int64_t>(targets.size());
    RegionalBasinInput basin;
    basin.staid = staid;
    basin.xSeq = torch::rand({n, seqLen, features}, torch::kFloat32);  // irrelevant to the target-scaler check
    std::vector<float> y(targets.begin(), targets.end());
    basin.ySeq = torch::from_blob(y.data(), {n, 1}, torch::kFloat32).clone();
    return basin;
}

// Population mean/std of log1p(values) - matches TensorScaler::fit's own
// convention exactly (torch Tensor::std(dim, /*unbiased=*/false, keepdim),
// i.e. divide by N, not N-1) so this is a true independent reference, not
// an approximation that happens to be close.
std::pair<double, double> log1pPopulationMeanStd(const std::vector<double>& values) {
    double sum = 0.0;
    for (const double v : values) sum += std::log1p(v);
    const double mean = sum / static_cast<double>(values.size());
    double sumSquaredDeviation = 0.0;
    for (const double v : values) {
        const double deviation = std::log1p(v) - mean;
        sumSquaredDeviation += deviation * deviation;
    }
    const double variance = sumSquaredDeviation / static_cast<double>(values.size());
    return {mean, std::sqrt(variance)};
}

// Exact regression test for the 2026-10-20 pooled-variance fix: the
// pooled target scaler must match the MIXTURE distribution's true
// mean/variance (law of total variance), not a naive average of each
// basin's own std dev - confirmed as a real, large (not rounding-level)
// error before this fix: naive sigma-averaging understated the pooled
// scale by ~2.6x on a 3-basin synthetic check with deliberately different
// basin means/stds. Deterministic (non-random) target values throughout,
// so every reference below is an EXACT computation, not a statistical
// approximation expected to match only within sampling noise.
void testPooledTargetScalerMatchesConcatenatedReference() {
    const int64_t seqLen = 4;
    const int64_t features = 2;
    // Three EQUAL-SIZE (N=50) basins with deliberately different target
    // ranges - equal size means a literal concatenation of all three
    // naturally represents the equal-weighted mixture this function is
    // supposed to compute, with no resampling needed to make the
    // comparison fair.
    std::vector<double> targetsA, targetsB, targetsC;
    for (int i = 0; i < 50; ++i) {
        targetsA.push_back(1.0 + i);     // 1..50,   mean 25.5
        targetsB.push_back(101.0 + i);   // 101..150, mean 125.5
        targetsC.push_back(11.0 + i);    // 11..60,  mean 35.5
    }
    std::vector<RegionalBasinInput> trainingBasins = {
        makeBasinWithExactTargets("basinA", targetsA, seqLen, features),
        makeBasinWithExactTargets("basinB", targetsB, seqLen, features),
        makeBasinWithExactTargets("basinC", targetsC, seqLen, features),
    };

    // Reference 1: closed-form mixture formula, computed independently
    // here (not by calling the implementation under test).
    const auto [muA, sigmaA] = log1pPopulationMeanStd(targetsA);
    const auto [muB, sigmaB] = log1pPopulationMeanStd(targetsB);
    const auto [muC, sigmaC] = log1pPopulationMeanStd(targetsC);
    const double muPooledFormula = (muA + muB + muC) / 3.0;
    const double meanOfSquares = ((sigmaA * sigmaA + muA * muA) + (sigmaB * sigmaB + muB * muB) +
                                  (sigmaC * sigmaC + muC * muC)) / 3.0;
    const double sigmaPooledFormula = std::sqrt(meanOfSquares - muPooledFormula * muPooledFormula);

    // Reference 2: literal concatenation of all three basins' raw target
    // values (equal size, so this IS the equal-weighted mixture directly,
    // with no formula involved at all) - the "explicitly concatenated
    // equal-basin synthetic reference" check.
    std::vector<double> concatenated = targetsA;
    concatenated.insert(concatenated.end(), targetsB.begin(), targetsB.end());
    concatenated.insert(concatenated.end(), targetsC.begin(), targetsC.end());
    const auto [muConcat, sigmaConcat] = log1pPopulationMeanStd(concatenated);

    // The two independent references must agree with each other (this is
    // a mathematical identity for equal-size basins, not an empirical
    // approximation) before either is trusted as ground truth for the
    // implementation check below.
    assert(std::abs(muPooledFormula - muConcat) < 1.0e-9);
    assert(std::abs(sigmaPooledFormula - sigmaConcat) < 1.0e-9);

    // Now check the actual implementation against both references.
    const TensorScaler pooledScaler = fitPooledTargetScalerFromTrainingPool(trainingBasins, "log_standardize");
    const HydroScalerState state = pooledScaler.exportState();
    assert(state.method == "log_standardize");
    assert(std::abs(state.offset[0] - muConcat) < 1.0e-6);
    assert(std::abs(state.scale[0] - sigmaConcat) < 1.0e-6);

    // The old (wrong) implementation's naive sigma-average would have
    // given approximately (sigmaA+sigmaB+sigmaC)/3 - confirm the fixed
    // implementation is NOT that value (guards against silently
    // reverting to the bug).
    const double naiveWrongAverage = (sigmaA + sigmaB + sigmaC) / 3.0;
    assert(std::abs(state.scale[0] - naiveWrongAverage) > 0.05);
}

void testPooledTrainingRejectsMalformedInput() {
    RegionalPretrainConfig config;
    config.epochs = 1;

    bool emptyTrainingRejected = false;
    try { (void)trainLstmRegionalPool({}, {makeSyntheticBasin("v", 10, 4, 2, 1.0, 0.0, 1)}, config); }
    catch (const std::invalid_argument&) { emptyTrainingRejected = true; }
    assert(emptyTrainingRejected);

    bool emptyValidationRejected = false;
    try { (void)trainLstmRegionalPool({makeSyntheticBasin("t", 10, 4, 2, 1.0, 0.0, 1)}, {}, config); }
    catch (const std::invalid_argument&) { emptyValidationRejected = true; }
    assert(emptyValidationRejected);

    // A basin with a different feature width than the pool must be
    // rejected, not silently truncated/padded.
    bool shapeMismatchRejected = false;
    try {
        std::vector<RegionalBasinInput> training = {
            makeSyntheticBasin("t1", 10, 4, 2, 1.0, 0.0, 1),
            makeSyntheticBasin("t2", 10, 4, 3, 1.0, 0.0, 2),  // 3 features, not 2
        };
        std::vector<RegionalBasinInput> validation = {makeSyntheticBasin("v", 10, 4, 2, 1.0, 0.0, 3)};
        (void)trainLstmRegionalPool(training, validation, config);
    } catch (const std::invalid_argument&) { shapeMismatchRejected = true; }
    assert(shapeMismatchRejected);
}

// ---------- Part B: integration test with tiny synthetic GIStoOHQ packages (file I/O) ----------

std::string isoHour(int64_t hour) {
    // hour 0 = 2020-01-01T00:00:00Z, strictly hourly thereafter - matches
    // the GIStoOHQ hourly-grid contract (buildGisToOhqModelRows requires a
    // strictly regular UTC grid).
    const int64_t daysPerYear = 365;  // test window never crosses a leap day
    int64_t day = hour / 24;
    const int64_t hourOfDay = hour % 24;
    int year = 2020;
    while (day >= daysPerYear) { day -= daysPerYear; ++year; }
    static const int monthLengths[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    int month = 0;
    while (day >= monthLengths[month]) { day -= monthLengths[month]; ++month; }
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%04d-%02d-%02dT%02d:00:00Z", year, month + 1, static_cast<int>(day) + 1,
                 static_cast<int>(hourOfDay));
    return std::string(buffer);
}

// Writes a tiny but real GIStoOHQ HydroPINNExport package (manifest.json +
// variables.json + observations/temporal_1..3.csv) - the identical format
// ingest_camelsh_basin.py produces and prepareGisToOhqPackage consumes, so
// this integration test exercises the real file-loading path, not a
// shortcut. `catchmentAreaM2` and `dischargeBaseCms` differ per call so the
// test can confirm per-basin normalization actually engages.
fs::path writeSyntheticGisToOhqPackage(const fs::path& root, const std::string& name, int64_t hours,
                                       double catchmentAreaM2, double dischargeBaseCms, uint64_t seed) {
    const fs::path dir = root / name;
    fs::create_directories(dir / "observations");
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> precipDist(0.0, 5.0);
    std::uniform_real_distribution<double> tempDist(-5.0, 25.0);
    std::uniform_real_distribution<double> humidityDist(30.0, 90.0);
    std::uniform_real_distribution<double> windDist(0.0, 8.0);
    std::uniform_real_distribution<double> solarDist(0.0, 3.0);
    std::uniform_real_distribution<double> petDist(0.0, 0.3);
    std::uniform_real_distribution<double> dischargeNoise(0.5, 1.5);

    std::ofstream temporal1(dir / "observations" / "temporal_1.csv");
    temporal1 << "timestamp_utc,variable,value,native_unit,provider_qualifiers\n";
    std::ofstream temporal2(dir / "observations" / "temporal_2.csv");
    temporal2 << "timestamp_utc,variable,value,native_unit,provider_qualifiers\n";
    std::ofstream temporal3(dir / "observations" / "temporal_3.csv");
    temporal3 << "timestamp_utc,variable,value,native_unit,provider_qualifiers\n";

    for (int64_t h = 0; h < hours; ++h) {
        const std::string ts = isoHour(h);
        temporal1 << ts << ",PRECTOTCORR," << precipDist(rng) << ",mm/hour,TEST\n";
        temporal1 << ts << ",T2M," << tempDist(rng) << ",C,TEST\n";
        temporal1 << ts << ",RH2M," << humidityDist(rng) << ",%,TEST\n";
        temporal1 << ts << ",WS2M," << windDist(rng) << ",m/s,TEST\n";
        temporal1 << ts << ",ALLSKY_SFC_SW_DWN," << solarDist(rng) << ",MJ/hr,TEST\n";
        // EVPTRNS must be once-per-UTC-day in MJ/m^2/day (see
        // ingest_camelsh_basin.py's header comment for why this convention
        // exists) - only emit it at hour-of-day 0.
        if (h % 24 == 0) temporal2 << ts << ",EVPTRNS," << (petDist(rng) * 24.0) << ",MJ/m^2/day,TEST\n";
        const double dischargeCms = std::max(0.01, dischargeBaseCms * dischargeNoise(rng));
        const double dischargeCfs = dischargeCms * 35.3146667;
        temporal3 << ts << ",00060," << dischargeCfs << ",ft3/s,TEST\n";
    }

    std::ofstream variables(dir / "variables.json");
    variables << "{\n  \"schema_name\": \"HydroPINNVariables\",\n  \"schema_version\": \"1.0\",\n"
              << "  \"variables\": [\n"
              << "    {\"name\": \"PRECTOTCORR\", \"units\": [\"mm/hour\"]},\n"
              << "    {\"name\": \"T2M\", \"units\": [\"C\"]},\n"
              << "    {\"name\": \"RH2M\", \"units\": [\"%\"]},\n"
              << "    {\"name\": \"WS2M\", \"units\": [\"m/s\"]},\n"
              << "    {\"name\": \"ALLSKY_SFC_SW_DWN\", \"units\": [\"MJ/hr\"]},\n"
              << "    {\"name\": \"EVPTRNS\", \"units\": [\"MJ/m^2/day\"]},\n"
              << "    {\"name\": \"00060\", \"units\": [\"ft3/s\"]}\n"
              << "  ]\n}\n";

    const std::string studyEnd = isoHour(hours - 1);
    std::ofstream manifest(dir / "manifest.json");
    manifest << "{\n  \"schema_name\": \"HydroPINNExport\",\n  \"schema_version\": \"1.2\",\n"
             << "  \"site_id\": \"" << name << "\",\n  \"profile\": \"water-balance-v1\",\n"
             << "  \"study_start\": \"2020-01-01T00:00:00Z\",\n  \"study_end\": \"" << studyEnd << "\",\n"
             << "  \"catchment_area_m2\": " << catchmentAreaM2 << ",\n"
             << "  \"qc_gate\": \"not_run\",\n  \"generated_at\": \"2020-01-01T00:00:00Z\"\n}\n";

    return dir;
}

void writeBasinListFile(const fs::path& path, const std::vector<fs::path>& basinDirs) {
    std::ofstream out(path);
    for (const auto& dir : basinDirs) out << dir.string() << "\n";
}

void testRegionalPretrainIntegrationFromRealPackages() {
    const fs::path root = fs::temp_directory_path() / "lstm_regional_pretrain_test_packages";
    fs::remove_all(root);
    fs::create_directories(root);

    // Two training basins, one validation basin - small (600h = 25 days,
    // enough for sequence_length=24 windowing to produce ~576 sequences
    // each) but real files through the real loader, with deliberately
    // different catchment areas/discharge magnitudes so basin-normalization
    // actually has something to normalize away.
    const auto train1 = writeSyntheticGisToOhqPackage(root, "train1", 600, 5.0e6, 0.5, 101);
    const auto train2 = writeSyntheticGisToOhqPackage(root, "train2", 600, 5.0e7, 8.0, 102);
    const auto val1 = writeSyntheticGisToOhqPackage(root, "val1", 600, 2.0e7, 3.0, 103);

    const fs::path trainingList = root / "training_basins.txt";
    const fs::path validationList = root / "validation_basins.txt";
    writeBasinListFile(trainingList, {train1, train2});
    writeBasinListFile(validationList, {val1});

    HydroRunConfig config;
    config.epochs = 2;
    config.batch_size = 16;
    config.learning_rate = 0.01;
    config.optimizer = "adam";
    config.normalization = "log_standardize";
    config.hidden_layers_csv = "6";
    config.lstm_sequence_length = 24;
    config.regional_training_basin_list_path = trainingList.string();
    config.regional_validation_basin_list_path = validationList.string();
    config.regional_per_basin_sequence_count = -1;

    const HydroRunResult result = trainLstmRegionalPretrainFromConfig(config);

    assert(result.success);
    assert(!result.model_checkpoint.empty());
    assert(result.model_checkpoint_format == "torch-module-v1");
    assert(static_cast<int>(result.training_loss_history.size()) == config.epochs);
    assert(static_cast<int>(result.validation_loss_history.size()) == config.epochs);
    for (const double loss : result.training_loss_history) assert(std::isfinite(loss));
    for (const double loss : result.validation_loss_history) assert(std::isfinite(loss));
    assert(std::isfinite(result.validation_mse));

    // Missing basin list paths must be rejected, not silently defaulted.
    HydroRunConfig missingPaths = config;
    missingPaths.regional_training_basin_list_path = "";
    bool rejected = false;
    try { (void)trainLstmRegionalPretrainFromConfig(missingPaths); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);

    fs::remove_all(root);
}

// Exact-row-count regression test for the regional_window_start/
// regional_window_end filter (the 2015-2022 enforcement gap found in
// review, 2026-10-20): a 1000-hour synthetic package, windowed to exactly
// [hour 96, hour 216) = [2020-01-05T00:00:00Z, 2020-01-10T00:00:00Z) = 120
// hours. With sequenceLength=24 and no gaps inside the window, the kept
// sequence count must be EXACTLY 120-24+1=97 - not "fewer than the full
// 1000-hour package" (which would pass a sloppier check), the precise
// number, matching this session's established standard for test_end_x.
void testRegionalWindowFilterExactRowCount() {
    const fs::path root = fs::temp_directory_path() / "lstm_regional_pretrain_test_window_filter";
    fs::remove_all(root);
    fs::create_directories(root);
    const auto packageDir = writeSyntheticGisToOhqPackage(root, "window_basin", 1000, 1.0e7, 2.0, 201);

    HydroRunConfig config;
    config.lstm_sequence_length = 24;
    config.regional_window_start = "2020-01-05T00:00:00Z";
    config.regional_window_end = "2020-01-10T00:00:00Z";

    const RegionalBasinInput windowed = loadRegionalBasinInput(packageDir.string(), config);
    assert(windowed.xSeq.size(0) == 97);
    assert(windowed.ySeq.size(0) == 97);

    // Unbounded (both fields empty) must recover the full package's
    // sequence count. Confirmed by direct diagnostic (not assumed): the
    // raw, pre-sequencing row count is 999, not the naive 1000 - the
    // harmonizer's discharge-coverage computation for the hourly GIStoOHQ
    // path drops exactly the series' own literal last row (a structural
    // edge effect: establishing discharge coverage for an hour needs a
    // following interval, which the very last row of any series cannot
    // have), confirmed via plotX spanning a gap-free [0, 998] - 999
    // contiguous rows, not 16 or 24 missing at a day boundary as a first
    // hypothesis about the EVPTRNS daily broadcast assumed. So:
    // 999 - sequenceLength(24) + 1 = 976.
    HydroRunConfig unbounded = config;
    unbounded.regional_window_start = "";
    unbounded.regional_window_end = "";
    const RegionalBasinInput full = loadRegionalBasinInput(packageDir.string(), unbounded);
    assert(full.xSeq.size(0) == 976);

    // A window entirely outside the package's data range must be
    // rejected, not silently produce a near-empty or empty result.
    HydroRunConfig outOfRange = config;
    outOfRange.regional_window_start = "2030-01-01T00:00:00Z";
    outOfRange.regional_window_end = "2030-02-01T00:00:00Z";
    bool outOfRangeRejected = false;
    try { (void)loadRegionalBasinInput(packageDir.string(), outOfRange); }
    catch (const std::runtime_error&) { outOfRangeRejected = true; }
    assert(outOfRangeRejected);

    // Only one of the two window fields set must be rejected - both or
    // neither, never a silently one-sided bound.
    HydroRunConfig halfSet = config;
    halfSet.regional_window_end = "";
    bool halfSetRejected = false;
    try { (void)loadRegionalBasinInput(packageDir.string(), halfSet); }
    catch (const std::runtime_error&) { halfSetRejected = true; }
    assert(halfSetRejected);

    fs::remove_all(root);
}

}  // namespace

int main() {
    testPooledTrainingTensorLevel();
    testPooledScalerRejectsOutOfDomainValidationBasin();
    testPooledTargetScalerIndependentOfValidationData();
    testPooledTargetScalerMatchesConcatenatedReference();
    testPooledTrainingRejectsMalformedInput();
    testRegionalWindowFilterExactRowCount();
    testRegionalPretrainIntegrationFromRealPackages();
    std::printf("lstm_regional_pretrain_test: all assertions passed\n");
    return 0;
}
