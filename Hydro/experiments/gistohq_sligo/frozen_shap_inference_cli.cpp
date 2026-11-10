// Frozen-inference CLI bridge for SHAP/attribution work (improvement_loop_20261023,
// Task: SHAP feature attribution -- see CLAUDE_DIAGNOSTIC_FIGURES_TASKS_20261105.md).
//
// Two subcommands, both operating on an already-trained, already-verified checkpoint
// directory (no training happens here):
//
//   dump --experiment-dir DIR --output CSV [--max-x N]
//     Writes the frozen checkpoint's own underlying physical (unscaled) feature series
//     -- x (elapsed_hours), the 6 stable-order features (P,T,RH,wind,solar,PET),
//     segment_id, target, target_valid -- using the EXACT SAME loading call
//     (prepareGisToOhqPackage) that hydro_tensor_builder.h uses for training, so the
//     ground truth this CLI reports can never silently diverge from what the model
//     -- was actually trained/evaluated against. Used by the Python driver to build real
//     background and explanation windows and to parity-check against predictions.csv.
//
//   predict --experiment-dir DIR --approach APPROACH --input CSV --output CSV
//     Reads rows of row_id + 144 flattened physical values (24 hours x 6 features,
//     hour-major: h0f0..h0f5,h1f0..h1f5,...,h23f0..h23f5), runs them through
//     HydroInferenceSession::predict (the same frozen-checkpoint bare forward-pass
//     engine already used elsewhere in this program), applies clamp_min(0.0) to match
//     the training backend's own post-inverse-transform clamp (lstmnetworkwrapper.cpp:
//     "predFull = targetScaler.inverseTransform(...).clamp_min(0.0)" -- HydroInference-
//     Session::predict does NOT apply this clamp itself, confirmed by direct source
//     reading; skipping it here would silently diverge from the real production output
//     exactly where it matters most, at small/negative raw predictions), and writes
//     row_id,prediction.
//
// Refuses reserved-period access (x >= 77016) in both modes. No training. No gradient
// computation (NoGradGuard, forward-only).
#include "../evaluation/artifact_loader.h"
#include "../evaluation/inference_runner.h"
#include "../dataset/gistohq_package_adapter.h"
#include "../dataset/hydro_package_directory.h"

#include <torch/torch.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kReservedPeriodXCutoff = 77016.0;

std::string requireArg(const std::vector<std::string>& args, const std::string& flag) {
    for (std::size_t i = 0; i + 1 < args.size(); ++i) {
        if (args[i] == flag) return args[i + 1];
    }
    throw std::runtime_error("Missing required argument: " + flag);
}

bool hasArg(const std::vector<std::string>& args, const std::string& flag) {
    return std::find(args.begin(), args.end(), flag) != args.end();
}

std::vector<std::string> splitCsvLine(const std::string& line) {
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string field;
    while (std::getline(ss, field, ',')) out.push_back(field);
    return out;
}

int runDump(const std::vector<std::string>& args) {
    const std::string experimentDir = requireArg(args, "--experiment-dir");
    const std::string outputPath = requireArg(args, "--output");
    const bool hasMax = hasArg(args, "--max-x");
    const double maxX = hasMax ? std::stod(requireArg(args, "--max-x")) : 1e300;

    std::ifstream exists(outputPath);
    if (exists.good()) throw std::runtime_error("Refusing to overwrite existing output: " + outputPath);

    HydroArtifactLoader loader;
    const auto artifacts = loader.loadForInference(experimentDir);
    const auto packageRoot = resolveHydroPackageDirectory(artifacts.experiment.config.hydro_package_path);
    const auto prepared = prepareGisToOhqPackage(packageRoot, true);
    if (prepared.model_rows.empty()) throw std::runtime_error("Package has no usable model rows.");

    std::ofstream out(outputPath);
    out << "x,segment_id,feature0,feature1,feature2,feature3,feature4,feature5,target,target_valid\n";
    std::size_t written = 0;
    for (const auto& row : prepared.model_rows) {
        if (row.elapsed_hours >= kReservedPeriodXCutoff) continue;
        if (row.elapsed_hours >= maxX) continue;
        out << row.elapsed_hours << ',' << row.segment_id;
        for (const auto value : row.features) out << ',' << value;
        out << ',' << row.target_runoff_mm_per_hour << ',' << (row.target_valid ? 1 : 0) << '\n';
        ++written;
    }
    std::cout << "wrote " << written << " rows to " << outputPath << "\n";
    return 0;
}

int runPredict(const std::vector<std::string>& args) {
    const std::string experimentDir = requireArg(args, "--experiment-dir");
    const std::string approach = requireArg(args, "--approach");
    const std::string inputPath = requireArg(args, "--input");
    const std::string outputPath = requireArg(args, "--output");

    std::ifstream exists(outputPath);
    if (exists.good()) throw std::runtime_error("Refusing to overwrite existing output: " + outputPath);

    HydroArtifactLoader loader;
    const auto artifacts = loader.loadForInference(experimentDir);
    HydroInferenceSession session(artifacts, approach);

    std::ifstream in(inputPath);
    if (!in.good()) throw std::runtime_error("Cannot open input: " + inputPath);
    std::string header;
    std::getline(in, header);

    std::vector<std::string> rowIds;
    std::vector<std::vector<float>> windows;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const auto fields = splitCsvLine(line);
        if (fields.size() != 1 + 24 * 6) {
            throw std::runtime_error("Expected row_id + 144 values, got " + std::to_string(fields.size()) +
                                     " fields.");
        }
        rowIds.push_back(fields[0]);
        std::vector<float> values;
        values.reserve(144);
        for (std::size_t i = 1; i < fields.size(); ++i) values.push_back(std::stof(fields[i]));
        windows.push_back(std::move(values));
    }
    if (windows.empty()) throw std::runtime_error("Input contains no prediction windows.");

    const auto n = static_cast<int64_t>(windows.size());
    torch::Tensor batch = torch::empty({n, 24, 6}, torch::kFloat32);
    auto accessor = batch.accessor<float, 3>();
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t h = 0; h < 24; ++h) {
            for (int64_t f = 0; f < 6; ++f) {
                accessor[i][h][f] = windows[static_cast<std::size_t>(i)][static_cast<std::size_t>(h * 6 + f)];
            }
        }
    }

    const torch::Tensor predictions = session.predict(batch).clamp_min(0.0);
    if (predictions.size(0) != n) throw std::runtime_error("Prediction count mismatch.");

    std::ofstream out(outputPath);
    out << "row_id,prediction\n";
    for (int64_t i = 0; i < n; ++i) {
        out << rowIds[static_cast<std::size_t>(i)] << ',' << predictions[i][0].item<double>() << '\n';
    }
    std::cout << "wrote " << n << " predictions to " << outputPath << "\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) throw std::runtime_error("Usage: frozen_shap_inference_cli <dump|predict> [args...]");
        const std::string mode = argv[1];
        std::vector<std::string> args(argv + 2, argv + argc);
        if (mode == "dump") return runDump(args);
        if (mode == "predict") return runPredict(args);
        throw std::runtime_error("Unknown mode: " + mode);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
