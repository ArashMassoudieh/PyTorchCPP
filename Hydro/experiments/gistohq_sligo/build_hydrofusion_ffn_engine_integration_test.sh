#!/usr/bin/env bash
# Builds hydrofusion_ffn_engine_integration_test (HydroFusion Task 4, real-
# engine half: a fitted NeuralNetworkWrapper exported/reloaded through the
# unmodified HydroExperimentExporter/HydroArtifactLoader/HydroInferenceSession
# path). Dependency list copied verbatim from the already-working
# build_synthetic_multiscale_smoke_test.sh in this same directory -- same
# libtorch/Qt/armadillo link requirements, just a different main() source
# file, since this test links against the same evaluation/dataset/models
# translation units (most are pulled in transitively by experiment_exporter.cpp/
# inference_runner.cpp even though this test only exercises the "ffn" approach).
#
# Usage: Hydro/experiments/gistohq_sligo/build_hydrofusion_ffn_engine_integration_test.sh [output_path]
#   Then run the resulting binary directly -- it builds its own synthetic
#   fixture, fits a small feedforward net on it, exports/reloads through the
#   real engine, and exits nonzero on any checkpoint-reload prediction
#   mismatch or rejected-input-width failure.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
libtorch="${LIBTORCH_PATH:-/mnt/3rd900/Projects/libtorch}"
output="${1:-/tmp/hydrofusion_ffn_engine_integration_test}"

if [ -z "${HYDROFUSION_FROZEN_SOURCE_ROOT:-}" ]; then
  exec python3 "$repo_root/Hydro/experiments/gistohq_sligo/build_hydrofusion_frozen.py" "$(basename "$0")" "$output"
fi
repo_root="$HYDROFUSION_FROZEN_SOURCE_ROOT"

python3 "$repo_root/Hydro/generate_reduced_reservoir_supervised_wrappers.py"
for f in "$repo_root/Hydro/generated/ffn_wrapper_reduced_reservoir.cpp" \
         "$repo_root/Hydro/generated/lstmnetworkwrapper_reduced_reservoir.cpp"; do
  if [ ! -f "$f" ]; then
    echo "Generator ran but did not produce $f" >&2
    exit 2
  fi
done

"${CXX:-g++}" -pipe -fopenmp -D_GLIBCXX_USE_CXX11_ABI=1 -O2 -std=gnu++1z -D_REENTRANT -fPIC \
  -DTORCH_SUPPORT -D_arma -DARMA_USE_OPENMP \
  -I"$repo_root" -I"$libtorch/include/torch/csrc/api/include" -I"$libtorch/include" \
  -I"$repo_root/Utilities" -I"$repo_root/Hydro" -I"$repo_root/Hydro/dataset" \
  -I"$repo_root/Hydro/models" -I"$repo_root/Hydro/physics" \
  -I/usr/include/x86_64-linux-gnu/qt5 -I/usr/include/x86_64-linux-gnu/qt5/QtCore \
  "$repo_root/Hydro/experiments/gistohq_sligo/hydrofusion_ffn_engine_integration_test.cpp" \
  "$repo_root/Hydro/evaluation/inference_runner.cpp" \
  "$repo_root/Hydro/evaluation/shared_scale_inference.cpp" \
  "$repo_root/Hydro/evaluation/experiment_loader.cpp" \
  "$repo_root/Hydro/evaluation/experiment_exporter.cpp" \
  "$repo_root/Hydro/evaluation/artifact_loader.cpp" \
  "$repo_root/Hydro/dataset/ddrr_loader.cpp" \
  "$repo_root/Hydro/dataset/gistohq_hourly_harmonizer.cpp" \
  "$repo_root/Hydro/dataset/gistohq_package_adapter.cpp" \
  "$repo_root/Hydro/dataset/gistohq_model_rows.cpp" \
  "$repo_root/Hydro/dataset/gistohq_temporal_csv.cpp" \
  "$repo_root/Hydro/dataset/hydro_checksum.cpp" \
  "$repo_root/Hydro/dataset/hydro_dataset_contract.cpp" \
  "$repo_root/Hydro/dataset/lag_builder.cpp" \
  "$repo_root/Hydro/dataset/sequence_builder.cpp" \
  "$repo_root/Hydro/models/lstm_wrapper.cpp" \
  "$repo_root/Hydro/models/lstm_pinn_wrapper.cpp" \
  "$repo_root/Hydro/models/pinn_wrapper.cpp" \
  "$repo_root/Hydro/models/ffn_pinn_wrapper.cpp" \
  "$repo_root/Hydro/generated/ffn_wrapper_reduced_reservoir.cpp" \
  "$repo_root/Hydro/generated/lstmnetworkwrapper_reduced_reservoir.cpp" \
  "$repo_root/Hydro/physics/physics_config.cpp" \
  "$repo_root/Hydro/physics/rr_physics.cpp" \
  "$repo_root/neuralnetworkwrapper.cpp" "$repo_root/neuralnetworkfactory.cpp" "$repo_root/hyperparameters.cpp" \
  "$repo_root/Utilities/Distribution.cpp" "$repo_root/Utilities/Matrix.cpp" "$repo_root/Utilities/Matrix_arma.cpp" \
  "$repo_root/Utilities/Matrix_arma_sp.cpp" "$repo_root/Utilities/QuickSort.cpp" "$repo_root/Utilities/Utilities.cpp" \
  "$repo_root/Utilities/Vector.cpp" "$repo_root/Utilities/Vector_arma.cpp" \
  -L"$libtorch/lib" -ltorch -ltorch_cpu -lc10 -lgomp -larmadillo -lcrypto -lpthread \
  -L/usr/lib/x86_64-linux-gnu -lQt5Core \
  -Wl,-rpath,"$libtorch/lib" \
  -o "$output"

echo "Built: $output"
