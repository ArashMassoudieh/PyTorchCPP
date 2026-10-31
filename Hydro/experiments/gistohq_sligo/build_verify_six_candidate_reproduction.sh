#!/usr/bin/env bash
# Builds verify_six_candidate_reproduction from a clean checkout, listing every
# dependency this session's trial-and-error found necessary (two earlier build
# attempts failed on undefined references to HydroInferenceRunner::
# runSharedScaleAblation, HydroArtifactLoader::loadScalers/loadModels,
# FFNPINNWrapper::train, LSTMNetworkWrapper::train, and qt_version_tag, each
# fixed by adding exactly the source file the linker named -- the list below
# is the result of that process, not a guess).
#
# Requires libtorch at $LIBTORCH_PATH (default: /mnt/3rd900/Projects/libtorch)
# and the Qt5 Core headers (for a QString symbol pulled in transitively via
# commontypes.h) -- both match this project's other build scripts/HydroPINN.pro.
#
# Submodule dependency: Utilities/ (Distribution.cpp, Matrix*.cpp, Vector*.cpp,
# TimeSeries.h, TimeSeriesSet.h, etc.) is a git submodule (.gitmodules), not
# plain tracked content. A bare `git clone` or `git worktree add` does NOT
# populate it -- confirmed by a clean-worktree test where every Utilities/*.cpp
# path was simply missing until `git submodule update --init` was run. Any
# genuinely clean checkout needs that step (or `git clone --recurse-submodules`)
# before this script, not just this script's own generator call below.
#
# Generated-wrapper dependency: Hydro/generated/ffn_wrapper_reduced_reservoir.cpp
# and Hydro/generated/lstmnetworkwrapper_reduced_reservoir.cpp are listed below
# as compile inputs but are NOT tracked in git (see Hydro/generated/.gitignore)
# -- in a genuinely clean checkout they do not exist until generated. HydroPINN.pro
# produces them as a qmake-time side effect (system(python3 $$REDUCED_RESERVOIR_GENERATOR),
# HydroPINN.pro lines ~88-91), which is why a prior `qmake && make` masked this
# dependency in earlier manual testing. This script no longer relies on that:
# it invokes the generator itself, below, so it is correct from a bare clean
# checkout with no qmake step at all.
#
# Usage: Hydro/experiments/gistohq_sligo/build_verify_six_candidate_reproduction.sh [output_path]
#   Then run the resulting binary directly:
#     <output_path> lstm_pinn <job_dir> [<job_dir> ...]
#   where each <job_dir> is an already-completed, frozen shared-scale-ablation
#   job directory (experiment_config.json, models/<approach>.pt, scalers.csv,
#   predictions.csv, artifact_manifest.csv). Paths are never hardcoded here --
#   every frozen job lives outside this repository.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
libtorch="${LIBTORCH_PATH:-/mnt/3rd900/Projects/libtorch}"
output="${1:-/tmp/verify_six_candidate_reproduction}"

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
  "$repo_root/Hydro/experiments/gistohq_sligo/verify_six_candidate_reproduction.cpp" \
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
echo "Run:   $output lstm_pinn <job_dir> [<job_dir> ...]"
