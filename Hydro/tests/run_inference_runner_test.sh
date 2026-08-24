#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
libtorch="${LIBTORCH_PATH:-/usr/local/libtorch}"
if [[ ! -f "$libtorch/include/torch/csrc/api/include/torch/torch.h" ]]; then
  echo "SKIP: LibTorch headers not found; set LIBTORCH_PATH to run inference_runner_test."
  exit 0
fi
binary="${TMPDIR:-/tmp}/hydro_inference_runner_test"
"${CXX:-c++}" -std=c++17 -O0 -Wall -Wextra -Werror -fopenmp -D_GLIBCXX_USE_CXX11_ABI="${TORCH_CXX11_ABI:-1}" \
  -I"$repo_root" -I"$repo_root/Hydro" -I"$repo_root/Utilities" \
  -I"$libtorch/include" -I"$libtorch/include/torch/csrc/api/include" \
  "$repo_root/Hydro/tests/inference_runner_test.cpp" \
  "$repo_root/Hydro/evaluation/inference_runner.cpp" \
  "$repo_root/Hydro/evaluation/artifact_loader.cpp" \
  "$repo_root/Hydro/evaluation/experiment_exporter.cpp" \
  "$repo_root/Hydro/evaluation/experiment_loader.cpp" \
  "$repo_root/Hydro/dataset/hydro_checksum.cpp" \
  "$repo_root/neuralnetworkwrapper.cpp" "$repo_root/neuralnetworkfactory.cpp" "$repo_root/hyperparameters.cpp" \
  "$repo_root/Utilities/Distribution.cpp" "$repo_root/Utilities/Matrix.cpp" \
  "$repo_root/Utilities/Matrix_arma.cpp" "$repo_root/Utilities/Matrix_arma_sp.cpp" \
  "$repo_root/Utilities/QuickSort.cpp" "$repo_root/Utilities/Utilities.cpp" \
  "$repo_root/Utilities/Vector.cpp" "$repo_root/Utilities/Vector_arma.cpp" \
  -L"$libtorch/lib" -Wl,-rpath,"$libtorch/lib" -ltorch -ltorch_cpu -lc10 -larmadillo -lcrypto -lgomp -lpthread -o "$binary"
"$binary"
