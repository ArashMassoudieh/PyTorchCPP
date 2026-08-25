#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
libtorch="${LIBTORCH_PATH:-/usr/local/libtorch}"
if [[ ! -f "$libtorch/include/torch/csrc/api/include/torch/torch.h" ]]; then
  echo "SKIP: LibTorch headers not found; set LIBTORCH_PATH to run tensor_scaler_test." >&2
  if [[ "${HYDRO_REQUIRE_LIBTORCH_TESTS:-0}" == "1" ]]; then exit 1; fi
  exit 77
fi
binary="${TMPDIR:-/tmp}/hydropinn_tensor_scaler_test"
"${CXX:-c++}" -std=c++17 -Wall -Wextra \
  -I"$repo_root/Hydro" -I"$libtorch/include" -I"$libtorch/include/torch/csrc/api/include" \
  "$repo_root/Hydro/tests/tensor_scaler_test.cpp" -L"$libtorch/lib" \
  -Wl,-rpath,"$libtorch/lib" -ltorch -ltorch_cpu -lc10 -o "$binary"
"$binary"
