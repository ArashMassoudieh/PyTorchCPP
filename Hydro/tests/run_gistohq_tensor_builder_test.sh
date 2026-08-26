#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
libtorch="${LIBTORCH_PATH:-/usr/local/libtorch}"
if [[ ! -f "$libtorch/include/torch/csrc/api/include/torch/torch.h" ]]; then
  echo "SKIP: LibTorch headers not found; set LIBTORCH_PATH to run gistohq_tensor_builder_test."
  if [[ "${HYDRO_REQUIRE_LIBTORCH_TESTS:-0}" == "1" ]]; then exit 1; fi
  exit 0
fi
binary="${TMPDIR:-/tmp}/gistohq_tensor_builder_test"
"${CXX:-c++}" -std=c++17 -O0 -Wall -Wextra -Werror -D_GLIBCXX_USE_CXX11_ABI="${TORCH_CXX11_ABI:-1}" \
  -I"$repo_root" -I"$libtorch/include" -I"$libtorch/include/torch/csrc/api/include" \
  "$repo_root/Hydro/tests/gistohq_tensor_builder_test.cpp" \
  -L"$libtorch/lib" -Wl,-rpath,"$libtorch/lib" -ltorch -ltorch_cpu -lc10 -lpthread -o "$binary"
"$binary"
