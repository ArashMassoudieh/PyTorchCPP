#!/usr/bin/env bash
# Builds and runs multiscale_daily_history_test (improvement_loop_20261023
# reports 66-68). Header-only new code (multiscale_lstm_module.h,
# daily_history_builder.h) plus the existing hydro_lstm_module.h -- no
# dataset/package loaders needed, so this has a much smaller dependency
# list than the shared-inference verifier scripts.
#
# Usage: Hydro/tests/build_multiscale_daily_history_test.sh [output_path]
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
libtorch="${LIBTORCH_PATH:-/mnt/3rd900/Projects/libtorch}"
output="${1:-/tmp/multiscale_daily_history_test}"

"${CXX:-g++}" -pipe -fopenmp -D_GLIBCXX_USE_CXX11_ABI=1 -O2 -std=gnu++1z -D_REENTRANT -fPIC \
  -DTORCH_SUPPORT \
  -I"$repo_root" -I"$libtorch/include/torch/csrc/api/include" -I"$libtorch/include" \
  -I"$repo_root/Hydro" -I"$repo_root/Hydro/dataset" -I"$repo_root/Hydro/models" \
  "$repo_root/Hydro/tests/multiscale_daily_history_test.cpp" \
  -L"$libtorch/lib" -ltorch -ltorch_cpu -lc10 -lgomp -lpthread \
  -Wl,-rpath,"$libtorch/lib" \
  -o "$output"

echo "Built: $output"
"$output"
