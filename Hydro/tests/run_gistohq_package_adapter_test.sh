#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
binary="${TMPDIR:-/tmp}/gistohq_package_adapter_test"
"${CXX:-c++}" -std=c++17 -Wall -Wextra -Werror -I"$repo_root/Hydro" \
  "$repo_root/Hydro/tests/gistohq_package_adapter_test.cpp" \
  "$repo_root/Hydro/dataset/gistohq_package_adapter.cpp" \
  "$repo_root/Hydro/dataset/gistohq_temporal_csv.cpp" \
  "$repo_root/Hydro/dataset/gistohq_hourly_harmonizer.cpp" \
  "$repo_root/Hydro/dataset/gistohq_model_rows.cpp" -o "$binary"
"$binary"
