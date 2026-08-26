#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
binary="${TMPDIR:-/tmp}/gistohq_model_rows_test"
"${CXX:-c++}" -std=c++17 -O2 -Wall -Wextra -Werror \
  "$repo_root/Hydro/tests/gistohq_model_rows_test.cpp" \
  "$repo_root/Hydro/dataset/gistohq_model_rows.cpp" \
  -o "$binary"
"$binary"
