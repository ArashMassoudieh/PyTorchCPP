#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
binary="${TMPDIR:-/tmp}/gistohq_temporal_csv_test"
"${CXX:-c++}" -std=c++17 -O2 -Wall -Wextra -Werror \
  "$repo_root/Hydro/tests/gistohq_temporal_csv_test.cpp" \
  "$repo_root/Hydro/dataset/gistohq_temporal_csv.cpp" \
  -o "$binary"
"$binary"
