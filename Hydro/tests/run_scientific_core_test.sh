#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
binary="${TMPDIR:-/tmp}/hydropinn_scientific_core_test"
"${CXX:-c++}" -std=c++17 -Wall -Wextra -Werror \
  -I"$repo_root/Hydro" \
  "$repo_root/Hydro/tests/scientific_core_test.cpp" -o "$binary"
"$binary"
