#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
binary="${TMPDIR:-/tmp}/hydropinn_dataset_contract_test"
"${CXX:-c++}" -std=c++17 -Wall -Wextra -Werror -I"$repo_root/Hydro" \
  "$repo_root/Hydro/tests/dataset_contract_test.cpp" \
  "$repo_root/Hydro/dataset/hydro_dataset_contract.cpp" -o "$binary"
"$binary"
