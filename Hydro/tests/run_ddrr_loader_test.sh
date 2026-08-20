#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
binary="${TMPDIR:-/tmp}/hydro_ddrr_loader_test"
"${CXX:-c++}" -std=c++17 -Wall -Wextra -Werror -I"$repo_root/Hydro" \
  "$repo_root/Hydro/tests/ddrr_loader_test.cpp" \
  "$repo_root/Hydro/dataset/ddrr_loader.cpp" \
  "$repo_root/Hydro/dataset/hydro_dataset_contract.cpp" \
  "$repo_root/Hydro/dataset/hydro_checksum.cpp" -lcrypto -o "$binary"
"$binary"
