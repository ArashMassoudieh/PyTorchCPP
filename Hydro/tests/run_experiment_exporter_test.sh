#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
binary="${TMPDIR:-/tmp}/hydro_experiment_exporter_test"
"${CXX:-c++}" -std=c++17 -Wall -Wextra -Werror -I"$repo_root/Hydro" \
  "$repo_root/Hydro/tests/experiment_exporter_test.cpp" \
  "$repo_root/Hydro/evaluation/experiment_exporter.cpp" \
  "$repo_root/Hydro/evaluation/experiment_loader.cpp" \
  "$repo_root/Hydro/evaluation/artifact_loader.cpp" \
  "$repo_root/Hydro/dataset/hydro_checksum.cpp" -lcrypto -o "$binary"
"$binary"
