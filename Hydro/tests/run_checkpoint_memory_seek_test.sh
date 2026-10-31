#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
test_dir="$(mktemp -d "${TMPDIR:-/tmp}/hydro-checkpoint-seek.XXXXXX")"
trap 'rm -rf "$test_dir"' EXIT
"${CXX:-c++}" -std=c++17 -O0 -g -Wall -Wextra -Werror \
  -fsanitize=address,undefined -fno-sanitize-recover=all \
  "$repo_root/Hydro/tests/checkpoint_memory_seek_test.cpp" \
  -o "$test_dir/checkpoint_memory_seek_test"
"$test_dir/checkpoint_memory_seek_test"
