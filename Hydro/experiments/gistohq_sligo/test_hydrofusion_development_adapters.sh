#!/usr/bin/env bash
set -euo pipefail
src=$(cd -- "$(dirname -- "$0")" && pwd)
build=$(mktemp -d "${TMPDIR:-/tmp}/hydrofusion-adapters.XXXXXXXX")
# Own only this freshly-created directory, never caller-provided outputs.
trap 'rm -rf -- "$build"' EXIT
for name in native_support_adapter sequence_builder; do
    "${CXX:-g++}" -std=c++17 -O2 -Wall -Wextra -Werror -fsanitize=undefined,address \
        "$src/test_hydrofusion_${name}.cpp" -o "$build/$name"
    # LeakSanitizer cannot operate under the host's ptrace restrictions.
    ASAN_OPTIONS=detect_leaks=0 "$build/$name"
done
