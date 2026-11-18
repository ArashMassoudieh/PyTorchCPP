#!/usr/bin/env bash
# Compares RAW per-row predictions.csv output (not just aggregate metrics
# in batch_summary.csv) for the 5 pre-existing methods (ffn/lstm/pinn/
# ffn_pinn/lstm_pinn) between two HydroBatch binaries, using the same
# deterministic (seed=42) synthetic jobs_regression.batch. Aggregate
# metrics matching does not prove the predictions themselves are
# unchanged -- this is the actual proof, row for row.
#
# Typical use: build one HydroBatch from the current working tree and a
# second from a prior commit (e.g. via `git worktree add`), then:
#   compare_regression_against_baseline_build.sh <after-binary> <baseline-binary> [output-root]
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
after_binary="${1:-}"
baseline_binary="${2:-}"
out_root="${3:-}"

if [ -z "$after_binary" ] || [ ! -x "$after_binary" ] || \
   [ -z "$baseline_binary" ] || [ ! -x "$baseline_binary" ]; then
  echo "Usage: $0 <after-HydroBatch-binary> <baseline-HydroBatch-binary> [output-root]" >&2
  exit 2
fi

# Hash inputs before creating any output. Never delete a caller's directory.
after_binary="$(realpath -e -- "$after_binary")"
baseline_binary="$(realpath -e -- "$baseline_binary")"
after_hash="$(sha256sum -- "$after_binary")"
baseline_hash="$(sha256sum -- "$baseline_binary")"
if [ -n "$out_root" ]; then
  mkdir -- "$out_root" || { echo "Output must be a NEW directory: $out_root" >&2; exit 2; }
else
  out_root="$(mktemp -d "${TMPDIR:-/tmp}/hydrofusion_regression.XXXXXXXX")"
fi
out_root="$(realpath -e -- "$out_root")"
after_out="$out_root/after"
baseline_out="$out_root/baseline"
mkdir -- "$after_out" "$baseline_out"
{
  printf 'after_binary_path=%s\nafter_binary_sha256=%s\n' "$after_binary" "${after_hash:0:64}"
  printf 'baseline_binary_path=%s\nbaseline_binary_sha256=%s\n' "$baseline_binary" "${baseline_hash:0:64}"
} | tee "$out_root/binary_hashes.txt"
echo "Evidence directory: $out_root"

echo "Running regression batch through AFTER binary: $after_binary"
after_exit=0
"$after_binary" "$here/jobs_regression.batch" "$after_out" > "$out_root/after.log" 2>&1 || after_exit=$?

echo "Running regression batch through BASELINE binary: $baseline_binary"
baseline_exit=0
"$baseline_binary" "$here/jobs_regression.batch" "$baseline_out" > "$out_root/baseline.log" 2>&1 || baseline_exit=$?

if [ "$after_exit" -ne 0 ] || [ "$baseline_exit" -ne 0 ]; then
  echo "FAIL: at least one binary did not complete all jobs successfully (after_exit=$after_exit baseline_exit=$baseline_exit)"
  echo "See $out_root/after.log and $out_root/baseline.log"
  exit 1
fi

modes=(ffn lstm pinn ffn_pinn lstm_pinn)
pass=0
fail=0
for mode in "${modes[@]}"; do
  exp="hydrofusion_regression_${mode}"
  a="$after_out/$exp/predictions.csv"
  b="$baseline_out/$exp/predictions.csv"
  if [ ! -f "$a" ] || [ ! -f "$b" ]; then
    echo "FAIL: $mode -- missing predictions.csv (a exists: $([ -f "$a" ] && echo yes || echo no), b exists: $([ -f "$b" ] && echo yes || echo no))"
    fail=$((fail + 1))
    continue
  fi
  # Two identical empty files are not evidence that either method ran.
  if [ "$(head -n 1 "$a")" != "approach,index,split,x,observed,predicted,residual" ] ||
     [ "$(wc -l < "$a")" -lt 2 ] || [ "$(wc -l < "$b")" -lt 2 ]; then
    echo "FAIL: $mode -- missing prediction header or data rows"
    fail=$((fail + 1))
    continue
  fi
  if diff -q "$a" "$b" > /dev/null; then
    rows=$(($(wc -l < "$a") - 1))
    echo "PASS: $mode -- predictions.csv byte-identical across $rows rows"
    pass=$((pass + 1))
  else
    echo "FAIL: $mode -- predictions.csv differs"
    diff "$a" "$b" | head -10 || true
    fail=$((fail + 1))
  fi
done

echo
echo "=== $pass passed, $fail failed ==="
[ "$fail" -eq 0 ]
