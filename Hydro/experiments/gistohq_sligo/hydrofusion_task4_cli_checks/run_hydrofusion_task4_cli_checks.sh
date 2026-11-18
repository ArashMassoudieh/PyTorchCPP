#!/usr/bin/env bash
# Repeatable CLI checks for HydroFusion Task 4's "ffn_hydrofusion" dispatch
# path, run through a real HydroBatch binary (not source review). Lives
# under Hydro/experiments/gistohq_sligo/ (not under any /.scratch_*/
# directory, which .gitignore excludes) specifically so a clean checkout
# of this repo has these configs/batch files available, not just whoever
# happened to create them in an ignored scratch dir.
#
# Checks:
#   1. ffn_hydrofusion, hydrofusion_target_mode=log_residual   -> expect success
#   2. ffn_hydrofusion, hydrofusion_target_mode=direct_log_q   -> expect success
#   3. ffn_hydrofusion, use_hydro_package=true (bogus path)    -> expect failure w/ guard message
#   4. ffn_hydrofusion, use_csv_data=true (bogus path)         -> expect failure w/ guard message
#   5. ffn/lstm/pinn/ffn_pinn/lstm_pinn regression smoke       -> expect all 5 success
#
# This script does NOT prove the regression jobs are unchanged relative to
# a prior build -- that requires comparing raw predictions.csv rows against
# a second binary, which compare_regression_against_baseline_build.sh does.
#
# Usage: run_hydrofusion_task4_cli_checks.sh <path-to-HydroBatch-binary> [output-root]
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
binary="${1:-}"
out_root="${2:-}"

if [ -z "$binary" ] || [ ! -x "$binary" ]; then
  echo "Usage: $0 <path-to-HydroBatch-binary> [output-root]" >&2
  exit 2
fi

binary="$(realpath -e -- "$binary")"
binary_hash="$(sha256sum -- "$binary")"
if [ -n "$out_root" ]; then
  mkdir -- "$out_root" || { echo "Output must be a NEW directory: $out_root" >&2; exit 2; }
else
  out_root="$(mktemp -d "${TMPDIR:-/tmp}/hydrofusion_cli_checks.XXXXXXXX")"
fi
out_root="$(realpath -e -- "$out_root")"
printf 'binary_path=%s\nbinary_sha256=%s\n' "$binary" "${binary_hash:0:64}" > "$out_root/binary_hashes.txt"
echo "Evidence directory: $out_root"

pass=0
fail=0

expect_success() {
  local name="$1" batch="$2"
  local out="$out_root/$name"
  if "$binary" "$here/$batch" "$out" > "$out_root/$name.log" 2>&1; then
    echo "PASS: $name (expected success, got success)"
    pass=$((pass + 1))
  else
    echo "FAIL: $name (expected success, got failure) -- see $out_root/$name.log"
    fail=$((fail + 1))
  fi
}

expect_failure_with_message() {
  local name="$1" batch="$2" needle="$3"
  local out="$out_root/$name"
  if "$binary" "$here/$batch" "$out" > "$out_root/$name.log" 2>&1; then
    echo "FAIL: $name (expected failure, got success) -- see $out_root/$name.log"
    fail=$((fail + 1))
  elif grep -qF "$needle" "$out_root/$name.log"; then
    echo "PASS: $name (expected failure with guard message, got it)"
    pass=$((pass + 1))
  else
    echo "FAIL: $name (failed, but not with the expected guard message) -- see $out_root/$name.log"
    fail=$((fail + 1))
  fi
}

expect_success           "log_residual"        "jobs.batch"
expect_success           "direct_log_q"         "jobs_direct_log_q.batch"
expect_failure_with_message "guard_hydro_package" "jobs_guard_test.batch" \
  "this mode is synthetic-fixture-only for now"
expect_failure_with_message "guard_csv_data"      "jobs_guard_test_csv.batch" \
  "this mode is synthetic-fixture-only for now"
expect_failure_with_message "missing_optin" "jobs_missing_optin.batch"   "explicit hydrofusion_synthetic_verification=true is required"
expect_failure_with_message "invalid_mode" "jobs_invalid_mode.batch"   "hydrofusion_target_mode must be"
expect_failure_with_message "missing_cutoff" "jobs_missing_cutoff.batch"   "must all be set explicitly"
expect_failure_with_message "fractional_cutoff" "jobs_fractional_cutoff.batch"   "cutoff must be an exact int64"
expect_failure_with_message "overflow_cutoff" "jobs_overflow_cutoff.batch"   "cutoff must be an exact int64"
expect_success           "regression_5method"  "jobs_regression.batch"

echo
echo "=== $pass passed, $fail failed ==="
[ "$fail" -eq 0 ]
