#!/usr/bin/env bash
# Bounded launcher for the 9 phase-1 multiscale history-length selection jobs
# (improvement_loop_20261023, Task 4 preflight -> Task 5's gated launch).
#
# Enforces, per job: a wall-clock timeout and a virtual-memory ceiling, plus
# a capped number of concurrent workers overall. Stops DISPATCHING NEW jobs
# as soon as any job fails (nonzero exit, timeout, or missing expected
# output) -- it never silently retries, replaces, or skips a failed job, and
# never broadens the grid. Whatever partial output a failed job produced is
# left untouched for inspection.
#
# This script performs no training by being read, sourced, or syntax-checked.
# It must be invoked explicitly, and only after the 27-job budget this phase
# belongs to has been separately approved by the user -- not by this script.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <phase1_selection_manifest_dir> <output_root> [max_workers]" >&2
  exit 2
fi
MANIFESTS_DIR="$1"
OUTPUT_ROOT="$2"
MAX_WORKERS="${3:-4}"
HYDROBATCH_BIN="${HYDROBATCH_BIN:-$ROOT/build-hydrobatch/HydroBatch}"

# Per-job resource limits -- generous safety nets against a runaway/hung
# job, not tuned expected-usage bounds (this architecture is small: 24h
# sequence, hidden=32, 1 layer, D-day daily branch, hidden=4; each process
# already pins torch to 1 internal thread, hydro_batch_runner.cpp).
PER_JOB_TIMEOUT_SECONDS="${PER_JOB_TIMEOUT_SECONDS:-14400}"   # 4 hours
PER_JOB_MEMORY_LIMIT_KB="${PER_JOB_MEMORY_LIMIT_KB:-8388608}" # 8 GiB virtual memory

if [[ ! -x "$HYDROBATCH_BIN" ]]; then
  echo "HydroBatch binary not found or not executable: $HYDROBATCH_BIN" >&2
  exit 2
fi
if [[ ! -d "$MANIFESTS_DIR" ]]; then
  echo "Manifest directory not found: $MANIFESTS_DIR" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"
if [[ -n "$(ls -A "$OUTPUT_ROOT" 2>/dev/null)" ]]; then
  echo "Refusing to reuse a non-empty output root (avoid clobbering/duplicating a batch): $OUTPUT_ROOT" >&2
  exit 2
fi

shopt -s nullglob
BATCH_FILES=("$MANIFESTS_DIR"/*.batch)
shopt -u nullglob
if [[ ${#BATCH_FILES[@]} -eq 0 ]]; then
  echo "No .batch files found in $MANIFESTS_DIR" >&2
  exit 2
fi

{
  echo "launched_utc=$(date -u +%FT%TZ)"
  echo "repo_commit=$(git -C "$ROOT" rev-parse HEAD)"
  echo "hydrobatch_bin=$HYDROBATCH_BIN"
  echo "hydrobatch_sha256=$(sha256sum "$HYDROBATCH_BIN" | cut -d' ' -f1)"
  echo "manifests_dir=$MANIFESTS_DIR"
  echo "max_workers=$MAX_WORKERS"
  echo "per_job_timeout_seconds=$PER_JOB_TIMEOUT_SECONDS"
  echo "per_job_memory_limit_kb=$PER_JOB_MEMORY_LIMIT_KB"
  echo "job_count=${#BATCH_FILES[@]}"
} | tee "$OUTPUT_ROOT/batch_launch_metadata.txt"

FAILED_SENTINEL="$OUTPUT_ROOT/.failed"

run_one_job() {
  local batch_file="$1"
  local name job_out status
  name="$(basename "$batch_file" .batch)"
  job_out="$OUTPUT_ROOT/$name"
  mkdir -p "$job_out"
  # Written as the condition of an `if` so `set -e` (errexit) does not abort
  # this function the instant the inner command fails -- a failure here is
  # an expected, handled outcome, not a script bug. A bare command whose
  # status is captured via `status=$?` on the next line would instead kill
  # the function before that line ever ran (confirmed by actually testing
  # the failure path against a fake stub, not by reasoning about it).
  if (
    ulimit -v "$PER_JOB_MEMORY_LIMIT_KB"
    timeout "$PER_JOB_TIMEOUT_SECONDS" "$HYDROBATCH_BIN" "$batch_file" "$job_out"
  ) > "$job_out/run.log" 2>&1; then
    status=0
  else
    status=$?
  fi
  echo "$status" > "$job_out/exit_status.txt"
  if [[ "$status" -ne 0 ]]; then
    echo "FAILED (exit $status): $name -- see $job_out/run.log" >&2
    touch "$FAILED_SENTINEL"
    return 1
  fi
  if [[ ! -f "$job_out/experiment_config.json" || ! -f "$job_out/predictions.csv" || ! -f "$job_out/artifact_manifest.csv" ]]; then
    echo "FAILED (missing expected output despite exit 0): $name" >&2
    echo "missing_expected_output" >> "$job_out/exit_status.txt"
    touch "$FAILED_SENTINEL"
    return 1
  fi
  echo "OK: $name"
}

overall_status=0
for batch_file in "${BATCH_FILES[@]}"; do
  if [[ -f "$FAILED_SENTINEL" ]]; then
    echo "Stopping dispatch: a prior job already failed. Not launching $batch_file." >&2
    overall_status=1
    break
  fi
  while [[ "$(jobs -rp | wc -l)" -ge "$MAX_WORKERS" ]]; do
    wait -n || true
  done
  run_one_job "$batch_file" &
done
wait

if [[ -f "$FAILED_SENTINEL" ]]; then
  overall_status=1
fi

echo "finished_utc=$(date -u +%FT%TZ)" >> "$OUTPUT_ROOT/batch_launch_metadata.txt"
if [[ "$overall_status" -ne 0 ]]; then
  echo "One or more selection jobs failed; stopped without retrying/replacing/broadening the grid. See $OUTPUT_ROOT/*/run.log" >&2
  exit 1
fi
echo "All ${#BATCH_FILES[@]} selection jobs completed successfully."
