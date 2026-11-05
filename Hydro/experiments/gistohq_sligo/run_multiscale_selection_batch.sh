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

# MAX_WORKERS must be a positive integer: 0 would make the dispatch loop's
# "while running jobs >= MAX_WORKERS" condition (0 >= 0) true forever,
# hanging before the first job is ever launched; a negative or non-numeric
# value is equally nonsensical here.
if ! [[ "$MAX_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "max_workers must be a positive integer, got: '$MAX_WORKERS'" >&2
  exit 2
fi

# Per-job resource limits -- generous safety nets against a runaway/hung
# job, not tuned expected-usage bounds (this architecture is small: 24h
# sequence, hidden=32, 1 layer, D-day daily branch, hidden=4; each process
# already pins torch to 1 internal thread, hydro_batch_runner.cpp).
PER_JOB_TIMEOUT_SECONDS="${PER_JOB_TIMEOUT_SECONDS:-14400}"   # 4 hours
PER_JOB_MEMORY_LIMIT_KB="${PER_JOB_MEMORY_LIMIT_KB:-8388608}" # 8 GiB virtual memory

# Same validation as MAX_WORKERS: these feed directly into `timeout` and
# `ulimit -v`, both of which accept nonsensical input silently (`timeout 0`
# kills a job instantly with no useful error; `ulimit -v` with a
# non-positive-integer argument either errors opaquely or, worse, is
# accepted and means something other than what was intended).
if ! [[ "$PER_JOB_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "PER_JOB_TIMEOUT_SECONDS must be a positive integer, got: '$PER_JOB_TIMEOUT_SECONDS'" >&2
  exit 2
fi
if ! [[ "$PER_JOB_MEMORY_LIMIT_KB" =~ ^[1-9][0-9]*$ ]]; then
  echo "PER_JOB_MEMORY_LIMIT_KB must be a positive integer, got: '$PER_JOB_MEMORY_LIMIT_KB'" >&2
  exit 2
fi

if [[ ! -x "$HYDROBATCH_BIN" ]]; then
  echo "HydroBatch binary not found or not executable: $HYDROBATCH_BIN" >&2
  exit 2
fi
if [[ ! -d "$MANIFESTS_DIR" ]]; then
  echo "Manifest directory not found: $MANIFESTS_DIR" >&2
  exit 2
fi

ACTUAL_HYDROBATCH_SHA256="$(sha256sum "$HYDROBATCH_BIN" | cut -d' ' -f1)"
# Recording a hash alone does not verify anything; compare it against a
# frozen expected value. EXPECTED_HYDROBATCH_SHA256 takes precedence if set
# directly; otherwise HASH_PIN_JSON (a path to a HASH_PIN.json produced by
# the hash-pinning step) is read for its binary_sha256 field. This check is
# FAIL-CLOSED: a real launch requires one of these two inputs. The only way
# to skip it is UNVERIFIED_HYDROBATCH_TEST_RUN=1, an explicit,
# unmistakably-named, test-only escape hatch (named for what it is, not a
# generic "skip" flag, so it cannot be set by accident or habit) -- intended
# for fake-stub/non-training test invocations like the ones used to verify
# this script's own behavior, never for a real research launch.
EXPECTED_HYDROBATCH_SHA256="${EXPECTED_HYDROBATCH_SHA256:-}"
if [[ -z "$EXPECTED_HYDROBATCH_SHA256" && -n "${HASH_PIN_JSON:-}" ]]; then
  if [[ ! -f "$HASH_PIN_JSON" ]]; then
    echo "HASH_PIN_JSON set but not found: $HASH_PIN_JSON" >&2
    exit 2
  fi
  EXPECTED_HYDROBATCH_SHA256="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['binary_sha256'])" "$HASH_PIN_JSON")"
fi
if [[ -z "$EXPECTED_HYDROBATCH_SHA256" ]]; then
  if [[ "${UNVERIFIED_HYDROBATCH_TEST_RUN:-0}" != "1" ]]; then
    echo "Refusing to launch: no expected HydroBatch hash supplied (set" >&2
    echo "EXPECTED_HYDROBATCH_SHA256 or HASH_PIN_JSON). A real research launch" >&2
    echo "must verify the binary against a frozen expected hash. Set" >&2
    echo "UNVERIFIED_HYDROBATCH_TEST_RUN=1 only for a non-training test" >&2
    echo "invocation (e.g. against a fake stub) -- never for a real launch." >&2
    exit 2
  fi
  echo "WARNING: UNVERIFIED_HYDROBATCH_TEST_RUN=1 -- launching with NO hash" >&2
  echo "verification. This run's output is not provenance-verified and must" >&2
  echo "never be treated as a real research result." >&2
elif [[ "$EXPECTED_HYDROBATCH_SHA256" != "$ACTUAL_HYDROBATCH_SHA256" ]]; then
  echo "HydroBatch binary hash mismatch -- refusing to launch." >&2
  echo "  expected: $EXPECTED_HYDROBATCH_SHA256" >&2
  echo "  actual:   $ACTUAL_HYDROBATCH_SHA256 ($HYDROBATCH_BIN)" >&2
  exit 2
fi

# Hash config/batch manifest inputs against frozen expectations too, not
# only the executable -- a verified binary run against a tampered or
# accidentally-stale manifest is still not a verified result. FAIL-CLOSED,
# same as the binary check above and for the same reason: an "optional"
# inventory check still permits an unverified real launch. The only way to
# skip it is the same UNVERIFIED_HYDROBATCH_TEST_RUN=1 escape hatch used for
# the binary check -- one flag, since a test invocation that skips
# provenance verification skips it for everything, not just the binary.
#
# `sha256sum --check` alone is NOT sufficient here and was replaced after
# being directly disproven: it only verifies that every file *listed* in
# the hash file matches -- it returns exit 0 even when an EXTRA, unlisted
# .batch file sits in MANIFESTS_DIR, because --check never looks at what
# exists on disk beyond its own list. Since this launcher enumerates ALL
# *.batch files via glob (a few lines below), an unlisted extra job would
# still be dispatched even with a passing --check. The verification below
# instead requires the *.batch file set on disk to equal, file for file,
# the .batch entries in MANIFEST_HASHES_FILE (no extra, none missing), and
# additionally hashes each .batch file's own referenced JSON config (not
# just the .batch file itself), since a verified .batch file pointing at a
# tampered config is still not a verified job.
#
# MANIFEST_HASHES_FILE format: a required first line "ROOT: <absolute_dir>"
# declaring the one directory every entry must resolve inside -- this makes
# the permitted input set an explicit, checked property of the file, not an
# assumption, and is what rejects a path-escape entry (e.g. one hashing
# something outside the intended tree via `..` or a symlink). Every
# subsequent non-blank, non-comment line is a standard `sha256sum`-style
# "<hash>  <path>" pair; a bare filename resolves relative to MANIFESTS_DIR
# (for *.batch files), an absolute path is used as-is (for each .batch
# file's own referenced config). Every path is canonicalized with
# Path.resolve() before any comparison, so relative segments and symlinks
# cannot produce two different string spellings of the same file bypassing
# the duplicate/root checks. A duplicate entry (same resolved path listed
# twice, with the same or a conflicting hash) is rejected outright -- an
# ambiguous inventory is not a frozen one.
if [[ -z "${MANIFEST_HASHES_FILE:-}" ]]; then
  if [[ "${UNVERIFIED_HYDROBATCH_TEST_RUN:-0}" != "1" ]]; then
    echo "Refusing to launch: no MANIFEST_HASHES_FILE supplied. A real" >&2
    echo "research launch must verify every .batch file and its referenced" >&2
    echo "config against a frozen expected-hash inventory. Set" >&2
    echo "UNVERIFIED_HYDROBATCH_TEST_RUN=1 only for a non-training test" >&2
    echo "invocation -- never for a real launch." >&2
    exit 2
  fi
  echo "WARNING: UNVERIFIED_HYDROBATCH_TEST_RUN=1 -- launching with NO manifest" >&2
  echo "inventory verification. This run's inputs are not provenance-verified" >&2
  echo "and must never be treated as a real research result." >&2
else
  if [[ ! -f "$MANIFEST_HASHES_FILE" ]]; then
    echo "MANIFEST_HASHES_FILE set but not found: $MANIFEST_HASHES_FILE" >&2
    exit 2
  fi
  if ! python3 - "$MANIFESTS_DIR" "$MANIFEST_HASHES_FILE" <<'PYEOF'
import hashlib
import sys
from pathlib import Path

manifests_dir = Path(sys.argv[1]).resolve()
hashes_file = Path(sys.argv[2])


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


lines = [l.strip() for l in hashes_file.read_text().splitlines()]
lines = [l for l in lines if l and not l.startswith("#")]
if not lines or not lines[0].startswith("ROOT:"):
    print(f"{hashes_file} must start with a 'ROOT: <absolute_dir>' line "
          "declaring the one permitted directory -- refusing an inventory "
          "with no explicit scope.", file=sys.stderr)
    sys.exit(2)
root = Path(lines[0][len("ROOT:"):].strip())
if not root.is_absolute():
    print(f"ROOT in {hashes_file} must be absolute, got: {root}", file=sys.stderr)
    sys.exit(2)
root = root.resolve()

expected = {}
for line in lines[1:]:
    expected_hash, _, raw_path = line.partition(" ")
    raw_path = raw_path.lstrip(" *")
    if not expected_hash or not raw_path:
        print(f"Malformed line in {hashes_file}: {line!r}", file=sys.stderr)
        sys.exit(2)
    candidate = Path(raw_path) if Path(raw_path).is_absolute() else manifests_dir / raw_path
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        print(f"Entry resolves outside the declared ROOT ({root}) in "
              f"{hashes_file}: {raw_path!r} -> {resolved}", file=sys.stderr)
        sys.exit(2)
    key = str(resolved)
    if key in expected and expected[key] != expected_hash:
        print(f"Duplicate, conflicting entry for {key} in {hashes_file}: "
              f"{expected[key]} vs {expected_hash}", file=sys.stderr)
        sys.exit(2)
    if key in expected:
        print(f"Duplicate entry for {key} in {hashes_file} -- an ambiguous "
              "inventory is not a frozen one, refusing.", file=sys.stderr)
        sys.exit(2)
    expected[key] = expected_hash

actual_batch_files = {str(p.resolve()) for p in manifests_dir.glob("*.batch")}
expected_batch_files = {p for p in expected if p.endswith(".batch")}

extra = actual_batch_files - expected_batch_files
missing = expected_batch_files - actual_batch_files
if extra or missing:
    if extra:
        print(f"Unlisted .batch file(s) present in {manifests_dir}, not in {hashes_file}: {sorted(extra)}", file=sys.stderr)
    if missing:
        print(f"Expected .batch file(s) missing from {manifests_dir}: {sorted(missing)}", file=sys.stderr)
    sys.exit(2)

referenced_configs = set()
for batch_path in sorted(actual_batch_files):
    line = Path(batch_path).read_text().strip()
    parts = line.split(None, 1)
    if len(parts) != 2:
        print(f"Malformed .batch file {batch_path}: expected '<approach> <config_path>', got {line!r}", file=sys.stderr)
        sys.exit(2)
    referenced_configs.add(str(Path(parts[1]).resolve()))

missing_config_entries = referenced_configs - set(expected)
if missing_config_entries:
    print(f"Referenced config(s) not covered by {hashes_file}: {sorted(missing_config_entries)}", file=sys.stderr)
    sys.exit(2)

mismatches = []
for path_str, expected_hash in expected.items():
    path = Path(path_str)
    if not path.is_file():
        mismatches.append(f"{path_str}: file not found")
        continue
    actual_hash = sha256_of(path)
    if actual_hash != expected_hash:
        mismatches.append(f"{path_str}: expected {expected_hash}, got {actual_hash}")

if mismatches:
    for m in mismatches:
        print(f"Manifest hash mismatch: {m}", file=sys.stderr)
    sys.exit(2)

print(f"Manifest verification OK: root={root}, {len(expected_batch_files)} .batch file(s), {len(referenced_configs)} referenced config(s), exact set match, no duplicates, all hashes verified.")
PYEOF
  then
    echo "Manifest hash verification failed against $MANIFEST_HASHES_FILE -- refusing to launch." >&2
    exit 2
  fi
fi

mkdir -p "$OUTPUT_ROOT"
if [[ -n "$(ls -A "$OUTPUT_ROOT" 2>/dev/null)" ]]; then
  echo "Refusing to reuse a non-empty output root (avoid clobbering/duplicating a batch): $OUTPUT_ROOT" >&2
  exit 2
fi

# Prevent two concurrent invocations of this script (any output root) from
# double-dispatching against the same HydroBatch binary/compute resources.
# The non-empty-output-root check above only catches reuse of one specific
# output directory; it does not stop a second, independent invocation
# (different output root, or the same one in the brief window before
# mkdir) from running at the same time. A single, well-known lock file
# scoped to this script makes "only one batch launch active at a time"
# an enforced property, not an assumption.
LOCK_FILE="$ROOT/build-hydrobatch/.multiscale_selection_batch.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "Refusing to launch: another run_multiscale_selection_batch.sh invocation" \
       "already holds the lock ($LOCK_FILE). Only one batch launch may be active" \
       "at a time." >&2
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
  echo "hydrobatch_sha256=$ACTUAL_HYDROBATCH_SHA256"
  echo "hydrobatch_sha256_expected=${EXPECTED_HYDROBATCH_SHA256:-<not supplied>}"
  echo "manifests_dir=$MANIFESTS_DIR"
  echo "max_workers=$MAX_WORKERS"
  echo "per_job_timeout_seconds=$PER_JOB_TIMEOUT_SECONDS"
  echo "per_job_memory_limit_kb=$PER_JOB_MEMORY_LIMIT_KB"
  echo "job_count=${#BATCH_FILES[@]}"
} | tee "$OUTPUT_ROOT/batch_launch_metadata.txt"

# Live record of in-flight jobs, so a crash or external monitor can see what
# was actually dispatched and still running, not just the static launch-time
# snapshot above. One file per job under $OUTPUT_ROOT/.running/; a job's
# marker is removed the instant run_one_job finishes (success or failure).
#
# Signal/crash behavior, documented rather than left implicit: if THIS
# script's own process is killed (SIGKILL, power loss, OOM), markers for
# whatever jobs were in flight at that moment are never removed -- they
# become stale. A marker's presence alone is therefore not sufficient
# evidence that a job is still active.
#
# `kill -0 <pid>` from the marker is a LIVENESS HINT, not worker-identity
# verification -- stated precisely, not overclaimed: it only answers "does
# a process with this PID exist right now," which can be wrong in both
# directions. PID reuse means a dead job's PID can be recycled by an
# unrelated process, making a stale marker look live. Conversely, the
# recorded PID is $BASHPID of the backgrounded run_one_job subshell, which
# wraps the real HydroBatch invocation inside `ulimit -v`/`timeout` -- that
# wrapper dying does not by itself prove HydroBatch's own child process has
# also stopped (an orphaned child can persist under init). A marker's PID
# being dead is therefore evidence the wrapper is gone, not a certified
# statement that no training work remains from that job -- this script does
# not reconcile that automatically (no later invocation reuses
# a non-empty output root, so there is nothing for it to reconcile against
# within its own run), but the information needed to do so externally
# (PID, start timestamp) is recorded in every marker.
RUNNING_DIR="$OUTPUT_ROOT/.running"
mkdir -p "$RUNNING_DIR"

FAILED_SENTINEL="$OUTPUT_ROOT/.failed"

run_one_job() {
  local batch_file="$1"
  local name job_out status
  name="$(basename "$batch_file" .batch)"
  job_out="$OUTPUT_ROOT/$name"
  mkdir -p "$job_out"
  # $BASHPID, not $$: this function runs in a backgrounded subshell
  # (`run_one_job ... &`), and bash's $$ is documented to retain the PID of
  # the top-level invoking shell even inside a subshell -- it would record
  # the same (wrong) PID for every job. $BASHPID gives this subshell's own,
  # actual PID, confirmed by direct testing.
  echo "$BASHPID $(date -u +%FT%TZ)" > "$RUNNING_DIR/$name"
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
  rm -f "$RUNNING_DIR/$name"
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
  # Re-check: a concurrently running job can fail and set the sentinel while
  # this iteration was blocked waiting for a free worker slot above -- the
  # check at the top of the loop body cannot see that, since it only runs
  # once per iteration, before the wait. Without this second check, a job
  # can still be dispatched one failure too late (confirmed by testing: a
  # 1-worker run with the 2nd of 3 jobs set to fail still launched the 3rd,
  # because the 3rd iteration's top-of-loop check passed before job 2 had
  # failed, and nothing re-checked after the wait that followed).
  if [[ -f "$FAILED_SENTINEL" ]]; then
    echo "Stopping dispatch: a prior job failed while waiting for a free worker slot. Not launching $batch_file." >&2
    overall_status=1
    break
  fi
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
