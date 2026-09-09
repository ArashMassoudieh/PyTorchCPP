#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
BATCH="${HYDROBATCH_BIN:-$ROOT/build-hydrobatch/HydroBatch}"
SLIGO_PACKAGE="${SLIGO_HYDROPINN_PACKAGE:-$ROOT/../GIStoOHQ/examples/SligoCreek/outputs/sligocreekdemo_data/hydropinn}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${1:-$HERE/batch_outputs/paper_run_$STAMP}"
mkdir -p "$OUT"
if [[ ! -x "$BATCH" ]]; then echo "HydroBatch not found: $BATCH" >&2; exit 2; fi
if [[ ! -f "$SLIGO_PACKAGE/manifest.json" ]]; then echo "Sligo HydroPINN package not found: $SLIGO_PACKAGE" >&2; exit 2; fi
{
  echo "paper_run=$OUT"
  echo "repo_commit=$(git -C "$ROOT" rev-parse HEAD)"
  echo "hydrobatch=$BATCH"
  echo "sligo_package=$SLIGO_PACKAGE"
  echo "started_utc=$(date -u +%FT%TZ)"
} > "$OUT/paper_run_metadata.txt"

echo "[paper] 0/3 controlled-method preflight"
HYDROBATCH_BIN="$BATCH" bash "$HERE/run_synthetic_method_regression.sh" | tee "$OUT/00_synthetic_preflight.log"

echo "[paper] 1/3 adaptive controlled synthetic study"
python3 "$HERE/run_adaptive_full_pipeline.py" \
  --hydrobatch "$BATCH" --output-root "$OUT/01_synthetic_controlled" \
  --data-source synthetic --synthetic-profile reduced_reservoir --synthetic-truth-k 0.08 \
  --sample-count 240 --t-start 0 --t-end 5 | tee "$OUT/01_synthetic_controlled.log"
python3 "$HERE/postprocess_metric_status.py" "$OUT/01_synthetic_controlled"

echo "[paper] 2/3 adaptive Sligo Creek study"
python3 "$HERE/run_adaptive_full_pipeline.py" \
  --hydrobatch "$BATCH" --output-root "$OUT/02_sligo_hydro" \
  --data-source hydro --hydro-package-path "$SLIGO_PACKAGE" --hydro-package-profile rainfall-runoff \
  | tee "$OUT/02_sligo_hydro.log"
python3 "$HERE/postprocess_metric_status.py" "$OUT/02_sligo_hydro"

echo "[paper] 3/3 manuscript tables and figures"
python3 "$HERE/build_paper_comparison.py" "$OUT"
python3 "$HERE/make_paper_figures.py" "$OUT"
echo "finished_utc=$(date -u +%FT%TZ)" >> "$OUT/paper_run_metadata.txt"

echo "[paper] FINAL PAPER RUN COMPLETE"
echo "[paper] output=$OUT"
echo "[paper] main table=$OUT/paper_final_method_comparison.csv"
echo "[paper] markdown=$OUT/paper_final_tables.md"
