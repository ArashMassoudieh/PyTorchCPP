#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
BATCH_BIN="${HYDROBATCH_BIN:-$ROOT/build-hydrobatch/HydroBatch}"
OUT="$HERE/batch_outputs/synthetic_method_regression_$(date +%Y%m%d_%H%M%S)"

if [[ ! -x "$BATCH_BIN" ]]; then
  echo "HydroBatch executable not found: $BATCH_BIN" >&2
  echo "Build with: cd $ROOT/build-hydrobatch && qmake ../HydroBatch.pro CONFIG+=PowerEdge && make -j4" >&2
  exit 2
fi

cd "$HERE"
python3 generate_unified_sweep.py \
  --data-source synthetic \
  --synthetic-profile reduced_reservoir \
  --sample-count 240 \
  --t-start 0 \
  --t-end 5 \
  --methods ffn,ffn_pinn,lstm,lstm_pinn,pinn \
  --ffn-architectures "16,16" \
  --ffn-activations relu \
  --ffn-lags "1,2,3,4,5,6" \
  --lstm-architectures 32 \
  --lstm-sequences 12 \
  --pinn-architectures "24,24" \
  --learning-rates 0.003 \
  --batch-sizes 32 \
  --seeds 42 \
  --physics-weights 0.005,0.1 \
  --recession-k 0.04

python3 - <<'PY'
import csv
import json
from pathlib import Path

root = Path("generated_unified")
manifest = list(csv.DictReader((root / "unified_manifest.csv").open()))
if len(manifest) != 7:
    raise SystemExit(f"Expected 7 generated jobs, got {len(manifest)}")
for row in manifest:
    if row["data_source"] != "synthetic":
        raise SystemExit(f"Source leak in manifest: {row['experiment_id']} -> {row['data_source']}")
    if row["synthetic_profile"] != "reduced_reservoir":
        raise SystemExit(f"Wrong synthetic profile: {row['experiment_id']} -> {row['synthetic_profile']}")
    if row["hydro_package_path"] or row["csv_path"]:
        raise SystemExit(f"External path leaked into synthetic manifest: {row['experiment_id']}")
    cfg = json.loads((root / f"{row['experiment_id']}.json").read_text())
    if cfg.get("use_hydro_package") or cfg.get("use_csv_data"):
        raise SystemExit(f"External source flag leaked into synthetic config: {row['experiment_id']}")
    if cfg.get("hydro_package_path") or cfg.get("csv_path"):
        raise SystemExit(f"External path leaked into synthetic config: {row['experiment_id']}")
    if cfg.get("synthetic_profile") != "reduced_reservoir":
        raise SystemExit(f"Wrong config profile: {row['experiment_id']}")
print("[source-regression] PASS: all 7 configs are controlled reduced_reservoir Synthetic jobs with no external paths.")
PY

mkdir -p "$OUT"
"$BATCH_BIN" unified_sweep.batch "$OUT" | tee "$OUT/run.log"

python3 - "$OUT/batch_summary.csv" <<'PY'
import csv
import math
import sys
from pathlib import Path

p = Path(sys.argv[1])
rows = list(csv.DictReader(p.open()))
if len(rows) != 7:
    raise SystemExit(f"Expected 7 successful summary rows, got {len(rows)}")
for r in rows:
    if r.get("success", "").lower() != "true":
        raise SystemExit(f"Unsuccessful job: {r.get('experiment_id')}")
    for f in ("test_mse", "rmse", "mae"):
        if not math.isfinite(float(r[f])):
            raise SystemExit(f"Non-finite {f} for {r['experiment_id']}")

hybrids = [r for r in rows if r["mode"] == "lstm_pinn"]
if len(hybrids) != 2:
    raise SystemExit(f"Expected two LSTM+PINN physics-weight checks, got {len(hybrids)}")
fields = ["test_mse", "rmse", "nse", "pbias", "physics_loss"]
if all(hybrids[0][f] == hybrids[1][f] for f in fields):
    raise SystemExit("LSTM+PINN still ignores physics_weight in controlled synthetic regression")

print("[synthetic-regression] PASS: 7 jobs exported successfully and LSTM+PINN responds to physics_weight.")
PY

echo "[synthetic-regression] output=$OUT"
