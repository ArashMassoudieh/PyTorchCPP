#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
BATCH_BIN="${HYDROBATCH_BIN:-$ROOT/build-hydrobatch/HydroBatch}"
OUT="$HERE/batch_outputs/method_regression_$(date +%Y%m%d_%H%M%S)"

if [[ ! -x "$BATCH_BIN" ]]; then
  echo "HydroBatch executable not found: $BATCH_BIN" >&2
  echo "Build with: cd $ROOT/build-hydrobatch && qmake ../HydroBatch.pro CONFIG+=PowerEdge && make -j4" >&2
  exit 2
fi

cd "$HERE"
python3 generate_unified_sweep.py \
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

mkdir -p "$OUT"
"$BATCH_BIN" unified_sweep.batch "$OUT" | tee "$OUT/run.log"

echo
python3 - "$OUT/batch_summary.csv" <<'PY'
import csv
import math
import sys
from pathlib import Path

p = Path(sys.argv[1])
rows = list(csv.DictReader(p.open()))
print(f"[regression] rows={len(rows)} summary={p}")
if len(rows) != 7:
    raise SystemExit(f"Expected 7 successful summary rows, got {len(rows)}")

for r in rows:
    if r.get("success", "").lower() != "true":
        raise SystemExit(f"Unsuccessful job in summary: {r.get('experiment_id')}")

hybrids = [r for r in rows if r["mode"] == "lstm_pinn"]
if len(hybrids) != 2:
    raise SystemExit(f"Expected two LSTM+PINN weight checks, got {len(hybrids)}")

# Exact equality here was the prior regression: the restored checkpoint came
# from the data-only warm-up, making physics_weight irrelevant.
fields = ["test_mse", "rmse", "nse", "pbias", "physics_loss"]
identical = all(hybrids[0][f] == hybrids[1][f] for f in fields)
if identical:
    raise SystemExit("LSTM+PINN regression failed: physics_weight=0.005 and 0.1 still produced identical fitted metrics.")

for r in rows:
    for f in ("test_mse", "rmse", "mae"):
        v = float(r[f])
        if not math.isfinite(v):
            raise SystemExit(f"Non-finite {f} for {r['experiment_id']}")

print("[regression] PASS: 7 jobs exported successfully and LSTM+PINN responds to physics_weight.")
PY

echo "[regression] output=$OUT"
