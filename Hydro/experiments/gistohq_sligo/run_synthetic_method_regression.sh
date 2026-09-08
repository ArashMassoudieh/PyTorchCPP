#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
BATCH_BIN="${HYDROBATCH_BIN:-$ROOT/build-hydrobatch/HydroBatch}"
OUT="$HERE/batch_outputs/synthetic_method_regression_$(date +%Y%m%d_%H%M%S)"
TRUTH_K=0.08

if [[ ! -x "$BATCH_BIN" ]]; then
  echo "HydroBatch executable not found: $BATCH_BIN" >&2
  echo "Build with: cd $ROOT/build-hydrobatch && qmake ../HydroBatch.pro CONFIG+=PowerEdge && make -j4" >&2
  exit 2
fi

cd "$HERE"
python3 generate_unified_sweep.py \
  --data-source synthetic \
  --synthetic-profile reduced_reservoir \
  --synthetic-truth-k "$TRUTH_K" \
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
  --recession-k "$TRUTH_K"

python3 - "$TRUTH_K" <<'PY'
import csv
import json
import sys
from pathlib import Path

truth_k = float(sys.argv[1])
root = Path("generated_unified")
manifest = list(csv.DictReader((root / "unified_manifest.csv").open()))
if len(manifest) != 7:
    raise SystemExit(f"Expected 7 generated jobs, got {len(manifest)}")
for row in manifest:
    if row["data_source"] != "synthetic":
        raise SystemExit(f"Source leak in manifest: {row['experiment_id']} -> {row['data_source']}")
    if row["synthetic_profile"] != "reduced_reservoir":
        raise SystemExit(f"Wrong synthetic profile: {row['experiment_id']} -> {row['synthetic_profile']}")
    if abs(float(row["synthetic_truth_k"]) - truth_k) > 1e-15:
        raise SystemExit(f"Truth k changed in manifest: {row['experiment_id']} -> {row['synthetic_truth_k']}")
    if row["hydro_package_path"] or row["csv_path"]:
        raise SystemExit(f"External path leaked into synthetic manifest: {row['experiment_id']}")
    cfg = json.loads((root / f"{row['experiment_id']}.json").read_text())
    if cfg.get("use_hydro_package") or cfg.get("use_csv_data"):
        raise SystemExit(f"External source flag leaked into synthetic config: {row['experiment_id']}")
    if cfg.get("hydro_package_path") or cfg.get("csv_path"):
        raise SystemExit(f"External path leaked into synthetic config: {row['experiment_id']}")
    if cfg.get("synthetic_profile") != "reduced_reservoir":
        raise SystemExit(f"Wrong config profile: {row['experiment_id']}")
    if abs(float(cfg.get("synthetic_reservoir_truth_k")) - truth_k) > 1e-15:
        raise SystemExit(f"Truth k changed in config: {row['experiment_id']}")
    if row["mode"] in {"ffn_pinn", "lstm_pinn", "pinn"}:
        candidate_k = float(row["recession_k"])
        if abs(candidate_k - truth_k) > 1e-15:
            raise SystemExit(
                f"Controlled known-truth regression requires model k=truth k; "
                f"{row['experiment_id']} has model k={candidate_k}, truth k={truth_k}"
            )
print(
    f"[source-regression] PASS: all 7 configs use one Synthetic truth, "
    f"physics models use matched k={truth_k}, and no external paths leaked."
)
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
    if r["mode"] in {"ffn_pinn", "lstm_pinn", "pinn"}:
        for f in ("physics_loss", "physics_residual_rmse"):
            if not math.isfinite(float(r[f])):
                raise SystemExit(f"Non-finite {f} for physics job {r['experiment_id']}")

by_mode = {}
for r in rows:
    by_mode.setdefault(r["mode"], []).append(r)

for mode in ("ffn_pinn", "lstm_pinn"):
    hybrids = by_mode.get(mode, [])
    if len(hybrids) != 2:
        raise SystemExit(f"Expected two {mode} physics-weight checks, got {len(hybrids)}")
    fields = ["test_mse", "rmse", "nse", "pbias", "physics_loss"]
    if all(hybrids[0][f] == hybrids[1][f] for f in fields):
        raise SystemExit(f"{mode} still ignores physics_weight in controlled synthetic regression")

pinn = by_mode.get("pinn", [])
if len(pinn) != 1:
    raise SystemExit(f"Expected one standalone PINN result, got {len(pinn)}")
pinn_rmse = float(pinn[0]["rmse"])
pinn_pbias = abs(float(pinn[0]["pbias"]))
if pinn_rmse >= 0.02:
    raise SystemExit(f"Known-truth standalone PINN RMSE is unexpectedly high: {pinn_rmse}")
if pinn_pbias >= 10.0:
    raise SystemExit(f"Known-truth standalone PINN |PBIAS| is unexpectedly high: {pinn_pbias}%")

ffn = by_mode.get("ffn", [])
ffn_pinn = by_mode.get("ffn_pinn", [])
if len(ffn) == 1 and ffn_pinn:
    best_hybrid_rmse = min(float(r["rmse"]) for r in ffn_pinn)
    if best_hybrid_rmse >= float(ffn[0]["rmse"]):
        print(
            "[synthetic-regression] NOTE: FFN+PINN did not beat FFN for this fixed architecture/seed; "
            "this is diagnostic, not a hard failure."
        )

print(
    "[synthetic-regression] PASS: 7 jobs exported; both hybrid families respond to physics_weight; "
    "standalone PINN satisfies loose known-truth quality gates."
)
PY

echo "[synthetic-regression] output=$OUT"
