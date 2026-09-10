#!/usr/bin/env python3
"""Fast runtime preflight for the process-aware real-data LSTM+PINN path.

This intentionally runs one small two-reservoir hybrid experiment before the
expensive paper pipeline.  It catches stale HydroBatch builds, profile-routing
rewrites, missing GIStoOHQ fields, non-finite predictions, and complete output
collapse without using the test set for model selection.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GENERATOR = HERE / "generate_unified_sweep.py"
BATCH = HERE / "unified_sweep.batch"


def run(cmd: list[object], cwd: Path | None = None) -> None:
    print("[process-preflight] $", " ".join(str(v) for v in cmd), flush=True)
    subprocess.run([str(v) for v in cmd], cwd=cwd or HERE, check=True)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hydrobatch", required=True, type=Path)
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--data-source", choices=("hydro", "csv"), required=True)
    p.add_argument("--hydro-package-path", default="")
    p.add_argument("--hydro-catchment-id", default="")
    p.add_argument("--hydro-package-profile", default="rainfall-runoff")
    p.add_argument("--csv-path", default="")
    p.add_argument("--csv-x-column", type=int, default=0)
    p.add_argument("--csv-y-column", type=int, default=3)
    p.add_argument("--csv-has-header", default="true")
    p.add_argument("--epochs", type=int, default=20)
    return p


def source_args(a: argparse.Namespace) -> list[str]:
    return [
        "--data-source", a.data_source,
        "--hydro-package-path", a.hydro_package_path,
        "--hydro-catchment-id", a.hydro_catchment_id,
        "--hydro-package-profile", a.hydro_package_profile,
        "--csv-path", a.csv_path,
        "--csv-x-column", str(a.csv_x_column),
        "--csv-y-column", str(a.csv_y_column),
        "--csv-has-header", a.csv_has_header,
    ]


def read_one_summary(path: Path) -> dict[str, str]:
    if not path.exists():
        raise RuntimeError(f"Preflight summary was not created: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise RuntimeError(f"Expected one preflight result, found {len(rows)}")
    return rows[0]


def prediction_std(path: Path) -> float:
    values: list[float] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                value = float(row["predicted"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
    if len(values) < 2:
        raise RuntimeError("Preflight produced too few finite predictions")
    return statistics.pstdev(values)


def main() -> int:
    a = parser().parse_args()
    if not a.hydrobatch.exists():
        raise SystemExit(f"HydroBatch not found: {a.hydrobatch}")
    if a.data_source == "hydro" and not a.hydro_package_path.strip():
        raise SystemExit("Hydro package path is required")
    if a.data_source == "csv" and not a.csv_path.strip():
        raise SystemExit("CSV path is required")

    out = a.output_root.resolve()
    out.mkdir(parents=True, exist_ok=True)

    run([
        sys.executable, GENERATOR,
        "--methods", "lstm_pinn",
        "--lstm-architectures", "32",
        "--lstm-sequences", "6",
        "--learning-rates", "0.003",
        "--batch-sizes", "32",
        "--seeds", "42",
        "--physics-weights", "0.001",
        "--lstm-pinn-profile", "two_reservoir_hybrid",
        "--fast-k", "0.25",
        "--slow-k", "0.01",
        "--routing-alpha", "0.65",
        "--epochs", str(max(10, a.epochs)),
        *source_args(a),
    ])
    run([a.hydrobatch.resolve(), BATCH.resolve(), out])

    row = read_one_summary(out / "batch_summary.csv")
    if row.get("success", "").lower() != "true":
        raise RuntimeError("Process-aware LSTM+PINN preflight did not report success")
    if row.get("physics_profile") != "two_reservoir_hybrid":
        raise RuntimeError(
            "Process-aware profile was rewritten unexpectedly: " + row.get("physics_profile", "<missing>")
        )
    for field in ("validation_mse", "rmse", "mae"):
        try:
            value = float(row[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Preflight has invalid {field}") from exc
        if not math.isfinite(value):
            raise RuntimeError(f"Preflight has non-finite {field}")

    experiment_id = row.get("experiment_id", "")
    predictions = out / experiment_id / "predictions.csv"
    if not predictions.exists():
        raise RuntimeError(f"Preflight predictions are missing: {predictions}")
    pred_std = prediction_std(predictions)
    if pred_std <= 1.0e-10:
        raise RuntimeError("Process-aware preflight collapsed to a constant prediction")

    print("[process-preflight] PASS", flush=True)
    print(f"[process-preflight] profile={row.get('physics_profile')}", flush=True)
    print(f"[process-preflight] validation_mse={row.get('validation_mse')}", flush=True)
    print(f"[process-preflight] rmse={row.get('rmse')}", flush=True)
    print(f"[process-preflight] prediction_std={pred_std:.9g}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
