#!/usr/bin/env python3
"""Run the complete HydroPINN paper workflow from one command/GUI action.

The workflow always creates a controlled reduced-reservoir synthetic verification
and a real-data experiment using the selected Hydro package or CSV source.  It
then generates metric-definition diagnostics, combined paper tables, frozen
configs, and publication figures.  Model selection is delegated to the adaptive
pipeline and therefore uses validation data only; held-out test metrics are not
used to choose hyperparameters.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ADAPTIVE = HERE / "run_adaptive_full_pipeline.py"
POSTPROCESS = HERE / "postprocess_metric_status.py"
TABLES = HERE / "build_paper_comparison.py"
FIGURES = HERE / "make_paper_figures.py"


def run(cmd: list[str]) -> None:
    print("[full-paper] $", " ".join(str(v) for v in cmd), flush=True)
    subprocess.run([str(v) for v in cmd], cwd=HERE, check=True)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hydrobatch", required=True, type=Path)
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--data-source", choices=("hydro", "csv"), required=True,
                   help="Real-data source; controlled synthetic verification is run automatically")
    p.add_argument("--csv-path", default="")
    p.add_argument("--csv-x-column", type=int, default=0)
    p.add_argument("--csv-y-column", type=int, default=3)
    p.add_argument("--csv-has-header", default="true")
    p.add_argument("--hydro-package-path", default="")
    p.add_argument("--hydro-catchment-id", default="")
    p.add_argument("--hydro-package-profile", default="rainfall-runoff")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--robust-seeds", default="42,123,2026,31415,27182")
    p.add_argument("--synthetic-sample-count", type=int, default=240)
    p.add_argument("--synthetic-t-start", type=float, default=0.0)
    p.add_argument("--synthetic-t-end", type=float, default=5.0)
    p.add_argument("--synthetic-truth-k", type=float, default=0.08)
    return p


def real_source_args(a: argparse.Namespace) -> list[str]:
    return [
        "--data-source", a.data_source,
        "--csv-path", a.csv_path,
        "--csv-x-column", str(a.csv_x_column),
        "--csv-y-column", str(a.csv_y_column),
        "--csv-has-header", a.csv_has_header,
        "--hydro-package-path", a.hydro_package_path,
        "--hydro-catchment-id", a.hydro_catchment_id,
        "--hydro-package-profile", a.hydro_package_profile,
        "--epochs", str(a.epochs),
        "--robust-seeds", a.robust_seeds,
    ]


def main() -> int:
    a = parser().parse_args()
    root = a.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not a.hydrobatch.exists():
        raise SystemExit(f"HydroBatch not found: {a.hydrobatch}")
    if a.data_source == "hydro" and not a.hydro_package_path.strip():
        raise SystemExit("Hydro package path is required")
    if a.data_source == "csv" and not a.csv_path.strip():
        raise SystemExit("CSV path is required")

    metadata = root / "paper_run_metadata.txt"
    metadata.write_text(
        "started_utc=" + datetime.now(timezone.utc).isoformat() + "\n" +
        f"hydrobatch={a.hydrobatch.resolve()}\n" +
        f"real_data_source={a.data_source}\n" +
        f"hydro_package_path={a.hydro_package_path}\n" +
        f"csv_path={a.csv_path}\n" +
        "selection=validation only; test metrics not used for tuning\n" +
        "comparison_domain=common longest contiguous GIStoOHQ segment when HydroPINNExport is used\n",
        encoding="utf-8",
    )

    synthetic = root / "01_synthetic_controlled"
    real = root / "02_sligo_hydro"  # retained name for backward-compatible table/figure scripts

    print("\n[full-paper] 1/4 Controlled reduced-reservoir verification", flush=True)
    run([
        sys.executable, ADAPTIVE,
        "--hydrobatch", a.hydrobatch.resolve(),
        "--output-root", synthetic,
        "--data-source", "synthetic",
        "--synthetic-profile", "reduced_reservoir",
        "--synthetic-truth-k", str(a.synthetic_truth_k),
        "--sample-count", str(a.synthetic_sample_count),
        "--t-start", str(a.synthetic_t_start),
        "--t-end", str(a.synthetic_t_end),
        "--epochs", str(a.epochs),
        "--robust-seeds", a.robust_seeds,
    ])

    print("\n[full-paper] 2/4 Real-data five-method adaptive study", flush=True)
    run([
        sys.executable, ADAPTIVE,
        "--hydrobatch", a.hydrobatch.resolve(),
        "--output-root", real,
        *real_source_args(a),
    ])

    print("\n[full-paper] 3/4 Diagnostics and final tables", flush=True)
    run([sys.executable, POSTPROCESS, real])
    run([sys.executable, TABLES, root])

    print("\n[full-paper] 4/4 Publication figures", flush=True)
    run([sys.executable, FIGURES, root])

    with metadata.open("a", encoding="utf-8") as f:
        f.write("finished_utc=" + datetime.now(timezone.utc).isoformat() + "\n")
        f.write("status=complete\n")

    print("\n[full-paper] COMPLETE")
    print("[full-paper] output:", root)
    print("[full-paper] table:", root / "paper_final_method_comparison.csv")
    print("[full-paper] markdown:", root / "paper_final_tables.md")
    print("[full-paper] figures: PNG 600 dpi + PDF + SVG")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
