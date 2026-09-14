#!/usr/bin/env python3
"""Run the complete HydroPINN paper workflow from one command/GUI action.

The workflow first executes a fast process-aware LSTM+PINN runtime preflight,
then creates a controlled reduced-reservoir synthetic verification and a
real-data experiment using the selected Hydro package or CSV source. It finally
generates split-shift diagnostics, metric-definition diagnostics, combined paper
tables, a post-hoc hybrid-gain assessment, frozen configs, and publication
figures. Real-data selection uses validation data only; held-out test metrics are
never used to choose hyperparameters.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREFLIGHT = HERE / "smoke_test_process_hybrid.py"
ADAPTIVE = HERE / "run_improved_adaptive_pipeline.py"
POSTPROCESS = HERE / "postprocess_metric_status.py"
SPLIT_DIAGNOSTICS = HERE / "diagnose_split_shift.py"
TABLES = HERE / "build_paper_comparison.py"
ASSESS = HERE / "assess_hybrid_gain.py"
FIGURES = HERE / "make_paper_figures.py"
LOG_PATH: Path | None = None


def append_log(text: str) -> None:
    if LOG_PATH is None:
        return
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(text)
        if text and not text.endswith("\n"):
            f.write("\n")


def say(text: str = "") -> None:
    print(text, flush=True)
    append_log(text)


def run(cmd: list[str]) -> None:
    command = "[full-paper] $ " + " ".join(str(v) for v in cmd)
    say(command)
    process = subprocess.Popen(
        [str(v) for v in cmd],
        cwd=HERE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        append_log(line.rstrip("\n"))
    code = process.wait()
    if code != 0:
        raise subprocess.CalledProcessError(code, [str(v) for v in cmd])


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
    p.add_argument("--skip-process-preflight", action="store_true",
                   help="Skip the fast process-aware LSTM+PINN smoke test")
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


def preflight_source_args(a: argparse.Namespace) -> list[str]:
    return [
        "--data-source", a.data_source,
        "--csv-path", a.csv_path,
        "--csv-x-column", str(a.csv_x_column),
        "--csv-y-column", str(a.csv_y_column),
        "--csv-has-header", a.csv_has_header,
        "--hydro-package-path", a.hydro_package_path,
        "--hydro-catchment-id", a.hydro_catchment_id,
        "--hydro-package-profile", a.hydro_package_profile,
    ]


def main() -> int:
    global LOG_PATH
    a = parser().parse_args()
    root = a.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    LOG_PATH = root / "full_pipeline.log"
    LOG_PATH.write_text(
        "HydroPINN full paper pipeline log\n"
        f"started_utc={datetime.now(timezone.utc).isoformat()}\n"
        f"output_root={root}\n\n",
        encoding="utf-8",
    )
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
        f"full_pipeline_log={LOG_PATH}\n" +
        "selection=validation only; test metrics not used for tuning\n" +
        "selection_order_real=nondegenerate KGE -> NSE -> |PBIAS| -> RMSE\n" +
        "r2_definition=squared Pearson correlation; NSE retained separately\n" +
        "comparison_domain=common longest contiguous GIStoOHQ segment when HydroPINNExport is used\n" +
        f"process_hybrid_preflight={'skipped' if a.skip_process_preflight else 'required'}\n",
        encoding="utf-8",
    )

    preflight = root / "00_process_hybrid_preflight"
    synthetic = root / "01_synthetic_controlled"
    real = root / "02_sligo_hydro"

    if not a.skip_process_preflight:
        say("\n[full-paper] 0/4 Process-aware LSTM+PINN preflight")
        run([
            sys.executable, PREFLIGHT,
            "--hydrobatch", a.hydrobatch.resolve(),
            "--output-root", preflight,
            "--epochs", "20",
            *preflight_source_args(a),
        ])
        with metadata.open("a", encoding="utf-8") as f:
            f.write("process_hybrid_preflight_status=pass\n")

    say("\n[full-paper] 1/4 Controlled reduced-reservoir verification")
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

    say("\n[full-paper] 2/4 Real-data five-method adaptive study")
    run([
        sys.executable, ADAPTIVE,
        "--hydrobatch", a.hydrobatch.resolve(),
        "--output-root", real,
        *real_source_args(a),
    ])

    say("\n[full-paper] 3/4 Split-shift diagnostics, metric definitions, final tables, and hybrid assessment")
    run([sys.executable, SPLIT_DIAGNOSTICS, real])
    run([sys.executable, POSTPROCESS, synthetic])
    run([sys.executable, POSTPROCESS, real])
    run([sys.executable, TABLES, root])
    run([sys.executable, ASSESS, root])

    say("\n[full-paper] 4/4 Publication figures")
    run([sys.executable, FIGURES, root])

    with metadata.open("a", encoding="utf-8") as f:
        f.write("finished_utc=" + datetime.now(timezone.utc).isoformat() + "\n")
        f.write("status=complete\n")

    say("\n[full-paper] COMPLETE")
    say("[full-paper] output: " + str(root))
    say("[full-paper] log: " + str(LOG_PATH))
    say("[full-paper] table: " + str(root / "paper_final_method_comparison.csv"))
    say("[full-paper] hybrid assessment: " + str(root / "paper_hybrid_gain_assessment.csv"))
    say("[full-paper] split diagnostics: " + str(real / "paper_split_shift_diagnostics.csv"))
    say("[full-paper] markdown: " + str(root / "paper_final_tables.md"))
    say("[full-paper] figures: PNG 600 dpi + PDF + SVG")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException as exc:
        append_log(f"[full-paper] FAILED: {type(exc).__name__}: {exc}")
        raise
