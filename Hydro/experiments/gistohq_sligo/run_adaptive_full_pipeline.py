#!/usr/bin/env python3
"""Adaptive, validation-selected HydroPINN tuning pipeline for paper-grade runs.

Stages:
1. Supervised architecture/memory tuning (FFN, LSTM).
2. Physics/process tuning. Synthetic verification keeps the reduced single
   reservoir. Real LSTM+PINN uses learned runoff generation plus differentiable
   fast/slow routing, with architecture/memory screening before routing tuning.
3. Optimizer tuning per method inheriting prior-stage winners.
4. Multi-seed robustness per method using frozen Stage-3 settings.

For real Hydro/CSV rainfall-runoff runs, selection is based only on VALIDATION
hydrologic skill: finite/non-degenerate KGE first, then NSE, then RMSE. Test
metrics are exported for final reporting but never used to choose candidates.
"""
from __future__ import annotations

import argparse
import csv
import math
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GENERATOR = HERE / "generate_unified_sweep.py"
BATCH_FILE = HERE / "unified_sweep.batch"
POSTPROCESS = HERE / "postprocess_metric_status.py"
MODES = ("ffn", "ffn_pinn", "lstm", "lstm_pinn", "pinn")
PHYSICS = {"ffn_pinn", "lstm_pinn", "pinn"}
PROCESS_PROFILE = "two_reservoir_hybrid"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hydrobatch", required=True, type=Path)
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--data-source", choices=("synthetic", "csv", "hydro"), required=True)
    p.add_argument("--synthetic-profile", default="reduced_reservoir")
    p.add_argument("--synthetic-truth-k", type=float, default=0.08)
    p.add_argument("--sample-count", type=int, default=240)
    p.add_argument("--t-start", type=float, default=0.0)
    p.add_argument("--t-end", type=float, default=5.0)
    p.add_argument("--csv-path", default="")
    p.add_argument("--csv-x-column", type=int, default=0)
    p.add_argument("--csv-y-column", type=int, default=3)
    p.add_argument("--csv-has-header", default="true")
    p.add_argument("--hydro-package-path", default="")
    p.add_argument("--hydro-catchment-id", default="")
    p.add_argument("--hydro-package-profile", default="rainfall-runoff")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--robust-seeds", default="42,123,2026,31415,27182")
    return p.parse_args()


def source_args(a: argparse.Namespace) -> list[str]:
    return [
        "--data-source", a.data_source,
        "--synthetic-profile", a.synthetic_profile,
        "--synthetic-truth-k", str(a.synthetic_truth_k),
        "--sample-count", str(a.sample_count),
        "--t-start", str(a.t_start),
        "--t-end", str(a.t_end),
        "--csv-path", a.csv_path,
        "--csv-x-column", str(a.csv_x_column),
        "--csv-y-column", str(a.csv_y_column),
        "--csv-has-header", a.csv_has_header,
        "--hydro-package-path", a.hydro_package_path,
        "--hydro-catchment-id", a.hydro_catchment_id,
        "--hydro-package-profile", a.hydro_package_profile,
        "--epochs", str(a.epochs),
    ]


def run(cmd: list[str], cwd: Path = HERE) -> None:
    print("[adaptive] $", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], cwd=cwd, check=True)


def load_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r.get("success", "").lower() == "true"]
    if not rows:
        raise RuntimeError(f"No successful experiments in {path}")
    return rows


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                keys.append(k); seen.add(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def finite_float(v: str | None, default: float = math.inf) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def corr(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return math.nan
    mx, my = statistics.fmean(x), statistics.fmean(y)
    sx = math.sqrt(sum((v - mx) ** 2 for v in x))
    sy = math.sqrt(sum((v - my) ** 2 for v in y))
    if sx <= 1.0e-15 or sy <= 1.0e-15:
        return math.nan
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def metrics(obs: list[float], pred: list[float]) -> dict[str, float]:
    n = len(obs)
    if n < 2 or n != len(pred):
        return {k: math.nan for k in ("mse", "rmse", "mae", "nse", "kge", "correlation", "pbias", "pred_std")}
    err = [p - o for o, p in zip(obs, pred)]
    mse = sum(e * e for e in err) / n
    rmse = math.sqrt(mse)
    mae = sum(abs(e) for e in err) / n
    mo, mp = statistics.fmean(obs), statistics.fmean(pred)
    ssto = sum((o - mo) ** 2 for o in obs)
    nse = 1.0 - sum(e * e for e in err) / ssto if ssto > 0 else math.nan
    r = corr(obs, pred)
    so = statistics.pstdev(obs)
    sp = statistics.pstdev(pred)
    alpha = sp / so if so > 0 else math.nan
    beta = mp / mo if mo != 0 else math.nan
    kge = 1.0 - math.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2) \
        if all(math.isfinite(v) for v in (r, alpha, beta)) else math.nan
    total = sum(obs)
    pbias = 100.0 * sum(err) / total if total != 0 else math.nan
    return {"mse": mse, "rmse": rmse, "mae": mae, "nse": nse,
            "kge": kge, "correlation": r, "pbias": pbias, "pred_std": sp}


def split_metrics(predictions: Path, split: str) -> dict[str, float]:
    if not predictions.exists():
        return {k: math.nan for k in ("mse", "rmse", "mae", "nse", "kge", "correlation", "pbias", "pred_std")}
    obs: list[float] = []
    pred: list[float] = []
    with predictions.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("split") != split:
                continue
            try:
                o = float(row["observed"]); p = float(row["predicted"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(o) and math.isfinite(p):
                obs.append(o); pred.append(p)
    return metrics(obs, pred)


def annotate_validation(rows: list[dict[str, str]], out: Path) -> None:
    for row in rows:
        exp = row.get("experiment_id", "")
        mode = row.get("mode", "")
        candidates = [out / exp / "predictions.csv", out / mode / exp / "predictions.csv"]
        path = next((p for p in candidates if p.exists()), candidates[0])
        m = split_metrics(path, "validation")
        for key, value in m.items():
            row[f"validation_{key}"] = str(value)
        row["validation_near_constant"] = "true" if math.isfinite(m["pred_std"]) and m["pred_std"] <= 1.0e-10 else "false"


def winner(a: argparse.Namespace, rows: list[dict[str, str]], mode: str) -> dict[str, str]:
    candidates = [r for r in rows if r.get("mode") == mode]
    if not candidates:
        raise RuntimeError(f"No candidates for mode={mode}")
    if a.data_source == "synthetic":
        return min(candidates, key=lambda r: (finite_float(r.get("validation_mse")), r.get("experiment_id", "")))

    def key(r: dict[str, str]):
        degenerate = r.get("validation_near_constant", "false") == "true"
        kge = finite_float(r.get("validation_kge"), -math.inf)
        nse = finite_float(r.get("validation_nse"), -math.inf)
        rmse = finite_float(r.get("validation_rmse"), math.inf)
        return (1 if degenerate else 0, -kge, -nse, rmse, r.get("experiment_id", ""))
    return min(candidates, key=key)


def run_generated(a: argparse.Namespace, generator_args: list[str], out: Path) -> list[dict[str, str]]:
    out.mkdir(parents=True, exist_ok=True)
    run([sys.executable, str(GENERATOR), *generator_args, *source_args(a)])
    run([str(a.hydrobatch.resolve()), str(BATCH_FILE.resolve()), str(out.resolve())], cwd=HERE.parent.parent.parent)
    rows = load_summary(out / "batch_summary.csv")
    annotate_validation(rows, out)
    write_rows(out / "batch_summary.csv", rows)
    return rows


def q(row: dict[str, str], field: str, fallback: str) -> str:
    value = row.get(field, "").strip()
    return value if value else fallback


def unique_semicolon(values: list[str]) -> str:
    out: list[str] = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return ";".join(out)


def stage1(a: argparse.Namespace, root: Path) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    print("\n[adaptive] STAGE 1: supervised architecture / memory", flush=True)
    rows = run_generated(a, [
        "--methods", "ffn,lstm",
        "--ffn-architectures", "16;24;32;48;16,16;24,24;32,16;32,32;48,24",
        "--ffn-activations", "tanh,relu",
        "--ffn-lags", "1;1,2;1,2,3;1,2,3,4;1,2,3,4,5;1,2,3,4,5,6",
        "--lstm-architectures", "16;24;32;48;24,24;32,32",
        "--lstm-sequences", "6,12,24,48",
        "--learning-rates", "0.003", "--batch-sizes", "32", "--seeds", "42",
    ], root / "01_stage1_supervised")
    winners = {m: winner(a, rows, m) for m in ("ffn", "lstm")}
    return rows, winners


def stage2(a: argparse.Namespace, root: Path, s1: dict[str, dict[str, str]]) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    print("\n[adaptive] STAGE 2: physics/process tuning", flush=True)
    ffn, lstm = s1["ffn"], s1["lstm"]
    stage_root = root / "02_stage2_physics"

    if a.data_source == "synthetic":
        rows = run_generated(a, [
            "--methods", "ffn_pinn,lstm_pinn,pinn",
            "--ffn-architectures", q(ffn, "hidden_layers", "16,16"),
            "--ffn-activations", q(ffn, "activation", "relu"),
            "--lstm-architectures", q(lstm, "hidden_layers", "32"),
            "--lstm-sequences", q(lstm, "lstm_sequence_length", "12"),
            "--pinn-architectures", "16,16;24,24;32,32",
            "--learning-rates", "0.003", "--batch-sizes", "32", "--seeds", "42",
            "--physics-weights", "0.001,0.005,0.01,0.025,0.05,0.1",
            "--recession-k", "0.01,0.02,0.04,0.08,0.16",
            "--lstm-pinn-profile", "linear_reservoir",
        ], stage_root)
        winners = {m: winner(a, rows, m) for m in ("ffn_pinn", "lstm_pinn", "pinn")}
        return rows, winners

    # Real-data legacy reduced-reservoir comparators remain unchanged.
    legacy_rows = run_generated(a, [
        "--methods", "ffn_pinn,pinn",
        "--ffn-architectures", q(ffn, "hidden_layers", "16,16"),
        "--ffn-activations", q(ffn, "activation", "relu"),
        "--pinn-architectures", "16,16;24,24;32,32",
        "--learning-rates", "0.003", "--batch-sizes", "32", "--seeds", "42",
        "--physics-weights", "0.000001,0.00001,0.0001,0.0005,0.001,0.005,0.01,0.025",
        "--recession-k", "0.0025,0.005,0.01,0.02,0.04,0.08,0.16",
    ], stage_root / "legacy")

    # First choose a recurrent memory/architecture for the process-aware hybrid
    # using fixed, moderate routing values. This keeps the expensive routing grid
    # separate from architecture selection and avoids a combinatorial sweep.
    hybrid_architectures = unique_semicolon([
        q(lstm, "hidden_layers", "48"), "32,16", "48,24", "48,32,16", "64,32,16"
    ])
    arch_rows = run_generated(a, [
        "--methods", "lstm_pinn",
        "--lstm-architectures", hybrid_architectures,
        "--lstm-sequences", "6,12,24,48",
        "--learning-rates", "0.003", "--batch-sizes", "32", "--seeds", "42",
        "--physics-weights", "0.001",
        "--lstm-pinn-profile", PROCESS_PROFILE,
        "--fast-k", "0.25", "--slow-k", "0.01", "--routing-alpha", "0.65",
    ], stage_root / "lstm_pinn_architecture")
    arch_winner = winner(a, arch_rows, "lstm_pinn")

    # Then tune routing and only the weak runoff-availability prior. The routing
    # layer itself is a hard differentiable process constraint.
    routing_rows = run_generated(a, [
        "--methods", "lstm_pinn",
        "--lstm-architectures", q(arch_winner, "hidden_layers", "48,24"),
        "--lstm-sequences", q(arch_winner, "lstm_sequence_length", "12"),
        "--learning-rates", "0.003", "--batch-sizes", "32", "--seeds", "42",
        "--physics-weights", "0,0.0001,0.001,0.01",
        "--lstm-pinn-profile", PROCESS_PROFILE,
        "--fast-k", "0.1,0.25,0.5",
        "--slow-k", "0.0025,0.01,0.04",
        "--routing-alpha", "0.35,0.65,0.85",
    ], stage_root / "lstm_pinn_routing")

    rows = legacy_rows + arch_rows + routing_rows
    write_rows(stage_root / "batch_summary.csv", rows)
    winners = {
        "ffn_pinn": winner(a, legacy_rows, "ffn_pinn"),
        "pinn": winner(a, legacy_rows, "pinn"),
        "lstm_pinn": winner(a, routing_rows, "lstm_pinn"),
    }
    return rows, winners


def method_args(mode: str, base: dict[str, str], *, lrs: str, batches: str, seeds: str) -> list[str]:
    args = ["--methods", mode, "--learning-rates", lrs, "--batch-sizes", batches, "--seeds", seeds]
    if mode in ("ffn", "ffn_pinn"):
        args += ["--ffn-architectures", q(base, "hidden_layers", "16,16"),
                 "--ffn-activations", q(base, "activation", "relu")]
        args += ["--ffn-lags", q(base, "input_lags", "1,2,3,4,5,6") if mode == "ffn" else "1"]
    if mode in ("lstm", "lstm_pinn"):
        args += ["--lstm-architectures", q(base, "hidden_layers", "32"),
                 "--lstm-sequences", q(base, "lstm_sequence_length", "12")]
    if mode == "pinn":
        args += ["--pinn-architectures", q(base, "hidden_layers", "24,24")]

    if mode == "lstm_pinn" and q(base, "physics_profile", "") == PROCESS_PROFILE:
        args += [
            "--lstm-pinn-profile", PROCESS_PROFILE,
            "--physics-weights", q(base, "physics_weight", "0.001"),
            "--fast-k", q(base, "storage_coeff", "0.25"),
            "--slow-k", q(base, "lambda_decay", "0.01"),
            "--routing-alpha", q(base, "runoff_coeff", "0.65"),
        ]
    elif mode in PHYSICS:
        args += ["--physics-weights", q(base, "physics_weight", "1.0" if mode == "pinn" else "0.01"),
                 "--recession-k", q(base, "latent_recession_per_hour", "0.08")]
    return args


def stage3(a: argparse.Namespace, root: Path, s1: dict[str, dict[str, str]], s2: dict[str, dict[str, str]]) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    print("\n[adaptive] STAGE 3: optimizer tuning inherited per method", flush=True)
    rows: list[dict[str, str]] = []
    bases = {"ffn": s1["ffn"], "lstm": s1["lstm"], **s2}
    stage_root = root / "03_stage3_optimizer"
    for mode in MODES:
        mode_rows = run_generated(a, method_args(mode, bases[mode], lrs="0.001,0.003,0.005", batches="16,32,64", seeds="42"), stage_root / mode)
        rows.extend(mode_rows)
    write_rows(stage_root / "batch_summary.csv", rows)
    winners = {m: winner(a, rows, m) for m in MODES}
    return rows, winners


def stage4(a: argparse.Namespace, root: Path, s3: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    print("\n[adaptive] STAGE 4: multi-seed robustness with frozen Stage-3 settings", flush=True)
    rows: list[dict[str, str]] = []
    stage_root = root / "04_stage4_robustness"
    for mode in MODES:
        b = s3[mode]
        mode_rows = run_generated(a, method_args(mode, b, lrs=q(b, "learning_rate", "0.003"), batches=q(b, "batch_size", "32"), seeds=a.robust_seeds), stage_root / mode)
        rows.extend(mode_rows)
    write_rows(stage_root / "batch_summary.csv", rows)
    return rows


def mean_sd(values: list[float]) -> tuple[float, float]:
    values = [v for v in values if math.isfinite(v)]
    if not values:
        return math.nan, math.nan
    return statistics.fmean(values), statistics.pstdev(values) if len(values) > 1 else 0.0


def whole_domain_metrics(predictions: Path) -> dict[str, float]:
    with predictions.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    obs = [float(r["observed"]) for r in rows]
    pred = [float(r["predicted"]) for r in rows]
    m = metrics(obs, pred)
    return {"whole_mse": m["mse"], "whole_rmse": m["rmse"], "whole_mae": m["mae"],
            "whole_nse": m["nse"], "whole_kge": m["kge"],
            "whole_correlation": m["correlation"], "whole_pbias": m["pbias"]}


def selection_label(a: argparse.Namespace) -> str:
    return "validation_mse" if a.data_source == "synthetic" else "validation_KGE_then_NSE_then_RMSE_non_degenerate"


def finalize(a: argparse.Namespace, root: Path, s1: dict[str, dict[str, str]], s2: dict[str, dict[str, str]], s3: dict[str, dict[str, str]], s4: list[dict[str, str]]) -> None:
    selected: list[dict[str, str]] = []
    sel = selection_label(a)
    for stage, ws in (("stage1", s1), ("stage2", s2), ("stage3", s3)):
        for mode, row in ws.items():
            r = dict(row); r["stage"] = stage; r["selection_metric"] = sel; selected.append(r)
    write_rows(root / "adaptive_validation_winners.csv", selected)

    robust_rows: list[dict[str, str]] = []
    for mode in MODES:
        members = [r for r in s4 if r.get("mode") == mode]
        out = {"mode": mode, "seed_count": str(len(members))}
        for field in ("validation_mse", "validation_kge", "validation_nse", "validation_rmse",
                      "test_mse", "rmse", "mae", "nse", "kge", "pbias",
                      "physics_loss", "physics_residual_rmse"):
            vals = [finite_float(r.get(field), math.nan) for r in members]
            m, sd = mean_sd(vals); out[field + "_mean"] = str(m); out[field + "_std"] = str(sd)
        robust_rows.append(out)
    write_rows(root / "paper_robustness_summary.csv", robust_rows)

    frozen = root / "frozen_configs"; frozen.mkdir(exist_ok=True)
    stage4_root = root / "04_stage4_robustness"
    for mode in MODES:
        members = [r for r in s4 if r.get("mode") == mode]
        chosen = next((r for r in members if r.get("random_seed") == "42"), members[0])
        cfg = stage4_root / mode / chosen["experiment_id"] / "experiment_config.json"
        shutil.copy2(cfg, frozen / f"{mode}.json")

    whole_rows: list[dict[str, str]] = []
    if a.data_source == "synthetic" and a.synthetic_profile == "reduced_reservoir":
        for row in s4:
            mode = row["mode"]
            mm = whole_domain_metrics(stage4_root / mode / row["experiment_id"] / "predictions.csv")
            whole_rows.append({"experiment_id": row["experiment_id"], "mode": mode, "seed": row.get("random_seed", ""),
                               **{k: str(v) for k, v in mm.items()}})
        write_rows(root / "synthetic_whole_domain_metrics.csv", whole_rows)
        summary = []
        for mode in MODES:
            members = [r for r in whole_rows if r["mode"] == mode]
            out = {"mode": mode, "seed_count": str(len(members))}
            for field in ("whole_rmse", "whole_mae", "whole_nse", "whole_kge", "whole_correlation", "whole_pbias"):
                m, sd = mean_sd([float(r[field]) for r in members]); out[field + "_mean"] = str(m); out[field + "_std"] = str(sd)
            summary.append(out)
        write_rows(root / "paper_synthetic_known_truth_summary.csv", summary)

    paper_rows = []
    for mode in MODES:
        b = s3[mode]; rr = next(r for r in robust_rows if r["mode"] == mode)
        out = {
            "mode": mode,
            "hidden_layers": b.get("hidden_layers", ""),
            "activation": b.get("activation", ""),
            "input_lags": b.get("input_lags", ""),
            "lstm_sequence_length": b.get("lstm_sequence_length", ""),
            "learning_rate": b.get("learning_rate", ""),
            "batch_size": b.get("batch_size", ""),
            "physics_profile": b.get("physics_profile", ""),
            "physics_weight": b.get("physics_weight", ""),
            "reservoir_k": b.get("latent_recession_per_hour", ""),
            "fast_k": b.get("storage_coeff", ""),
            "slow_k": b.get("lambda_decay", ""),
            "routing_alpha": b.get("runoff_coeff", ""),
        }
        out.update(rr); paper_rows.append(out)
    write_rows(root / "paper_method_summary.csv", paper_rows)

    lines = [
        "HydroPINN adaptive paper-run manifest",
        "====================================",
        f"data_source={a.data_source}",
        f"output_root={root}",
        f"selection_metric={sel} (test metrics never used for tuning)",
        "domain=common longest contiguous GIStoOHQ segment for all five methods when applicable",
        "real_lstm_pinn=learned runoff generation + differentiable fast/slow reservoir routing",
        "process_hybrid_scaling=train-only standardization for neural predictor; routing remains in physical units",
        "stage4=multi-seed robustness of frozen Stage-3 configurations",
        "",
    ]
    for mode in MODES:
        b = s3[mode]
        lines.append(
            f"{mode}: hidden={b.get('hidden_layers','')} act={b.get('activation','')} "
            f"lags={b.get('input_lags','')} seq={b.get('lstm_sequence_length','')} "
            f"lr={b.get('learning_rate','')} batch={b.get('batch_size','')} "
            f"profile={b.get('physics_profile','')} w={b.get('physics_weight','')} "
            f"k={b.get('latent_recession_per_hour','')} fast_k={b.get('storage_coeff','')} "
            f"slow_k={b.get('lambda_decay','')} alpha={b.get('runoff_coeff','')}"
        )
    lines += ["", "Paper artifacts:", "  adaptive_validation_winners.csv", "  paper_robustness_summary.csv",
              "  paper_method_summary.csv", "  frozen_configs/*.json"]
    if whole_rows:
        lines += ["  synthetic_whole_domain_metrics.csv", "  paper_synthetic_known_truth_summary.csv"]
    (root / "PAPER_RUN_MANIFEST.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def main() -> int:
    a = parse_args(); root = a.output_root.resolve(); root.mkdir(parents=True, exist_ok=True)
    if not a.hydrobatch.exists(): raise SystemExit(f"HydroBatch not found: {a.hydrobatch}")
    if a.data_source == "synthetic" and a.synthetic_profile != "reduced_reservoir": raise SystemExit("Paper five-method synthetic run requires reduced_reservoir")
    if a.data_source == "hydro" and not a.hydro_package_path: raise SystemExit("Hydro package path is required")
    if a.data_source == "csv" and not a.csv_path: raise SystemExit("CSV path is required")
    _, s1 = stage1(a, root)
    _, s2 = stage2(a, root, s1)
    _, s3 = stage3(a, root, s1, s2)
    s4 = stage4(a, root, s3)
    finalize(a, root, s1, s2, s3, s4)
    if a.data_source != "synthetic" and POSTPROCESS.exists():
        run([sys.executable, str(POSTPROCESS), str(root)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
