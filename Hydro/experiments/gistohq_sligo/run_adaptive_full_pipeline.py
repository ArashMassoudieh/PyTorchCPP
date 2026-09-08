#!/usr/bin/env python3
"""Adaptive, validation-selected HydroPINN tuning pipeline for paper-grade runs.

Stages:
1. Supervised architecture/memory tuning (FFN, LSTM).
2. Physics tuning inheriting Stage-1 architectures (FFN+PINN, LSTM+PINN, PINN).
3. Optimizer tuning per method inheriting prior-stage winners.
4. Multi-seed robustness per method using frozen Stage-3 settings.

Selection uses validation_mse only. Test metrics are never used to choose candidates.
For controlled reduced-reservoir synthetic runs, whole-domain known-truth metrics are
computed separately from predictions.csv to avoid misleading low-variance tail NSE.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GENERATOR = HERE / "generate_unified_sweep.py"
BATCH_FILE = HERE / "unified_sweep.batch"
MODES = ("ffn", "ffn_pinn", "lstm", "lstm_pinn", "pinn")
PHYSICS = {"ffn_pinn", "lstm_pinn", "pinn"}


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


def winner(rows: list[dict[str, str]], mode: str) -> dict[str, str]:
    candidates = [r for r in rows if r.get("mode") == mode]
    if not candidates:
        raise RuntimeError(f"No candidates for mode={mode}")
    return min(candidates, key=lambda r: (finite_float(r.get("validation_mse")), r.get("experiment_id", "")))


def run_generated(a: argparse.Namespace, generator_args: list[str], out: Path) -> list[dict[str, str]]:
    out.mkdir(parents=True, exist_ok=True)
    run([sys.executable, str(GENERATOR), *generator_args, *source_args(a)])
    run([str(a.hydrobatch.resolve()), str(BATCH_FILE.resolve()), str(out.resolve())], cwd=HERE.parent.parent.parent)
    return load_summary(out / "batch_summary.csv")


def q(row: dict[str, str], field: str, fallback: str) -> str:
    value = row.get(field, "").strip()
    return value if value else fallback


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
    winners = {m: winner(rows, m) for m in ("ffn", "lstm")}
    return rows, winners


def stage2(a: argparse.Namespace, root: Path, s1: dict[str, dict[str, str]]) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    print("\n[adaptive] STAGE 2: physics tuning inherited from Stage 1", flush=True)
    ffn, lstm = s1["ffn"], s1["lstm"]
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
    ], root / "02_stage2_physics")
    winners = {m: winner(rows, m) for m in ("ffn_pinn", "lstm_pinn", "pinn")}
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
    if mode in PHYSICS:
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
    winners = {m: winner(rows, m) for m in MODES}
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


def corr(x: list[float], y: list[float]) -> float:
    if len(x) < 2: return math.nan
    mx, my = statistics.fmean(x), statistics.fmean(y)
    sx = math.sqrt(sum((v-mx)**2 for v in x)); sy = math.sqrt(sum((v-my)**2 for v in y))
    if sx == 0 or sy == 0: return math.nan
    return sum((a-mx)*(b-my) for a,b in zip(x,y))/(sx*sy)


def whole_domain_metrics(predictions: Path) -> dict[str, float]:
    with predictions.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    obs = [float(r["observed"]) for r in rows]; pred = [float(r["predicted"]) for r in rows]
    n = len(obs)
    if n == 0: raise RuntimeError(f"Empty predictions: {predictions}")
    err = [p-o for o,p in zip(obs,pred)]
    mse = sum(e*e for e in err)/n; rmse = math.sqrt(mse); mae = sum(abs(e) for e in err)/n
    mean_o = statistics.fmean(obs); mean_p = statistics.fmean(pred)
    denom = sum((o-mean_o)**2 for o in obs)
    nse = 1.0 - sum(e*e for e in err)/denom if denom > 0 else math.nan
    pbias = 100.0*sum(err)/sum(obs) if sum(obs) != 0 else math.nan
    r = corr(obs,pred); std_o = statistics.pstdev(obs); std_p = statistics.pstdev(pred)
    alpha = std_p/std_o if std_o > 0 else math.nan; beta = mean_p/mean_o if mean_o != 0 else math.nan
    kge = 1.0-math.sqrt((r-1)**2+(alpha-1)**2+(beta-1)**2) if all(math.isfinite(v) for v in (r,alpha,beta)) else math.nan
    return {"whole_mse":mse,"whole_rmse":rmse,"whole_mae":mae,"whole_nse":nse,"whole_kge":kge,"whole_correlation":r,"whole_pbias":pbias}


def finalize(a: argparse.Namespace, root: Path, s1: dict[str, dict[str, str]], s2: dict[str, dict[str, str]], s3: dict[str, dict[str, str]], s4: list[dict[str, str]]) -> None:
    selected: list[dict[str, str]] = []
    for stage, ws in (("stage1", s1), ("stage2", s2), ("stage3", s3)):
        for mode, row in ws.items():
            r = dict(row); r["stage"] = stage; r["selection_metric"] = "validation_mse"; selected.append(r)
    write_rows(root / "adaptive_validation_winners.csv", selected)

    robust_rows: list[dict[str, str]] = []
    for mode in MODES:
        members = [r for r in s4 if r.get("mode") == mode]
        out = {"mode": mode, "seed_count": str(len(members))}
        for field in ("validation_mse","test_mse","rmse","mae","nse","kge","pbias","physics_loss","physics_residual_rmse"):
            vals = [finite_float(r.get(field), math.nan) for r in members]
            m, sd = mean_sd(vals); out[field+"_mean"] = str(m); out[field+"_std"] = str(sd)
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
            metrics = whole_domain_metrics(stage4_root / mode / row["experiment_id"] / "predictions.csv")
            whole_rows.append({"experiment_id":row["experiment_id"],"mode":mode,"seed":row.get("random_seed",""),**{k:str(v) for k,v in metrics.items()}})
        write_rows(root / "synthetic_whole_domain_metrics.csv", whole_rows)
        summary = []
        for mode in MODES:
            members = [r for r in whole_rows if r["mode"] == mode]; out = {"mode":mode,"seed_count":str(len(members))}
            for field in ("whole_rmse","whole_mae","whole_nse","whole_kge","whole_correlation","whole_pbias"):
                m,sd=mean_sd([float(r[field]) for r in members]); out[field+"_mean"]=str(m); out[field+"_std"]=str(sd)
            summary.append(out)
        write_rows(root / "paper_synthetic_known_truth_summary.csv", summary)

    paper_rows=[]
    for mode in MODES:
        b=s3[mode]; rr=next(r for r in robust_rows if r["mode"]==mode)
        out={"mode":mode,"hidden_layers":b.get("hidden_layers",""),"activation":b.get("activation",""),"input_lags":b.get("input_lags",""),"lstm_sequence_length":b.get("lstm_sequence_length",""),"learning_rate":b.get("learning_rate",""),"batch_size":b.get("batch_size",""),"physics_weight":b.get("physics_weight",""),"reservoir_k":b.get("latent_recession_per_hour","")}
        out.update(rr); paper_rows.append(out)
    write_rows(root / "paper_method_summary.csv", paper_rows)

    lines=["HydroPINN adaptive paper-run manifest","====================================",f"data_source={a.data_source}",f"output_root={root}","selection_metric=validation_mse (test metrics never used for tuning)","stage4=multi-seed robustness of frozen Stage-3 configurations",""]
    for mode in MODES:
        b=s3[mode]
        lines.append(f"{mode}: hidden={b.get('hidden_layers','')} act={b.get('activation','')} lags={b.get('input_lags','')} seq={b.get('lstm_sequence_length','')} lr={b.get('learning_rate','')} batch={b.get('batch_size','')} w={b.get('physics_weight','')} k={b.get('latent_recession_per_hour','')}")
    lines += ["", "Paper artifacts:", "  adaptive_validation_winners.csv", "  paper_robustness_summary.csv", "  paper_method_summary.csv", "  frozen_configs/*.json"]
    if whole_rows: lines += ["  synthetic_whole_domain_metrics.csv", "  paper_synthetic_known_truth_summary.csv"]
    (root / "PAPER_RUN_MANIFEST.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def main() -> int:
    a=parse_args(); root=a.output_root.resolve(); root.mkdir(parents=True, exist_ok=True)
    if not a.hydrobatch.exists(): raise SystemExit(f"HydroBatch not found: {a.hydrobatch}")
    if a.data_source=="synthetic" and a.synthetic_profile!="reduced_reservoir": raise SystemExit("Paper five-method synthetic run requires reduced_reservoir")
    if a.data_source=="hydro" and not a.hydro_package_path: raise SystemExit("Hydro package path is required")
    if a.data_source=="csv" and not a.csv_path: raise SystemExit("CSV path is required")
    _,s1=stage1(a,root); _,s2=stage2(a,root,s1); _,s3=stage3(a,root,s1,s2); s4=stage4(a,root,s3); finalize(a,root,s1,s2,s3,s4)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
