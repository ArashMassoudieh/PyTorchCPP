#!/usr/bin/env python3
"""Run three leakage-safe rolling-origin Sligo evaluations and aggregate metrics.

Each fold reruns the established adaptive five-method pipeline with an expanding
training window and a contiguous 10% validation block. Hyperparameters are
selected from validation only. HydroBatch's remaining post-validation rows are
held out; for cross-fold comparability this driver reports metrics only on the
predeclared 10%-record test window from paper_rolling_fold_diagnostics.csv.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FOLD_RUNNER = HERE / "run_rolling_fold_pipeline.py"
MODES = ("ffn", "ffn_pinn", "lstm", "lstm_pinn", "pinn")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hydrobatch", required=True, type=Path)
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--diagnostics-csv", required=True, type=Path)
    p.add_argument("--hydro-package-path", required=True)
    p.add_argument("--hydro-catchment-id", default="")
    p.add_argument("--hydro-package-profile", default="rainfall-runoff")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--robust-seeds", default="42,123,2026,31415,27182")
    return p.parse_args()


def read_folds(path: Path):
    with path.open(newline="", encoding="utf-8") as f: rows = list(csv.DictReader(f))
    folds = []
    for fold in sorted({int(r["fold"]) for r in rows}):
        group = [r for r in rows if int(r["fold"]) == fold]
        train = next(r for r in group if r["split"] == "train")
        val = next(r for r in group if r["split"] == "validation")
        test = next(r for r in group if r["split"] == "test")
        folds.append({"fold": fold, "train_fraction": float(train["train_end_fraction"]),
                      "validation_fraction": int(val["count"]) / sum(int(r["count"]) for r in rows if int(r["fold"]) == 1) if False else 0.10,
                      "test_start_index": int(test["start_index"]), "test_end_index": int(test["end_index"]),
                      "test_q90_events": int(test["q90_event_count"]), "test_q95_events": int(test["q95_event_count"])})
    return folds


def corr(x, y):
    if len(x) < 2: return math.nan
    mx, my = statistics.fmean(x), statistics.fmean(y)
    sx = math.sqrt(sum((v-mx)**2 for v in x)); sy = math.sqrt(sum((v-my)**2 for v in y))
    if sx <= 1e-15 or sy <= 1e-15: return math.nan
    return sum((a-mx)*(b-my) for a,b in zip(x,y))/(sx*sy)


def metrics(obs, pred):
    n=len(obs)
    if n < 2: return {k: math.nan for k in ("rmse","mae","nse","kge","pearson_r2","pbias","prediction_std")}
    err=[p-o for o,p in zip(obs,pred)]; mse=sum(e*e for e in err)/n; mo=statistics.fmean(obs); mp=statistics.fmean(pred)
    ssto=sum((o-mo)**2 for o in obs); so=statistics.pstdev(obs); sp=statistics.pstdev(pred); r=corr(obs,pred)
    nse=1-sum(e*e for e in err)/ssto if ssto>0 else math.nan
    alpha=sp/so if so>0 else math.nan; beta=mp/mo if mo!=0 else math.nan
    kge=1-math.sqrt((r-1)**2+(alpha-1)**2+(beta-1)**2) if all(math.isfinite(v) for v in (r,alpha,beta)) else math.nan
    pbias=100*sum(err)/sum(obs) if sum(obs)!=0 else math.nan
    return {"rmse":math.sqrt(mse),"mae":sum(abs(e) for e in err)/n,"nse":nse,"kge":kge,
            "pearson_r2":r*r if math.isfinite(r) else math.nan,"pbias":pbias,"prediction_std":sp}


def predictions_for_window(path, lo, hi):
    obs,pred=[],[]
    with path.open(newline="",encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try: idx=int(r["index"]); o=float(r["observed"]); p=float(r["predicted"])
            except (KeyError,ValueError,TypeError): continue
            if lo <= idx <= hi and math.isfinite(o) and math.isfinite(p): obs.append(o); pred.append(p)
    return obs,pred


def write_csv(path, rows):
    if not rows: return
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)


def main():
    a=parse_args(); root=a.output_root.resolve(); root.mkdir(parents=True,exist_ok=True); folds=read_folds(a.diagnostics_csv.resolve())
    all_rows=[]
    for spec in folds:
        fold=spec["fold"]; out=root/f"fold_{fold:02d}"
        env=os.environ.copy(); env["HYDROPINN_ROLLING_TRAIN_FRACTION"]=str(spec["train_fraction"]); env["HYDROPINN_ROLLING_VALIDATION_FRACTION"]="0.10"
        cmd=[sys.executable,str(FOLD_RUNNER),"--hydrobatch",str(a.hydrobatch.resolve()),"--output-root",str(out),"--data-source","hydro",
             "--hydro-package-path",a.hydro_package_path,"--hydro-catchment-id",a.hydro_catchment_id,"--hydro-package-profile",a.hydro_package_profile,
             "--epochs",str(a.epochs),"--robust-seeds",a.robust_seeds]
        print(f"\n[rolling-cv] FOLD {fold}: train={spec['train_fraction']:.0%}, validation=10%, reported test indices={spec['test_start_index']}..{spec['test_end_index']}",flush=True)
        subprocess.run(cmd,cwd=HERE,env=env,check=True)
        stage4=out/"04_stage4_robustness"
        for mode in MODES:
            for pred_path in sorted((stage4/mode).glob("*/predictions.csv")):
                obs,pred=predictions_for_window(pred_path,spec["test_start_index"],spec["test_end_index"])
                m=metrics(obs,pred); all_rows.append({"fold":fold,"train_fraction":spec["train_fraction"],"validation_fraction":0.10,
                    "test_start_index":spec["test_start_index"],"test_end_index":spec["test_end_index"],"test_q90_events":spec["test_q90_events"],"test_q95_events":spec["test_q95_events"],
                    "mode":mode,"experiment_id":pred_path.parent.name,"n_test":len(obs),"near_constant":math.isfinite(m["prediction_std"]) and m["prediction_std"]<=1e-10,**m})
        write_csv(root/"paper_rolling_cv_seed_metrics.csv",all_rows)

    summary=[]
    for fold in sorted({int(r["fold"]) for r in all_rows}):
        for mode in MODES:
            members=[r for r in all_rows if int(r["fold"])==fold and r["mode"]==mode]
            row={"fold":fold,"mode":mode,"seed_count":len(members),"near_constant_seed_count":sum(bool(r["near_constant"]) for r in members)}
            for key in ("rmse","mae","nse","kge","pearson_r2","pbias","prediction_std"):
                vals=[float(r[key]) for r in members if math.isfinite(float(r[key]))]
                row[key+"_mean"]=statistics.fmean(vals) if vals else math.nan; row[key+"_std"]=statistics.pstdev(vals) if len(vals)>1 else (0.0 if vals else math.nan)
            summary.append(row)
    write_csv(root/"paper_rolling_cv_summary.csv",summary)
    print(f"\n[rolling-cv] COMPLETE: {root}",flush=True); print(f"[rolling-cv] seed metrics: {root/'paper_rolling_cv_seed_metrics.csv'}",flush=True); print(f"[rolling-cv] summary: {root/'paper_rolling_cv_summary.csv'}",flush=True)
    return 0


if __name__=="__main__": raise SystemExit(main())
