#!/usr/bin/env python3
"""Freeze the common-scale reduced-urban physics screen.

Each year uses parameters selected only on that year's validation window. The
managed and no-detention arms are identical except for the detention-area
fraction, while the zero arm removes the physical baseline entirely. All arms
share the same 24-hour LSTM, discharge scale, seed, optimizer, and partitions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


YEARS = (2018, 2020, 2023)
ARMS = ("zero", "urban_no_detention", "urban_managed")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finalist-configs", type=Path, required=True)
    parser.add_argument("--selected-params", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seeds", default="42", help="Comma-separated random seeds")
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    seeds = tuple(int(token.strip()) for token in args.seeds.split(",") if token.strip())
    if not seeds or any(seed < 0 for seed in seeds):
        raise ValueError("seeds must contain non-negative integers")
    selected = json.loads(args.selected_params.read_text())
    finalist_manifest_path = args.finalist_configs / "run_manifest.json"
    finalist_manifest = json.loads(finalist_manifest_path.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = []
    for year in YEARS:
        template_path = args.finalist_configs / f"finalist_{year}_baseline24_s42.json"
        template = json.loads(template_path.read_text())
        source_jobs = [j for j in finalist_manifest["jobs"]
                       if int(j["year"]) == year and j["variant"] == "baseline24" and int(j["seed"]) == 42]
        if len(source_jobs) != 1:
            raise RuntimeError(f"Expected one finalist control source for {year}; found {len(source_jobs)}")
        source_job = source_jobs[0]
        # Calibrate the complete managed structure on validation, then remove
        # only its detention fraction for the causal structural control.
        p = selected[str(year)]["managed"]
        for arm in ARMS:
          for seed in seeds:
            run_id = f"urbanphysics_{year}_{arm}_s{seed}"
            config = dict(template)
            config.update({
                "experiment_id": run_id,
                "epochs": args.epochs,
                "random_seed": seed,
                "physics_profile": (
                    "lstm_shared_scale_ablation_zero" if arm == "zero"
                    else "lstm_shared_scale_ablation_urban"
                ),
                "urban_quick_k": p["quick_k"],
                "urban_slow_k": p["slow_k"],
                "urban_detention_k": p["detention_k"],
                "urban_channel_k": p["channel_k"],
                "urban_quick_fraction": p["quick_fraction"],
                "urban_detention_fraction": 0.0 if arm == "urban_no_detention" else p["detention_fraction"],
                "urban_runoff_coefficient": p["runoff_coefficient"],
                "pinn_routing_lag_hours": float(p["lag_steps"]),
                # These profiles do not use the weighted-loss knobs.
                "physical_error_weight": 0.0,
                "high_flow_loss_weight": 0.0,
            })
            config_path = (args.output / f"{run_id}.json").resolve()
            batch_path = (args.output / f"{run_id}.batch").resolve()
            config_path.write_text(json.dumps(config, indent=2) + "\n")
            batch_path.write_text(f"lstm_pinn {config_path}\n")
            jobs.append({
                "id": run_id,
                "year": str(year),
                "model": f"lstm_urbanphysics_{arm}",
                "variant": arm,
                "seed": seed,
                "test_start_x": source_job["test_start_x"],
                "config_sha256": sha(config_path),
                "config": str(config_path),
                "batch": str(batch_path),
                "bounds": {
                    "train_end_x": template["train_end_x"],
                    "validation_end_x": template["validation_end_x"],
                    "test_end_x": template["test_end_x"],
                },
            })
    manifest = {
        "purpose": "Common-scale 24-hour LSTM reduced-urban physics screen",
        "selection_warning": "Parameters selected on development-year validation windows; test years remain development evidence",
        "parameter_source": str(args.selected_params.resolve()),
        "parameter_source_sha256": sha(args.selected_params),
        "source_finalist_manifest": str(finalist_manifest_path.resolve()),
        "source_finalist_manifest_sha256": sha(finalist_manifest_path),
        "causal_contrast": "managed and no-detention arms differ only in urban_detention_fraction",
        "years": list(YEARS),
        "seeds": list(seeds),
        "epochs": args.epochs,
        "arms": list(ARMS),
        "jobs": jobs,
    }
    manifest_path = args.output / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(jobs)} frozen jobs to {args.output}")
    print(f"manifest_sha256={sha(manifest_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
