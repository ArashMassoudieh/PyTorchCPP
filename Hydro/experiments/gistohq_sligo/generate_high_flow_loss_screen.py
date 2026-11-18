#!/usr/bin/env python3
"""Freeze a bounded training-only high-flow-loss screen for the 24 h LSTM."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


YEARS = (2018, 2020, 2023)
VARIANTS = {
    "control": (0.99, 0.0),
    "q95_w025": (0.95, 0.25),
    "q95_w1": (0.95, 1.0),
    "q99_w025": (0.99, 0.25),
    "q99_w1": (0.99, 1.0),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finalist-configs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--id-prefix", default="highflow",
                        help="Experiment-ID prefix, used to keep corrected reruns distinct")
    args = parser.parse_args()
    if args.epochs < 1 or args.seed < 0:
        raise ValueError("epochs must be positive and seed non-negative")
    source_manifest_path = args.finalist_configs / "run_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = []
    for year in YEARS:
        template_path = args.finalist_configs / f"finalist_{year}_baseline24_s42.json"
        template = json.loads(template_path.read_text())
        matches = [j for j in source_manifest["jobs"]
                   if int(j["year"]) == year and j["variant"] == "baseline24" and int(j["seed"]) == 42]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one source control for {year}; found {len(matches)}")
        source = matches[0]
        for variant, (quantile, weight) in VARIANTS.items():
            run_id = f"{args.id_prefix}_{year}_{variant}_s{args.seed}"
            config = dict(template)
            config.update({
                "experiment_id": run_id,
                "epochs": args.epochs,
                "random_seed": args.seed,
                "high_flow_loss_quantile": quantile,
                "high_flow_loss_weight": weight,
                "physical_error_weight": 0.0,
            })
            config_path = (args.output / f"{run_id}.json").resolve()
            batch_path = (args.output / f"{run_id}.batch").resolve()
            config_path.write_text(json.dumps(config, indent=2) + "\n")
            batch_path.write_text(f"lstm {config_path}\n")
            jobs.append({
                "id": run_id,
                "year": str(year),
                "model": f"lstm_highflow_{variant}",
                "variant": variant,
                "seed": args.seed,
                "test_start_x": source["test_start_x"],
                "config_sha256": sha(config_path),
                "config": str(config_path),
                "batch": str(batch_path),
                "bounds": source["bounds"],
            })
    manifest = {
        "purpose": "Bounded training-only high-flow-loss screen on frozen 24-hour LSTM",
        "selection_warning": "Thresholds are fitted from training discharge only; test years are development evidence",
        "source_finalist_manifest": str(source_manifest_path.resolve()),
        "source_finalist_manifest_sha256": sha(source_manifest_path),
        "years": list(YEARS),
        "seeds": [args.seed],
        "epochs": args.epochs,
        "id_prefix": args.id_prefix,
        "variants": {k: {"quantile": q, "weight": w} for k, (q, w) in VARIANTS.items()},
        "jobs": jobs,
    }
    manifest_path = args.output / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(jobs)} frozen jobs to {args.output}")
    print(f"manifest_sha256={sha(manifest_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
