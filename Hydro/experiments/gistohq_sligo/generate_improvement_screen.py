#!/usr/bin/env python3
"""Generate the first bounded LSTM improvement screen from frozen annual folds.

The screen is intentionally cheap (30 epochs, one seed, three diagnostic
years).  It is a development experiment, not final evidence.  Candidates are
promoted only after validation/test diagnostics and later paired-seed reruns.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


YEARS = (2018, 2020, 2023)
SEED = 42
VARIANTS = {
    "baseline24": {},
    "seq48": {"lstm_sequence_length": 48},
    "seq72": {"lstm_sequence_length": 72},
    "seq120": {"lstm_sequence_length": 120},
    "seq72_shuffle": {"lstm_sequence_length": 72, "shuffle_training": True},
    "seq72_b64_lr003": {"lstm_sequence_length": 72, "batch_size": 64, "learning_rate": 0.003},
    "seq72_api24": {"lstm_sequence_length": 72, "antecedent_index_halflife_hours": 24.0},
    "seq72_api72": {"lstm_sequence_length": 72, "antecedent_index_halflife_hours": 72.0},
    "seq72_api168": {"lstm_sequence_length": 72, "antecedent_index_halflife_hours": 168.0},
    "seq72_phys1": {"lstm_sequence_length": 72, "physical_error_weight": 1.0},
    "seq72_h64": {"lstm_sequence_length": 72, "hidden_layers": "64"},
    "seq72_h64x2_drop02": {"lstm_sequence_length": 72, "hidden_layers": "64,64", "lstm_dropout": 0.2},
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annual-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.annual_manifest.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = []
    for year in YEARS:
        matches = [job for job in source["jobs"]
                   if int(job["year"]) == year and job["model"] == "lstm_24h" and int(job["seed"]) == SEED]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one frozen LSTM source job for {year}; found {len(matches)}")
        original = matches[0]
        template = json.loads(Path(original["config"]).read_text())
        for variant, changes in VARIANTS.items():
            run_id = f"screen_{year}_{variant}_s{SEED}"
            config = dict(template)
            config.update(changes)
            config.update({"experiment_id": run_id, "epochs": 30, "random_seed": SEED})
            config_path = (args.output / f"{run_id}.json").resolve()
            batch_path = (args.output / f"{run_id}.batch").resolve()
            config_path.write_text(json.dumps(config, indent=2) + "\n")
            batch_path.write_text(f"lstm {config_path}\n")
            jobs.append({
                "id": run_id,
                "year": str(year),
                "model": f"lstm_screen_{variant}",
                "variant": variant,
                "seed": SEED,
                "test_start_x": original["test_start_x"],
                "config_sha256": sha(config_path),
                "config": str(config_path),
                "batch": str(batch_path),
                "bounds": original["bounds"],
            })
    manifest = {
        "source_annual_manifest": str(args.annual_manifest.resolve()),
        "source_annual_manifest_sha256": sha(args.annual_manifest),
        "purpose": "successive-halving screen; development-only; test years already exploratory",
        "selection_rule": "promote only multi-year gains; rerun finalists at 150 epochs with paired seeds",
        "years": list(YEARS),
        "seed": SEED,
        "epochs": 30,
        "variants": VARIANTS,
        "jobs": jobs,
    }
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(jobs)} frozen screening jobs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
