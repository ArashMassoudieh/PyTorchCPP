#!/usr/bin/env python3
"""Freeze the Stage-A HydroPINN finalists for paired-seed 150-epoch runs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


YEARS = (2018, 2020, 2023)
SEEDS = (42, 123, 27182)
FINALISTS = ("baseline24", "seq72", "seq72_api168")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.screen_manifest.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = []
    for year in YEARS:
        for variant in FINALISTS:
            matches = [job for job in source["jobs"]
                       if int(job["year"]) == year and job["variant"] == variant]
            if len(matches) != 1:
                raise RuntimeError(f"Expected one screen source for {year}/{variant}; found {len(matches)}")
            original = matches[0]
            template = json.loads(Path(original["config"]).read_text())
            for seed in SEEDS:
                run_id = f"finalist_{year}_{variant}_s{seed}"
                config = dict(template)
                config.update({"experiment_id": run_id, "epochs": 150, "random_seed": seed})
                config_path = (args.output / f"{run_id}.json").resolve()
                batch_path = (args.output / f"{run_id}.batch").resolve()
                config_path.write_text(json.dumps(config, indent=2) + "\n")
                batch_path.write_text(f"lstm {config_path}\n")
                jobs.append({
                    "id": run_id,
                    "year": str(year),
                    "model": f"lstm_finalist_{variant}",
                    "variant": variant,
                    "seed": seed,
                    "test_start_x": original["test_start_x"],
                    "config_sha256": sha(config_path),
                    "config": str(config_path),
                    "batch": str(batch_path),
                    "bounds": original["bounds"],
                })
    manifest = {
        "source_screen_manifest": str(args.screen_manifest.resolve()),
        "source_screen_manifest_sha256": sha(args.screen_manifest),
        "purpose": "Stage-A finalists; 150 epochs; paired seeds; development diagnostic years",
        "selection_basis": {
            "baseline24": "control",
            "seq72": "highest pooled common-support NSE (0.651841) and gains in 2018/2023",
            "seq72_api168": "best 2023 NSE (0.781326), gains in 2018/2023, and best large-event median NSE",
        },
        "selection_warning": "2018, 2020, and 2023 have been inspected and are development evidence, not untouched holdouts",
        "years": list(YEARS),
        "seeds": list(SEEDS),
        "epochs": 150,
        "finalists": list(FINALISTS),
        "jobs": jobs,
    }
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(jobs)} frozen finalist jobs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
