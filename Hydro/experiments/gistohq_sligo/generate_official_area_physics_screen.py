#!/usr/bin/env python3
"""Freeze the drainage-area sensitivity of the supported urban physics arm."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


YEARS = (2018, 2020, 2023)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urban-screen-configs", type=Path, required=True)
    parser.add_argument("--base-package", type=Path, required=True)
    parser.add_argument("--official-area-package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.seed < 0:
        raise ValueError("epochs must be positive and seed non-negative")
    source_manifest_path = args.urban_screen_configs / "run_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    variants = {
        "nldi_area": args.base_package.resolve(),
        "official_area": args.official_area_package.resolve(),
    }
    jobs = []
    for year in YEARS:
        template_path = args.urban_screen_configs / f"urbanphysics_{year}_urban_no_detention_s42.json"
        template = json.loads(template_path.read_text())
        matches = [j for j in source_manifest["jobs"]
                   if int(j["year"]) == year and j["variant"] == "urban_no_detention"
                   and int(j["seed"]) == 42]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one no-detention source for {year}")
        source = matches[0]
        for variant, package in variants.items():
            run_id = f"area_physics_{year}_{variant}_s{args.seed}"
            config = dict(template)
            config.update({
                "experiment_id": run_id,
                "epochs": args.epochs,
                "random_seed": args.seed,
                "hydro_package_path": str(package),
                "high_flow_loss_weight": 0.0,
                "physical_error_weight": 0.0,
                "use_mrms_spatial_cv_feature": False,
            })
            config_path = (args.output / f"{run_id}.json").resolve()
            batch_path = (args.output / f"{run_id}.batch").resolve()
            config_path.write_text(json.dumps(config, indent=2) + "\n")
            batch_path.write_text(f"lstm_pinn {config_path}\n")
            jobs.append({
                "id": run_id,
                "year": str(year),
                "model": f"lstm_urbanphysics_{variant}",
                "variant": variant,
                "seed": args.seed,
                "test_start_x": source["test_start_x"],
                "config_sha256": sha(config_path),
                "config": str(config_path),
                "batch": str(batch_path),
                "bounds": source["bounds"],
            })
    manifest = {
        "purpose": "Drainage-area sensitivity of supported no-detention urban physics",
        "selection_warning": "Development-period sensitivity; reserved target period remains sealed",
        "source_urban_manifest": str(source_manifest_path.resolve()),
        "source_urban_manifest_sha256": sha(source_manifest_path),
        "base_package": str(args.base_package.resolve()),
        "base_area_m2": json.loads((args.base_package / "manifest.json").read_text())["catchment_area_m2"],
        "base_package_manifest_sha256": sha(args.base_package / "manifest.json"),
        "official_area_package": str(args.official_area_package.resolve()),
        "official_area_m2": json.loads((args.official_area_package / "manifest.json").read_text())["catchment_area_m2"],
        "official_area_package_manifest_sha256": sha(args.official_area_package / "manifest.json"),
        "years": list(YEARS),
        "seeds": [args.seed],
        "epochs": args.epochs,
        "jobs": jobs,
    }
    manifest_path = args.output / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(jobs)} frozen jobs to {args.output}")
    print(f"manifest_sha256={sha(manifest_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
