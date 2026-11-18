#!/usr/bin/env python3
"""Copy a development package with an explicit alternative catchment area."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-package", type=Path, required=True)
    parser.add_argument("--output-package", type=Path, required=True)
    parser.add_argument("--area-m2", type=float, required=True)
    parser.add_argument("--area-source", required=True)
    args = parser.parse_args()
    if args.area_m2 <= 0:
        raise ValueError("area-m2 must be positive")
    if args.output_package.exists():
        raise RuntimeError(f"Refusing to overwrite existing output: {args.output_package}")
    shutil.copytree(args.source_package, args.output_package)
    manifest_path = args.output_package / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    original_area = float(manifest["catchment_area_m2"])
    manifest["site_id"] += "_area_sensitivity"
    manifest["catchment_area_m2"] = args.area_m2
    manifest["catchment_area_source"] = args.area_source
    manifest["qc_gate"] = "not_run"
    manifest["provenance_note"] = (
        manifest.get("provenance_note", "") + f" Area-sensitivity variant: existing forcing and "
        f"discharge observations are byte-for-byte copies; only catchment_area_m2 changed from "
        f"{original_area:.17g} to {args.area_m2:.17g}. The consumer re-derives runoff depth from "
        "the unchanged discharge using this declared area. Source package remains unchanged."
    ).strip()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output_package),
        "original_area_m2": original_area,
        "alternative_area_m2": args.area_m2,
        "runoff_depth_scale_factor": original_area / args.area_m2,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
