#!/usr/bin/env python3
"""Build a seal-safe package variant carrying MRMS spatial variability.

The acquired MRMS catalog already stores each hour's within-basin cell
coefficient of variation. This script copies an existing development package
and appends that diagnostic under an honest, dimensionless variable name. It
never modifies the source package and never reads target observations beyond
the source package's declared study end.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


VARIABLE = "PRECIP_MRMS_SPATIAL_CV"


def parse_time(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-package", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-package", type=Path, required=True)
    args = parser.parse_args()
    if args.output_package.exists():
        raise RuntimeError(f"Refusing to overwrite existing output: {args.output_package}")

    manifest = json.loads((args.source_package / "manifest.json").read_text())
    end = parse_time(manifest["study_end"])
    catalog = json.loads(args.catalog.read_text())
    rows: dict[str, float] = {}
    null_count = 0
    for asset in catalog["assets"]:
        digest = asset["content_digest"]
        obj = args.cache / "objects" / "sha256" / digest[:2] / digest[2:]
        for row in json.loads(obj.read_text())["rows"]:
            timestamp = row["timestamp_utc"]
            if parse_time(timestamp) > end:
                continue
            value = row.get("cell_cv")
            if value is None:
                null_count += 1
                continue
            value = float(value)
            if value < 0.0:
                raise RuntimeError(f"Negative MRMS spatial CV at {timestamp}")
            if timestamp in rows and rows[timestamp] != value:
                raise RuntimeError(f"Conflicting MRMS spatial CV at {timestamp}")
            rows[timestamp] = value

    shutil.copytree(args.source_package, args.output_package)
    temporal = args.output_package / "observations" / "temporal_5.csv"
    with temporal.open("w") as stream:
        stream.write("timestamp_utc,variable,value,native_unit,provider_qualifiers\n")
        for timestamp in sorted(rows, key=parse_time):
            stream.write(f"{timestamp},{VARIABLE},{rows[timestamp]:.17g},1,MRMS_CELL_CV\n")

    variables_path = args.output_package / "variables.json"
    variables = json.loads(variables_path.read_text())
    if any(item.get("name") == VARIABLE for item in variables["variables"]):
        raise RuntimeError(f"Source package already declares {VARIABLE}")
    variables["variables"].append({
        "missing_values": "zero_fallback_when_absent_or_mean_rain_is_zero",
        "name": VARIABLE,
        "normalization": None,
        "units": ["1"],
    })
    variables_path.write_text(json.dumps(variables, indent=2) + "\n")

    manifest_path = args.output_package / "manifest.json"
    manifest["site_id"] = manifest["site_id"] + "_mrms_spatial_cv"
    manifest["qc_gate"] = "not_run"
    manifest["provenance_note"] = (
        manifest.get("provenance_note", "") + " Spatial-feature variant: copied without "
        "changing existing values and appended PRECIP_MRMS_SPATIAL_CV from the already-acquired "
        "MRMS catalog's per-hour cell_cv field. Null/undefined CV hours are absent and the HydroPINN "
        "optional feature maps them to zero. Source package remains unchanged."
    ).strip()
    manifest.setdefault("assets", []).append({
        "path": "observations/temporal_5.csv",
        "description": "MRMS within-basin grid-cell coefficient of variation (dimensionless)",
    })
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output_package),
        "rows": len(rows),
        "null_or_undefined_hours": null_count,
        "first": min(rows) if rows else None,
        "last": max(rows) if rows else None,
        "source_study_end": manifest["study_end"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
