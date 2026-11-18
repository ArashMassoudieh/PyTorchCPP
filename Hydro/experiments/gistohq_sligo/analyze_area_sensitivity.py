#!/usr/bin/env python3
"""Score catchment-area sensitivity runs in common native-discharge units.

HydroBatch exports the shared-scale physics predictions as catchment-normalized
depth (mm/h).  When catchment area itself is the experimental factor, those
files necessarily contain different observed depths for the same native gauge
discharge.  This analyzer converts each arm back to m3/s before checking that
observations agree and before computing common-support metrics.
"""
import argparse
import csv
import json
from pathlib import Path

from audit_development_predictions import metrics
from run_isolated_batch import sha


def parse_model_areas(values):
    areas = {}
    for value in values:
        model, area = value.rsplit("=", 1)
        areas[model] = float(area)
    return areas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-support", type=Path, required=True)
    parser.add_argument(
        "--model-area-km2",
        action="append",
        required=True,
        metavar="MODEL=AREA",
        help="Catchment area used by one manifest model arm; repeat per arm",
    )
    args = parser.parse_args()

    jobs = json.loads(args.manifest.read_text())["jobs"]
    model_areas = parse_model_areas(args.model_area_km2)
    missing_areas = sorted({job["model"] for job in jobs} - set(model_areas))
    if missing_areas:
        raise ValueError(f"Missing model-area mappings: {missing_areas}")

    support = json.loads(args.native_support.read_text())
    if any(check["interpolation_spans"] for check in support["boundary_interpolation_checks"]):
        raise ValueError("Target interpolation crosses a partition boundary")
    native = set(support["strict_observed_hour_x"])

    incomplete = [job["id"] for job in jobs if not (args.runs / job["id"] / "VERIFIED.json").exists()]
    if incomplete:
        raise RuntimeError(f"{len(incomplete)} incomplete runs; refusing analysis")

    data = {}
    for job in jobs:
        root = args.runs / job["id"]
        path = root / job["id"] / "predictions.csv"
        verified = json.loads((root / "VERIFIED.json").read_text())
        if sha(path) != verified["predictions_sha256"]:
            raise ValueError(f"Predictions changed since verification: {job['id']}")
        if sha(job["config"]) != verified["config_sha256"]:
            raise ValueError(f"Configuration changed since verification: {job['id']}")

        # 1 mm/h over 1 km2 equals 1/3.6 m3/s.
        depth_to_cms = model_areas[job["model"]] / 3.6
        rows = {}
        with path.open() as handle:
            for row in csv.DictReader(handle):
                if row["split"] == "test" and float(row["x"]) >= job["test_start_x"]:
                    x = float(row["x"])
                    if x in rows:
                        raise ValueError(f"Duplicate test time in {job['id']}: {x}")
                    rows[x] = (
                        float(row["observed"]) * depth_to_cms,
                        float(row["predicted"]) * depth_to_cms,
                    )
        data[job["id"]] = rows

    coverage = {}
    pooled = {}
    annual = []
    # HydroBatch writes float32 depth.  Reversing the two area conversions
    # produces at most 3.2e-6 m3/s disagreement in this frozen batch.
    observation_tolerance_cms = 1e-5
    for year in sorted({job["year"] for job in jobs}):
        group = [job for job in jobs if job["year"] == year]
        common = set.intersection(*(set(data[job["id"]]) for job in group))
        before_native = len(common)
        common &= native
        if not common:
            raise ValueError(f"No common test support for {year}")
        coverage[year] = {
            "common_hours": len(common),
            "excluded_non_native_hours": before_native - len(common),
            "expected_calendar_hours": group[0]["bounds"]["test_end_x"] - group[0]["test_start_x"] + 1,
            "individual_test_hours": {job["id"]: len(data[job["id"]]) for job in group},
        }
        reference = data[group[0]["id"]]
        for job in group:
            rows = []
            for x in sorted(common):
                observed, predicted = data[job["id"]][x]
                if abs(observed - reference[x][0]) > observation_tolerance_cms:
                    raise ValueError(
                        f"Native-discharge observation disagreement at {x}: "
                        f"{observed} versus {reference[x][0]}"
                    )
                # Use one common observation value so paired metric differences
                # cannot contain floating-point conversion noise.
                rows.append((x, reference[x][0], predicted))
            annual.append(
                dict(year=year, model=job["model"], seed=job["seed"], **metrics(rows))
            )
            pooled.setdefault((job["model"], job["seed"]), []).extend(rows)

    args.output.mkdir(parents=True, exist_ok=True)
    aggregate = []
    for (model, seed), rows in pooled.items():
        xs = [row[0] for row in rows]
        if len(xs) != len(set(xs)):
            raise ValueError("Overlapping annual tests")
        aggregate.append(dict(model=model, seed=seed, **metrics(rows)))
        with (args.output / f"out_of_sample_{model}_s{seed}.csv").open("w") as handle:
            writer = csv.writer(handle)
            writer.writerow(["x", "observed", "predicted"])
            writer.writerows(rows)

    for name, rows in (("annual_metrics.csv", annual), ("pooled_metrics.csv", aggregate)):
        with (args.output / name).open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (args.output / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")
    (args.output / "area_conversion.json").write_text(
        json.dumps(
            {
                "input_units": "mm/h",
                "scored_units": "m3/s",
                "formula": "Q_m3_s = depth_mm_h * area_km2 / 3.6",
                "model_area_km2": model_areas,
                "observation_tolerance_m3_s": observation_tolerance_cms,
            },
            indent=2,
        )
        + "\n"
    )
    print("Wrote common-support metrics in native discharge units (m3/s).")


if __name__ == "__main__":
    main()
