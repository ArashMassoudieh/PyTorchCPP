#!/usr/bin/env python3
"""Rank leakage-safe CAMELSH donor basins for Sligo Creek transfer tests."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


TARGET = {
    "staid": "01650800",
    "lat": 38.986222,
    "lon": -77.004861,
    "area_km2": 16.7013,
    "developed_pct": 82.166873,
}
EXCLUDED = {"01650800", "01651000"}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    with args.selection.open() as stream:
        for source in csv.DictReader(stream):
            staid = source["STAID"]
            if staid in EXCLUDED:
                continue
            lat, lon = float(source["LAT_GAGE"]), float(source["LNG_GAGE"])
            area = float(source["DRAIN_SQKM"])
            developed = float(source["developed_pct"])
            distance = haversine_km(TARGET["lat"], TARGET["lon"], lat, lon)
            area_distance = abs(math.log(area / TARGET["area_km2"]))
            developed_distance = abs(developed - TARGET["developed_pct"]) / 100.0
            # Location matters most because it preserves climate/storm type;
            # log-area and developed cover keep response scales comparable.
            score = distance / 100.0 + area_distance + developed_distance
            rows.append({
                "rank": 0,
                "STAID": staid,
                "STANAME": source["STANAME"],
                "STATE": source["STATE"],
                "distance_km": distance,
                "area_km2": area,
                "area_ratio_to_target": area / TARGET["area_km2"],
                "developed_pct": developed,
                "coverage_years_2015_2022": int(source["years_passing_2015_2022"]),
                "similarity_score": score,
            })
    rows.sort(key=lambda row: (row["similarity_score"], row["STAID"]))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Ranked {len(rows)} leakage-safe candidates; top 20 written in {args.output}")
    for row in rows[:20]:
        print(row["rank"], row["STAID"], f"{row['distance_km']:.1f} km", row["STANAME"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
