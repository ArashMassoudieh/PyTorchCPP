#!/usr/bin/env python3
"""Diagnose which events dominate the frozen annual HydroPINN backtest.

This is a reporting diagnostic.  It consumes only the already-scored,
common-support out-of-sample series and the development-period MRMS forcing.
It never reads reserved-period target observations and never selects a model.
Neural-model event summaries use the median prediction across frozen seeds;
seed-level instability is exported separately.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


EPOCH = datetime(2015, 1, 1, tzinfo=timezone.utc)


def quantile(values: list[float], q: float) -> float:
    xs = sorted(values)
    if not xs:
        return math.nan
    if len(xs) == 1:
        return xs[0]
    pos = q * (len(xs) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return xs[lo]
    weight = pos - lo
    return xs[lo] * (1 - weight) + xs[hi] * weight


def read_series(path: Path) -> dict[int, tuple[float, float]]:
    rows: dict[int, tuple[float, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            x = int(round(float(row["x"])))
            observed, predicted = float(row["observed"]), float(row["predicted"])
            if not (math.isfinite(observed) and math.isfinite(predicted)):
                raise ValueError(f"Non-finite prediction in {path} at x={x}")
            if x in rows:
                raise ValueError(f"Duplicate x={x} in {path}")
            rows[x] = observed, predicted
    return rows


def parse_prediction_name(path: Path) -> tuple[str, int]:
    stem = path.stem.removeprefix("out_of_sample_")
    model, seed = stem.rsplit("_s", 1)
    return model, int(seed)


def load_ensembles(analysis: Path):
    individual: dict[tuple[str, int], dict[int, tuple[float, float]]] = {}
    by_model: dict[str, list[int]] = defaultdict(list)
    reference: dict[int, float] | None = None
    for path in sorted(analysis.glob("out_of_sample_*_s*.csv")):
        model, seed = parse_prediction_name(path)
        series = read_series(path)
        observations = {x: values[0] for x, values in series.items()}
        if reference is None:
            reference = observations
        elif observations != reference:
            raise ValueError(f"Common-support observations differ in {path}")
        individual[model, seed] = series
        by_model[model].append(seed)
    if reference is None:
        raise FileNotFoundError(f"No out-of-sample series in {analysis}")
    ensemble: dict[str, dict[int, float]] = {}
    for model, seeds in by_model.items():
        ensemble[model] = {
            x: statistics.median(individual[model, seed][x][1] for seed in seeds)
            for x in reference
        }
    return reference, individual, dict(by_model), ensemble


def load_hourly_rain(path: Path) -> dict[int, float]:
    rain: dict[int, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("variable") != "PRECIP_MRMS_BASIN_AVG":
                continue
            stamp = datetime.fromisoformat(row["timestamp_utc"].replace("Z", "+00:00"))
            x = int(round((stamp - EPOCH).total_seconds() / 3600.0))
            value = float(row["value"])
            if math.isfinite(value):
                rain[x] = value
    return rain


def event_cores(observed: dict[int, float], threshold: float, merge_gap: int) -> list[tuple[int, int]]:
    peaks = sorted(x for x, value in observed.items() if value >= threshold)
    if not peaks:
        return []
    groups: list[list[int]] = [[peaks[0]]]
    for x in peaks[1:]:
        if x - groups[-1][-1] <= merge_gap:
            groups[-1].append(x)
        else:
            groups.append([x])
    return [(group[0], group[-1]) for group in groups]


def expanded_events(cores: list[tuple[int, int]], before: int, after: int) -> list[tuple[int, int]]:
    expanded: list[list[int]] = []
    for start, end in cores:
        candidate = [start - before, end + after]
        if expanded and candidate[0] <= expanded[-1][1]:
            expanded[-1][1] = max(expanded[-1][1], candidate[1])
        else:
            expanded.append(candidate)
    return [(start, end) for start, end in expanded]


def stamp(x: int) -> str:
    from datetime import timedelta
    return (EPOCH + timedelta(hours=x)).isoformat().replace("+00:00", "Z")


def event_metrics(xs: list[int], observed: dict[int, float], predicted: dict[int, float],
                  peak_match_radius_hours: int = 12) -> dict[str, float | int]:
    obs = [observed[x] for x in xs]
    pred = [predicted[x] for x in xs]
    residual = [p - o for o, p in zip(obs, pred)]
    sse = sum(value * value for value in residual)
    mean_obs = statistics.fmean(obs)
    denominator = sum((value - mean_obs) ** 2 for value in obs)
    peak_obs_i = max(range(len(xs)), key=lambda i: obs[i])
    # Complex wet periods can contain several distinct rainfall pulses.  Pair
    # the simulated peak with the observed event peak locally so a preceding
    # pulse cannot create a meaningless multi-day timing error.
    peak_candidates = [i for i, x in enumerate(xs)
                       if abs(x - xs[peak_obs_i]) <= peak_match_radius_hours]
    peak_pred_i = max(peak_candidates, key=lambda i: pred[i])
    observed_sum = sum(obs)
    return {
        "n": len(xs),
        "sse": sse,
        "rmse": math.sqrt(sse / len(xs)),
        "nse": 1 - sse / denominator if denominator > 0 else math.nan,
        "observed_peak": obs[peak_obs_i],
        "predicted_peak": pred[peak_pred_i],
        "peak_magnitude_error_percent": 100 * (pred[peak_pred_i] - obs[peak_obs_i]) / obs[peak_obs_i] if obs[peak_obs_i] else math.nan,
        "peak_timing_error_hours": xs[peak_pred_i] - xs[peak_obs_i],
        "volume_error_percent": 100 * (sum(pred) - observed_sum) / observed_sum if observed_sum else math.nan,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--mrms", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
    parser.add_argument("--merge-gap-hours", type=int, default=24)
    parser.add_argument("--hours-before", type=int, default=24)
    parser.add_argument("--hours-after", type=int, default=48)
    parser.add_argument("--figure-count", type=int, default=10)
    parser.add_argument("--peak-match-radius-hours", type=int, default=12)
    args = parser.parse_args()
    if not 0 < args.threshold_quantile < 1:
        raise SystemExit("threshold quantile must lie in (0, 1)")

    observed, individual, by_model, ensemble = load_ensembles(args.analysis)
    rain = load_hourly_rain(args.mrms)
    threshold = quantile(list(observed.values()), args.threshold_quantile)
    cores = event_cores(observed, threshold, args.merge_gap_hours)
    events = expanded_events(cores, args.hours_before, args.hours_after)
    args.output.mkdir(parents=True, exist_ok=True)
    figures = args.output / "figures"
    figures.mkdir(exist_ok=True)

    total_sse = {
        model: sum((prediction[x] - observed[x]) ** 2 for x in observed)
        for model, prediction in ensemble.items()
    }
    event_rows: list[dict] = []
    summary_rows: list[dict] = []
    for number, (start, end) in enumerate(events, 1):
        xs = [x for x in sorted(observed) if start <= x <= end]
        if not xs:
            continue
        obs_peak_x = max(xs, key=lambda x: observed[x])
        model_metrics = {}
        for model in sorted(ensemble):
            values = event_metrics(xs, observed, ensemble[model], args.peak_match_radius_hours)
            values["sse_share_percent"] = 100 * float(values["sse"]) / total_sse[model]
            model_metrics[model] = values
            event_rows.append({
                "event": number,
                "start_x": xs[0],
                "end_x": xs[-1],
                "start_utc": stamp(xs[0]),
                "end_utc": stamp(xs[-1]),
                "model": model,
                **values,
            })
        lstm_key = "lstm_residual_hybrid" if "lstm_residual_hybrid" in model_metrics else sorted(model_metrics)[0]
        lstm_values = model_metrics[lstm_key]
        summary_rows.append({
            "event": number,
            "start_utc": stamp(xs[0]),
            "end_utc": stamp(xs[-1]),
            "observed_peak_utc": stamp(obs_peak_x),
            "observed_peak": observed[obs_peak_x],
            "rain_window_mm": sum(rain.get(x, 0.0) for x in range(xs[0], xs[-1] + 1)),
            "rain_antecedent_72h_mm": sum(rain.get(x, 0.0) for x in range(xs[0] - 72, xs[0])),
            "lstm_hybrid_sse": lstm_values["sse"],
            "lstm_hybrid_sse_share_percent": lstm_values["sse_share_percent"],
            "lstm_hybrid_peak_error_percent": lstm_values["peak_magnitude_error_percent"],
            "lstm_hybrid_peak_timing_hours": lstm_values["peak_timing_error_hours"],
        })

    summary_rows.sort(key=lambda row: float(row["lstm_hybrid_sse"]), reverse=True)
    for rank, row in enumerate(summary_rows, 1):
        row["lstm_hybrid_error_rank"] = rank
    write_csv(args.output / "event_summary.csv", summary_rows)
    write_csv(args.output / "event_metrics.csv", event_rows)

    seed_rows: list[dict] = []
    for number, (start, end) in enumerate(events, 1):
        xs = [x for x in sorted(observed) if start <= x <= end]
        for (model, seed), series in sorted(individual.items()):
            prediction = {x: series[x][1] for x in xs}
            seed_rows.append({"event": number, "model": model, "seed": seed,
                              **event_metrics(xs, observed, prediction, args.peak_match_radius_hours)})
    write_csv(args.output / "event_seed_metrics.csv", seed_rows)

    hourly_rows: list[dict] = []
    for model, prediction in sorted(ensemble.items()):
        ranked = sorted(observed, key=lambda x: (prediction[x] - observed[x]) ** 2, reverse=True)
        cumulative = 0.0
        for rank, x in enumerate(ranked, 1):
            squared_error = (prediction[x] - observed[x]) ** 2
            cumulative += squared_error
            hourly_rows.append({
                "model": model, "rank": rank, "x": x, "timestamp_utc": stamp(x),
                "observed": observed[x], "predicted": prediction[x], "squared_error": squared_error,
                "cumulative_sse_percent": 100 * cumulative / total_sse[model],
                "rain_mm": rain.get(x, math.nan),
            })
    write_csv(args.output / "hourly_error_contributions.csv", hourly_rows)

    flow_edges = [0.0, 0.5, 0.9, 0.95, 0.99, 1.0]
    flow_rows: list[dict] = []
    for model, prediction in sorted(ensemble.items()):
        for lower, upper in zip(flow_edges, flow_edges[1:]):
            lo = quantile(list(observed.values()), lower)
            hi = quantile(list(observed.values()), upper)
            xs = [x for x, value in observed.items() if value >= lo and (value <= hi if upper == 1 else value < hi)]
            squared = [(prediction[x] - observed[x]) ** 2 for x in xs]
            flow_rows.append({
                "model": model, "quantile_lower": lower, "quantile_upper": upper,
                "flow_lower": lo, "flow_upper": hi, "n": len(xs),
                "rmse": math.sqrt(statistics.fmean(squared)),
                "sse_share_percent": 100 * sum(squared) / total_sse[model],
                "mean_error": statistics.fmean(prediction[x] - observed[x] for x in xs),
            })
    write_csv(args.output / "flow_bin_errors.csv", flow_rows)

    try:
        import matplotlib.pyplot as plt
        event_lookup = {int(row["event"]): row for row in summary_rows}
        selected = [int(row["event"]) for row in summary_rows[:args.figure_count]]
        for event_no in selected:
            start, end = events[event_no - 1]
            xs = [x for x in sorted(observed) if start <= x <= end]
            labels = [datetime.fromisoformat(stamp(x).replace("Z", "+00:00")) for x in xs]
            fig, (ax_rain, ax_flow) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,
                                                  gridspec_kw={"height_ratios": [1, 3]})
            ax_rain.bar(labels, [rain.get(x, 0.0) for x in xs], width=0.025, color="#3f7fbf")
            ax_rain.invert_yaxis(); ax_rain.set_ylabel("MRMS\n(mm h$^{-1}$)")
            ax_flow.plot(labels, [observed[x] for x in xs], color="black", linewidth=1.8, label="Observed")
            for model, color in [("lstm_24h", "#d95f02"), ("lstm_residual_hybrid", "#1b9e77"),
                                 ("ffn_residual_hybrid", "#7570b3"), ("two_reservoir", "#666666")]:
                if model in ensemble:
                    ax_flow.plot(labels, [ensemble[model][x] for x in xs], linewidth=1.1, label=model.replace("_", " "), color=color)
            ax_flow.set_ylabel("Discharge (m$^3$ s$^{-1}$)"); ax_flow.set_xlabel("UTC")
            ax_flow.legend(fontsize=8, ncol=2)
            meta = event_lookup[event_no]
            fig.suptitle(f"Event {event_no}: {meta['observed_peak_utc']} | hybrid error rank {meta['lstm_hybrid_error_rank']}")
            fig.tight_layout(); fig.savefig(figures / f"event_{event_no:03d}.png", dpi=180); plt.close(fig)
    except ImportError:
        pass

    metadata = {
        "status": "diagnostic-only; no model selection",
        "common_hours": len(observed),
        "event_threshold_quantile": args.threshold_quantile,
        "event_threshold_flow": threshold,
        "event_count": len(events),
        "event_merge_gap_hours": args.merge_gap_hours,
        "event_window_hours_before": args.hours_before,
        "event_window_hours_after": args.hours_after,
        "peak_match_radius_hours": args.peak_match_radius_hours,
        "ensemble": "median prediction across frozen seeds; deterministic model unchanged",
        "reserved_target_observations_read": False,
    }
    (args.output / "diagnostic_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote failure atlas for {len(events)} events and {len(observed)} common-support hours.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
