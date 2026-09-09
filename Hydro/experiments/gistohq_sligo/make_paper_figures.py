#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except Exception as e:
    raise SystemExit(f"matplotlib is required for paper figures: {e}")

MODES = ("ffn", "ffn_pinn", "lstm", "lstm_pinn", "pinn")
LABEL = {"ffn": "FFN", "ffn_pinn": "FFN + PINN", "lstm": "LSTM", "lstm_pinn": "LSTM + PINN", "pinn": "PINN"}


def read(path: Path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def seed42(stage4: Path, mode: str):
    rows = [r for r in read(stage4 / "batch_summary.csv") if r["mode"] == mode]
    if not rows:
        raise RuntimeError(f"No Stage-4 rows for {mode}: {stage4}")
    return next((r for r in rows if r.get("random_seed") == "42"), rows[0])


def pred(stage4: Path, mode: str):
    r = seed42(stage4, mode)
    return read(stage4 / mode / r["experiment_id"] / "predictions.csv")


def save_publication(fig, root: Path, stem: str):
    # 600-dpi raster for journal submission plus vector PDF/SVG for editing/typesetting.
    fig.savefig(root / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(root / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(root / f"{stem}.svg", bbox_inches="tight")


def keyed(rows, split=None):
    out = {}
    for r in rows:
        if split and r.get("split") != split:
            continue
        try:
            x = float(r["x"]); obs = float(r["observed"]); pr = float(r["predicted"])
        except (KeyError, TypeError, ValueError):
            continue
        if not all(math.isfinite(v) for v in (x, obs, pr)):
            continue
        # Physical times are hourly for GIStoOHQ and regular for synthetic.  A
        # rounded key avoids float32 text serialization noise without changing x.
        out[round(x, 8)] = (x, obs, pr)
    return out


def common_prediction_series(stage4: Path, split=None):
    maps = {m: keyed(pred(stage4, m), split) for m in MODES}
    common = None
    for m in MODES:
        keys = set(maps[m])
        common = keys if common is None else common & keys
    common = sorted(common or [])
    if len(common) < 5:
        raise RuntimeError(
            "The five methods do not share a sufficient common prediction time domain. "
            "Rebuild/rerun the corrected common-contiguous-domain pipeline before making paper figures."
        )
    x = [maps["ffn"][k][0] for k in common]
    observed = [maps["ffn"][k][1] for k in common]
    predictions = {m: [maps[m][k][2] for k in common] for m in MODES}
    return x, observed, predictions


def plot_predictions(root: Path, subdir: str, stem: str, title: str, split=None, hydro=False):
    stage4 = root / subdir / "04_stage4_robustness"
    x, observed, predictions = common_prediction_series(stage4, split)

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.plot(x, observed, label="Observed" if hydro else "Known truth", linewidth=2.4, zorder=6)
    for mode in MODES:
        ax.plot(x, predictions[mode], label=LABEL[mode], linewidth=1.25, alpha=0.92)
    ax.set_xlabel("Elapsed time (h)" if hydro else "Synthetic time")
    ax.set_ylabel("Runoff (mm h$^{-1}$)" if hydro else "Runoff")
    ax.set_title(title, pad=10)
    ax.legend(ncol=3, fontsize=9, frameon=False, loc="best")
    ax.grid(alpha=0.16, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_publication(fig, root, stem)
    plt.close(fig)


def numeric(row, key):
    try:
        v = float(row.get(key, "nan"))
        return v if math.isfinite(v) else math.nan
    except (TypeError, ValueError):
        return math.nan


def plot_skill_summary(root: Path):
    rows = read(root / "02_sligo_hydro" / "paper_method_summary.csv")
    by_mode = {r["mode"]: r for r in rows}
    labels = [LABEL[m] for m in MODES]
    xs = list(range(len(MODES)))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    panels = [
        ("RMSE", "rmse_mean", "rmse_std", "RMSE (mm h$^{-1}$)", None),
        ("NSE", "nse_mean", "nse_std", "NSE", 0.0),
        ("PBIAS", "pbias_mean", "pbias_std", "PBIAS (%)", 0.0),
    ]
    for ax, (_, mean_key, std_key, ylabel, reference) in zip(axes, panels):
        means = [numeric(by_mode[m], mean_key) for m in MODES]
        errs = [numeric(by_mode[m], std_key) for m in MODES]
        ax.errorbar(xs, means, yerr=errs, fmt="o", capsize=4, linewidth=1.2, markersize=6)
        if reference is not None:
            ax.axhline(reference, linewidth=0.9, linestyle="--", alpha=0.6)
        ax.set_xticks(xs, labels, rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.16, linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Sligo Creek held-out performance across five random seeds", y=1.02)
    fig.tight_layout()
    save_publication(fig, root, "paper_sligo_hydrologic_skill")
    plt.close(fig)

    # Keep a dedicated RMSE figure for compatibility, but make it a restrained
    # point/error-bar plot rather than a bar chart that can overemphasize low RMSE.
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    means = [numeric(by_mode[m], "rmse_mean") for m in MODES]
    errs = [numeric(by_mode[m], "rmse_std") for m in MODES]
    ax.errorbar(xs, means, yerr=errs, fmt="o", capsize=5, linewidth=1.3, markersize=7)
    ax.set_xticks(xs, labels, rotation=20, ha="right")
    ax.set_ylabel("RMSE (mm h$^{-1}$)")
    ax.set_title("Sligo Creek five-seed RMSE robustness")
    ax.grid(axis="y", alpha=0.16, linewidth=0.7)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout(); save_publication(fig, root, "paper_sligo_rmse_robustness"); plt.close(fig)


def plot_k_recovery(root: Path):
    s2 = root / "01_synthetic_controlled" / "02_stage2_physics"
    rows = [r for r in read(s2 / "batch_summary.csv") if r["mode"] == "pinn"]
    groups = {}
    for r in rows:
        k = float(r["latent_recession_per_hour"])
        groups.setdefault(k, []).append(float(r["validation_mse"]))
    ks = sorted(groups)
    vals = [min(groups[k]) for k in ks]
    if not ks:
        raise RuntimeError("No standalone PINN Stage-2 candidates for k recovery figure")

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(ks, vals, marker="o", linewidth=1.6, markersize=6, label="Best validation MSE at candidate k")
    ax.axvline(0.08, linestyle="--", linewidth=1.3, label="Known synthetic $k=0.08$")
    best = min(range(len(vals)), key=lambda i: vals[i])
    ax.scatter([ks[best]], [vals[best]], s=55, zorder=5)
    ax.annotate(f"minimum at k={ks[best]:.3g}", (ks[best], vals[best]),
                xytext=(8, 12), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Candidate reservoir coefficient $k$")
    ax.set_ylabel("Validation MSE")
    ax.set_title("Standalone PINN known-parameter recovery")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.16, linewidth=0.7)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout(); save_publication(fig, root, "paper_synthetic_k_recovery"); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("paper_root", type=Path)
    a = p.parse_args(); root = a.paper_root.resolve()
    plot_predictions(root, "01_synthetic_controlled", "paper_synthetic_known_truth_predictions",
                     "Controlled reduced-reservoir known-truth validation")
    plot_predictions(root, "02_sligo_hydro", "paper_sligo_test_predictions",
                     "Sligo Creek held-out test predictions", split="test", hydro=True)
    plot_skill_summary(root)
    plot_k_recovery(root)
    print("[paper-figures] wrote publication figures as PNG (600 dpi), PDF, and SVG to", root)


if __name__ == "__main__":
    main()
