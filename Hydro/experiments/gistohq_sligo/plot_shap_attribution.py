#!/usr/bin/env python3
"""Plot the grouped Shapley-attribution results from compute_shap_attribution.py's
output JSON. Read-only (no new computation); matches this project's existing figure
style (plot_rcParams below mirrors generate_development_diagnostics.py)."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CHANNEL_LABELS = {
    "precip": "Precipitation",
    "temp": "Temperature",
    "rh": "Relative humidity",
    "wind": "Wind speed",
    "solar": "Solar radiation",
    "pet": "PET",
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-prefix", type=Path, required=True)
    args = p.parse_args()

    if args.output_prefix.with_suffix(".pdf").exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_prefix}.pdf")

    report = json.loads(args.input.read_text())
    mean_abs = report["channel_mean_abs_phi"]
    mean_signed = report["channel_mean_phi"]

    order = sorted(mean_abs, key=lambda c: mean_abs[c], reverse=True)
    labels = [CHANNEL_LABELS[c] for c in order]
    abs_vals = [mean_abs[c] for c in order]
    signed_vals = [mean_signed[c] for c in order]

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.4))
    y = np.arange(len(order))

    ax1.barh(y, abs_vals, color="#3985ab")
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels)
    ax1.invert_yaxis()
    ax1.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
    ax1.set_xlabel("Mean |SHAP value| (mm/h)")
    ax1.set_title("Attribution magnitude")

    colors = ["#c95b4d" if v < 0 else "#188877" for v in signed_vals]
    ax2.barh(y, signed_vals, color=colors)
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels)
    ax2.invert_yaxis()
    ax2.axvline(0, color="#444444", linewidth=0.8)
    ax2.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
    ax2.set_xlabel("Mean signed SHAP value (mm/h)")
    ax2.set_title("Attribution direction")

    fig.suptitle(
        f"Grouped SHAP attribution, plain LSTM (n_explained={report['n_explained']}, "
        f"n_background={report['n_background']}, seed={report['seed']})",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_prefix.with_suffix(".pdf"))
    fig.savefig(args.output_prefix.with_suffix(".png"), dpi=200)
    print("wrote", args.output_prefix.with_suffix(".pdf"), "and .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
