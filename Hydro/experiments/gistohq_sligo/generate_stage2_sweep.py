#!/usr/bin/env python3
"""Generate Stage-2 learning-rate / batch-size sweeps for the selected Sligo Creek finalists.

Stage 2 keeps the Stage-1-selected architecture, memory setting, normalization,
split, epochs, and seed fixed while varying:
  learning_rate in {0.001, 0.003, 0.005}
  batch_size    in {16, 32, 64}

Finalists:
  * FFN: 6 h lags, hidden 16,16, ReLU
  * LSTM: 12 h, hidden 32
  * LSTM: 24 h, hidden 24,24
  * LSTM: 24 h, hidden 32

This produces 4 x 3 x 3 = 36 runs by default.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
GENERATED = HERE / "generated_stage2"
BATCH_PATH = HERE / "hyperparameter_stage2.batch"
MANIFEST_PATH = GENERATED / "stage2_manifest.csv"

FFN_BASE = HERE / "ffn_standardize_lag6.json"
LSTM_BASE = HERE / "lstm_standardize_seq12.json"

DEFAULT_LRS = [0.001, 0.003, 0.005]
DEFAULT_BATCHES = [16, 32, 64]

FINALISTS = [
    {
        "name": "ffn_lag6_h16x16_relu",
        "mode": "ffn",
        "hidden_layers": "16,16",
        "activation": "relu",
        "input_lags": "1,2,3,4,5,6",
        "use_time_lagged_ffn": True,
        "lstm_sequence_length": 6,
    },
    {
        "name": "lstm_seq12_h32",
        "mode": "lstm",
        "hidden_layers": "32",
        "activation": "tanh",
        "input_lags": "1",
        "use_time_lagged_ffn": False,
        "lstm_sequence_length": 12,
    },
    {
        "name": "lstm_seq24_h24x24",
        "mode": "lstm",
        "hidden_layers": "24,24",
        "activation": "tanh",
        "input_lags": "1",
        "use_time_lagged_ffn": False,
        "lstm_sequence_length": 24,
    },
    {
        "name": "lstm_seq24_h32",
        "mode": "lstm",
        "hidden_layers": "32",
        "activation": "tanh",
        "input_lags": "1",
        "use_time_lagged_ffn": False,
        "lstm_sequence_length": 24,
    },
]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_float_list(value: str) -> list[float]:
    values = []
    for token in value.split(","):
        token = token.strip()
        if token:
            values.append(float(token))
    if not values:
        raise ValueError("At least one learning rate is required.")
    return values


def parse_int_list(value: str) -> list[int]:
    values = []
    for token in value.split(","):
        token = token.strip()
        if token:
            parsed = int(token)
            if parsed < 1:
                raise ValueError("Batch sizes must be positive integers.")
            values.append(parsed)
    if not values:
        raise ValueError("At least one batch size is required.")
    return values


def lr_slug(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def write_json(path: Path, config: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learning-rates", default=",".join(str(v) for v in DEFAULT_LRS))
    parser.add_argument("--batch-sizes", default=",".join(str(v) for v in DEFAULT_BATCHES))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    learning_rates = parse_float_list(args.learning_rates)
    batch_sizes = parse_int_list(args.batch_sizes)

    ffn_base = load_json(FFN_BASE)
    lstm_base = load_json(LSTM_BASE)

    GENERATED.mkdir(parents=True, exist_ok=True)
    for old in GENERATED.glob("stage2_*.json"):
        old.unlink()

    batch_lines = [
        "# Sligo Creek supervised hyperparameter Stage 2",
        "# Selected Stage-1 finalists; learning-rate x batch-size sweep",
        "# Seed fixed unless overridden when generating",
        "",
    ]
    manifest_rows = []

    for finalist in FINALISTS:
        base = ffn_base if finalist["mode"] == "ffn" else lstm_base
        for lr in learning_rates:
            for batch_size in batch_sizes:
                experiment_id = (
                    f"sligo_stage2_{finalist['name']}_lr{lr_slug(lr)}_b{batch_size}_s{args.seed}"
                )
                filename = f"stage2_{experiment_id}.json"
                config = dict(base)
                config.update(
                    {
                        "experiment_id": experiment_id,
                        "epochs": args.epochs,
                        "learning_rate": lr,
                        "batch_size": batch_size,
                        "random_seed": args.seed,
                        "hidden_layers": finalist["hidden_layers"],
                        "activation": finalist["activation"],
                        "input_lags": finalist["input_lags"],
                        "use_time_lagged_ffn": finalist["use_time_lagged_ffn"],
                        "lstm_sequence_length": finalist["lstm_sequence_length"],
                    }
                )
                write_json(GENERATED / filename, config)
                batch_lines.append(f"{finalist['mode']} generated_stage2/{filename}")
                manifest_rows.append(
                    [
                        experiment_id,
                        finalist["mode"],
                        finalist["name"],
                        finalist["hidden_layers"],
                        finalist["activation"],
                        finalist["input_lags"],
                        finalist["lstm_sequence_length"],
                        lr,
                        batch_size,
                        args.seed,
                        args.epochs,
                    ]
                )
        batch_lines.append("")

    BATCH_PATH.write_text("\n".join(batch_lines).rstrip() + "\n", encoding="utf-8")

    with MANIFEST_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "experiment_id",
                "mode",
                "finalist",
                "hidden_layers",
                "activation",
                "input_lags",
                "lstm_sequence_length",
                "learning_rate",
                "batch_size",
                "random_seed",
                "epochs",
            ]
        )
        writer.writerows(manifest_rows)

    print(f"Generated {len(manifest_rows)} Stage-2 experiments")
    print(f"Finalists: {len(FINALISTS)}")
    print(f"Learning rates: {learning_rates}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Seed: {args.seed}")
    print(f"Batch: {BATCH_PATH}")
    print(f"Configs: {GENERATED}")
    print(f"Manifest: {MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
