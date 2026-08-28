#!/usr/bin/env python3
"""Generate Stage-2 learning-rate / batch-size sweeps for Sligo Creek finalists.

Stage 2 keeps architecture/memory choices fixed while varying learning rate and
batch size. The FFN finalist set is configurable so a sigmoid candidate can be
included after the Stage-1 sigmoid diagnostic without repeating the entire
architecture sweep.
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
VALID_FFN_ACTIVATIONS = {"relu", "tanh", "sigmoid"}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_float_list(value: str) -> list[float]:
    values = [float(token.strip()) for token in value.split(",") if token.strip()]
    if not values or any(v <= 0 for v in values):
        raise ValueError("Learning rates must contain at least one positive value.")
    return values


def parse_int_list(value: str) -> list[int]:
    values = [int(token.strip()) for token in value.split(",") if token.strip()]
    if not values or any(v < 1 for v in values):
        raise ValueError("Batch sizes must contain at least one positive integer.")
    return values


def parse_activations(value: str) -> list[str]:
    values = [token.strip().lower() for token in value.split(",") if token.strip()]
    if not values:
        raise ValueError("At least one FFN activation is required.")
    invalid = [v for v in values if v not in VALID_FFN_ACTIVATIONS]
    if invalid:
        raise ValueError("Unsupported FFN activation(s): " + ", ".join(invalid))
    return values


def architecture_slug(value: str) -> str:
    return "x".join(part.strip() for part in value.split(","))


def lr_slug(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def write_json(path: Path, config: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")


def make_finalists(args) -> list[dict]:
    finalists = []
    for activation in parse_activations(args.ffn_activations):
        hidden = args.sigmoid_hidden if activation == "sigmoid" else args.ffn_hidden
        finalists.append({
            "name": f"ffn_lag6_h{architecture_slug(hidden)}_{activation}",
            "mode": "ffn",
            "hidden_layers": hidden,
            "activation": activation,
            "input_lags": "1,2,3,4,5,6",
            "use_time_lagged_ffn": True,
            "lstm_sequence_length": 6,
        })

    finalists.extend([
        {
            "name": f"lstm_seq12_h{architecture_slug(args.lstm12_hidden)}",
            "mode": "lstm",
            "hidden_layers": args.lstm12_hidden,
            "activation": "tanh",  # metadata only; native LSTM nonlinearities are used
            "input_lags": "1",
            "use_time_lagged_ffn": False,
            "lstm_sequence_length": 12,
        },
        {
            "name": f"lstm_seq24_h{architecture_slug(args.lstm24_deep_hidden)}",
            "mode": "lstm",
            "hidden_layers": args.lstm24_deep_hidden,
            "activation": "tanh",
            "input_lags": "1",
            "use_time_lagged_ffn": False,
            "lstm_sequence_length": 24,
        },
        {
            "name": f"lstm_seq24_h{architecture_slug(args.lstm24_hidden)}",
            "mode": "lstm",
            "hidden_layers": args.lstm24_hidden,
            "activation": "tanh",
            "input_lags": "1",
            "use_time_lagged_ffn": False,
            "lstm_sequence_length": 24,
        },
    ])
    return finalists


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learning-rates", default=",".join(str(v) for v in DEFAULT_LRS))
    parser.add_argument("--batch-sizes", default=",".join(str(v) for v in DEFAULT_BATCHES))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ffn-activations", default="relu",
                        help="Comma-separated Stage-2 FFN finalists, e.g. relu,sigmoid.")
    parser.add_argument("--ffn-hidden", default="16,16",
                        help="Hidden architecture for non-sigmoid FFN finalist(s).")
    parser.add_argument("--sigmoid-hidden", default="16,16",
                        help="Hidden architecture for sigmoid FFN finalist, if selected.")
    parser.add_argument("--lstm12-hidden", default="32")
    parser.add_argument("--lstm24-deep-hidden", default="24,24")
    parser.add_argument("--lstm24-hidden", default="32")
    args = parser.parse_args()

    learning_rates = parse_float_list(args.learning_rates)
    batch_sizes = parse_int_list(args.batch_sizes)
    finalists = make_finalists(args)

    ffn_base = load_json(FFN_BASE)
    lstm_base = load_json(LSTM_BASE)

    GENERATED.mkdir(parents=True, exist_ok=True)
    for old in GENERATED.glob("stage2_*.json"):
        old.unlink()

    batch_lines = [
        "# Sligo Creek supervised hyperparameter Stage 2",
        "# Finalist architecture/memory fixed; learning-rate x batch-size sweep",
        "# FFN activation finalists may include sigmoid",
        "",
    ]
    manifest_rows = []

    for finalist in finalists:
        base = ffn_base if finalist["mode"] == "ffn" else lstm_base
        for lr in learning_rates:
            for batch_size in batch_sizes:
                experiment_id = f"sligo_stage2_{finalist['name']}_lr{lr_slug(lr)}_b{batch_size}_s{args.seed}"
                filename = f"stage2_{experiment_id}.json"
                config = dict(base)
                config.update({
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
                })
                write_json(GENERATED / filename, config)
                batch_lines.append(f"{finalist['mode']} generated_stage2/{filename}")
                manifest_rows.append([
                    experiment_id, finalist["mode"], finalist["name"], finalist["hidden_layers"],
                    finalist["activation"], finalist["input_lags"], finalist["lstm_sequence_length"],
                    lr, batch_size, args.seed, args.epochs,
                ])
        batch_lines.append("")

    BATCH_PATH.write_text("\n".join(batch_lines).rstrip() + "\n", encoding="utf-8")
    with MANIFEST_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "experiment_id", "mode", "finalist", "hidden_layers", "activation", "input_lags",
            "lstm_sequence_length", "learning_rate", "batch_size", "random_seed", "epochs",
        ])
        writer.writerows(manifest_rows)

    print(f"Generated {len(manifest_rows)} Stage-2 experiments")
    print(f"Finalists: {[f['name'] for f in finalists]}")
    print(f"Learning rates: {learning_rates}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Seed: {args.seed}")
    print(f"Batch: {BATCH_PATH}")
    print(f"Configs: {GENERATED}")
    print(f"Manifest: {MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
