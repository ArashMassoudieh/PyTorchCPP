#!/usr/bin/env python3
"""Generate Stage-3 multi-seed robustness runs for selected Sligo Creek winners.

Stage 3 freezes the selected Stage-2 hyperparameters and varies only random seed.
Defaults represent the current provisional winners and can be overridden from the
GUI after Stage 2 is complete.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
GENERATED = HERE / "generated_stage3"
BATCH_PATH = HERE / "hyperparameter_stage3.batch"
MANIFEST_PATH = GENERATED / "stage3_manifest.csv"
FFN_BASE = HERE / "ffn_standardize_lag6.json"
LSTM_BASE = HERE / "lstm_standardize_seq12.json"

DEFAULT_SEEDS = [42, 123, 2026, 31415, 27182]
VALID_ACTIVATIONS = {"relu", "tanh", "sigmoid"}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, config: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")


def parse_int_list(value: str) -> list[int]:
    values = [int(token.strip()) for token in value.split(",") if token.strip()]
    if not values:
        raise ValueError("At least one seed is required.")
    return values


def architecture_slug(value: str) -> str:
    return "x".join(part.strip() for part in value.split(","))


def lr_slug(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def finalist(name: str, mode: str, hidden: str, activation: str, lr: float,
             batch: int, sequence: int, enabled: bool) -> dict | None:
    if not enabled:
        return None
    return {
        "name": name,
        "mode": mode,
        "hidden_layers": hidden,
        "activation": activation,
        "learning_rate": lr,
        "batch_size": batch,
        "lstm_sequence_length": sequence,
        "input_lags": "1,2,3,4,5,6" if mode == "ffn" else "1",
        "use_time_lagged_ffn": mode == "ffn",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default=",".join(str(v) for v in DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=150)

    parser.add_argument("--ffn-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ffn-hidden", default="16,16")
    parser.add_argument("--ffn-activation", default="relu")
    parser.add_argument("--ffn-lr", type=float, default=0.003)
    parser.add_argument("--ffn-batch", type=int, default=32)

    parser.add_argument("--lstm1-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lstm1-sequence", type=int, default=12)
    parser.add_argument("--lstm1-hidden", default="32")
    parser.add_argument("--lstm1-lr", type=float, default=0.003)
    parser.add_argument("--lstm1-batch", type=int, default=32)

    parser.add_argument("--lstm2-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lstm2-sequence", type=int, default=24)
    parser.add_argument("--lstm2-hidden", default="32")
    parser.add_argument("--lstm2-lr", type=float, default=0.003)
    parser.add_argument("--lstm2-batch", type=int, default=32)
    args = parser.parse_args()

    seeds = parse_int_list(args.seeds)
    activation = args.ffn_activation.lower().strip()
    if activation not in VALID_ACTIVATIONS:
        parser.error(f"Unsupported FFN activation: {activation}")

    finalists = [
        finalist(
            f"ffn_lag6_h{architecture_slug(args.ffn_hidden)}_{activation}",
            "ffn", args.ffn_hidden, activation, args.ffn_lr, args.ffn_batch, 6, args.ffn_enabled,
        ),
        finalist(
            f"lstm_seq{args.lstm1_sequence}_h{architecture_slug(args.lstm1_hidden)}",
            "lstm", args.lstm1_hidden, "tanh", args.lstm1_lr, args.lstm1_batch,
            args.lstm1_sequence, args.lstm1_enabled,
        ),
        finalist(
            f"lstm_seq{args.lstm2_sequence}_h{architecture_slug(args.lstm2_hidden)}",
            "lstm", args.lstm2_hidden, "tanh", args.lstm2_lr, args.lstm2_batch,
            args.lstm2_sequence, args.lstm2_enabled,
        ),
    ]
    finalists = [f for f in finalists if f is not None]
    if not finalists:
        parser.error("At least one Stage-3 finalist must be enabled.")

    ffn_base = load_json(FFN_BASE)
    lstm_base = load_json(LSTM_BASE)
    GENERATED.mkdir(parents=True, exist_ok=True)
    for old in GENERATED.glob("stage3_*.json"):
        old.unlink()

    batch_lines = [
        "# Sligo Creek Stage 3 multi-seed robustness sweep",
        "# Hyperparameters fixed from Stage 2; only random seed varies",
        "",
    ]
    manifest_rows = []

    for item in finalists:
        base = ffn_base if item["mode"] == "ffn" else lstm_base
        for seed in seeds:
            experiment_id = (
                f"sligo_stage3_{item['name']}_lr{lr_slug(item['learning_rate'])}"
                f"_b{item['batch_size']}_s{seed}"
            )
            filename = f"stage3_{experiment_id}.json"
            config = dict(base)
            config.update({
                "experiment_id": experiment_id,
                "epochs": args.epochs,
                "learning_rate": item["learning_rate"],
                "batch_size": item["batch_size"],
                "random_seed": seed,
                "hidden_layers": item["hidden_layers"],
                "activation": item["activation"],
                "input_lags": item["input_lags"],
                "use_time_lagged_ffn": item["use_time_lagged_ffn"],
                "lstm_sequence_length": item["lstm_sequence_length"],
            })
            write_json(GENERATED / filename, config)
            batch_lines.append(f"{item['mode']} generated_stage3/{filename}")
            manifest_rows.append([
                experiment_id, item["mode"], item["name"], item["hidden_layers"],
                item["activation"], item["lstm_sequence_length"], item["learning_rate"],
                item["batch_size"], seed, args.epochs,
            ])
        batch_lines.append("")

    BATCH_PATH.write_text("\n".join(batch_lines).rstrip() + "\n", encoding="utf-8")
    with MANIFEST_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "experiment_id", "mode", "finalist", "hidden_layers", "activation",
            "lstm_sequence_length", "learning_rate", "batch_size", "random_seed", "epochs",
        ])
        writer.writerows(manifest_rows)

    print(f"Generated {len(manifest_rows)} Stage-3 experiments")
    print(f"Finalists: {[f['name'] for f in finalists]}")
    print(f"Seeds: {seeds}")
    print(f"Batch: {BATCH_PATH}")
    print(f"Configs: {GENERATED}")
    print(f"Manifest: {MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
