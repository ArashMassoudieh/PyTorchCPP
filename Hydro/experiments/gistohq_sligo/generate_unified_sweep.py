#!/usr/bin/env python3
"""Generate a method-aware HydroPINN sweep for all five approaches.

The generator deliberately avoids a blind Cartesian product. Parameters are
applied only to methods for which they are meaningful:
  FFN             : hidden x activation x lag x optimizer grid x seeds
  FFN + PINN      : hidden x activation x physics-weight x recession-k x optimizer grid x seeds
  LSTM            : hidden x sequence x optimizer grid x seeds
  LSTM + PINN     : hidden x sequence x physics-weight x recession-k x optimizer grid x seeds
  PINN            : hidden x recession-k x optimizer grid x seeds

Physics-informed GIStoOHQ runs use the independent reduced-reservoir equation
    dQ/dt = k (Peff - Q),  Peff=max(P-PET,0),
rather than a precomputed latent-storage trajectory.  storage_coeff remains the
serialized k field for backward compatibility with existing experiment files.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
FFN_BASE = HERE / "ffn_standardize_lag6.json"
LSTM_BASE = HERE / "lstm_standardize_seq12.json"
OUT = HERE / "generated_unified"
BATCH = HERE / "unified_sweep.batch"
MANIFEST = OUT / "unified_manifest.csv"

ALL_METHODS = ("ffn", "ffn_pinn", "lstm", "lstm_pinn", "pinn")


def csv_values(text: str, cast=str):
    values = [x.strip() for x in text.split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("list cannot be empty")
    return [cast(x) for x in values]


def semi_values(text: str):
    values = [x.strip() for x in text.split(";") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("list cannot be empty")
    return values


def slug(value: object) -> str:
    s = str(value).strip().lower().replace(".", "p").replace(",", "x")
    return re.sub(r"[^a-z0-9_+-]+", "_", s).strip("_")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_config(cfg: dict) -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    filename = cfg["experiment_id"] + ".json"
    path = OUT / filename
    path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    return f"generated_unified/{filename}"


def common(cfg: dict, args, lr: float, batch: int, seed: int) -> dict:
    cfg = dict(cfg)
    cfg.update({
        "epochs": args.epochs,
        "learning_rate": lr,
        "batch_size": batch,
        "random_seed": seed,
        "shuffle_training": False,
        "evaluate_metrics": True,
    })
    return cfg


def physics_common(cfg: dict, k: float) -> dict:
    cfg = dict(cfg)
    cfg.update({
        "normalization": "none",
        "physics_profile": "linear_reservoir",
        "physics_dt": 1.0,
        "lambda_decay": k,
        "forcing_gain": k,
        "storage_coeff": k,
        "pinn_collocation_points": 0,
        "hydro_package_profile": "rainfall-runoff",
        "use_time_lagged_ffn": False,
        "input_lags": "1",
    })
    return cfg


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--methods", default=",".join(ALL_METHODS))
    p.add_argument("--ffn-hidden", default="16,16")
    p.add_argument("--ffn-architectures", default="16,16")
    p.add_argument("--ffn-activations", default="relu")
    p.add_argument("--ffn-lags", default="1,2,3,4,5,6", help="Semicolon-separated lag specifications")
    p.add_argument("--lstm-architectures", default="32")
    p.add_argument("--lstm-sequences", default="12")
    p.add_argument("--pinn-architectures", default="24,24")
    p.add_argument("--learning-rates", default="0.003")
    p.add_argument("--batch-sizes", default="32")
    p.add_argument("--seeds", default="42")
    p.add_argument("--physics-weights", default="0.005,0.01,0.025,0.05")
    p.add_argument("--recession-k", default="0.01,0.02,0.04,0.08,0.16")
    p.add_argument("--data-weight", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=150)
    return p


def main() -> int:
    args = parser().parse_args()
    methods = csv_values(args.methods)
    unknown = sorted(set(methods) - set(ALL_METHODS))
    if unknown:
        raise SystemExit("Unknown method(s): " + ", ".join(unknown))
    methods = [m for m in ALL_METHODS if m in methods]
    if not methods:
        raise SystemExit("Select at least one method")

    ffn_arch = semi_values(args.ffn_architectures or args.ffn_hidden)
    activations = csv_values(args.ffn_activations)
    lags = semi_values(args.ffn_lags)
    lstm_arch = semi_values(args.lstm_architectures)
    sequences = csv_values(args.lstm_sequences, int)
    pinn_arch = semi_values(args.pinn_architectures)
    lrs = csv_values(args.learning_rates, float)
    batches = csv_values(args.batch_sizes, int)
    seeds = csv_values(args.seeds, int)
    physics_weights = csv_values(args.physics_weights, float)
    ks = csv_values(args.recession_k, float)

    if args.epochs < 1 or any(v <= 0 for v in lrs) or any(v < 1 for v in batches + sequences):
        raise SystemExit("epochs/LR/batch/sequence settings must be positive")
    if any(v <= 0 for v in ks):
        raise SystemExit("reservoir k values must be positive")
    if any(v < 0 for v in physics_weights):
        raise SystemExit("physics weights cannot be negative")

    ffn_base = load(FFN_BASE)
    lstm_base = load(LSTM_BASE)
    jobs: list[tuple[str, str, dict]] = []

    grid = list(itertools.product(lrs, batches, seeds))

    if "ffn" in methods:
        for hidden, act, lag, (lr, batch, seed) in itertools.product(ffn_arch, activations, lags, grid):
            cfg = common(ffn_base, args, lr, batch, seed)
            cfg.update({"hidden_layers": hidden, "activation": act,
                        "input_lags": lag, "use_time_lagged_ffn": True,
                        "normalization": "standardize"})
            cfg["experiment_id"] = f"unified_ffn_h{slug(hidden)}_{slug(act)}_lag{slug(lag)}_lr{slug(lr)}_b{batch}_s{seed}"
            jobs.append(("ffn", write_config(cfg), cfg))

    if "ffn_pinn" in methods:
        for hidden, act, w, k, (lr, batch, seed) in itertools.product(ffn_arch, activations, physics_weights, ks, grid):
            cfg = physics_common(common(ffn_base, args, lr, batch, seed), k)
            cfg.update({"hidden_layers": hidden, "activation": act,
                        "data_weight": args.data_weight, "physics_weight": w})
            cfg["experiment_id"] = f"unified_ffn_pinn_h{slug(hidden)}_{slug(act)}_w{slug(w)}_k{slug(k)}_lr{slug(lr)}_b{batch}_s{seed}"
            jobs.append(("ffn_pinn", write_config(cfg), cfg))

    if "lstm" in methods:
        for hidden, seq, (lr, batch, seed) in itertools.product(lstm_arch, sequences, grid):
            cfg = common(lstm_base, args, lr, batch, seed)
            cfg.update({"hidden_layers": hidden, "lstm_sequence_length": seq,
                        "normalization": "standardize"})
            cfg["experiment_id"] = f"unified_lstm_h{slug(hidden)}_seq{seq}_lr{slug(lr)}_b{batch}_s{seed}"
            jobs.append(("lstm", write_config(cfg), cfg))

    if "lstm_pinn" in methods:
        for hidden, seq, w, k, (lr, batch, seed) in itertools.product(lstm_arch, sequences, physics_weights, ks, grid):
            cfg = physics_common(common(lstm_base, args, lr, batch, seed), k)
            cfg.update({"hidden_layers": hidden, "lstm_sequence_length": seq,
                        "data_weight": args.data_weight, "physics_weight": w})
            cfg["experiment_id"] = f"unified_lstm_pinn_h{slug(hidden)}_seq{seq}_w{slug(w)}_k{slug(k)}_lr{slug(lr)}_b{batch}_s{seed}"
            jobs.append(("lstm_pinn", write_config(cfg), cfg))

    if "pinn" in methods:
        for hidden, k, (lr, batch, seed) in itertools.product(pinn_arch, ks, grid):
            cfg = physics_common(common(ffn_base, args, lr, batch, seed), k)
            cfg.update({"hidden_layers": hidden, "activation": "tanh",
                        "data_weight": 0.0, "physics_weight": 1.0})
            cfg["experiment_id"] = f"unified_pinn_h{slug(hidden)}_k{slug(k)}_lr{slug(lr)}_b{batch}_s{seed}"
            jobs.append(("pinn", write_config(cfg), cfg))

    BATCH.write_text(
        "# Unified HydroPINN five-method sweep; generated file\n" +
        "\n".join(f"{mode} {path}" for mode, path, _ in jobs) + "\n",
        encoding="utf-8",
    )

    OUT.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", encoding="utf-8") as out:
        out.write("experiment_id,mode,hidden_layers,activation,lstm_sequence_length,input_lags,learning_rate,batch_size,seed,physics_weight,recession_k\n")
        for mode, _, cfg in jobs:
            out.write(
                f"{cfg['experiment_id']},{mode},\"{cfg.get('hidden_layers','')}\",{cfg.get('activation','')},"
                f"{cfg.get('lstm_sequence_length','')},\"{cfg.get('input_lags','')}\",{cfg['learning_rate']},"
                f"{cfg['batch_size']},{cfg['random_seed']},{cfg.get('physics_weight','')},{cfg.get('storage_coeff','')}\n"
            )

    counts = {m: sum(1 for mode, _, _ in jobs if mode == m) for m in ALL_METHODS}
    print(f"Generated {len(jobs)} valid experiment(s)")
    for method in ALL_METHODS:
        if counts[method]:
            print(f"  {method}: {counts[method]}")
    print(f"Batch: {BATCH}")
    print(f"Manifest: {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
