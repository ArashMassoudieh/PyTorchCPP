#!/usr/bin/env python3
"""Generate a method-aware HydroPINN sweep for all five approaches.

The selected data source is explicit and is applied to every generated config.
For controlled Synthetic validation, every method and every candidate sees one
fixed reduced-reservoir truth hydrograph. The truth coefficient
``synthetic_reservoir_truth_k`` is independent from the candidate/model k that
is swept through ``lambda_decay/storage_coeff``.
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
PHYSICS_METHODS = {"ffn_pinn", "lstm_pinn", "pinn"}
DATA_SOURCES = ("synthetic", "csv", "hydro")


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


def bool_text(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean: {value}")


def source_name(cfg: dict) -> str:
    if cfg.get("use_hydro_package"):
        return "hydro"
    if cfg.get("use_csv_data"):
        return "csv"
    return "synthetic"


def apply_source(cfg: dict, args) -> dict:
    """Overwrite every source-specific field; never inherit source state."""
    cfg = dict(cfg)
    cfg.update({
        "use_hydro_package": False,
        "use_csv_data": False,
        "hydro_package_path": "",
        "hydro_catchment_id": "",
        "hydro_package_profile": args.hydro_package_profile,
        "csv_path": "",
        "csv_x_column": args.csv_x_column,
        "csv_y_column": args.csv_y_column,
        "csv_has_header": args.csv_has_header,
        "synthetic_profile": args.synthetic_profile,
        "sample_count": args.sample_count,
        "t_start": args.t_start,
        "t_end": args.t_end,
        "synthetic_reservoir_truth_k": args.synthetic_truth_k,
    })

    if args.data_source == "hydro":
        cfg.update({
            "use_hydro_package": True,
            "hydro_package_path": args.hydro_package_path,
            "hydro_catchment_id": args.hydro_catchment_id,
        })
    elif args.data_source == "csv":
        cfg.update({
            "use_csv_data": True,
            "csv_path": args.csv_path,
        })

    actual = source_name(cfg)
    if actual != args.data_source:
        raise RuntimeError(f"source routing invariant failed: requested={args.data_source}, config={actual}")
    if args.data_source == "synthetic" and (cfg["hydro_package_path"] or cfg["csv_path"]):
        raise RuntimeError("Synthetic config unexpectedly retained an external data path")
    return cfg


def common(cfg: dict, args, lr: float, batch: int, seed: int) -> dict:
    cfg = apply_source(cfg, args)
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

    p.add_argument("--data-source", choices=DATA_SOURCES, default="hydro")
    p.add_argument("--synthetic-profile", default="reduced_reservoir")
    p.add_argument("--sample-count", type=int, default=240)
    p.add_argument("--t-start", type=float, default=0.0)
    p.add_argument("--t-end", type=float, default=5.0)
    p.add_argument("--synthetic-truth-k", type=float, default=0.08,
                   help="Fixed ground-truth reservoir k used to generate Synthetic reduced_reservoir data")
    p.add_argument("--hydro-package-path", default="../GIStoOHQ/examples/SligoCreek/outputs/sligocreekdemo_data/hydropinn")
    p.add_argument("--hydro-catchment-id", default="")
    p.add_argument("--hydro-package-profile", default="rainfall-runoff")
    p.add_argument("--csv-path", default="")
    p.add_argument("--csv-x-column", type=int, default=0)
    p.add_argument("--csv-y-column", type=int, default=3)
    p.add_argument("--csv-has-header", type=bool_text, default=True)
    return p


def validate_source_args(args, methods: list[str]) -> None:
    if args.synthetic_truth_k <= 0:
        raise SystemExit("--synthetic-truth-k must be positive")
    if args.data_source == "hydro" and not args.hydro_package_path.strip():
        raise SystemExit("--hydro-package-path is required for --data-source hydro")
    if args.data_source == "csv" and not args.csv_path.strip():
        raise SystemExit("--csv-path is required for --data-source csv")
    if args.data_source == "csv" and any(m in PHYSICS_METHODS for m in methods):
        if args.csv_x_column != 0 or args.csv_y_column < 3:
            raise SystemExit("Reduced-reservoir CSV physics requires time column 0, P column 1, PET column 2, and runoff target column >=3")
    if args.data_source == "synthetic":
        if args.sample_count < 32 or not args.t_end > args.t_start:
            raise SystemExit("Synthetic source requires sample_count>=32 and t_end>t_start")
        if any(m in PHYSICS_METHODS for m in methods) and args.synthetic_profile != "reduced_reservoir":
            raise SystemExit(
                "The five-method synthetic physics pipeline requires --synthetic-profile reduced_reservoir so all methods share one controlled truth."
            )


def main() -> int:
    args = parser().parse_args()
    methods = csv_values(args.methods)
    unknown = sorted(set(methods) - set(ALL_METHODS))
    if unknown:
        raise SystemExit("Unknown method(s): " + ", ".join(unknown))
    methods = [m for m in ALL_METHODS if m in methods]
    if not methods:
        raise SystemExit("Select at least one method")
    validate_source_args(args, methods)

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

    for mode, _, cfg in jobs:
        actual = source_name(cfg)
        if actual != args.data_source:
            raise RuntimeError(f"generated job {cfg['experiment_id']} has source={actual}, expected={args.data_source}")
        if args.data_source == "synthetic":
            if cfg.get("hydro_package_path") or cfg.get("csv_path"):
                raise RuntimeError(f"synthetic job {cfg['experiment_id']} leaked an external path")
            if cfg.get("synthetic_reservoir_truth_k") != args.synthetic_truth_k:
                raise RuntimeError(f"synthetic job {cfg['experiment_id']} changed truth k")

    BATCH.write_text(
        f"# Unified HydroPINN five-method sweep; data_source={args.data_source}; generated file\n" +
        "\n".join(f"{mode} {path}" for mode, path, _ in jobs) + "\n",
        encoding="utf-8",
    )

    OUT.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", encoding="utf-8") as out:
        out.write("experiment_id,mode,data_source,synthetic_profile,synthetic_truth_k,hydro_package_path,csv_path,hidden_layers,activation,lstm_sequence_length,input_lags,learning_rate,batch_size,seed,physics_weight,recession_k\n")
        for mode, _, cfg in jobs:
            out.write(
                f"{cfg['experiment_id']},{mode},{source_name(cfg)},{cfg.get('synthetic_profile','')},{cfg.get('synthetic_reservoir_truth_k','')},"
                f"\"{cfg.get('hydro_package_path','')}\",\"{cfg.get('csv_path','')}\","
                f"\"{cfg.get('hidden_layers','')}\",{cfg.get('activation','')},"
                f"{cfg.get('lstm_sequence_length','')},\"{cfg.get('input_lags','')}\",{cfg['learning_rate']},"
                f"{cfg['batch_size']},{cfg['random_seed']},{cfg.get('physics_weight','')},{cfg.get('storage_coeff','')}\n"
            )

    counts = {m: sum(1 for mode, _, _ in jobs if mode == m) for m in ALL_METHODS}
    print(f"Data source: {args.data_source}")
    if args.data_source == "synthetic":
        print(f"Synthetic profile: {args.synthetic_profile}; truth_k={args.synthetic_truth_k}; samples={args.sample_count}; t=[{args.t_start},{args.t_end}]")
    elif args.data_source == "hydro":
        print(f"Hydro package: {args.hydro_package_path}")
    else:
        print(f"CSV: {args.csv_path}; x={args.csv_x_column}; y={args.csv_y_column}")
    print(f"Generated {len(jobs)} valid experiment(s)")
    for method in ALL_METHODS:
        if counts[method]:
            print(f"  {method}: {counts[method]}")
    print(f"Batch: {BATCH}")
    print(f"Manifest: {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
