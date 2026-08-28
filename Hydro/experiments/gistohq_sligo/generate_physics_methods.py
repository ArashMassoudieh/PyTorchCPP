#!/usr/bin/env python3
"""Generate initial GIStoOHQ physics-informed experiments for Sligo Creek.

The generated suite exercises the three methods that were previously blocked by
missing observed storage:
  * FFN + PINN
  * LSTM + PINN
  * standalone PINN

HydroBatch enables the latent-storage water-balance adapter for these modes at
runtime.  The JSON configs deliberately use normalization=none because the
current residual is evaluated in physical rainfall/PET/runoff/storage units.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "generated_physics"
BATCH = HERE / "gistohq_physics_methods.batch"

FFN_BASE = HERE / "ffn_standardize_lag6.json"
LSTM_BASE = HERE / "lstm_standardize_seq12.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(name: str, cfg: dict) -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    return f"generated_physics/{name}"


def common(cfg: dict) -> dict:
    cfg = dict(cfg)
    cfg.update({
        "epochs": 150,
        "batch_size": 32,
        "learning_rate": 0.003,
        "random_seed": 42,
        "shuffle_training": False,
        "normalization": "none",
        "physics_profile": "water_balance",
        "data_weight": 1.0,
        "physics_weight": 0.2,
        "physics_dt": 1.0,
        "storage_coeff": 0.08,
        "pinn_collocation_points": 0,
        "hydro_package_profile": "rainfall-runoff",
    })
    return cfg


def main() -> int:
    ffn = common(load(FFN_BASE))
    ffn.update({
        "experiment_id": "sligo_ffn_pinn_latent_storage",
        "hidden_layers": "16,16",
        "activation": "relu",
        "use_time_lagged_ffn": False,
        "input_lags": "1",
    })

    lstm = common(load(LSTM_BASE))
    lstm.update({
        "experiment_id": "sligo_lstm_pinn_latent_storage_seq12_h32",
        "hidden_layers": "32",
        "activation": "tanh",
        "lstm_sequence_length": 12,
        "use_time_lagged_ffn": False,
        "input_lags": "1",
    })

    pinn = common(load(FFN_BASE))
    pinn.update({
        "experiment_id": "sligo_pinn_latent_storage",
        "hidden_layers": "24,24",
        "activation": "tanh",
        "use_time_lagged_ffn": False,
        "input_lags": "1",
        "data_weight": 0.0,
        "physics_weight": 1.0,
    })

    entries = [
        ("ffn_pinn", write("ffn_pinn_latent_storage.json", ffn)),
        ("lstm_pinn", write("lstm_pinn_latent_storage.json", lstm)),
        ("pinn", write("pinn_latent_storage.json", pinn)),
    ]
    BATCH.write_text(
        "# GIStoOHQ Sligo Creek latent-storage physics methods\n" +
        "\n".join(f"{mode} {path}" for mode, path in entries) + "\n",
        encoding="utf-8",
    )
    print("Generated 3 GIStoOHQ physics-informed experiments")
    print(f"Batch: {BATCH}")
    print(f"Configs: {OUT}")
    print("Latent storage is enabled by HydroBatch for all three physics modes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
