#!/usr/bin/env python3
"""Generate a normal unified sweep, then apply rolling chronological split ratios.

This wrapper deliberately leaves generate_unified_sweep.py unchanged. The split
fractions come from HYDROPINN_ROLLING_TRAIN_FRACTION and
HYDROPINN_ROLLING_VALIDATION_FRACTION. Every generated config is checked and
rewritten before HydroBatch can consume unified_sweep.batch.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE / "generate_unified_sweep.py"
GENERATED = HERE / "generated_unified"


def env_fraction(name: str) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        raise SystemExit(f"{name} is required for rolling-CV generation")
    try:
        value = float(raw)
    except ValueError as exc:
        raise SystemExit(f"{name} must be numeric") from exc
    if not 0.0 < value < 1.0:
        raise SystemExit(f"{name} must lie strictly between 0 and 1")
    return value


def main() -> int:
    train = env_fraction("HYDROPINN_ROLLING_TRAIN_FRACTION")
    validation = env_fraction("HYDROPINN_ROLLING_VALIDATION_FRACTION")
    if train + validation >= 1.0:
        raise SystemExit("rolling train_fraction + validation_fraction must be < 1")

    subprocess.run([sys.executable, str(BASE), *sys.argv[1:]], cwd=HERE, check=True)
    configs = sorted(GENERATED.glob("*.json"))
    if not configs:
        raise RuntimeError("unified sweep generated no JSON configs")
    for path in configs:
        cfg = json.loads(path.read_text(encoding="utf-8"))
        cfg["train_fraction"] = train
        cfg["validation_fraction"] = validation
        cfg["shuffle_training"] = False
        path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    print(f"[rolling-generator] applied chronological train={train:g}, validation={validation:g} to {len(configs)} configs", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
