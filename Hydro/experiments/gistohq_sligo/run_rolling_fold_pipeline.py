#!/usr/bin/env python3
"""Run one adaptive five-method paper pipeline with rolling split generation."""
from __future__ import annotations

from pathlib import Path

import run_improved_adaptive_pipeline as improved

HERE = Path(__file__).resolve().parent
ROLLING_GENERATOR = HERE / "generate_rolling_sweep.py"

# run_improved_adaptive_pipeline delegates generation through the imported base
# module. Redirect only that generator hook; all tuning, validation selection,
# process physics, parallel HydroBatch execution, and five-seed robustness stay
# identical to the established paper pipeline.
improved.base.GENERATOR = ROLLING_GENERATOR

if __name__ == "__main__":
    raise SystemExit(improved.base.main())
