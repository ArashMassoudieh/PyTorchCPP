#!/usr/bin/env python3
"""Fast regression checks for HydroPINN build-time routing and synthetic physics.

No model training is performed. The checks are intentionally cheap enough to run
before qmake/full-paper runs.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HYDRO = HERE.parents[1]
GENERATOR = HYDRO / "generate_gistohq_pinn_window.py"
GENERATED = HYDRO / "generated" / "hydropinnwindow_gistohq_pinn.cpp"
ROUTING_HEADER = HYDRO / "models" / "hydro_lstm_module.h"


def check_generated_dispatch() -> None:
    subprocess.run([sys.executable, str(GENERATOR)], check=True, cwd=HYDRO.parent)
    text = GENERATED.read_text(encoding="utf-8")

    direct = '''        } else if (mode == "ffn_pinn") {\n            FFNReservoirPINNWrapper runner;\n            result = runner.train(cfg);\n'''
    ga_trial = '''            } else {\n                FFNPINNWrapper runner;\n                trial = runner.train(trialCfg);\n'''
    ga_confirm = '''                } else {\n                    FFNPINNWrapper runner;\n                    confirmTrial = runner.train(confirmCfg);\n'''

    if text.count(direct) != 1:
        raise SystemExit("FAIL: direct FFN+PINN dispatch is not uniquely routed to FFNReservoirPINNWrapper")
    if text.count(ga_trial) != 1:
        raise SystemExit("FAIL: GA trial dispatch was rewritten away from lag-capable FFNPINNWrapper")
    if text.count(ga_confirm) != 1:
        raise SystemExit("FAIL: GA confirmation dispatch was rewritten away from lag-capable FFNPINNWrapper")

    print("PASS: direct FFN+PINN uses reservoir wrapper; both GA lag-search dispatches remain lag-capable")


def check_two_reservoir_routing_contract() -> None:
    text = ROUTING_HEADER.read_text(encoding="utf-8")
    required = (
        "qFast = qFast + fastStep * (fast_fraction * r - qFast);",
        "qSlow = qSlow + slowStep * ((1.0 - fast_fraction) * r - qSlow);",
        "const torch::Tensor fast = torch::cat(fastValues, 0);",
        "const torch::Tensor slow = torch::cat(slowValues, 0);",
        "return std::make_tuple(fast + slow, fast, slow);",
    )
    for snippet in required:
        if snippet not in text:
            raise SystemExit(f"FAIL: two-reservoir routing contract changed or optimization missing: {snippet}")
    if "totalValues.push_back" in text:
        raise SystemExit("FAIL: two-reservoir routing still creates a per-timestep total-runoff tensor")

    # Numerically verify the optimized bookkeeping against the original recurrence.
    dt, fast_k, slow_k, alpha = 1.0, 0.10, 0.04, 0.85
    runoff = [0.0, 0.3, 1.1, 0.7, 0.2, 0.0, 0.5, 0.1]
    qf = qs = 0.0
    original: list[tuple[float, float, float]] = []
    for r in runoff:
        qf = qf + dt * fast_k * (alpha * r - qf)
        qs = qs + dt * slow_k * ((1.0 - alpha) * r - qs)
        original.append((qf + qs, qf, qs))

    qf = qs = 0.0
    optimized: list[tuple[float, float, float]] = []
    fast_step = dt * fast_k
    slow_step = dt * slow_k
    for r in runoff:
        qf = qf + fast_step * (alpha * r - qf)
        qs = qs + slow_step * ((1.0 - alpha) * r - qs)
        optimized.append((qf + qs, qf, qs))

    max_abs = max(abs(a - b) for left, right in zip(original, optimized) for a, b in zip(left, right))
    if max_abs > 1.0e-15:
        raise SystemExit(f"FAIL: two-reservoir routing refactor changed recurrence values: max_abs={max_abs:.3e}")
    print("PASS: two-reservoir routing keeps the recurrence and removes per-step total tensor allocation")


def check_backward_euler_truth() -> None:
    samples = 240
    t0, t1 = 0.0, 5.0
    dt = (t1 - t0) / (samples - 1)
    k = 0.08
    q = 0.15
    qs: list[float] = []
    peffs: list[float] = []

    for i in range(samples):
        r = i / (samples - 1)
        storm1 = 1.6 * math.exp(-0.5 * ((r - 0.25) / 0.055) ** 2)
        storm2 = 1.1 * math.exp(-0.5 * ((r - 0.62) / 0.085) ** 2)
        precipitation = storm1 + storm2 + 0.12 * max(0.0, math.sin(6.0 * math.pi * r))
        pet = 0.035 + 0.02 * (1.0 + math.sin(2.0 * math.pi * r - 0.5))
        peff = max(0.0, precipitation - pet)
        if i > 0:
            q = (q + dt * k * peff) / (1.0 + dt * k)
        q = max(0.0, q)
        qs.append(q)
        peffs.append(peff)

    residuals = [
        (qs[i] - qs[i - 1]) / dt - k * (peffs[i] - qs[i])
        for i in range(1, samples)
    ]
    rmse = math.sqrt(sum(v * v for v in residuals) / len(residuals))
    max_abs = max(abs(v) for v in residuals)
    print(f"synthetic truth residual: rmse={rmse:.3e}, max_abs={max_abs:.3e}")
    if rmse > 1.0e-11 or max_abs > 1.0e-10:
        raise SystemExit("FAIL: controlled synthetic truth is inconsistent with the PINN backward-Euler residual")
    print("PASS: controlled synthetic truth is discretely consistent with the PINN residual")


def main() -> int:
    check_generated_dispatch()
    check_two_reservoir_routing_contract()
    check_backward_euler_truth()
    print("PASS: HydroPINN GUI/physics regression checks complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
