#!/usr/bin/env python3
"""Paper tuning refinement with bias-aware validation selection for real catchments.

This is a thin override around run_adaptive_full_pipeline. Synthetic verification
is unchanged. For real rainfall-runoff data it:
  * keeps test data fully held out;
  * ranks non-degenerate candidates by validation KGE, NSE, |PBIAS|, then RMSE;
  * refines the process-aware LSTM+PINN routing grid around the boundary solution
    found by the previous paper run, without increasing the sweep size drastically.
"""
from __future__ import annotations

import math
import sys

import run_adaptive_full_pipeline as base


def winner(a, rows, mode):
    candidates = [r for r in rows if r.get("mode") == mode]
    if not candidates:
        raise RuntimeError(f"No candidates for mode={mode}")
    if a.data_source == "synthetic":
        return min(candidates, key=lambda r: (base.finite_float(r.get("validation_mse")), r.get("experiment_id", "")))

    def key(r):
        degenerate = r.get("validation_near_constant", "false") == "true"
        kge = base.finite_float(r.get("validation_kge"), -math.inf)
        nse = base.finite_float(r.get("validation_nse"), -math.inf)
        pbias = abs(base.finite_float(r.get("validation_pbias"), math.inf))
        rmse = base.finite_float(r.get("validation_rmse"), math.inf)
        return (1 if degenerate else 0, -kge, -nse, pbias, rmse, r.get("experiment_id", ""))

    return min(candidates, key=key)


def stage2(a, root, s1):
    if a.data_source == "synthetic":
        return ORIGINAL_STAGE2(a, root, s1)

    print("\n[adaptive] STAGE 2: bias-aware process/physics tuning", flush=True)
    ffn, lstm = s1["ffn"], s1["lstm"]
    stage_root = root / "02_stage2_physics"

    legacy_rows = base.run_generated(a, [
        "--methods", "ffn_pinn,pinn",
        "--ffn-architectures", base.q(ffn, "hidden_layers", "16,16"),
        "--ffn-activations", base.q(ffn, "activation", "relu"),
        "--pinn-architectures", "16,16;24,24;32,32",
        "--learning-rates", "0.003", "--batch-sizes", "32", "--seeds", "42",
        "--physics-weights", "0.000001,0.00001,0.0001,0.0005,0.001,0.005,0.01,0.025",
        "--recession-k", "0.0025,0.005,0.01,0.02,0.04,0.08,0.16",
    ], stage_root / "legacy")

    hybrid_architectures = base.unique_semicolon([
        base.q(lstm, "hidden_layers", "48"),
        "32,16", "48,24", "48,32,16", "64,32,16"
    ])
    arch_rows = base.run_generated(a, [
        "--methods", "lstm_pinn",
        "--lstm-architectures", hybrid_architectures,
        "--lstm-sequences", "6,12,24,48",
        "--learning-rates", "0.003", "--batch-sizes", "32", "--seeds", "42",
        "--physics-weights", "0.0001",
        "--lstm-pinn-profile", base.PROCESS_PROFILE,
        "--fast-k", "0.10", "--slow-k", "0.04", "--routing-alpha", "0.85",
    ], stage_root / "lstm_pinn_architecture")
    arch_winner = winner(a, arch_rows, "lstm_pinn")

    # Previous validation-selected solution landed on fast_k=0.10, slow_k=0.04,
    # alpha=0.85 and w=1e-4, several of them at/near grid boundaries. Refine that
    # neighborhood while keeping the number of expensive jobs controlled.
    routing_rows = base.run_generated(a, [
        "--methods", "lstm_pinn",
        "--lstm-architectures", base.q(arch_winner, "hidden_layers", "64,32,16"),
        "--lstm-sequences", base.q(arch_winner, "lstm_sequence_length", "48"),
        "--learning-rates", "0.003", "--batch-sizes", "32", "--seeds", "42",
        "--physics-weights", "0,0.00003,0.0001,0.0003",
        "--lstm-pinn-profile", base.PROCESS_PROFILE,
        "--fast-k", "0.05,0.10,0.20",
        "--slow-k", "0.02,0.04,0.06",
        "--routing-alpha", "0.75,0.85,0.95",
    ], stage_root / "lstm_pinn_routing")

    rows = legacy_rows + arch_rows + routing_rows
    base.write_rows(stage_root / "batch_summary.csv", rows)
    winners = {
        "ffn_pinn": winner(a, legacy_rows, "ffn_pinn"),
        "pinn": winner(a, legacy_rows, "pinn"),
        "lstm_pinn": winner(a, routing_rows, "lstm_pinn"),
    }
    return rows, winners


ORIGINAL_STAGE2 = base.stage2
base.winner = winner
base.stage2 = stage2
base.selection_label = lambda a: (
    "validation_mse" if a.data_source == "synthetic"
    else "validation_KGE_then_NSE_then_absPBIAS_then_RMSE_non_degenerate"
)

if __name__ == "__main__":
    raise SystemExit(base.main())
