#!/usr/bin/env python3
"""Generate the HydroPINN window translation unit with GIStoOHQ PINN enabled.

The main GUI source is intentionally kept as the canonical implementation.  This
small build-time transform removes the historical hard block on GIStoOHQ PINN
runs and enables the forcing-only physics adapter for physics-informed modes.
The generated file is not intended to be edited by hand.
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "hydropinnwindow.cpp"
OUT_DIR = HERE / "generated"
OUTPUT = OUT_DIR / "hydropinnwindow_gistohq_pinn.cpp"

OLD = '''            if (isGisToOhqHydroPinnExport(packageRoot) && mode != "ffn" && mode != "lstm") {
                throw std::runtime_error(
                    "GIStoOHQ HydroPINNExport has no observed storage; only FFN and LSTM are enabled. "
                    "PINN approaches require a separately versioned rainfall-runoff physics profile.");
            }
'''

NEW = '''            if (isGisToOhqHydroPinnExport(packageRoot) && mode != "ffn" && mode != "lstm") {
                // GIStoOHQ physics modes use the reduced runoff-reservoir equation
                // dQ/dt = k(Peff-Q), Peff=max(P-PET,0).  No observed or generated
                // storage enters the model inputs, so the residual is independent
                // rather than a restatement of a precomputed storage trajectory.
                cfg.use_latent_storage_physics = true; // legacy flag name: selects contiguous physics forcing layout
                cfg.pinn_physics_profile = "linear_reservoir";
                cfg.lambda_decay = cfg.latent_storage_recession_per_hour;
                cfg.forcing_gain = cfg.latent_storage_recession_per_hour;
                if (cfg.normalization != "none") {
                    cfg.normalization = "none";
                    appendLog("GIStoOHQ PINN residuals require physical units; normalization was set to none for this physics-informed run.");
                }
                appendLog(QString("GIStoOHQ physics mode enabled: runoff reservoir dQ/dt=k(Peff-Q), k=%1 1/h; no storage input is used.")
                              .arg(cfg.latent_storage_recession_per_hour, 0, 'g', 6));
            }
'''

OLD_STANDALONE = '''            appendLog(cfg.pinn_physics_profile == "water_balance"
                          ? "Standalone PINN uses physics-only water-balance loss with P/ET/total-storage features."
                          : "Standalone PINN uses the explicit physics-only wrapper (data_weight=0).");
'''

NEW_STANDALONE = '''            appendLog(cfg.use_latent_storage_physics
                          ? "Standalone PINN uses the runoff-reservoir physics residual with an initial-condition anchor and no storage input."
                          : (cfg.pinn_physics_profile == "water_balance"
                                 ? "Standalone PINN uses physics-only water-balance loss with explicit observed storage."
                                 : "Standalone PINN uses the explicit physics-only wrapper."));
'''

OLD_FFN_INCLUDE = '#include "models/ffn_pinn_wrapper.h"\n'
NEW_FFN_INCLUDE = '#include "models/ffn_pinn_wrapper.h"\n#include "models/ffn_reservoir_pinn_wrapper.h"\n'


def main() -> int:
    text = SOURCE.read_text(encoding="utf-8")
    if OLD not in text:
        raise SystemExit("Expected GIStoOHQ PINN hard-block text was not found; update the generator for the current GUI source.")
    text = text.replace(OLD, NEW, 1)
    if OLD_STANDALONE not in text:
        raise SystemExit("Expected standalone PINN log block was not found; update the generator for the current GUI source.")
    text = text.replace(OLD_STANDALONE, NEW_STANDALONE, 1)
    if OLD_FFN_INCLUDE not in text:
        raise SystemExit("Expected FFN-PINN include was not found; update the generator for the current GUI source.")
    text = text.replace(OLD_FFN_INCLUDE, NEW_FFN_INCLUDE, 1)
    text = text.replace("FFNPINNWrapper runner;", "FFNReservoirPINNWrapper runner;")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(text, encoding="utf-8")
    print(f"Generated {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
