#!/usr/bin/env python3
"""Generate the HydroPINN window translation unit with current physics routing.

The canonical GUI source stays readable and stable. This build-time transform:
- enables GIStoOHQ reduced-reservoir physics without reconstructed storage,
- routes FFN+PINN to the corrected reservoir wrapper,
- exposes an explicit ``reduced_reservoir`` synthetic validation profile, and
- makes that profile use the same shared truth generator as all five methods.
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
                // dQ/dt = k(Peff-Q), Peff=max(P-PET,0). No observed or generated
                // storage enters the model inputs.
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

NEW_STANDALONE = '''            appendLog(cfg.pinn_physics_profile == "linear_reservoir"
                          ? "Standalone PINN uses reduced-reservoir physics with one observed initial-condition anchor; no storage input is used."
                          : (cfg.pinn_physics_profile == "water_balance"
                                 ? "Standalone PINN uses physics-only water-balance loss with explicit observed storage."
                                 : "Standalone PINN uses the explicit physics-only wrapper."));
'''

OLD_FFN_INCLUDE = '#include "models/ffn_pinn_wrapper.h"\n'
NEW_FFN_INCLUDE = '#include "models/ffn_pinn_wrapper.h"\n#include "models/ffn_reservoir_pinn_wrapper.h"\n'
OLD_DATASET_INCLUDE = '#include "dataset/csv_tensor_builder.h"\n'
NEW_DATASET_INCLUDE = '#include "dataset/csv_tensor_builder.h"\n#include "dataset/reservoir_physics_tensor_builder.h"\n'

OLD_PROFILE_LIST = 'profileCombo_->addItems({"watershed_balance", "rainfall_runoff", "neuroforge_inputs_target", "exp_decay", "damped_sine", "mixed_wave"});'
NEW_PROFILE_LIST = 'profileCombo_->addItems({"watershed_balance", "rainfall_runoff", "reduced_reservoir", "neuroforge_inputs_target", "exp_decay", "damped_sine", "mixed_wave"});'

OLD_CFG = '''    HydroRunConfig cfg = currentConfig();
    if (cfg.use_hydro_package) {
'''
NEW_CFG = '''    HydroRunConfig cfg = currentConfig();
    if (!cfg.use_hydro_package && !cfg.use_csv_data && cfg.synthetic_profile == "reduced_reservoir" &&
        (mode == "ffn_pinn" || mode == "lstm_pinn" || mode == "pinn")) {
        cfg.pinn_physics_profile = "linear_reservoir";
        cfg.latent_storage_recession_per_hour = std::max(1.0e-8, cfg.lambda_decay);
        cfg.forcing_gain = cfg.latent_storage_recession_per_hour;
        cfg.use_time_lagged_ffn = false;
        if (cfg.normalization != "none") {
            cfg.normalization = "none";
            appendLog("Reduced-reservoir PINN validation uses physical units; normalization was set to none for this physics-informed run.");
        }
        appendLog(QString("Controlled reduced-reservoir physics enabled: dQ/dt=k(Peff-Q), k=%1.")
                      .arg(cfg.latent_storage_recession_per_hour, 0, 'g', 6));
    }
    if (cfg.use_hydro_package) {
'''

OLD_PREVIEW_BRANCH = '''    } else if (profile == "watershed_balance") {
'''
NEW_PREVIEW_BRANCH = '''    } else if (profile == "reduced_reservoir") {
        HydroRunConfig previewCfg = currentConfig();
        previewCfg.synthetic_profile = "reduced_reservoir";
        previewCfg.latent_storage_recession_per_hour = std::max(1.0e-8, previewCfg.lambda_decay);
        torch::Tensor rx, ry, rt;
        buildReducedReservoirSyntheticTensors(previewCfg, rx, ry, rt);
        auto xc = rx.to(torch::kCPU).contiguous();
        auto yc = ry.to(torch::kCPU).contiguous();
        for (int64_t i = 0; i < xc.size(0); ++i) {
            xs.push_back(xc[i][0].item<double>());
            rainfall.push_back(xc[i][2].item<double>());
            evapotranspiration.push_back(xc[i][3].item<double>());
            ys.push_back(yc[i][0].item<double>());
        }
    } else if (profile == "watershed_balance") {
'''

OLD_INPUT_MAP = '''    } else if (profile == "watershed_balance" || profile == "rainfall_runoff") {
        lastSyntheticInputs_["effective_precipitation"] = rainfall;
'''
NEW_INPUT_MAP = '''    } else if (profile == "reduced_reservoir") {
        std::vector<double> peff;
        peff.reserve(rainfall.size());
        for (size_t i = 0; i < rainfall.size(); ++i) peff.push_back(std::max(0.0, rainfall[i] - evapotranspiration[i]));
        lastSyntheticInputs_["precipitation"] = rainfall;
        lastSyntheticInputs_["PET"] = evapotranspiration;
        lastSyntheticInputs_["effective_precipitation"] = peff;
    } else if (profile == "watershed_balance" || profile == "rainfall_runoff") {
        lastSyntheticInputs_["effective_precipitation"] = rainfall;
'''

OLD_EXPORT = '''        } else if (profile == "watershed_balance" || profile == "rainfall_runoff") {
            if (profile == "watershed_balance") {
'''
NEW_EXPORT = '''        } else if (profile == "reduced_reservoir") {
            out << "t,precipitation,PET,runoff\\n";
            for (int i = 0; i < samples; ++i) {
                const size_t k = static_cast<size_t>(i);
                out << xs[k] << "," << rainfall[k] << "," << evapotranspiration[k] << "," << ys[k] << "\\n";
            }
        } else if (profile == "watershed_balance" || profile == "rainfall_runoff") {
            if (profile == "watershed_balance") {
'''


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f"Expected {label} anchor was not found; update the GUI generator for the current source.")
    return text.replace(old, new, 1)


def main() -> int:
    text = SOURCE.read_text(encoding="utf-8")
    text = replace_once(text, OLD, NEW, "GIStoOHQ PINN")
    text = replace_once(text, OLD_STANDALONE, NEW_STANDALONE, "standalone PINN log")
    text = replace_once(text, OLD_FFN_INCLUDE, NEW_FFN_INCLUDE, "FFN-PINN include")
    text = replace_once(text, OLD_DATASET_INCLUDE, NEW_DATASET_INCLUDE, "dataset include")
    text = replace_once(text, OLD_PROFILE_LIST, NEW_PROFILE_LIST, "synthetic profile list")
    text = replace_once(text, OLD_CFG, NEW_CFG, "run-mode config")
    text = replace_once(text, OLD_PREVIEW_BRANCH, NEW_PREVIEW_BRANCH, "synthetic preview")
    text = replace_once(text, OLD_INPUT_MAP, NEW_INPUT_MAP, "synthetic input map")
    text = replace_once(text, OLD_EXPORT, NEW_EXPORT, "synthetic export")
    text = text.replace("FFNPINNWrapper runner;", "FFNReservoirPINNWrapper runner;")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(text, encoding="utf-8")
    print(f"Generated {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
