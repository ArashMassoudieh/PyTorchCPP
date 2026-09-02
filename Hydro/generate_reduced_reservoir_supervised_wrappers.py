#!/usr/bin/env python3
"""Generate supervised FFN/LSTM wrappers with the shared reduced-reservoir synthetic source.

The canonical wrappers retain their existing synthetic profiles.  At build time this
transform adds exactly one explicit profile, ``reduced_reservoir``, and routes it to
``buildReducedReservoirSyntheticTensors``.  This guarantees FFN and LSTM see the
same t/P/PET/Peff/Q realization used by FFN+PINN, LSTM+PINN, and PINN.
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"
OUT = HERE / "generated"

FILES = {
    "ffn_wrapper.cpp": "ffn_wrapper_reduced_reservoir.cpp",
    "lstmnetworkwrapper.cpp": "lstmnetworkwrapper_reduced_reservoir.cpp",
}

INCLUDE_ANCHOR = '#include "../dataset/hydro_tensor_builder.h"\n'
INCLUDE_REPLACEMENT = (
    '#include "../dataset/hydro_tensor_builder.h"\n'
    '#include "../dataset/reservoir_physics_tensor_builder.h"\n'
)

LOAD_BLOCK = '''    if (!loadHydroPackageTensors(config, x, y, plotX)) {
        if (config.use_csv_data) loadHydroCsvTensors(config, x, y, plotX);
        else buildSyntheticSeries(config, x, y, plotX);
    }
'''

LOAD_REPLACEMENT = '''    if (!loadHydroPackageTensors(config, x, y, plotX)) {
        if (config.use_csv_data) loadHydroCsvTensors(config, x, y, plotX);
        else if (config.synthetic_profile == "reduced_reservoir")
            buildReducedReservoirSyntheticTensors(config, x, y, plotX);
        else buildSyntheticSeries(config, x, y, plotX);
    }
'''


def transform(source: Path, output: Path) -> None:
    text = source.read_text(encoding="utf-8")
    if INCLUDE_ANCHOR not in text:
        raise SystemExit(f"Include anchor not found in {source}")
    text = text.replace(INCLUDE_ANCHOR, INCLUDE_REPLACEMENT, 1)
    if LOAD_BLOCK not in text:
        raise SystemExit(f"Synthetic load block not found in {source}")
    text = text.replace(LOAD_BLOCK, LOAD_REPLACEMENT, 1)
    output.write_text(text, encoding="utf-8")
    print(f"Generated {output}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for source_name, output_name in FILES.items():
        transform(MODELS / source_name, OUT / output_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
