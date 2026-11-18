#pragma once

// Opt-in FFN+HydroFusion one-hour-ahead causal forecasting mode (mode
// "ffn_hydrofusion" in Hydro/batch/hydro_batch_runner.cpp). A genuinely
// config-driven production dispatch for the HydroFusion pipeline verified
// in Hydro/experiments/gistohq_sligo/hydrofusion_ffn_engine_integration_test.cpp
// -- this wrapper fits the identical shared assembly/metadata code
// (hydrofusion_dataset_assembly.h, hydrofusion_metadata_sidecar.h) through
// this project's real CLI config loader -> dispatch -> train -> export
// path, with hyperparameters taken from the real HydroRunConfig instead
// of the test's hardcoded values. Still fits on the tiny synthetic
// fixture (hydrofusion_synthetic_fixture.h), not real GisToOhq data -- see
// CLAUDE_FIVE_METHOD_COMPATIBILITY_MATRIX_20261118.md's implementation
// plan for why wiring real ingestion is a separately-scoped, later step.
//
// No existing mode's dispatch/export/checkpoint behavior is touched by
// this file: "ffn_hydrofusion" is a brand-new mode string, itself the
// opt-in gate, so nothing reaches this wrapper unless a .batch file names
// it explicitly.

#include "hydro_run_types.h"

class FFNHydroFusionWrapper {
public:
    HydroRunResult train(const HydroRunConfig& config);
};
