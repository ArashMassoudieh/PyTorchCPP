#pragma once

#include "hydro_run_types.h"
#include "../dataset/tensor_scaler.h"

#include <torch/torch.h>

#include <cstdint>
#include <string>
#include <vector>

// One basin's raw (pre-normalization), already-sequenced LSTM input/target
// tensors - target is already basin-normalized runoff depth (mm/hour), the
// same field loadHydroPackageTensors always returns as `y` (see
// gistohq_hourly_harmonizer.cpp's catchment-area conversion), NOT raw
// discharge. xSeq: [N, seqLen, F]. ySeq: [N, 1].
struct RegionalBasinInput {
    std::string staid;
    torch::Tensor xSeq;
    torch::Tensor ySeq;
};

struct RegionalPretrainConfig {
    int epochs = 1;
    int batch_size = 16;
    double learning_rate = 0.005;
    double weight_decay = 0.0;
    std::string normalization = "log_standardize";
    std::vector<int> hidden_layers = {32};
    std::string activation = "tanh";
    double lstm_dropout = 0.0;
    int64_t random_seed = 42;
    // Sequences drawn per basin per epoch (see multi_basin_sampling.h). <= 0
    // means "auto": the smallest training basin's own sequence count, so
    // every basin contributes equally without ever padding the shortest one
    // beyond what it can supply without repeats - the stricter, more
    // conservative default. Pass an explicit value to force more epoch-level
    // coverage of larger basins at the cost of short basins repeating more.
    int64_t per_basin_sequence_count = -1;
};

struct RegionalPretrainOutput {
    bool success = false;
    std::string message;
    std::vector<std::uint8_t> model_checkpoint;
    std::string model_checkpoint_format;
    int best_epoch = 0;
    double best_validation_loss = std::numeric_limits<double>::infinity();
    std::vector<double> training_loss_history;
    std::vector<double> validation_loss_history;
};

// Pools `trainingBasins` into balanced, shuffled per-epoch minibatches (each
// basin contributing the identical sequence count per epoch regardless of
// its own record length - see multi_basin_sampling.h), with per-basin input
// and target normalization fit ONLY from that basin's own data (never a
// training basin's statistics applied to another basin, and never a
// validation basin's statistics borrowed from any training basin). Model
// selection uses pooled validation loss over `validationBasins` - basins
// entirely held out from training, each normalized by its OWN independently
// fit scaler. Deterministic given `config.random_seed` (drives both the
// per-epoch sampling and this LSTM's own weight initialization).
//
// Pure at the tensor level - no file I/O, no HydroRunConfig dependency -
// so it is directly unit-testable with tiny synthetic tensors (see
// scientific_core_test.cpp's resolveTestEndSequenceCount-style tests for
// the precedent this follows). The thin, file-I/O-performing wrapper that
// loads real basin packages into RegionalBasinInput lives in
// lstm_regional_pretrain.cpp as trainLstmRegionalPretrainFromConfig.
RegionalPretrainOutput trainLstmRegionalPool(
    const std::vector<RegionalBasinInput>& trainingBasins,
    const std::vector<RegionalBasinInput>& validationBasins,
    const RegionalPretrainConfig& config);

// Loads and sequences ONE basin package (loadHydroPackageTensors +
// makeLstmSequences), applying the exact calendar window
// (templateConfig.regional_window_start/regional_window_end) if set.
// Exposed publicly (not just used internally by
// trainLstmRegionalPretrainFromConfig) so the window filter's exact
// row-count behavior is independently testable against a real package,
// not just inferred from whether the full pipeline happens to succeed.
RegionalBasinInput loadRegionalBasinInput(const std::string& packagePath, const HydroRunConfig& templateConfig);

// Fits ONE target scaler from the training pool only (basin-balanced
// equally-weighted mixture, not per-basin, not touching validation data -
// see lstm_regional_pretrain.cpp's doc comment for the leakage bug this
// exists to avoid). Exposed publicly so the mixture-variance formula is
// independently testable against an explicitly concatenated equal-basin
// reference, not just inferred from end-to-end training behavior.
TensorScaler fitPooledTargetScalerFromTrainingPool(const std::vector<RegionalBasinInput>& trainingBasins,
                                                   const std::string& normalization);

// File-I/O entry point for HydroBatch's dispatcher: reads
// config.regional_training_basin_list_path/regional_validation_basin_list_path
// (plain text, one GIStoOHQ package path per line - the same convention
// already used by urban_tier.txt/all_candidates.txt), loads and sequences
// each basin via loadHydroPackageTensors + makeLstmSequences (the identical
// gap-aware windowing single-basin LSTM training already uses - see
// lstm_sequence_builder.h), and calls trainLstmRegionalPool. Returns a
// HydroRunResult so it slots into the existing batch/export machinery
// unchanged; most physical-unit metrics fields are left at their NaN
// default since they do not apply to a basin-pooled pretraining run (no
// single x/observed/predicted series to report) - model_checkpoint is the
// field this mode exists to produce.
HydroRunResult trainLstmRegionalPretrainFromConfig(const HydroRunConfig& config);
