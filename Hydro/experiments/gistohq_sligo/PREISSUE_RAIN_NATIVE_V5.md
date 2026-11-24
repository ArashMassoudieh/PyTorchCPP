# Native pre-issue rain v5

An opt-in **development-only** recipe binds the audited hourly rain CSV and its provenance SHA256 into the config, managed artifact manifest and model recipe tensor. No existing input or checkpoint recipe is replaced.

`zero_channel_v1` and `rate_hold_v1` require the same full 24-step weather history. Availability exclusions happen before training-only q_ref, input/target scaling and flow-loss scaling. Missing, late, malformed or low-coverage weather is excluded, never zero-filled. The zero channel is an explicit information ablation on the same eligible rows, not missing-weather imputation.

The available instantaneous rain rate is valid ten minutes before the forecast origin. Its archive HTTP Last-Modified must lie between valid time and origin. This is an archive release proxy; historical end-user operational delivery is unproven. Each historical step receives its own available value. No later rate is substituted.

Historical physical state remains the original causal valid-time v3 ledger driven by available lagged forcing. For `rate_hold_v1`, only a copy of that state forecasts the next hour using `max(0, available_rate - available_PET)`. The current-hour mean, storage and committed historical state are unchanged. This is a one-hour held-rate hypothesis, not knowledge of future rainfall. `zero_channel_v1` keeps the dry forecast copy.

Plain neural screens use zero physical-context channels and a persistence reference. Hybrid screens use physical-context channels with the discharge-anchored reference. Neural layouts have26channels in the planned matched pilot. Legacy20-channel modes, v4 25-channel modes and old checkpoints remain available when the rain recipe is disabled.

Acceptance requires strict synthetic CLI/export/fresh-reload, corruption rejection, disabled-recipe raw prediction parity and native feature-only support checks. The finite next pilot is eight new seed42 neural fits (four families times two rain recipes) and two fixed physical evaluations, validation only, with no automatic retries/expansion. Fifty epochs, eight hours per job, two CPU threads, no GPU,12GiB and250MB artifacts per job. Validation gates and both NSE/squared-Pearson-R² goals are frozen before launch. A passing development screen is not blind generalization or a manuscript promotion.

Scripts: `verify_hydrofusion_preissue_cli.py`, `compare_hydrofusion_preissue_legacy.py`, `hydrofusion_preissue_native_preflight.cpp`, `run_hydrofusion_preissue_screen.py`. Tests: `hydrofusion_preissue_rain_test.cpp`, `test_hydrofusion_preissue_screen.py`.
