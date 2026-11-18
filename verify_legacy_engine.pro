# Synthetic regression harness; share the headless production dependency list.
include(HydroBatch.pro)
TARGET = legacy_engine_regression_test
SOURCES -= Hydro/batch/hydro_batch_runner.cpp
SOURCES += Hydro/tests/legacy_engine_regression_test.cpp \
           Hydro/evaluation/artifact_loader.cpp \
           Hydro/evaluation/inference_runner.cpp \
           Hydro/evaluation/shared_scale_inference.cpp
