# Standalone console tool: semantic checkpoint reload-reproduction
# verification for the Stage B multiscale-memory-screen jobs
# (CLAUDE_SUPERVISOR_TASKS.md item 3). Reuses the exact non-GUI dependency
# list already proven to build and link in HydroBatch.pro, plus the three
# evaluation files (artifact_loader/inference_runner/shared_scale_inference)
# HydroPINN.pro additionally needs for checkpoint reload. No GUI/widgets.

QT += core

CONFIG += c++17 console
CONFIG -= app_bundle
TEMPLATE = app
TARGET = verify_stageb_checkpoint_reproduction

DEFINES += DEBUG_ TORCH_SUPPORT _arma ARMA_USE_OPENMP QT_NO_KEYWORDS

isEmpty(LIBTORCH_PATH) {
    exists(/mnt/3rd900/Projects/libtorch/include/torch/csrc/api/include/torch/torch.h) {
        LIBTORCH_PATH = /mnt/3rd900/Projects/libtorch
    } else: exists(/usr/local/libtorch/include/torch/csrc/api/include/torch/torch.h) {
        LIBTORCH_PATH = /usr/local/libtorch
    }
}
!exists($$LIBTORCH_PATH/include/torch/csrc/api/include/torch/torch.h) {
    error("LibTorch not found.")
}
message("Using LIBTORCH_PATH=$$LIBTORCH_PATH")

INCLUDEPATH += \
    $$LIBTORCH_PATH/include/torch/csrc/api/include \
    $$LIBTORCH_PATH/include \
    . Utilities Hydro Hydro/dataset Hydro/models Hydro/physics

LIBS += -L$$LIBTORCH_PATH/lib -ltorch -ltorch_cpu -lc10
LIBS += -lgomp -lpthread -larmadillo -lcrypto
QMAKE_CXXFLAGS += -fopenmp
QMAKE_LFLAGS += -fopenmp

isEmpty(TORCH_CXX11_ABI) { TORCH_CXX11_ABI = 1 }
QMAKE_CXXFLAGS += -D_GLIBCXX_USE_CXX11_ABI=$$TORCH_CXX11_ABI

unix:!macx { QMAKE_LFLAGS += -Wl,-rpath,$$LIBTORCH_PATH/lib }
CONFIG(release, debug|release) { QMAKE_CXXFLAGS += -O2 } else { QMAKE_CXXFLAGS += -O0 -g }

REDUCED_RESERVOIR_GENERATOR = $$PWD/Hydro/generate_reduced_reservoir_supervised_wrappers.py
GENERATED_FFN_WRAPPER = $$PWD/Hydro/generated/ffn_wrapper_reduced_reservoir.cpp
GENERATED_LSTM_NETWORK_WRAPPER = $$PWD/Hydro/generated/lstmnetworkwrapper_reduced_reservoir.cpp
system(python3 $$REDUCED_RESERVOIR_GENERATOR)
!exists($$GENERATED_FFN_WRAPPER) { error("Failed to generate reduced-reservoir FFN wrapper.") }
!exists($$GENERATED_LSTM_NETWORK_WRAPPER) { error("Failed to generate reduced-reservoir LSTM wrapper.") }

SOURCES += \
    Hydro/experiments/gistohq_sligo/verify_stageb_checkpoint_reproduction.cpp \
    neuralnetworkwrapper.cpp \
    neuralnetworkfactory.cpp \
    hyperparameters.cpp \
    Utilities/Distribution.cpp \
    Utilities/Matrix.cpp \
    Utilities/Matrix_arma.cpp \
    Utilities/Matrix_arma_sp.cpp \
    Utilities/QuickSort.cpp \
    Utilities/Utilities.cpp \
    Utilities/Vector.cpp \
    Utilities/Vector_arma.cpp \
    Hydro/dataset/ddrr_loader.cpp \
    Hydro/dataset/gistohq_hourly_harmonizer.cpp \
    Hydro/dataset/gistohq_package_adapter.cpp \
    Hydro/dataset/gistohq_model_rows.cpp \
    Hydro/dataset/gistohq_temporal_csv.cpp \
    Hydro/dataset/hydro_checksum.cpp \
    Hydro/dataset/hydro_dataset_contract.cpp \
    Hydro/dataset/lag_builder.cpp \
    Hydro/dataset/sequence_builder.cpp \
    Hydro/evaluation/experiment_exporter.cpp \
    Hydro/evaluation/experiment_loader.cpp \
    Hydro/evaluation/artifact_loader.cpp \
    Hydro/evaluation/inference_runner.cpp \
    Hydro/evaluation/shared_scale_inference.cpp \
    Hydro/models/pinn_wrapper.cpp \
    Hydro/physics/physics_config.cpp \
    Hydro/physics/rr_physics.cpp \
    $$GENERATED_FFN_WRAPPER \
    Hydro/models/ffn_pinn_wrapper.cpp \
    Hydro/models/lstm_wrapper.cpp \
    Hydro/models/lstm_pinn_wrapper.cpp \
    $$GENERATED_LSTM_NETWORK_WRAPPER
