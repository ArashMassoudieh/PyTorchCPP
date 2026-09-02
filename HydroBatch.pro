# Headless HydroPINN experiment batch runner.
QT += core
CONFIG += c++17 console
CONFIG -= app_bundle
TEMPLATE = app
TARGET = HydroBatch

isEmpty(HOST_PROFILE) {
    CONFIG += Jason
    DEFINES += Jason
}
contains(CONFIG, Jason) { DEFINES += Jason }
contains(CONFIG, Behzad) { DEFINES += Behzad }
contains(CONFIG, PowerEdge) { DEFINES += PowerEdge }
contains(CONFIG, Arash) { DEFINES += Arash }
contains(CONFIG, SligoCreek) { DEFINES += SligoCreek }

DEFINES += DEBUG_ TORCH_SUPPORT _arma ARMA_USE_OPENMP QT_NO_KEYWORDS

# Embed the source revision used by this binary. __DATE__/__TIME__ are captured
# by Hydro/build_identity.h during compilation.
GIT_COMMIT = $$system(git -C $$PWD rev-parse --short HEAD 2>/dev/null)
isEmpty(GIT_COMMIT) { GIT_COMMIT = unknown }
DEFINES += HYDRO_GIT_COMMIT=\"$$GIT_COMMIT\"
message("HydroBatch build commit=$$GIT_COMMIT")

isEmpty(LIBTORCH_PATH) {
    contains(DEFINES, Jason) { LIBTORCH_PATH = /usr/local/libtorch }
    contains(DEFINES, Arash) { LIBTORCH_PATH = /usr/local/libtorch }
    contains(DEFINES, PowerEdge) { LIBTORCH_PATH = /mnt/3rd900/Projects/libtorch }
    contains(DEFINES, Behzad) { LIBTORCH_PATH = /mnt/3rd900/Projects/libtorch }
}
!exists($$LIBTORCH_PATH/include/torch/csrc/api/include/torch/torch.h) {
    exists(/mnt/3rd900/Projects/libtorch/include/torch/csrc/api/include/torch/torch.h) {
        LIBTORCH_PATH = /mnt/3rd900/Projects/libtorch
    } else: exists(/usr/local/libtorch/include/torch/csrc/api/include/torch/torch.h) {
        LIBTORCH_PATH = /usr/local/libtorch
    } else: exists(/opt/libtorch/include/torch/csrc/api/include/torch/torch.h) {
        LIBTORCH_PATH = /opt/libtorch
    }
}
!exists($$LIBTORCH_PATH/include/torch/csrc/api/include/torch/torch.h) {
    error("LibTorch not found. Set LIBTORCH_PATH or install it in a supported location.")
}

INCLUDEPATH += \
    $$LIBTORCH_PATH/include/torch/csrc/api/include \
    $$LIBTORCH_PATH/include \
    . Utilities Hydro Hydro/dataset Hydro/models Hydro/physics Hydro/evaluation

LIBS += -L$$LIBTORCH_PATH/lib -ltorch -ltorch_cpu -lc10
LIBS += -lgomp -lpthread -larmadillo -lcrypto
QMAKE_CXXFLAGS += -fopenmp
QMAKE_LFLAGS += -fopenmp

isEmpty(TORCH_CXX11_ABI) { TORCH_CXX11_ABI = 1 }
QMAKE_CXXFLAGS += -D_GLIBCXX_USE_CXX11_ABI=$$TORCH_CXX11_ABI
unix:!macx { QMAKE_LFLAGS += -Wl,-rpath,$$LIBTORCH_PATH/lib }

CONFIG(release, debug|release) { QMAKE_CXXFLAGS += -O3 } else { QMAKE_CXXFLAGS += -O0 -g }

SOURCES += \
    Hydro/batch/hydro_batch_runner.cpp \
    Hydro/build_identity_batch.cpp \
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
    Hydro/models/pinn_wrapper.cpp \
    Hydro/physics/physics_config.cpp \
    Hydro/physics/rr_physics.cpp \
    Hydro/models/ffn_wrapper.cpp \
    Hydro/models/ffn_pinn_wrapper.cpp \
    Hydro/models/lstm_wrapper.cpp \
    Hydro/models/lstm_pinn_wrapper.cpp \
    Hydro/models/lstmnetworkwrapper.cpp

HEADERS += \
    neuralnetworkwrapper.h neuralnetworkfactory.h commontypes.h Normalization.h TestHyperParameters.h \
    ga.h ga.hpp hyperparameters.h individual.h \
    Utilities/TimeSeries.h Utilities/TimeSeries.hpp Utilities/TimeSeriesSet.h Utilities/TimeSeriesSet.hpp \
    Utilities/Distribution.h Utilities/Matrix.h Utilities/Matrix_arma.h Utilities/Matrix_arma_sp.h \
    Utilities/QuickSort.h Utilities/Utilities.h Utilities/Vector.h Utilities/Vector_arma.h \
    Hydro/build_identity.h \
    Hydro/dataset/ddrr_loader.h Hydro/dataset/gistohq_hourly_harmonizer.h \
    Hydro/dataset/gistohq_package_adapter.h Hydro/dataset/gistohq_model_rows.h \
    Hydro/dataset/gistohq_tensor_builder.h Hydro/dataset/gistohq_temporal_csv.h \
    Hydro/dataset/hydro_checksum.h Hydro/dataset/hydro_dataset_contract.h \
    Hydro/dataset/hydro_package_directory.h Hydro/dataset/hydro_units.h Hydro/dataset/forecast_alignment.h \
    Hydro/dataset/hydro_tensor_builder.h Hydro/dataset/csv_tensor_builder.h \
    Hydro/dataset/lag_builder.h Hydro/dataset/lagged_tensor_builder.h \
    Hydro/dataset/sequence_builder.h Hydro/dataset/chronological_split.h Hydro/dataset/tensor_scaler.h \
    Hydro/evaluation/hydro_metrics.h Hydro/evaluation/experiment_exporter.h Hydro/evaluation/experiment_loader.h \
    Hydro/physics/physics_config.h Hydro/physics/rr_physics.h \
    Hydro/models/hydro_run_types.h Hydro/models/hydro_lstm_module.h \
    Hydro/models/ffn_wrapper.h Hydro/models/ffn_pinn_wrapper.h Hydro/models/pinn_wrapper.h \
    Hydro/models/lstm_wrapper.h Hydro/models/lstm_pinn_wrapper.h Hydro/models/lstmnetworkwrapper.h
