# ============================================================
# HydroPINN.pro
# Separate Qt GUI project for hydrology / physics-informed ML.
# This keeps the existing GUI app/project files untouched.
# ============================================================

QT += core gui widgets charts

CONFIG += c++17 console
CONFIG -= app_bundle
TEMPLATE = app
TARGET = HydroPINN

# =========================
# Host profile
# =========================
# Pick ONE profile, or pass one from qmake:
#   qmake HydroPINN.pro CONFIG+=PowerEdge
# =========================
isEmpty(HOST_PROFILE) {
    CONFIG += Jason
    DEFINES += Jason
}

contains(CONFIG, Jason) {
    DEFINES += Jason
}
contains(CONFIG, Behzad) {
    DEFINES += Behzad
}
contains(CONFIG, PowerEdge) {
    DEFINES += PowerEdge
}
contains(CONFIG, Arash) {
    DEFINES += Arash
}
contains(CONFIG, SligoCreek) {
    DEFINES += SligoCreek
}

# =========================
# Build configuration
# =========================
DEFINES += DEBUG_
DEFINES += TORCH_SUPPORT
DEFINES += _arma
DEFINES += ARMA_USE_OPENMP
DEFINES += QT_NO_KEYWORDS

# Do not enable QT_GUI_SUPPORT here unless ProgressWindow.cpp is also linked.
# Some shared neural-network sources may otherwise reference ProgressWindow symbols.

# =========================
# LibTorch configuration
# =========================
# You can override this from terminal:
#   qmake HydroPINN.pro LIBTORCH_PATH=/path/to/libtorch
# =========================
isEmpty(LIBTORCH_PATH) {
    contains(DEFINES, Jason) {
        LIBTORCH_PATH = /usr/local/libtorch
    }
    contains(DEFINES, Arash) {
        LIBTORCH_PATH = /usr/local/libtorch
    }
    contains(DEFINES, PowerEdge) {
        LIBTORCH_PATH = /mnt/3rd900/Projects/libtorch
    }
    contains(DEFINES, Behzad) {
        LIBTORCH_PATH = /mnt/3rd900/Projects/libtorch
    }
}

# Automatic fallback detection.
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
    error("LibTorch not found. Run qmake with LIBTORCH_PATH=/path/to/libtorch or install it in /mnt/3rd900/Projects/libtorch, /usr/local/libtorch, or /opt/libtorch.")
}

message("Using LIBTORCH_PATH=$$LIBTORCH_PATH")

INCLUDEPATH += \
    $$LIBTORCH_PATH/include/torch/csrc/api/include \
    $$LIBTORCH_PATH/include \
    . \
    Utilities \
    Hydro \
    Hydro/dataset \
    Hydro/models \
    Hydro/physics

LIBS += -L$$LIBTORCH_PATH/lib -ltorch -ltorch_cpu -lc10
LIBS += -lgomp -lpthread -larmadillo

QMAKE_CXXFLAGS += -fopenmp
QMAKE_LFLAGS   += -fopenmp

# IMPORTANT:
# This ABI value must match the downloaded LibTorch package.
# Most prebuilt Linux LibTorch packages use 1, but older/special builds may use 0.
isEmpty(TORCH_CXX11_ABI) {
    TORCH_CXX11_ABI = 1
}
QMAKE_CXXFLAGS += -D_GLIBCXX_USE_CXX11_ABI=$$TORCH_CXX11_ABI
message("Using TORCH_CXX11_ABI=$$TORCH_CXX11_ABI")

unix:!macx {
    QMAKE_LFLAGS += -Wl,-rpath,$$LIBTORCH_PATH/lib
}

CONFIG(release, debug|release) {
    QMAKE_CXXFLAGS += -O3
} else {
    QMAKE_CXXFLAGS += -O0 -g
}

# =========================
# Sources
# =========================
SOURCES += \
    Hydro/main_hydropinn.cpp \
    Hydro/hydropinnwindow.cpp \
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
    Hydro/dataset/lag_builder.cpp \
    Hydro/dataset/sequence_builder.cpp \
    Hydro/physics/physics_config.cpp \
    Hydro/physics/rr_physics.cpp \
    Hydro/models/ffn_wrapper.cpp \
    Hydro/models/ffn_pinn_wrapper.cpp \
    Hydro/models/lstm_wrapper.cpp \
    Hydro/models/lstm_pinn_wrapper.cpp \
    Hydro/models/lstmnetworkwrapper.cpp

# =========================
# Headers
# =========================
HEADERS += \
    neuralnetworkwrapper.h \
    neuralnetworkfactory.h \
    commontypes.h \
    Normalization.h \
    TestHyperParameters.h \
    ga.h \
    ga.hpp \
    hyperparameters.h \
    individual.h \
    Utilities/TimeSeries.h \
    Utilities/TimeSeries.hpp \
    Utilities/TimeSeriesSet.h \
    Utilities/TimeSeriesSet.hpp \
    Utilities/Distribution.h \
    Utilities/Matrix.h \
    Utilities/Matrix_arma.h \
    Utilities/Matrix_arma_sp.h \
    Utilities/QuickSort.h \
    Utilities/Utilities.h \
    Utilities/Vector.h \
    Utilities/Vector_arma.h \
    Hydro/hydropinnwindow.h \
    Hydro/dataset/ddrr_loader.h \
    Hydro/dataset/lag_builder.h \
    Hydro/dataset/sequence_builder.h \
    Hydro/physics/physics_config.h \
    Hydro/physics/rr_physics.h \
    Hydro/models/hydro_run_types.h \
    Hydro/models/ffn_wrapper.h \
    Hydro/models/ffn_pinn_wrapper.h \
    Hydro/models/lstm_wrapper.h \
    Hydro/models/lstm_pinn_wrapper.h \
    Hydro/models/lstmnetworkwrapper.h

# =========================
# Install target
# =========================
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
