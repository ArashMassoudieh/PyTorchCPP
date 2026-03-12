# ============================================================
# HydroPINN.pro
# Separate window app project for hydrology / PINN work
# Keeps existing GUI app (e.g. NeuroForge.pro) untouched
# ============================================================

QT += core gui widgets

CONFIG += c++17 console
CONFIG -= app_bundle
TEMPLATE = app
TARGET = HydroPINN

# =========================
# Preferred host config style
# =========================
CONFIG += Jason
DEFINES += Jason

# CONFIG += Behzad
# DEFINES += Behzad

# CONFIG += PowerEdge
# DEFINES += PowerEdge

# CONFIG += Arash
# DEFINES += Arash

# CONFIG += SligoCreek
# DEFINES += SligoCreek

# =========================
# Build Configuration
# =========================
DEFINES += DEBUG_
DEFINES += TORCH_SUPPORT
DEFINES += _arma
DEFINES += ARMA_USE_OPENMP
DEFINES += QT_NO_KEYWORDS
DEFINES += QT_GUI_SUPPORT

# =========================
# LibTorch Configuration
# =========================
LIBTORCH_PATH =

# Host-preferred defaults
contains(DEFINES, Jason) {
    LIBTORCH_PATH = /usr/local/libtorch
}

contains(DEFINES, Arash) {
    LIBTORCH_PATH = /usr/local/libtorch
}

contains(DEFINES, PowerEdge) {
    LIBTORCH_PATH = /mnt/3rd900/Projects/libtorch
}

# Add others here later if needed
# contains(DEFINES, Behzad) {
#     LIBTORCH_PATH = /path/to/libtorch
# }
#
# contains(DEFINES, SligoCreek) {
#     LIBTORCH_PATH = /path/to/libtorch
# }

# Automatic fallback detection when host default is missing/unset.
!exists($$LIBTORCH_PATH/include/torch/torch.h) {
    exists(/mnt/3rd900/Projects/libtorch/include/torch/torch.h) {
        LIBTORCH_PATH = /mnt/3rd900/Projects/libtorch
    } else: exists(/usr/local/libtorch/include/torch/torch.h) {
        LIBTORCH_PATH = /usr/local/libtorch
    } else: exists(/opt/libtorch/include/torch/torch.h) {
        LIBTORCH_PATH = /opt/libtorch
    }
}

!exists($$LIBTORCH_PATH/include/torch/torch.h) {
    error("LibTorch not found. Set LIBTORCH_PATH or install to /mnt/3rd900/Projects/libtorch, /usr/local/libtorch, or /opt/libtorch.")
}

message("Using LIBTORCH_PATH=$$LIBTORCH_PATH")

INCLUDEPATH += $$LIBTORCH_PATH/include/torch/csrc/api/include
INCLUDEPATH += $$LIBTORCH_PATH/include

INCLUDEPATH += .
INCLUDEPATH += Utilities
INCLUDEPATH += Hydro
INCLUDEPATH += Hydro/dataset
INCLUDEPATH += Hydro/models
INCLUDEPATH += Hydro/physics

LIBS += -L$$LIBTORCH_PATH/lib -ltorch -ltorch_cpu -lc10
LIBS += -lgomp -lpthread -larmadillo

QMAKE_CXXFLAGS += -fopenmp
QMAKE_LFLAGS   += -fopenmp
QMAKE_CXXFLAGS += -D_GLIBCXX_USE_CXX11_ABI=1
QMAKE_LFLAGS   += -Wl,-rpath,$$LIBTORCH_PATH/lib

# Optional build tuning
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
    Utilities/Vector_arma.cpp

# Add these as you create them
SOURCES += \
    Hydro/dataset/ddrr_loader.cpp \
    Hydro/dataset/lag_builder.cpp \
    Hydro/dataset/sequence_builder.cpp \
    Hydro/physics/physics_config.cpp \
    Hydro/physics/rr_physics.cpp \
    Hydro/models/ffn_wrapper.cpp \
    Hydro/models/ffn_pinn_wrapper.cpp \
    Hydro/models/lstm_wrapper.cpp \
    Hydro/models/lstm_pinn_wrapper.cpp

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
    Utilities/Vector_arma.h

HEADERS += \
    Hydro/hydropinnwindow.h \
    Hydro/dataset/ddrr_loader.h \
    Hydro/dataset/lag_builder.h \
    Hydro/dataset/sequence_builder.h \
    Hydro/physics/physics_config.h \
    Hydro/physics/rr_physics.h \
    Hydro/models/ffn_wrapper.h \
    Hydro/models/ffn_pinn_wrapper.h \
    Hydro/models/lstm_wrapper.h \
    Hydro/models/lstm_pinn_wrapper.h

# =========================
# Install target
# =========================
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
