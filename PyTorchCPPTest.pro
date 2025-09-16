QT -= gui
CONFIG += c++17 console
CONFIG -= app_bundle

TEMPLATE = app
TARGET = torch_qt_test

DEFINES += PowerEdge

# =========================
# Project Sources & Headers
# =========================
SOURCES += \
    hyperparameters.cpp \
    main_TimeSeries_Training.cpp \
    Utilities/Distribution.cpp \
    Utilities/Matrix.cpp \
    Utilities/Matrix_arma.cpp \
    Utilities/Matrix_arma_sp.cpp \
    Utilities/QuickSort.cpp \
    Utilities/Utilities.cpp \
    Utilities/Vector.cpp \
    Utilities/Vector_arma.cpp \
    neuralnetworkwrapper.cpp

HEADERS += \
    Binary.h \
    Normalization.h \
    TestHyperParameters.h \
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
    ga.h \
    ga.hpp \
    hyperparameters.h \
    individual.h \
    neuralnetworkwrapper.h
    Utilities/Vector_arma.h


DEFINES += TORCH_SUPPORT
DEFINES += _arma
DEFINES += ARMA_USE_OPENMP
DEFINES += QT_NO_KEYWORDS

# =========================
# LibTorch configuration
# =========================

# Default (fallback)
LIBTORCH_PATH = /usr/local/libtorch

# If "Arash" is defined in DEFINES += Arash
contains(DEFINES, Arash) {
    LIBTORCH_PATH = /usr/local/libtorch
}

# If "PowerEdge" is defined in DEFINES += PowerEdge
contains(DEFINES, PowerEdge) {
    LIBTORCH_PATH = /mnt/3rd900/Projects/libtorch
}


# Includes (order matters!)
INCLUDEPATH += $$LIBTORCH_PATH/include/torch/csrc/api/include
INCLUDEPATH += $$LIBTORCH_PATH/include
INCLUDEPATH += Utilities


# Libraries
LIBS += -L$$LIBTORCH_PATH/lib -ltorch -ltorch_cpu -lc10

# =========================
# Extra Libraries
# =========================
LIBS += -lgomp -lpthread -larmadillo

# =========================
# ABI Compatibility
# =========================
# PyTorch 2.8.0 (GCC ≥7) usually needs ABI=1
QMAKE_CXXFLAGS += -D_GLIBCXX_USE_CXX11_ABI=1

# =========================
# Optional: embed RPATH so you don’t need LD_LIBRARY_PATH
# =========================
QMAKE_LFLAGS += -Wl,-rpath,$$LIBTORCH_PATH/lib
