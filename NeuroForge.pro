QT += core gui widgets charts
CONFIG += c++17
CONFIG -= app_bundle
TEMPLATE = app
TARGET = NeuroForge

# =========================
# Build Configuration
# =========================
DEFINES += Arash
DEFINES += DEBUG_
DEFINES += TORCH_SUPPORT
DEFINES += _arma
DEFINES += ARMA_USE_OPENMP
DEFINES += QT_NO_KEYWORDS  # Required for Qt + LibTorch compatibility
DEFINES += QT_GUI_SUPPORT
# =========================
# Project Sources & Headers
# =========================
SOURCES += \
    DataLoadDialog.cpp \
    GASettingsDialog.cpp \
    ProgressWindow.cpp \
    chartviewer.cpp \
    chartwindow.cpp \
    incrementaltrainingdialog.cpp \
    main.cpp \
    mainwindow.cpp \
    hyperparameters.cpp \
    networkarchitecturedialog.cpp \
    neuralnetworkfactory.cpp \
    neuralnetworkwrapper.cpp \
    Utilities/Distribution.cpp \
    Utilities/Matrix.cpp \
    Utilities/Matrix_arma.cpp \
    Utilities/Matrix_arma_sp.cpp \
    Utilities/QuickSort.cpp \
    Utilities/Utilities.cpp \
    Utilities/Vector.cpp \
    Utilities/Vector_arma.cpp \
    syntheticdatadialog.cpp

HEADERS += \
    DataLoadDialog.h \
    GASettingsDialog.h \
    ProgressWindow.h \
    chartviewer.h \
    chartwindow.h \
    commontypes.h \
    incrementaltrainingdialog.h \
    mainwindow.h \
    Binary.h \
    Normalization.h \
    TestHyperParameters.h \
    ga.h \
    ga.hpp \
    hyperparameters.h \
    individual.h \
    networkarchitecturedialog.h \
    neuralnetworkfactory.h \
    neuralnetworkwrapper.h \
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
    syntheticdatadialog.h

FORMS += \
    mainwindow.ui

# =========================
# LibTorch Configuration
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
QMAKE_CXXFLAGS += -fopenmp
QMAKE_LFLAGS   += -fopenmp

# =========================
# ABI Compatibility
# =========================
# PyTorch 2.8.0 (GCC ≥7) usually needs ABI=1
QMAKE_CXXFLAGS += -D_GLIBCXX_USE_CXX11_ABI=1

# =========================
# Optional: embed RPATH so you don't need LD_LIBRARY_PATH
# =========================
QMAKE_LFLAGS += -Wl,-rpath,$$LIBTORCH_PATH/lib

# =========================
# Deployment
# =========================
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
