QT -= gui
CONFIG += c++17 console
CONFIG -= app_bundle

TARGET = torch_qt_test
TEMPLATE = app

DEFINES += TORCH_SUPPORT
DEFINES += _arma
DEFINES += ARMA_USE_OPENMP

INCLUDEPATH += Utilities/

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
    hyperparameters.h \
    neuralnetworkwrapper.h

DEFINES += QT_NO_KEYWORDS

# LibTorch configuration
LIBTORCH_PATH = /usr/local/libtorch

INCLUDEPATH += $$LIBTORCH_PATH/include
INCLUDEPATH += $$LIBTORCH_PATH/include/torch/csrc/api/include

LIBS += -L$$LIBTORCH_PATH/lib
LIBS += -ltorch -ltorch_cpu -lc10
LIBS += -lgomp -lpthread -larmadillo

# Required for LibTorch
QMAKE_CXXFLAGS += -D_GLIBCXX_USE_CXX11_ABI=1
