TEMPLATE = app
CONFIG += console c++17
CONFIG -= app_bundle
CONFIG -= qt

# Path to libtorch
LIBTORCH_DIR = /home/arash/Projects/libtorch

# Include headers
INCLUDEPATH += $$LIBTORCH_DIR/include
INCLUDEPATH += $$LIBTORCH_DIR/include/torch/csrc/api/include

# Link against libtorch libraries
LIBS += -L$$LIBTORCH_DIR/lib -ltorch -ltorch_cpu -lc10

# Optional: If your compiler needs rpath for runtime lib loading
QMAKE_RPATHDIR += $$LIBTORCH_DIR/lib


SOURCES += \
        main.cpp
