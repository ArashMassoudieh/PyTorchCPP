QT += core gui widgets
TEMPLATE = app
CONFIG += c++17 console
CONFIG -= app_bundle

# Path to libtorch
LIBTORCH_DIR = /home/arash/Projects/libtorch

# Include headers
INCLUDEPATH += $$LIBTORCH_DIR/include
INCLUDEPATH += $$LIBTORCH_DIR/include/torch/csrc/api/include

# Link against libtorch libraries
LIBS += -L$$LIBTORCH_DIR/lib -ltorch -ltorch_cpu -lc10

# Optional: If your compiler needs rpath for runtime lib loading
QMAKE_RPATHDIR += $$LIBTORCH_DIR/lib

# Enable shared GUI-guarded code paths
DEFINES += QT_GUI_SUPPORT


SOURCES += \
        main.cpp
