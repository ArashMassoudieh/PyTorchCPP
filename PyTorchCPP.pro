QT += core gui widgets
TEMPLATE = app
CONFIG += c++17 console
CONFIG -= app_bundle

DEFINES += Arash
DEFINES += QT_GUI_SUPPORT

# =========================
# LibTorch configuration
# =========================
LIBTORCH_PATH = /usr/local/libtorch

contains(DEFINES, Arash) {
    LIBTORCH_PATH = /usr/local/libtorch
}

contains(DEFINES, PowerEdge) {
    LIBTORCH_PATH = /mnt/3rd900/Projects/libtorch
}

# Automatic fallback detection when host default is missing/unset.
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
    error("LibTorch not found. Set LIBTORCH_PATH or install to /mnt/3rd900/Projects/libtorch, /usr/local/libtorch, or /opt/libtorch.")
}

message("Using LIBTORCH_PATH=$$LIBTORCH_PATH")

# Include headers
INCLUDEPATH += $$LIBTORCH_PATH/include
INCLUDEPATH += $$LIBTORCH_PATH/include/torch/csrc/api/include

# Link against libtorch libraries
LIBS += -L$$LIBTORCH_PATH/lib -ltorch -ltorch_cpu -lc10

# Optional: If your compiler needs rpath for runtime lib loading
QMAKE_RPATHDIR += $$LIBTORCH_PATH/lib

SOURCES += \
        main.cpp
