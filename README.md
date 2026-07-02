# PyTorchCPP / NeuroForge

PyTorchCPP is a collection of Qt + LibTorch desktop applications for experimenting
with neural-network training workflows in C++. The repository currently centers on
three related apps:

- **NeuroForge** (`NeuroForge.pro`): a Qt Widgets GUI for loading tabular data,
  configuring feed-forward neural networks, training with LibTorch, and using a
  genetic algorithm (GA) to search hyperparameters.
- **HydroPINN** (`HydroPINN.pro`): a hydrology-focused GUI that compares pure
  data-driven models with physics-informed neural-network (PINN) variants.
- **LSTM Time Series Predictor** (`LSTM/`): a standalone Qt/Libtorch time-series
  predictor for sequence-model experiments.

The codebase is useful as both a runnable application and a C++ reference for
combining Qt, LibTorch, Armadillo, OpenMP, and model-selection utilities.

## Repository layout

| Path | Purpose |
| --- | --- |
| `main.cpp`, `mainwindow.*`, `*.ui` | NeuroForge application entry point and primary Qt UI. |
| `neuralnetworkwrapper.*`, `neuralnetworkfactory.*` | LibTorch model construction and training wrappers used by the GUI workflows. |
| `ga.*`, `individual.h`, `GADistribution.*` | Genetic algorithm primitives for hyperparameter search. |
| `Hydro/` | HydroPINN datasets, physics residuals, model wrappers, and hydrology GUI. |
| `LSTM/` | Standalone LSTM time-series predictor application. |
| `Utilities/` | Matrix/vector/time-series helper classes and Armadillo-backed implementations. |
| `Data/` | Small example input/output files and a sample NeuroForge project file. |
| `Doxygen/` | Generated API documentation artifacts. |

## Application capabilities

### NeuroForge

NeuroForge provides an interactive training workflow for small-to-medium tabular
machine-learning problems:

1. Load input and observed-output data from text files.
2. Configure network architecture, normalization, lags, and training parameters.
3. Train neural-network models through LibTorch-backed wrappers.
4. Use GA-driven optimization to explore network and training hyperparameters.
5. Inspect training progress and plot input/output data through Qt dialogs.

### HydroPINN

HydroPINN extends the shared training infrastructure to rainfall-runoff style
experiments. It includes dataset loaders, lag/sequence builders, hydrologic
physics residuals, and model wrappers for five approaches:

- feed-forward neural network (FFN),
- FFN + PINN residual,
- LSTM,
- LSTM + PINN residual,
- standalone PINN baseline.

See [`Hydro/APPROACHES.md`](Hydro/APPROACHES.md) for a concise description of
those approaches and how they differ.

### LSTM Time Series Predictor

The `LSTM/` folder contains a focused sequence-prediction app with its own main
window, dataset class, preprocessing utilities, early stopping, model trainer,
and result plotting components.

## Dependencies

Install the following before building any app:

- C++17-capable compiler.
- Qt 5 or Qt 6 with Widgets; Qt Charts is required by `NeuroForge.pro` and
  `HydroPINN.pro`.
- LibTorch C++ distribution matching your compiler ABI.
- Armadillo development package.
- OpenMP runtime (`libgomp` on GCC-based Linux systems).
- qmake or Qt Creator.

> **LibTorch ABI note:** The project files default to `_GLIBCXX_USE_CXX11_ABI=1`.
> If your LibTorch package was built with the old C++ ABI, pass
> `TORCH_CXX11_ABI=0` to qmake where supported.

## Building

Build out of tree so generated Makefiles and object files stay separate from the
source tree.

### NeuroForge

```bash
mkdir -p build-neuroforge
cd build-neuroforge
qmake ../NeuroForge.pro LIBTORCH_PATH=/path/to/libtorch
make -j"$(nproc)"
./NeuroForge
```

### HydroPINN

```bash
mkdir -p build-hydropinn
cd build-hydropinn
qmake ../HydroPINN.pro LIBTORCH_PATH=/path/to/libtorch TORCH_CXX11_ABI=1
make -j"$(nproc)"
./HydroPINN
```

### Minimal PyTorchCPP target

`PyTorchCPP.pro` is a smaller qmake target that currently builds the main Qt
entry point and is useful for validating LibTorch/Qt discovery.

```bash
mkdir -p build-pytorchcpp
cd build-pytorchcpp
qmake ../PyTorchCPP.pro LIBTORCH_PATH=/path/to/libtorch
make -j"$(nproc)"
```

## Running a quick NeuroForge workflow

1. Build and launch `NeuroForge`.
2. Use the data-loading dialog to select `Data/Inputs.txt` and `Data/Output.txt`
   or your own whitespace-delimited files.
3. Open the architecture and hyperparameter dialogs to define hidden layers,
   learning rate, epoch count, and normalization choices.
4. Start training and monitor progress in the progress/chart windows.
5. Optionally run the GA optimizer to search alternative hyperparameter sets.

## PINN progress

`NeuralNetworkWrapper` includes a PINN-oriented training method for the
exponential decay ODE:

\[
\frac{dy}{dt} + \lambda y = 0
\]

Use `trainPINNExponentialDecay(...)` to combine:

- supervised data loss (`MSE(y_pred, y_true)`), and
- physics residual loss (`MSE(dy/dt + lambda * y_pred, 0)`).

This is a practical baseline for continuing PINN work while keeping the current
Qt/LibTorch training workflow intact.

## Development notes

- Prefer editing the richer `NeuroForge.pro` or `HydroPINN.pro` targets when
  adding production features; `PyTorchCPP.pro` is intentionally minimal.
- Keep generated build directories out of the repository.
- Regenerate Doxygen output after changing public headers or major APIs.
- Avoid mixing Qt signal/slot macros with LibTorch headers unless
  `QT_NO_KEYWORDS` is enabled, as done in the main qmake project files.

## Doxygen notes

HydroPINN headers include Doxygen comments for key classes and APIs, including
dataset builders, model wrappers, physics configuration/residuals, and the main
window. Regenerate docs with your usual Doxygen configuration to include updated
pages.
