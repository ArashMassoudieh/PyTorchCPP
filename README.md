# PyTorchCPP

## PINN progress

`NeuralNetworkWrapper` now includes a PINN-oriented training method for the
exponential decay ODE:

\[
\frac{dy}{dt} + \lambda y = 0
\]

Use `trainPINNExponentialDecay(...)` to combine:

- supervised data loss (`MSE(y_pred, y_true)`), and
- physics residual loss (`MSE(dy/dt + lambda * y_pred, 0)`).

This is a practical baseline for continuing PINN work while keeping the current
Qt/libtorch training workflow intact.
