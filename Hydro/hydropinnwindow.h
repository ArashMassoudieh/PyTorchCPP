#pragma once

#include <QMainWindow>

class QLabel;
class QComboBox;
class QPushButton;

/**
 * @file hydropinnwindow.h
 * @brief Main GUI window for HydroPINN mode selection and run controls.
 */

/**
 * @brief Simple Qt main window for HydroPINN experiments.
 *
 * Provides a mode selector for baseline/PINN model variants and a run action
 * entry point. This class is intentionally lightweight while the Hydro module
 * is under active development.
 */
class HydroPINNWindow : public QMainWindow {
    Q_OBJECT

public:
    /**
     * @brief Construct the HydroPINN main window.
     * @param parent Optional parent widget.
     */
    explicit HydroPINNWindow(QWidget* parent = nullptr);

private:
    QLabel* statusLabel_;   ///< Status label with current mode information.
    QComboBox* modeCombo_;  ///< Mode selector (ffn, ffn_pinn, lstm, lstm_pinn).
    QPushButton* runButton_;

    /**
     * @brief Update status text according to the selected mode.
     */
    void updateStatus();
};
