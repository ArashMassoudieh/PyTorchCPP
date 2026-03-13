#pragma once

#include <QMainWindow>

class QLabel;
class QComboBox;
class QPushButton;
class QTextEdit;

/**
 * @file hydropinnwindow.h
 * @brief Main GUI window for HydroPINN mode selection and run controls.
 */

/**
 * @brief Simple Qt main window for HydroPINN experiments.
 *
 * Provides mode selection and run orchestration for baseline/PINN variants.
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
    QTextEdit* logText_;    ///< Runtime log output for mode execution.

    /**
     * @brief Update status text according to the selected mode.
     */
    void updateStatus();

    /**
     * @brief Dispatch and execute the currently selected mode.
     */
    void runSelectedMode();

    /**
     * @brief Append a timestamped line to the runtime log panel.
     */
    void appendLog(const QString& line);
};
