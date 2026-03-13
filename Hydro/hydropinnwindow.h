#pragma once

#include <QMainWindow>

#include "models/hydro_run_types.h"

class QLabel;
class QComboBox;
class QPushButton;
class QTextEdit;
class QSpinBox;
class QDoubleSpinBox;
class QCheckBox;
class QLineEdit;
class QChartView;

/**
 * @file hydropinnwindow.h
 * @brief Main GUI window for HydroPINN mode selection and run controls.
 */
class HydroPINNWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit HydroPINNWindow(QWidget* parent = nullptr);

private:
    QLabel* statusLabel_;
    QComboBox* modeCombo_;
    QPushButton* runButton_;
    QPushButton* runAllButton_;
    QPushButton* runFFNButton_;
    QPushButton* runFFNPINNButton_;
    QPushButton* runLSTMButton_;
    QPushButton* runLSTMPINNButton_;
    QTextEdit* logText_;
    QChartView* chartView_;

    // Config controls
    QSpinBox* epochsSpin_;
    QSpinBox* batchSpin_;
    QDoubleSpinBox* lrSpin_;
    QDoubleSpinBox* lambdaSpin_;
    QDoubleSpinBox* dataWeightSpin_;
    QDoubleSpinBox* physicsWeightSpin_;
    QSpinBox* sampleCountSpin_;
    QDoubleSpinBox* tStartSpin_;
    QDoubleSpinBox* tEndSpin_;
    QLineEdit* hiddenLayersEdit_;
    QComboBox* activationCombo_;
    QComboBox* profileCombo_;
    QCheckBox* evalCheck_;

    void updateStatus();
    void runSelectedMode();
    void runAllModes();
    void runMode(const QString& mode);
    void setRunningUiState(bool running);
    void appendLog(const QString& line);
    HydroRunConfig currentConfig() const;
    void updatePlot(const QString& mode, const HydroRunResult& result);
};
