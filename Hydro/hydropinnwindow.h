#pragma once

#include <QMainWindow>
#include <QString>

#include <map>

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
class QTextBrowser;
class QListWidget;

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
    QTextEdit* logText_;
    QChartView* chartView_;
    QTextBrowser* perfSummaryText_;

    // Training/network controls
    QSpinBox* epochsSpin_;
    QSpinBox* batchSpin_;
    QDoubleSpinBox* lrSpin_;
    QDoubleSpinBox* lambdaSpin_;
    QDoubleSpinBox* dataWeightSpin_;
    QDoubleSpinBox* physicsWeightSpin_;
    QLineEdit* hiddenLayersEdit_;
    QComboBox* activationCombo_;
    QSpinBox* layerSizeSpin_;
    QComboBox* layerActivationCombo_;
    QPushButton* addLayerButton_;
    QPushButton* removeLayerButton_;
    QListWidget* layersList_;
    QComboBox* outputActivationCombo_;
    QCheckBox* evalCheck_;

    // Data controls
    QComboBox* dataSourceCombo_;
    QLineEdit* csvPathEdit_;
    QPushButton* browseCsvButton_;
    QSpinBox* csvXColSpin_;
    QSpinBox* csvYColSpin_;
    QCheckBox* csvHeaderCheck_;
    QSpinBox* sampleCountSpin_;
    QDoubleSpinBox* tStartSpin_;
    QDoubleSpinBox* tEndSpin_;
    QComboBox* profileCombo_;
    QPushButton* generateSyntheticButton_;
    QLineEdit* syntheticExportPathEdit_;
    QPushButton* browseSyntheticExportButton_;

    // NeuroForge-style workflow actions
    QPushButton* runPredictionButton_;
    QPushButton* runAllPredictionButton_;
    QPushButton* runPredictionFFNButton_;
    QPushButton* runPredictionFFNPINNButton_;
    QPushButton* runPredictionLSTMButton_;
    QPushButton* runPredictionLSTMPINNButton_;
    QPushButton* runTrainingButton_;
    QPushButton* runAllTrainingButton_;
    QPushButton* runTrainingFFNButton_;
    QPushButton* runTrainingFFNPINNButton_;
    QPushButton* runTrainingLSTMButton_;
    QPushButton* runTrainingLSTMPINNButton_;
    QPushButton* configureGAButton_;
    QPushButton* startGAButton_;
    QPushButton* stopGAButton_;
    QPushButton* refreshPerformanceButton_;
    QPushButton* clearPlotButton_;
    std::map<QString, HydroRunResult> lastModeResults_;

    void updateStatus();
    void runSelectedMode();
    void runAllModes();
    void runMode(const QString& mode);
    void showSelectedPrediction();
    void showAllPredictions();
    void showPredictionForMode(const QString& mode);
    void setRunningUiState(bool running);
    void updateDataSourceUiState();
    void browseCsv();
    void browseSyntheticExportPath();
    void generateSyntheticDataPreview();
    void appendLog(const QString& line);
    HydroRunConfig currentConfig() const;
    QString selectedModeKey() const;
    void syncNetworkCsvFromLayerList();
    void updatePlot(const QString& mode, const HydroRunResult& result);
    void configureGAPlaceholder();
    void startGAPlaceholder();
    void stopGAPlaceholder();
    void refreshPerformanceAssessment();
    void clearPlot();
};
