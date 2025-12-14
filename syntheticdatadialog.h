#ifndef SYNTHETICDATADIALOG_H
#define SYNTHETICDATADIALOG_H

#include <QDialog>
#include <QLineEdit>
#include <QDoubleSpinBox>
#include <QPushButton>
#include <QLabel>
#include "TimeSeriesSet.h"
#include "TimeSeries.h"

/**
 * @brief Dialog for generating synthetic test data for GA optimization
 */
class SyntheticDataDialog : public QDialog
{
    Q_OBJECT

public:
    explicit SyntheticDataDialog(QWidget *parent = nullptr);
    ~SyntheticDataDialog();

    // Getters for generated data
    TimeSeriesSet<double> getInputData() const { return inputData_; }
    TimeSeries<double> getTargetData() const { return targetData_; }
    bool dataGenerated() const { return dataGenerated_; }

private Q_SLOTS:
    void onGenerate();
    void onPlotInputs();
    void onPlotTarget();

private:
    // UI Components
    QDoubleSpinBox *tStartSpin_;
    QDoubleSpinBox *tEndSpin_;
    QDoubleSpinBox *dtSpin_;
    QLineEdit *outputPathEdit_;
    QPushButton *browseButton_;
    QPushButton *generateButton_;
    QPushButton *plotInputsButton_;
    QPushButton *plotTargetButton_;
    QPushButton *okButton_;
    QPushButton *cancelButton_;
    QLabel *statusLabel_;

    // Data
    TimeSeriesSet<double> inputData_;
    TimeSeries<double> targetData_;
    bool dataGenerated_;

    // Helper methods
    void setupUI();
    void connectSignals();
    void generateSyntheticData(double t_start, double t_end, double dt, const std::string& output_path);
};

#endif // SYNTHETICDATADIALOG_H
