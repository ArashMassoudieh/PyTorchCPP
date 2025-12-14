#ifndef INCREMENTALTRAININGDIALOG_H
#define INCREMENTALTRAININGDIALOG_H

#include <QDialog>
#include <QDoubleSpinBox>
#include <QSpinBox>
#include <QCheckBox>
#include <QPushButton>
#include <QLabel>
#include "commontypes.h"



/**
 * @brief Dialog for configuring incremental training parameters
 */
class IncrementalTrainingDialog : public QDialog
{
    Q_OBJECT

public:
    explicit IncrementalTrainingDialog(IncrementalTrainingParams& params, QWidget *parent = nullptr);
    ~IncrementalTrainingDialog();

private Q_SLOTS:
    void onAccept();
    void onWindowSizeChanged();
    void onWindowStepChanged();
    void updateSummary();

private:
    void setupUI();
    void connectSignals();
    void loadParameters();
    void saveParameters();

    // UI Components
    QDoubleSpinBox* windowSizeSpin_;
    QDoubleSpinBox* windowStepSpin_;
    QSpinBox* epochsPerWindowSpin_;
    QSpinBox* batchSizeSpin_;
    QDoubleSpinBox* learningRateSpin_;
    QCheckBox* useOverlapCheck_;
    QCheckBox* resetOptimizerCheck_;
    QLabel* summaryLabel_;
    QPushButton* okButton_;
    QPushButton* cancelButton_;

    // Reference to parameters
    IncrementalTrainingParams& params_;

    // Temporary values (only saved on OK)
    IncrementalTrainingParams tempParams_;
};

#endif // INCREMENTALTRAININGDIALOG_H
