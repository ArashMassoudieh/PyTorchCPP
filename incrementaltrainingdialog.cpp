#include "incrementaltrainingdialog.h"
#include "mainwindow.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QGroupBox>
#include <QMessageBox>

IncrementalTrainingDialog::IncrementalTrainingDialog(IncrementalTrainingParams& params, QWidget *parent)
    : QDialog(parent)
    , params_(params)
    , tempParams_(params)  // Copy current values
{
    setWindowTitle("Configure Incremental Training");
    setModal(true);
    setupUI();
    connectSignals();
    loadParameters();
    updateSummary();
    resize(550, 500);
}

IncrementalTrainingDialog::~IncrementalTrainingDialog()
{
}

void IncrementalTrainingDialog::setupUI()
{
    QVBoxLayout* mainLayout = new QVBoxLayout(this);

    // Window configuration group
    QGroupBox* windowGroup = new QGroupBox("Window Configuration", this);
    QFormLayout* windowLayout = new QFormLayout(windowGroup);

    windowSizeSpin_ = new QDoubleSpinBox(this);
    windowSizeSpin_->setRange(10.0, 100000.0);
    windowSizeSpin_->setDecimals(1);
    windowSizeSpin_->setSuffix(" time units");
    windowLayout->addRow("Window Size:", windowSizeSpin_);

    windowStepSpin_ = new QDoubleSpinBox(this);
    windowStepSpin_->setRange(10.0, 100000.0);
    windowStepSpin_->setDecimals(1);
    windowStepSpin_->setSuffix(" time units");
    windowLayout->addRow("Window Step:", windowStepSpin_);

    useOverlapCheck_ = new QCheckBox("Allow window overlap", this);
    windowLayout->addRow("", useOverlapCheck_);

    mainLayout->addWidget(windowGroup);

    // Training configuration group
    QGroupBox* trainingGroup = new QGroupBox("Training Configuration", this);
    QFormLayout* trainingLayout = new QFormLayout(trainingGroup);

    epochsPerWindowSpin_ = new QSpinBox(this);
    epochsPerWindowSpin_->setRange(1, 10000);
    trainingLayout->addRow("Epochs per Window:", epochsPerWindowSpin_);

    batchSizeSpin_ = new QSpinBox(this);
    batchSizeSpin_->setRange(1, 1024);
    trainingLayout->addRow("Batch Size:", batchSizeSpin_);

    learningRateSpin_ = new QDoubleSpinBox(this);
    learningRateSpin_->setRange(0.000001, 1.0);
    learningRateSpin_->setDecimals(6);
    learningRateSpin_->setSingleStep(0.0001);
    trainingLayout->addRow("Learning Rate:", learningRateSpin_);

    resetOptimizerCheck_ = new QCheckBox("Reset optimizer state on new window", this);
    resetOptimizerCheck_->setToolTip("If checked, optimizer momentum is reset for each window");
    trainingLayout->addRow("", resetOptimizerCheck_);

    mainLayout->addWidget(trainingGroup);

    // Summary
    QGroupBox* summaryGroup = new QGroupBox("Summary", this);
    QVBoxLayout* summaryLayout = new QVBoxLayout(summaryGroup);

    summaryLabel_ = new QLabel(this);
    summaryLabel_->setWordWrap(true);
    summaryLabel_->setStyleSheet("QLabel { background-color: #ecf0f1; padding: 10px; border-radius: 5px; }");
    summaryLayout->addWidget(summaryLabel_);

    mainLayout->addWidget(summaryGroup);

    mainLayout->addStretch();

    // Dialog buttons
    QHBoxLayout* buttonLayout = new QHBoxLayout();
    buttonLayout->addStretch();

    okButton_ = new QPushButton("OK", this);
    okButton_->setDefault(true);
    buttonLayout->addWidget(okButton_);

    cancelButton_ = new QPushButton("Cancel", this);
    buttonLayout->addWidget(cancelButton_);

    mainLayout->addLayout(buttonLayout);
    setLayout(mainLayout);
}

void IncrementalTrainingDialog::connectSignals()
{
    connect(okButton_, &QPushButton::clicked, this, &IncrementalTrainingDialog::onAccept);
    connect(cancelButton_, &QPushButton::clicked, this, &QDialog::reject);

    // Update summary when values change
    connect(windowSizeSpin_, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &IncrementalTrainingDialog::onWindowSizeChanged);
    connect(windowStepSpin_, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &IncrementalTrainingDialog::onWindowStepChanged);
    connect(epochsPerWindowSpin_, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &IncrementalTrainingDialog::updateSummary);
    connect(useOverlapCheck_, &QCheckBox::stateChanged,
            this, &IncrementalTrainingDialog::updateSummary);

}

void IncrementalTrainingDialog::loadParameters()
{
    // Load from the actual params (not tempParams_)
    windowSizeSpin_->setValue(params_.windowSize);
    windowStepSpin_->setValue(params_.windowStep);
    epochsPerWindowSpin_->setValue(params_.epochsPerWindow);
    batchSizeSpin_->setValue(params_.batchSize);
    learningRateSpin_->setValue(params_.learningRate);
    useOverlapCheck_->setChecked(params_.useOverlap);
    resetOptimizerCheck_->setChecked(params_.resetOnNewWindow);

    tempParams_ = params_;
}

void IncrementalTrainingDialog::saveParameters()
{
    tempParams_.windowSize = windowSizeSpin_->value();
    tempParams_.windowStep = windowStepSpin_->value();
    tempParams_.epochsPerWindow = epochsPerWindowSpin_->value();
    tempParams_.batchSize = batchSizeSpin_->value();
    tempParams_.learningRate = learningRateSpin_->value();
    tempParams_.useOverlap = useOverlapCheck_->isChecked();
    tempParams_.resetOnNewWindow = resetOptimizerCheck_->isChecked();
}

void IncrementalTrainingDialog::onWindowSizeChanged()
{
    // Auto-adjust step if overlap is not desired
    if (!useOverlapCheck_->isChecked()) {
        windowStepSpin_->setValue(windowSizeSpin_->value());
    }
    updateSummary();
}

void IncrementalTrainingDialog::onWindowStepChanged()
{
    // Check if there's overlap
    double windowSize = windowSizeSpin_->value();
    double windowStep = windowStepSpin_->value();

    if (windowStep < windowSize) {
        useOverlapCheck_->setChecked(true);
    } else {
        useOverlapCheck_->setChecked(false);
    }
    updateSummary();
}

void IncrementalTrainingDialog::updateSummary()
{
    double windowSize = windowSizeSpin_->value();
    double windowStep = windowStepSpin_->value();
    int epochs = epochsPerWindowSpin_->value();

    QString summary;

    // Window overlap info
    if (windowStep < windowSize) {
        double overlap = windowSize - windowStep;
        double overlapPercent = (overlap / windowSize) * 100.0;
        summary += QString("<b>Window Overlap:</b> %1 time units (%2%)<br>")
                       .arg(overlap, 0, 'f', 1)
                       .arg(overlapPercent, 0, 'f', 1);
    } else {
        summary += QString("<b>Window Overlap:</b> None (sequential windows)<br>");
    }

    // Training info
    summary += QString("<b>Training per window:</b> %1 epochs<br>").arg(epochs);
    summary += QString("<b>Batch size:</b> %1<br>").arg(batchSizeSpin_->value());
    summary += QString("<b>Learning rate:</b> %1<br>").arg(learningRateSpin_->value(), 0, 'g', 4);

    // Optimizer reset
    if (resetOptimizerCheck_->isChecked()) {
        summary += QString("<br><i>Optimizer state will reset between windows</i>");
    } else {
        summary += QString("<br><i>Optimizer state will persist across windows</i>");
    }

    summaryLabel_->setText(summary);
}

void IncrementalTrainingDialog::onAccept()
{
    // Validate parameters
    if (windowSizeSpin_->value() <= 0) {
        QMessageBox::warning(this, "Invalid Parameter", "Window size must be positive.");
        return;
    }

    if (windowStepSpin_->value() <= 0) {
        QMessageBox::warning(this, "Invalid Parameter", "Window step must be positive.");
        return;
    }

    if (epochsPerWindowSpin_->value() <= 0) {
        QMessageBox::warning(this, "Invalid Parameter", "Epochs per window must be positive.");
        return;
    }

    // Save to temporary params
    saveParameters();

    // Copy to actual params (only on successful accept)
    params_ = tempParams_;

    accept();
}
