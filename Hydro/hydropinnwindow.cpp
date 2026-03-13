#include "hydropinnwindow.h"

#include "models/ffn_wrapper.h"
#include "models/ffn_pinn_wrapper.h"
#include "models/lstm_wrapper.h"
#include "models/lstm_pinn_wrapper.h"

#include <QCheckBox>
#include <QComboBox>
#include <QCoreApplication>
#include <QDateTime>
#include <QDoubleSpinBox>
#include <QElapsedTimer>
#include <QFormLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QMessageBox>
#include <QPushButton>
#include <QSpinBox>
#include <QTextEdit>
#include <QVBoxLayout>
#include <QWidget>

#include <exception>

HydroPINNWindow::HydroPINNWindow(QWidget* parent)
    : QMainWindow(parent), statusLabel_(new QLabel(this)), modeCombo_(new QComboBox(this)),
      runButton_(new QPushButton("Run Selected", this)), runAllButton_(new QPushButton("Run All", this)),
      runFFNButton_(new QPushButton("Run FFN", this)), runFFNPINNButton_(new QPushButton("Run FFN_PINN", this)),
      runLSTMButton_(new QPushButton("Run LSTM", this)), runLSTMPINNButton_(new QPushButton("Run LSTM_PINN", this)),
      logText_(new QTextEdit(this)), epochsSpin_(new QSpinBox(this)), batchSpin_(new QSpinBox(this)),
      lrSpin_(new QDoubleSpinBox(this)), lambdaSpin_(new QDoubleSpinBox(this)),
      dataWeightSpin_(new QDoubleSpinBox(this)), physicsWeightSpin_(new QDoubleSpinBox(this)),
      evalCheck_(new QCheckBox("Evaluate test metrics", this)) {
    setWindowTitle("HydroPINN - Experiment Runner");
    resize(920, 560);

    auto* central = new QWidget(this);
    auto* root = new QVBoxLayout(central);

    auto* title = new QLabel("HydroPINN Modes", central);
    title->setStyleSheet("font-size: 18px; font-weight: bold;");

    modeCombo_->addItems({"ffn", "ffn_pinn", "lstm", "lstm_pinn"});

    // Config panel
    auto* cfgBox = new QGroupBox("Run Configuration", central);
    auto* cfgForm = new QFormLayout(cfgBox);

    epochsSpin_->setRange(1, 10000);
    epochsSpin_->setValue(180);
    batchSpin_->setRange(1, 4096);
    batchSpin_->setValue(32);

    lrSpin_->setDecimals(6);
    lrSpin_->setRange(1e-6, 1.0);
    lrSpin_->setSingleStep(0.001);
    lrSpin_->setValue(0.002);

    lambdaSpin_->setDecimals(4);
    lambdaSpin_->setRange(0.0, 100.0);
    lambdaSpin_->setValue(0.8);

    dataWeightSpin_->setDecimals(4);
    dataWeightSpin_->setRange(0.0, 100.0);
    dataWeightSpin_->setValue(1.0);

    physicsWeightSpin_->setDecimals(4);
    physicsWeightSpin_->setRange(0.0, 100.0);
    physicsWeightSpin_->setValue(0.2);

    evalCheck_->setChecked(true);

    cfgForm->addRow("Epochs", epochsSpin_);
    cfgForm->addRow("Batch size", batchSpin_);
    cfgForm->addRow("Learning rate", lrSpin_);
    cfgForm->addRow("Lambda (PINN)", lambdaSpin_);
    cfgForm->addRow("Data weight", dataWeightSpin_);
    cfgForm->addRow("Physics weight", physicsWeightSpin_);
    cfgForm->addRow(evalCheck_);

    // Button panel
    auto* btnBox = new QGroupBox("Actions", central);
    auto* btnGrid = new QGridLayout(btnBox);
    btnGrid->addWidget(runButton_, 0, 0);
    btnGrid->addWidget(runAllButton_, 0, 1);
    btnGrid->addWidget(runFFNButton_, 1, 0);
    btnGrid->addWidget(runFFNPINNButton_, 1, 1);
    btnGrid->addWidget(runLSTMButton_, 2, 0);
    btnGrid->addWidget(runLSTMPINNButton_, 2, 1);

    auto* topRow = new QHBoxLayout();
    topRow->addWidget(modeCombo_, 1);

    logText_->setReadOnly(true);
    logText_->setPlaceholderText("Run logs will appear here...");

    root->addWidget(title);
    root->addLayout(topRow);
    root->addWidget(cfgBox);
    root->addWidget(btnBox);
    root->addWidget(statusLabel_);
    root->addWidget(logText_, 1);

    setCentralWidget(central);

    connect(runButton_, &QPushButton::clicked, this, &HydroPINNWindow::runSelectedMode);
    connect(runAllButton_, &QPushButton::clicked, this, &HydroPINNWindow::runAllModes);
    connect(runFFNButton_, &QPushButton::clicked, this, [this]() { runMode("ffn"); });
    connect(runFFNPINNButton_, &QPushButton::clicked, this, [this]() { runMode("ffn_pinn"); });
    connect(runLSTMButton_, &QPushButton::clicked, this, [this]() { runMode("lstm"); });
    connect(runLSTMPINNButton_, &QPushButton::clicked, this, [this]() { runMode("lstm_pinn"); });
    connect(modeCombo_, &QComboBox::currentTextChanged, this, [this](const QString&) { updateStatus(); });

    updateStatus();
    appendLog("HydroPINN ready.");
}

HydroRunConfig HydroPINNWindow::currentConfig() const {
    HydroRunConfig cfg;
    cfg.epochs = epochsSpin_->value();
    cfg.batch_size = batchSpin_->value();
    cfg.learning_rate = lrSpin_->value();
    cfg.lambda_decay = lambdaSpin_->value();
    cfg.data_weight = dataWeightSpin_->value();
    cfg.physics_weight = physicsWeightSpin_->value();
    cfg.evaluate_metrics = evalCheck_->isChecked();
    return cfg;
}

void HydroPINNWindow::setRunningUiState(bool running) {
    runButton_->setEnabled(!running);
    runAllButton_->setEnabled(!running);
    runFFNButton_->setEnabled(!running);
    runFFNPINNButton_->setEnabled(!running);
    runLSTMButton_->setEnabled(!running);
    runLSTMPINNButton_->setEnabled(!running);
    runButton_->setText(running ? "Running..." : "Run Selected");
}

void HydroPINNWindow::appendLog(const QString& line) {
    const QString ts = QDateTime::currentDateTime().toString("hh:mm:ss");
    logText_->append(QString("[%1] %2").arg(ts, line));
}

void HydroPINNWindow::updateStatus() {
    statusLabel_->setText(QString("Ready to run mode: %1").arg(modeCombo_->currentText()));
}

void HydroPINNWindow::runSelectedMode() {
    runMode(modeCombo_->currentText());
}

void HydroPINNWindow::runAllModes() {
    const QStringList modes = {"ffn", "ffn_pinn", "lstm", "lstm_pinn"};
    for (const QString& m : modes) {
        runMode(m);
    }
}

void HydroPINNWindow::runMode(const QString& mode) {
    appendLog(QString("Starting mode: %1").arg(mode));
    setRunningUiState(true);
    statusLabel_->setText(QString("Running mode: %1 ...").arg(mode));
    appendLog("Dispatch started.");
    QCoreApplication::processEvents();

    QElapsedTimer timer;
    timer.start();

    const HydroRunConfig cfg = currentConfig();
    HydroRunResult result;
    QString errorDetails;

    try {
        if (mode == "ffn") {
            FFNWrapper runner;
            result = runner.train(cfg);
        } else if (mode == "ffn_pinn") {
            FFNPINNWrapper runner;
            result = runner.train(cfg);
        } else if (mode == "lstm") {
            LSTMWrapper runner;
            result = runner.train(cfg);
        } else if (mode == "lstm_pinn") {
            LSTMPINNWrapper runner;
            result = runner.train(cfg);
        } else {
            result.success = false;
            errorDetails = QString("Unknown mode selected: %1").arg(mode);
        }
    } catch (const std::exception& e) {
        result.success = false;
        errorDetails = QString("Exception: %1").arg(e.what());
    } catch (...) {
        result.success = false;
        errorDetails = "Unknown non-std exception during mode execution.";
    }

    const qint64 elapsedMs = timer.elapsed();
    if (result.success) {
        statusLabel_->setText(QString("Completed mode: %1 (%2 ms)").arg(mode).arg(elapsedMs));
        appendLog(QString("Mode '%1' finished successfully in %2 ms.").arg(mode).arg(elapsedMs));
        appendLog(QString("  final_loss=%1, mse=%2, msg=%3")
                      .arg(result.final_loss, 0, 'g', 8)
                      .arg(result.mse, 0, 'g', 8)
                      .arg(QString::fromStdString(result.message)));
    } else {
        statusLabel_->setText(QString("Mode failed: %1").arg(mode));
        appendLog(QString("Mode '%1' failed.").arg(mode));
        if (!errorDetails.isEmpty()) {
            appendLog(QString("Failure details: %1").arg(errorDetails));
        }
        QMessageBox::warning(this, "HydroPINN",
                             QString("Mode '%1' failed.%2")
                                 .arg(mode)
                                 .arg(errorDetails.isEmpty() ? "" : QString("\n\n%1").arg(errorDetails)));
    }

    setRunningUiState(false);
}
