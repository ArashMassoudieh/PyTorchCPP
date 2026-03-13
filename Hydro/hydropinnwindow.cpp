#include "hydropinnwindow.h"

#include "models/ffn_wrapper.h"
#include "models/ffn_pinn_wrapper.h"
#include "models/lstm_wrapper.h"
#include "models/lstm_pinn_wrapper.h"

#include <QComboBox>
#include <QCoreApplication>
#include <QDateTime>
#include <QElapsedTimer>
#include <QLabel>
#include <QMessageBox>
#include <QPushButton>
#include <QTextEdit>
#include <QVBoxLayout>
#include <QWidget>

#include <exception>

HydroPINNWindow::HydroPINNWindow(QWidget* parent)
    : QMainWindow(parent), statusLabel_(new QLabel(this)), modeCombo_(new QComboBox(this)),
      runButton_(new QPushButton("Run", this)), logText_(new QTextEdit(this)) {
    setWindowTitle("HydroPINN - Experiment Runner");
    resize(720, 380);

    auto* central = new QWidget(this);
    auto* layout = new QVBoxLayout(central);

    auto* title = new QLabel("HydroPINN Modes", central);
    title->setStyleSheet("font-size: 18px; font-weight: bold;");

    modeCombo_->addItems({"ffn", "ffn_pinn", "lstm", "lstm_pinn"});
    logText_->setReadOnly(true);
    logText_->setPlaceholderText("Run logs will appear here...");

    layout->addWidget(title);
    layout->addWidget(modeCombo_);
    layout->addWidget(runButton_);
    layout->addWidget(statusLabel_);
    layout->addWidget(logText_);

    setCentralWidget(central);

    connect(runButton_, &QPushButton::clicked, this, &HydroPINNWindow::runSelectedMode);
    connect(modeCombo_, &QComboBox::currentTextChanged, this, [this](const QString&) { updateStatus(); });

    updateStatus();
    appendLog("HydroPINN ready.");
}

void HydroPINNWindow::appendLog(const QString& line) {
    const QString ts = QDateTime::currentDateTime().toString("hh:mm:ss");
    logText_->append(QString("[%1] %2").arg(ts, line));
}

void HydroPINNWindow::updateStatus() {
    statusLabel_->setText(QString("Ready to run mode: %1").arg(modeCombo_->currentText()));
}

void HydroPINNWindow::runSelectedMode() {
    const QString mode = modeCombo_->currentText();
    appendLog(QString("Starting mode: %1").arg(mode));

    runButton_->setEnabled(false);
    runButton_->setText("Running...");
    statusLabel_->setText(QString("Running mode: %1 ...").arg(mode));
    appendLog("Dispatch started.");

    QCoreApplication::processEvents();

    QElapsedTimer timer;
    timer.start();

    bool ok = false;
    QString errorDetails;

    try {
        if (mode == "ffn") {
            FFNWrapper runner;
            ok = runner.train();
        } else if (mode == "ffn_pinn") {
            FFNPINNWrapper runner;
            ok = runner.train();
        } else if (mode == "lstm") {
            LSTMWrapper runner;
            ok = runner.train();
        } else if (mode == "lstm_pinn") {
            LSTMPINNWrapper runner;
            ok = runner.train();
        } else {
            errorDetails = QString("Unknown mode selected: %1").arg(mode);
            appendLog(errorDetails);
        }
    } catch (const std::exception& e) {
        ok = false;
        errorDetails = QString("Exception: %1").arg(e.what());
        appendLog(errorDetails);
    } catch (...) {
        ok = false;
        errorDetails = "Unknown non-std exception during mode execution.";
        appendLog(errorDetails);
    }

    const qint64 elapsedMs = timer.elapsed();
    if (ok) {
        statusLabel_->setText(QString("Completed mode: %1 (%2 ms)").arg(mode).arg(elapsedMs));
        appendLog(QString("Mode '%1' finished successfully in %2 ms.").arg(mode).arg(elapsedMs));
        QMessageBox::information(this, "HydroPINN",
                                 QString("Mode '%1' completed in %2 ms.").arg(mode).arg(elapsedMs));
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

    runButton_->setText("Run");
    runButton_->setEnabled(true);
}
