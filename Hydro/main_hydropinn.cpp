#include "hydropinnwindow.h"

#include <QAction>
#include <QApplication>
#include <QCheckBox>
#include <QCoreApplication>
#include <QDialog>
#include <QDir>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QProcess>
#include <QPushButton>
#include <QRegularExpression>
#include <QSpinBox>
#include <QSurfaceFormat>
#include <QTextCursor>
#include <QTextEdit>
#include <QTimer>
#include <QToolBar>
#include <QVBoxLayout>
#include <torch/torch.h>

#include <exception>
#include <iostream>

namespace {
QPushButton* findButtonByText(HydroPINNWindow& window, const QString& text)
{
    const auto buttons = window.findChildren<QPushButton*>();
    for (QPushButton* button : buttons) {
        if (button->text() == text) return button;
    }
    return nullptr;
}

void configureContextSpecificPlotButtons(HydroPINNWindow& window)
{
    QPushButton* inputsOutput = findButtonByText(window, "Show Inputs + Output");
    QPushButton* cumulativePhysics = findButtonByText(window, "Cumulative Physics Residual");

    if (inputsOutput) {
        inputsOutput->setText("Inputs + Output");
        inputsOutput->setToolTip(
            "Plot the actual configured input columns together with the output. "
            "Synthetic uses generated channels; CSV uses configured model columns; "
            "GIStoOHQ Hydro packages plot precipitation, temperature, RH, wind, solar, PET, and observed runoff.");
        inputsOutput->setEnabled(true);
        QObject::disconnect(inputsOutput, nullptr, &window, nullptr);
        QObject::connect(inputsOutput, &QPushButton::clicked, &window,
                         [&window]() { window.showCurrentInputsOutputs(); });
    }

    if (cumulativePhysics) {
        cumulativePhysics->setText("Cumulative Physics Residual (PINN only)");
        cumulativePhysics->setToolTip(
            "Requires a successful physics-informed run with stored physics residuals. "
            "Plain FFN/LSTM runs do not produce this series. Current GIStoOHQ rainfall-runoff "
            "exports intentionally block PINN approaches until a separate rainfall-runoff "
            "physics profile is versioned.");
    }
}

void installR2LogEnhancer(HydroPINNWindow& window)
{
    QTextEdit* runLog = nullptr;
    const auto edits = window.findChildren<QTextEdit*>();
    for (QTextEdit* edit : edits) {
        if (edit->placeholderText().contains("Run logs", Qt::CaseInsensitive)) {
            runLog = edit;
            break;
        }
    }
    if (!runLog) return;

    QObject::connect(runLog, &QTextEdit::textChanged, runLog, [runLog]() {
        if (runLog->property("hydro_r2_rewrite").toBool()) return;
        QTextCursor cursor(runLog->document());
        cursor.movePosition(QTextCursor::End);
        cursor.select(QTextCursor::BlockUnderCursor);
        QString line = cursor.selectedText();
        if (!line.contains("final_loss=") || !line.contains("nse=") || line.contains("r2=")) return;

        const QRegularExpression rx(QStringLiteral("nse=([^,\\s]+)"));
        const auto match = rx.match(line);
        if (!match.hasMatch()) return;
        const QString nseValue = match.captured(1);
        line.replace(match.capturedStart(0), match.capturedLength(0),
                     QString("r2=%1, nse=%1").arg(nseValue));

        runLog->setProperty("hydro_r2_rewrite", true);
        cursor.insertText(line);
        runLog->setProperty("hydro_r2_rewrite", false);
    });
}

QString locateHydroBatchExecutable(QWidget* parent)
{
    const QStringList candidates = {
        QCoreApplication::applicationDirPath() + "/HydroBatch",
        QDir::currentPath() + "/HydroBatch",
        QDir::currentPath() + "/build-hydrobatch/HydroBatch",
        QDir::currentPath() + "/../build-hydrobatch/HydroBatch"
    };
    for (const QString& candidate : candidates) {
        const QFileInfo info(candidate);
        if (info.exists() && info.isFile() && info.isExecutable()) return info.absoluteFilePath();
    }
    return QFileDialog::getOpenFileName(parent, "Select HydroBatch executable", QDir::currentPath(),
                                        "HydroBatch executable (HydroBatch);;All files (*)");
}

QString locateRepositoryRoot()
{
    QStringList starts = {QDir::currentPath(), QCoreApplication::applicationDirPath()};
    for (const QString& start : starts) {
        QDir dir(start);
        for (int depth = 0; depth < 8; ++depth) {
            if (QFileInfo::exists(dir.filePath("HydroPINN.pro")) || QFileInfo::exists(dir.filePath("HydroBatch.pro"))) {
                return dir.absolutePath();
            }
            if (!dir.cdUp()) break;
        }
    }
    return QString();
}

void runConfigBatchFromGui(HydroPINNWindow& window, QAction* action)
{
    const QString batchPath = QFileDialog::getOpenFileName(
        &window, "Select Hydro config batch", QDir::currentPath() + "/Hydro/experiments",
        "Hydro batch files (*.batch);;All files (*)");
    if (batchPath.isEmpty()) return;

    const QString outputDirectory = QFileDialog::getExistingDirectory(
        &window, "Select batch output directory", QFileInfo(batchPath).absolutePath());
    if (outputDirectory.isEmpty()) return;

    const QString executable = locateHydroBatchExecutable(&window);
    if (executable.isEmpty()) {
        QMessageBox::information(&window, "HydroPINN Batch",
                                 "HydroBatch was not selected. Build it with HydroBatch.pro, then retry.");
        return;
    }

    auto* dialog = new QDialog(&window);
    dialog->setAttribute(Qt::WA_DeleteOnClose, true);
    dialog->setWindowTitle("HydroPINN Config Batch");
    dialog->resize(900, 520);

    auto* layout = new QVBoxLayout(dialog);
    auto* status = new QLabel(QString("Running batch:\n%1\n\nOutput:\n%2").arg(batchPath, outputDirectory), dialog);
    status->setWordWrap(true);
    layout->addWidget(status);

    auto* output = new QPlainTextEdit(dialog);
    output->setReadOnly(true);
    output->setMaximumBlockCount(6000);
    output->appendPlainText(QString("HydroBatch executable: %1").arg(executable));
    layout->addWidget(output, 1);

    auto* buttonRow = new QHBoxLayout();
    auto* stopButton = new QPushButton("Stop Batch", dialog);
    auto* closeButton = new QPushButton("Close", dialog);
    closeButton->setEnabled(false);
    buttonRow->addStretch(1);
    buttonRow->addWidget(stopButton);
    buttonRow->addWidget(closeButton);
    layout->addLayout(buttonRow);

    auto* process = new QProcess(dialog);
    process->setProcessChannelMode(QProcess::MergedChannels);
    process->setWorkingDirectory(QFileInfo(executable).absolutePath());

    action->setEnabled(false);
    QObject::connect(process, &QProcess::readyReadStandardOutput, dialog, [process, output]() {
        const QString text = QString::fromLocal8Bit(process->readAllStandardOutput());
        if (!text.isEmpty()) {
            output->moveCursor(QTextCursor::End);
            output->insertPlainText(text);
            output->moveCursor(QTextCursor::End);
        }
    });
    QObject::connect(stopButton, &QPushButton::clicked, dialog, [process, status, stopButton]() {
        if (process->state() == QProcess::NotRunning) return;
        status->setText("Stopping batch after terminating the active HydroBatch process...");
        stopButton->setEnabled(false);
        process->terminate();
        QTimer::singleShot(5000, process, [process]() {
            if (process->state() != QProcess::NotRunning) process->kill();
        });
    });
    QObject::connect(closeButton, &QPushButton::clicked, dialog, &QDialog::close);
    QObject::connect(process,
                     static_cast<void (QProcess::*)(int, QProcess::ExitStatus)>(&QProcess::finished), dialog,
                     [action, status, stopButton, closeButton, outputDirectory](int exitCode, QProcess::ExitStatus exitStatus) {
                         action->setEnabled(true);
                         stopButton->setEnabled(false);
                         closeButton->setEnabled(true);
                         const bool ok = exitStatus == QProcess::NormalExit && exitCode == 0;
                         status->setText(ok
                             ? QString("Batch completed successfully.\nSummary: %1/batch_summary.csv").arg(outputDirectory)
                             : QString("Batch finished with errors (exit code %1). Review the log below.\nSummary/output: %2")
                                   .arg(exitCode).arg(outputDirectory));
                     });
    QObject::connect(process, &QProcess::errorOccurred, dialog,
                     [action, status, stopButton, closeButton](QProcess::ProcessError error) {
                         if (error == QProcess::FailedToStart) {
                             action->setEnabled(true);
                             stopButton->setEnabled(false);
                             closeButton->setEnabled(true);
                             status->setText("HydroBatch failed to start. Check the executable path and build.");
                         }
                     });
    QObject::connect(dialog, &QObject::destroyed, &window, [action, process]() {
        action->setEnabled(true);
        if (process->state() != QProcess::NotRunning) process->terminate();
    });

    dialog->show();
    process->start(executable, {batchPath, outputDirectory});
}

void createTuningSweepFromGui(HydroPINNWindow& window)
{
    const QString repoRoot = locateRepositoryRoot();
    if (repoRoot.isEmpty()) {
        QMessageBox::critical(&window, "Tuning Sweep", "Unable to locate the PyTorchCPP repository root.");
        return;
    }
    const QString experimentDir = repoRoot + "/Hydro/experiments/gistohq_sligo";
    const QString generator = experimentDir + "/generate_hyperparameter_sweep.py";
    if (!QFileInfo::exists(generator)) {
        QMessageBox::critical(&window, "Tuning Sweep", "Sweep generator not found:\n" + generator);
        return;
    }

    QDialog dialog(&window);
    dialog.setWindowTitle("Sligo Creek Tuning Sweep");
    dialog.resize(650, 520);
    auto* root = new QVBoxLayout(&dialog);

    auto* intro = new QLabel(
        "Select the supervised hyperparameter combinations to generate. "
        "FFN activation is configurable; LSTM uses its native internal nonlinearities, so activation switching is intentionally disabled for LSTM.",
        &dialog);
    intro->setWordWrap(true);
    root->addWidget(intro);

    auto* familyBox = new QGroupBox("Model families", &dialog);
    auto* familyLayout = new QHBoxLayout(familyBox);
    auto* ffnCheck = new QCheckBox("FFN", familyBox);
    auto* lstmCheck = new QCheckBox("LSTM", familyBox);
    ffnCheck->setChecked(true);
    lstmCheck->setChecked(true);
    familyLayout->addWidget(ffnCheck);
    familyLayout->addWidget(lstmCheck);
    familyLayout->addStretch(1);
    root->addWidget(familyBox);

    auto* selectionBox = new QGroupBox("Search space", &dialog);
    auto* form = new QFormLayout(selectionBox);

    auto* ffnArchitectures = new QLineEdit("16;24;32;48;16,16;24,24;32,16;32,32;48,24", selectionBox);
    ffnArchitectures->setToolTip("Semicolon-separated hidden-layer architectures. Commas separate layers within one architecture.");
    form->addRow("FFN hidden layers", ffnArchitectures);

    auto* activationRow = new QWidget(selectionBox);
    auto* activationLayout = new QHBoxLayout(activationRow);
    activationLayout->setContentsMargins(0, 0, 0, 0);
    auto* tanhCheck = new QCheckBox("tanh", activationRow);
    auto* reluCheck = new QCheckBox("ReLU", activationRow);
    auto* sigmoidCheck = new QCheckBox("sigmoid", activationRow);
    tanhCheck->setChecked(true);
    reluCheck->setChecked(true);
    sigmoidCheck->setChecked(false);
    activationLayout->addWidget(tanhCheck);
    activationLayout->addWidget(reluCheck);
    activationLayout->addWidget(sigmoidCheck);
    activationLayout->addStretch(1);
    form->addRow("FFN activations", activationRow);

    auto* lstmArchitectures = new QLineEdit("16;24;32;48;24,24;32,32", selectionBox);
    form->addRow("LSTM hidden layers", lstmArchitectures);
    auto* lstmSequences = new QLineEdit("12,24", selectionBox);
    form->addRow("LSTM sequence lengths (h)", lstmSequences);

    auto* epochsSpin = new QSpinBox(selectionBox);
    epochsSpin->setRange(1, 100000);
    epochsSpin->setValue(150);
    form->addRow("Epochs", epochsSpin);

    auto* lrSpin = new QDoubleSpinBox(selectionBox);
    lrSpin->setDecimals(6);
    lrSpin->setRange(0.000001, 1.0);
    lrSpin->setSingleStep(0.001);
    lrSpin->setValue(0.003);
    form->addRow("Learning rate", lrSpin);

    auto* batchSpin = new QSpinBox(selectionBox);
    batchSpin->setRange(1, 100000);
    batchSpin->setValue(32);
    form->addRow("Batch size", batchSpin);

    auto* seedSpin = new QSpinBox(selectionBox);
    seedSpin->setRange(0, 2147483647);
    seedSpin->setValue(42);
    form->addRow("Random seed", seedSpin);
    root->addWidget(selectionBox);

    auto updateEnabled = [=]() {
        ffnArchitectures->setEnabled(ffnCheck->isChecked());
        activationRow->setEnabled(ffnCheck->isChecked());
        lstmArchitectures->setEnabled(lstmCheck->isChecked());
        lstmSequences->setEnabled(lstmCheck->isChecked());
    };
    QObject::connect(ffnCheck, &QCheckBox::toggled, &dialog, updateEnabled);
    QObject::connect(lstmCheck, &QCheckBox::toggled, &dialog, updateEnabled);
    updateEnabled();

    auto* note = new QLabel(
        "Tip: to run only the missing sigmoid FFN cases, turn LSTM off, turn tanh/ReLU off, and turn sigmoid on.", &dialog);
    note->setWordWrap(true);
    root->addWidget(note);

    auto* buttons = new QHBoxLayout();
    auto* cancelButton = new QPushButton("Cancel", &dialog);
    auto* generateButton = new QPushButton("Generate Sweep", &dialog);
    buttons->addStretch(1);
    buttons->addWidget(cancelButton);
    buttons->addWidget(generateButton);
    root->addLayout(buttons);

    QObject::connect(cancelButton, &QPushButton::clicked, &dialog, &QDialog::reject);
    QObject::connect(generateButton, &QPushButton::clicked, &dialog, [&]() {
        if (!ffnCheck->isChecked() && !lstmCheck->isChecked()) {
            QMessageBox::warning(&dialog, "Tuning Sweep", "Select at least one model family.");
            return;
        }

        QStringList activations;
        if (tanhCheck->isChecked()) activations << "tanh";
        if (reluCheck->isChecked()) activations << "relu";
        if (sigmoidCheck->isChecked()) activations << "sigmoid";
        if (ffnCheck->isChecked() && activations.isEmpty()) {
            QMessageBox::warning(&dialog, "Tuning Sweep", "Select at least one FFN activation.");
            return;
        }

        QStringList args;
        args << generator;
        if (ffnCheck->isChecked() && !lstmCheck->isChecked()) args << "--ffn-only";
        if (lstmCheck->isChecked() && !ffnCheck->isChecked()) args << "--lstm-only";
        if (ffnCheck->isChecked()) {
            args << "--ffn-architectures" << ffnArchitectures->text().trimmed();
            args << "--activations" << activations.join(',');
        }
        if (lstmCheck->isChecked()) {
            args << "--lstm-architectures" << lstmArchitectures->text().trimmed();
            args << "--lstm-sequences" << lstmSequences->text().trimmed();
        }
        args << "--epochs" << QString::number(epochsSpin->value());
        args << "--learning-rate" << QString::number(lrSpin->value(), 'g', 12);
        args << "--batch-size" << QString::number(batchSpin->value());
        args << "--seed" << QString::number(seedSpin->value());

        QProcess process;
        process.setWorkingDirectory(experimentDir);
        process.setProcessChannelMode(QProcess::MergedChannels);
        process.start("python3", args);
        if (!process.waitForStarted(5000)) {
            QMessageBox::critical(&dialog, "Tuning Sweep", "Unable to start python3 sweep generator.");
            return;
        }
        process.waitForFinished(-1);
        const QString output = QString::fromLocal8Bit(process.readAll());
        if (process.exitStatus() != QProcess::NormalExit || process.exitCode() != 0) {
            QMessageBox::critical(&dialog, "Tuning Sweep", "Sweep generation failed:\n\n" + output);
            return;
        }

        dialog.accept();
        QMessageBox::information(
            &window, "Tuning Sweep Generated",
            output + "\nUse Batch > Run Config Batch... and select:\n" + experimentDir + "/hyperparameter_stage1.batch");
    });

    dialog.exec();
}

void configureBatchGui(HydroPINNWindow& window)
{
    auto* batchToolBar = window.addToolBar("Batch");
    batchToolBar->setObjectName("HydroBatchToolBar");

    QAction* tuningAction = batchToolBar->addAction("Tuning Sweep...");
    tuningAction->setToolTip(
        "Choose FFN/LSTM families, architectures, FFN activation (including sigmoid), "
        "LSTM sequence lengths, learning rate, batch size, epochs, and seed, then generate the batch configuration.");

    QAction* runBatchAction = batchToolBar->addAction("Run Config Batch...");
    runBatchAction->setToolTip(
        "Run an FFN/LSTM experiment .batch file sequentially with HydroBatch, "
        "stream progress in the GUI, and write per-run artifacts plus batch_summary.csv.");

    QMenu* batchMenu = window.menuBar()->addMenu("Batch");
    batchMenu->addAction(tuningAction);
    batchMenu->addAction(runBatchAction);

    QObject::connect(tuningAction, &QAction::triggered, &window,
                     [&window]() { createTuningSweepFromGui(window); });
    QObject::connect(runBatchAction, &QAction::triggered, &window,
                     [&window, runBatchAction]() { runConfigBatchFromGui(window, runBatchAction); });
}
} // namespace

int main(int argc, char *argv[])
{
#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
    QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
    QCoreApplication::setAttribute(Qt::AA_UseHighDpiPixmaps);
#endif
    try {
        torch::set_num_threads(1);
        torch::set_num_interop_threads(1);
        QApplication app(argc, argv);
        QCoreApplication::setApplicationName("HydroPINN");
        QCoreApplication::setApplicationVersion("0.1");
        QCoreApplication::setOrganizationName("EnviroInformatics LLC");
        QApplication::setApplicationDisplayName("HydroPINN - Physics-Informed Hydrology");

        HydroPINNWindow window;
        configureContextSpecificPlotButtons(window);
        installR2LogEnhancer(window);
        configureBatchGui(window);
        window.show();
        return app.exec();
    }
    catch (const c10::Error &e) {
        std::cerr << "LibTorch error:\n" << e.what() << std::endl;
        QMessageBox::critical(nullptr, "HydroPINN - LibTorch error", QString::fromStdString(e.what()));
        return EXIT_FAILURE;
    }
    catch (const std::exception &e) {
        std::cerr << "Application error:\n" << e.what() << std::endl;
        QMessageBox::critical(nullptr, "HydroPINN - Error", QString::fromUtf8(e.what()));
        return EXIT_FAILURE;
    }
    catch (...) {
        std::cerr << "Unknown application error." << std::endl;
        QMessageBox::critical(nullptr, "HydroPINN - Error", "Unknown application error.");
        return EXIT_FAILURE;
    }
}
