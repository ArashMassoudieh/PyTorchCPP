#include "hydropinnwindow.h"

#include <QAction>
#include <QApplication>
#include <QCoreApplication>
#include <QDialog>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QLabel>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QProcess>
#include <QPushButton>
#include <QSurfaceFormat>
#include <QTextCursor>
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

    return QFileDialog::getOpenFileName(
        parent,
        "Select HydroBatch executable",
        QDir::currentPath(),
        "HydroBatch executable (HydroBatch);;All files (*)");
}

void runConfigBatchFromGui(HydroPINNWindow& window, QAction* action)
{
    const QString batchPath = QFileDialog::getOpenFileName(
        &window,
        "Select Hydro config batch",
        QDir::currentPath() + "/Hydro/experiments",
        "Hydro batch files (*.batch);;All files (*)");
    if (batchPath.isEmpty()) return;

    const QString outputDirectory = QFileDialog::getExistingDirectory(
        &window,
        "Select batch output directory",
        QFileInfo(batchPath).absolutePath());
    if (outputDirectory.isEmpty()) return;

    const QString executable = locateHydroBatchExecutable(&window);
    if (executable.isEmpty()) {
        QMessageBox::information(
            &window,
            "HydroPINN Batch",
            "HydroBatch was not selected. Build it with HydroBatch.pro, then retry.");
        return;
    }

    auto* dialog = new QDialog(&window);
    dialog->setAttribute(Qt::WA_DeleteOnClose, true);
    dialog->setWindowTitle("HydroPINN Config Batch");
    dialog->resize(900, 520);

    auto* layout = new QVBoxLayout(dialog);
    auto* status = new QLabel(
        QString("Running batch:\n%1\n\nOutput:\n%2").arg(batchPath, outputDirectory), dialog);
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
                     static_cast<void (QProcess::*)(int, QProcess::ExitStatus)>(&QProcess::finished),
                     dialog,
                     [action, status, stopButton, closeButton, outputDirectory](int exitCode, QProcess::ExitStatus exitStatus) {
                         action->setEnabled(true);
                         stopButton->setEnabled(false);
                         closeButton->setEnabled(true);
                         const bool ok = exitStatus == QProcess::NormalExit && exitCode == 0;
                         status->setText(
                             ok
                                 ? QString("Batch completed successfully.\nSummary: %1/batch_summary.csv").arg(outputDirectory)
                                 : QString("Batch finished with errors (exit code %1). Review the log below.\nSummary/output: %2")
                                       .arg(exitCode)
                                       .arg(outputDirectory));
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
        if (process->state() != QProcess::NotRunning) {
            process->terminate();
        }
    });

    dialog->show();
    process->start(executable, {batchPath, outputDirectory});
}

void configureBatchGui(HydroPINNWindow& window)
{
    auto* batchToolBar = window.addToolBar("Batch");
    batchToolBar->setObjectName("HydroBatchToolBar");
    QAction* runBatchAction = batchToolBar->addAction("Run Config Batch...");
    runBatchAction->setToolTip(
        "Run an FFN/LSTM experiment .batch file sequentially with HydroBatch, "
        "stream progress in the GUI, and write per-run artifacts plus batch_summary.csv.");

    QMenu* batchMenu = window.menuBar()->addMenu("Batch");
    batchMenu->addAction(runBatchAction);

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
