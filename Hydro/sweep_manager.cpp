#include <QAction>
#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QCoreApplication>
#include <QDateTime>
#include <QDialog>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMainWindow>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QProcess>
#include <QPushButton>
#include <QSpinBox>
#include <QTabWidget>
#include <QTextEdit>
#include <QTimer>
#include <QVBoxLayout>

#include <algorithm>

namespace {

QString findRepoRoot()
{
    const QStringList starts = {QDir::currentPath(), QCoreApplication::applicationDirPath()};
    for (const QString& start : starts) {
        QDir dir(start);
        for (int depth = 0; depth < 10; ++depth) {
            if (QFileInfo::exists(dir.filePath("HydroPINN.pro")) ||
                QFileInfo::exists(dir.filePath("HydroBatch.pro"))) return dir.absolutePath();
            if (!dir.cdUp()) break;
        }
    }
    return {};
}

QMenu* findBatchMenu(QMainWindow* window)
{
    if (!window || !window->menuBar()) return nullptr;
    for (QAction* action : window->menuBar()->actions()) {
        if (action && action->text().remove('&') == "Batch") return action->menu();
    }
    return nullptr;
}

int commaCount(const QString& text)
{
    int n = 0;
    for (const QString& part : text.split(',', Qt::SkipEmptyParts)) if (!part.trimmed().isEmpty()) ++n;
    return std::max(1, n);
}

int semiCount(const QString& text)
{
    int n = 0;
    for (const QString& part : text.split(';', Qt::SkipEmptyParts)) if (!part.trimmed().isEmpty()) ++n;
    return std::max(1, n);
}

QString hydroBatchExecutable(const QString& root)
{
    const QStringList candidates = {
        root + "/build-hydrobatch/HydroBatch",
        QCoreApplication::applicationDirPath() + "/HydroBatch",
        QDir::currentPath() + "/HydroBatch"
    };
    for (const QString& candidate : candidates) {
        QFileInfo info(candidate);
        if (info.exists() && info.isFile() && info.isExecutable()) return info.absoluteFilePath();
    }
    return {};
}

bool generateSweep(QWidget* parent, const QStringList& args, QString* outputText = nullptr)
{
    const QString root = findRepoRoot();
    if (root.isEmpty()) {
        QMessageBox::critical(parent, "Sweep Manager", "Unable to locate the PyTorchCPP repository root.");
        return false;
    }
    const QString dir = root + "/Hydro/experiments/gistohq_sligo";
    const QString generator = dir + "/generate_unified_sweep.py";
    if (!QFileInfo::exists(generator)) {
        QMessageBox::critical(parent, "Sweep Manager", "Unified sweep generator not found:\n" + generator);
        return false;
    }

    QProcess process;
    process.setWorkingDirectory(dir);
    process.setProcessChannelMode(QProcess::MergedChannels);
    QStringList processArgs{generator};
    processArgs << args;
    process.start("python3", processArgs);
    if (!process.waitForStarted(5000)) {
        QMessageBox::critical(parent, "Sweep Manager", "Unable to start python3.");
        return false;
    }
    process.waitForFinished(-1);
    const QString output = QString::fromLocal8Bit(process.readAll());
    if (outputText) *outputText = output;
    if (process.exitStatus() != QProcess::NormalExit || process.exitCode() != 0) {
        QMessageBox::critical(parent, "Sweep Manager", "Sweep generation failed:\n\n" + output);
        return false;
    }
    return true;
}

void launchBatch(QMainWindow* window, const QString& batchPath)
{
    const QString root = findRepoRoot();
    const QString executable = hydroBatchExecutable(root);
    if (executable.isEmpty()) {
        QMessageBox::critical(window, "Sweep Manager",
            "HydroBatch executable was not found. Build it first with qmake ../HydroBatch.pro CONFIG+=PowerEdge && make -j4.");
        return;
    }

    const QString defaultRoot = root + "/Hydro/experiments/gistohq_sligo/batch_outputs";
    QDir().mkpath(defaultRoot);
    const QString selected = QFileDialog::getExistingDirectory(window, "Choose Sweep Output Parent", defaultRoot);
    if (selected.isEmpty()) return;
    const QString stamp = QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss");
    const QString outputDir = selected + "/unified_sweep_" + stamp;
    QDir().mkpath(outputDir);

    auto* logDialog = new QDialog(window);
    logDialog->setAttribute(Qt::WA_DeleteOnClose);
    logDialog->setWindowTitle("Unified Sweep - Running");
    logDialog->resize(900, 650);
    auto* layout = new QVBoxLayout(logDialog);
    auto* status = new QLabel("Starting HydroBatch...", logDialog);
    auto* log = new QTextEdit(logDialog);
    log->setReadOnly(true);
    auto* buttons = new QHBoxLayout();
    auto* stop = new QPushButton("Stop", logDialog);
    auto* close = new QPushButton("Close", logDialog);
    close->setEnabled(false);
    buttons->addStretch(1); buttons->addWidget(stop); buttons->addWidget(close);
    layout->addWidget(status);
    layout->addWidget(log, 1);
    layout->addLayout(buttons);

    auto* process = new QProcess(logDialog);
    process->setWorkingDirectory(root);
    process->setProcessChannelMode(QProcess::MergedChannels);
    QObject::connect(process, &QProcess::readyReadStandardOutput, logDialog, [process, log]() {
        log->moveCursor(QTextCursor::End);
        log->insertPlainText(QString::fromLocal8Bit(process->readAllStandardOutput()));
        log->moveCursor(QTextCursor::End);
    });
    QObject::connect(stop, &QPushButton::clicked, logDialog, [process, status, stop]() {
        if (process->state() == QProcess::NotRunning) return;
        status->setText("Stopping...");
        stop->setEnabled(false);
        process->terminate();
        QTimer::singleShot(5000, process, [process]() {
            if (process->state() != QProcess::NotRunning) process->kill();
        });
    });
    QObject::connect(close, &QPushButton::clicked, logDialog, &QDialog::close);
    QObject::connect(process, qOverload<int, QProcess::ExitStatus>(&QProcess::finished), logDialog,
                     [status, stop, close, outputDir](int code, QProcess::ExitStatus exitStatus) {
        stop->setEnabled(false);
        close->setEnabled(true);
        if (exitStatus == QProcess::NormalExit && code == 0) {
            status->setText("Sweep completed successfully. Summary: " + outputDir + "/batch_summary.csv");
        } else {
            status->setText("Sweep finished with failures. Review the log and batch_summary.csv in " + outputDir);
        }
    });

    log->append("HydroBatch executable: " + executable);
    log->append("Batch: " + batchPath);
    log->append("Output: " + outputDir + "\n");
    process->start(executable, {batchPath, outputDir});
    if (!process->waitForStarted(5000)) {
        status->setText("Unable to start HydroBatch.");
        close->setEnabled(true);
        stop->setEnabled(false);
    }
    logDialog->show();
}

void showSweepManager(QMainWindow* window)
{
    QDialog dialog(window);
    dialog.setWindowTitle("HydroPINN Sweep Manager - All Five Methods");
    dialog.resize(820, 720);
    auto* root = new QVBoxLayout(&dialog);

    auto* intro = new QLabel(
        "Configure FFN, FFN + PINN, LSTM, LSTM + PINN, and standalone PINN from one place. "
        "Only parameters applicable to each method are multiplied into its run count.", &dialog);
    intro->setWordWrap(true);
    root->addWidget(intro);

    auto* presetRow = new QHBoxLayout();
    auto* preset = new QComboBox(&dialog);
    preset->addItems({"Five-method baseline", "Supervised architecture/memory", "Physics Stage 1", "Custom"});
    auto* applyPreset = new QPushButton("Apply Preset", &dialog);
    presetRow->addWidget(new QLabel("Preset", &dialog));
    presetRow->addWidget(preset, 1);
    presetRow->addWidget(applyPreset);
    root->addLayout(presetRow);

    auto* tabs = new QTabWidget(&dialog);

    auto* methodsTab = new QWidget(tabs);
    auto* methodsLayout = new QVBoxLayout(methodsTab);
    auto* ffn = new QCheckBox("FFN", methodsTab);
    auto* ffnPinn = new QCheckBox("FFN + PINN", methodsTab);
    auto* lstm = new QCheckBox("LSTM", methodsTab);
    auto* lstmPinn = new QCheckBox("LSTM + PINN", methodsTab);
    auto* pinn = new QCheckBox("PINN (physics-only)", methodsTab);
    for (QCheckBox* cb : {ffn, ffnPinn, lstm, lstmPinn, pinn}) { cb->setChecked(true); methodsLayout->addWidget(cb); }
    methodsLayout->addStretch(1);
    tabs->addTab(methodsTab, "Methods");

    auto* archTab = new QWidget(tabs);
    auto* archForm = new QFormLayout(archTab);
    auto* ffnArch = new QLineEdit("16,16", archTab);
    auto* ffnActs = new QLineEdit("relu", archTab);
    auto* ffnLags = new QLineEdit("1,2,3,4,5,6", archTab);
    auto* lstmArch = new QLineEdit("32", archTab);
    auto* lstmSeq = new QLineEdit("12", archTab);
    auto* pinnArch = new QLineEdit("24,24", archTab);
    archForm->addRow("FFN architectures (; separated)", ffnArch);
    archForm->addRow("FFN activations (, separated)", ffnActs);
    archForm->addRow("FFN lag specifications (; separated)", ffnLags);
    archForm->addRow("LSTM architectures (; separated)", lstmArch);
    archForm->addRow("LSTM sequences (, separated)", lstmSeq);
    archForm->addRow("PINN architectures (; separated)", pinnArch);
    tabs->addTab(archTab, "Architecture / Memory");

    auto* trainTab = new QWidget(tabs);
    auto* trainForm = new QFormLayout(trainTab);
    auto* lrs = new QLineEdit("0.003", trainTab);
    auto* batches = new QLineEdit("32", trainTab);
    auto* seeds = new QLineEdit("42", trainTab);
    auto* epochs = new QSpinBox(trainTab);
    epochs->setRange(1, 100000); epochs->setValue(150);
    trainForm->addRow("Learning rates", lrs);
    trainForm->addRow("Batch sizes", batches);
    trainForm->addRow("Random seeds", seeds);
    trainForm->addRow("Epochs", epochs);
    tabs->addTab(trainTab, "Training");

    auto* physicsTab = new QWidget(tabs);
    auto* physicsForm = new QFormLayout(physicsTab);
    auto* physicsWeights = new QLineEdit("0.05", physicsTab);
    auto* recession = new QLineEdit("0.08", physicsTab);
    auto* dataWeight = new QLineEdit("1.0", physicsTab);
    physicsForm->addRow("Physics weights (FFN+PINN/LSTM+PINN)", physicsWeights);
    physicsForm->addRow("Latent recession k [1/h]", recession);
    physicsForm->addRow("Data weight (hybrid PINNs)", dataWeight);
    auto* note = new QLabel(
        "Standalone PINN automatically uses data_weight=0 and physics_weight=1. "
        "Its run count changes with k, architecture, optimizer settings, and seed—not with hybrid physics weight.", physicsTab);
    note->setWordWrap(true);
    physicsForm->addRow(note);
    tabs->addTab(physicsTab, "Physics");

    root->addWidget(tabs, 1);

    auto* countLabel = new QLabel(&dialog);
    countLabel->setStyleSheet("font-weight: bold; padding: 6px;");
    root->addWidget(countLabel);

    auto selectedMethods = [&]() {
        QStringList methods;
        if (ffn->isChecked()) methods << "ffn";
        if (ffnPinn->isChecked()) methods << "ffn_pinn";
        if (lstm->isChecked()) methods << "lstm";
        if (lstmPinn->isChecked()) methods << "lstm_pinn";
        if (pinn->isChecked()) methods << "pinn";
        return methods;
    };

    auto updateCount = [&]() {
        const qint64 optimizerGrid = static_cast<qint64>(commaCount(lrs->text())) * commaCount(batches->text()) * commaCount(seeds->text());
        const qint64 ffnA = semiCount(ffnArch->text());
        const qint64 acts = commaCount(ffnActs->text());
        const qint64 lagN = semiCount(ffnLags->text());
        const qint64 lstmA = semiCount(lstmArch->text());
        const qint64 seqN = commaCount(lstmSeq->text());
        const qint64 pinnA = semiCount(pinnArch->text());
        const qint64 wN = commaCount(physicsWeights->text());
        const qint64 kN = commaCount(recession->text());
        qint64 total = 0;
        QStringList parts;
        if (ffn->isChecked()) { const qint64 n = ffnA * acts * lagN * optimizerGrid; total += n; parts << "FFN " + QString::number(n); }
        if (ffnPinn->isChecked()) { const qint64 n = ffnA * acts * wN * kN * optimizerGrid; total += n; parts << "FFN+PINN " + QString::number(n); }
        if (lstm->isChecked()) { const qint64 n = lstmA * seqN * optimizerGrid; total += n; parts << "LSTM " + QString::number(n); }
        if (lstmPinn->isChecked()) { const qint64 n = lstmA * seqN * wN * kN * optimizerGrid; total += n; parts << "LSTM+PINN " + QString::number(n); }
        if (pinn->isChecked()) { const qint64 n = pinnA * kN * optimizerGrid; total += n; parts << "PINN " + QString::number(n); }
        countLabel->setText("Valid experiments: " + QString::number(total) + "   [" + parts.join(" | ") + "]");
        countLabel->setStyleSheet(total > 500 ? "font-weight:bold; padding:6px; color:#b00020;" : "font-weight:bold; padding:6px;");
    };

    const QList<QLineEdit*> edits = {ffnArch, ffnActs, ffnLags, lstmArch, lstmSeq, pinnArch,
                                     lrs, batches, seeds, physicsWeights, recession, dataWeight};
    for (QLineEdit* edit : edits) QObject::connect(edit, &QLineEdit::textChanged, &dialog, [&](const QString&){ updateCount(); });
    for (QCheckBox* cb : {ffn, ffnPinn, lstm, lstmPinn, pinn}) QObject::connect(cb, &QCheckBox::toggled, &dialog, [&](bool){ updateCount(); });

    QObject::connect(applyPreset, &QPushButton::clicked, &dialog, [&]() {
        const QString p = preset->currentText();
        if (p == "Five-method baseline") {
            for (QCheckBox* cb : {ffn, ffnPinn, lstm, lstmPinn, pinn}) cb->setChecked(true);
            ffnArch->setText("16,16"); ffnActs->setText("relu"); ffnLags->setText("1,2,3,4,5,6");
            lstmArch->setText("32"); lstmSeq->setText("12"); pinnArch->setText("24,24");
            lrs->setText("0.003"); batches->setText("32"); seeds->setText("42");
            physicsWeights->setText("0.05"); recession->setText("0.08"); dataWeight->setText("1.0");
        } else if (p == "Supervised architecture/memory") {
            ffn->setChecked(true); lstm->setChecked(true); ffnPinn->setChecked(false); lstmPinn->setChecked(false); pinn->setChecked(false);
            ffnArch->setText("16;24;32;48;16,16;24,24;32,16;32,32;48,24");
            ffnActs->setText("tanh,relu,sigmoid"); ffnLags->setText("1,2,3,4,5,6");
            lstmArch->setText("16;24;32;48;24,24;32,32"); lstmSeq->setText("12,24");
            lrs->setText("0.003"); batches->setText("32"); seeds->setText("42");
        } else if (p == "Physics Stage 1") {
            ffn->setChecked(false); lstm->setChecked(false); ffnPinn->setChecked(true); lstmPinn->setChecked(true); pinn->setChecked(true);
            ffnArch->setText("16,16"); ffnActs->setText("relu"); lstmArch->setText("32"); lstmSeq->setText("12"); pinnArch->setText("24,24");
            lrs->setText("0.003"); batches->setText("32"); seeds->setText("42");
            physicsWeights->setText("0.001,0.005,0.01,0.025,0.05,0.1"); recession->setText("0.01,0.02,0.04,0.08,0.16"); dataWeight->setText("1.0");
        }
        updateCount();
    });

    auto buildArgs = [&]() {
        return QStringList{
            "--methods", selectedMethods().join(','),
            "--ffn-architectures", ffnArch->text().trimmed(),
            "--ffn-activations", ffnActs->text().trimmed(),
            "--ffn-lags", ffnLags->text().trimmed(),
            "--lstm-architectures", lstmArch->text().trimmed(),
            "--lstm-sequences", lstmSeq->text().trimmed(),
            "--pinn-architectures", pinnArch->text().trimmed(),
            "--learning-rates", lrs->text().trimmed(),
            "--batch-sizes", batches->text().trimmed(),
            "--seeds", seeds->text().trimmed(),
            "--physics-weights", physicsWeights->text().trimmed(),
            "--recession-k", recession->text().trimmed(),
            "--data-weight", dataWeight->text().trimmed(),
            "--epochs", QString::number(epochs->value())
        };
    };

    auto* buttons = new QHBoxLayout();
    auto* cancel = new QPushButton("Close", &dialog);
    auto* generate = new QPushButton("Generate Only", &dialog);
    auto* generateRun = new QPushButton("Generate && Run", &dialog);
    buttons->addStretch(1); buttons->addWidget(cancel); buttons->addWidget(generate); buttons->addWidget(generateRun);
    root->addLayout(buttons);
    QObject::connect(cancel, &QPushButton::clicked, &dialog, &QDialog::reject);
    QObject::connect(generate, &QPushButton::clicked, &dialog, [&]() {
        if (selectedMethods().isEmpty()) { QMessageBox::warning(&dialog, "Sweep Manager", "Select at least one method."); return; }
        QString output;
        if (generateSweep(&dialog, buildArgs(), &output)) {
            QMessageBox::information(&dialog, "Sweep Generated", output + "\nUse Generate & Run or Batch > Run Config Batch... to execute it.");
        }
    });
    QObject::connect(generateRun, &QPushButton::clicked, &dialog, [&]() {
        if (selectedMethods().isEmpty()) { QMessageBox::warning(&dialog, "Sweep Manager", "Select at least one method."); return; }
        QString output;
        if (!generateSweep(&dialog, buildArgs(), &output)) return;
        const QString rootPath = findRepoRoot();
        const QString batchPath = rootPath + "/Hydro/experiments/gistohq_sligo/unified_sweep.batch";
        launchBatch(window, batchPath);
        dialog.accept();
    });

    applyPreset->click();
    updateCount();
    dialog.exec();
}

void installSweepManager()
{
    QMainWindow* window = nullptr;
    for (QWidget* widget : QApplication::topLevelWidgets()) {
        if ((window = qobject_cast<QMainWindow*>(widget))) break;
    }
    if (!window) { QTimer::singleShot(100, [](){ installSweepManager(); }); return; }
    QMenu* batch = findBatchMenu(window);
    if (!batch) { QTimer::singleShot(100, [](){ installSweepManager(); }); return; }
    if (batch->findChild<QAction*>("HydroUnifiedSweepManagerAction")) return;

    QAction* action = new QAction("Sweep Manager...", batch);
    action->setObjectName("HydroUnifiedSweepManagerAction");
    action->setToolTip("Configure, generate, and run method-aware sweeps for all five HydroPINN approaches.");
    QAction* before = batch->actions().isEmpty() ? nullptr : batch->actions().first();
    batch->insertAction(before, action);
    batch->insertSeparator(before);
    QObject::connect(action, &QAction::triggered, window, [window](){ showSweepManager(window); });
}

void scheduleInstall() { QTimer::singleShot(0, [](){ installSweepManager(); }); }
}

Q_COREAPP_STARTUP_FUNCTION(scheduleInstall)
