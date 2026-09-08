#include <QAction>
#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QCoreApplication>
#include <QDateTime>
#include <QDialog>
#include <QDir>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFileInfo>
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
#include <QTextCursor>
#include <QTextEdit>
#include <QTimer>
#include <QToolBar>
#include <QVBoxLayout>

namespace {

QString repoRoot()
{
    const QStringList starts = {QDir::currentPath(), QCoreApplication::applicationDirPath()};
    for (const QString& start : starts) {
        QDir d(start);
        for (int depth = 0; depth < 10; ++depth) {
            if (QFileInfo::exists(d.filePath("HydroPINN.pro")) || QFileInfo::exists(d.filePath("HydroBatch.pro")))
                return d.absolutePath();
            if (!d.cdUp()) break;
        }
    }
    return {};
}

QMenu* batchMenu(QMainWindow* w)
{
    if (!w || !w->menuBar()) return nullptr;
    for (QAction* a : w->menuBar()->actions())
        if (a && a->menu() && QString(a->text()).remove('&') == "Batch") return a->menu();
    return nullptr;
}

QString hydroBatchExecutable(const QString& root)
{
    const QStringList candidates = {
        root + "/build-hydrobatch/HydroBatch",
        QCoreApplication::applicationDirPath() + "/HydroBatch",
        QDir::currentPath() + "/HydroBatch"
    };
    for (const QString& c : candidates) {
        QFileInfo f(c);
        if (f.exists() && f.isFile() && f.isExecutable()) return f.absoluteFilePath();
    }
    return {};
}

struct PipelineSourceSnapshot {
    QString source;
    QString sourceDisplay;
    QString syntheticProfile;
    int sampleCount = 240;
    double tStart = 0.0;
    double tEnd = 5.0;
    QString csvPath;
    int csvXColumn = 0;
    int csvYColumn = 3;
    bool csvHasHeader = true;
    QString hydroPackagePath;
    QString hydroCatchmentId;
    QString hydroPackageProfile = "rainfall-runoff";

    QString description() const
    {
        if (source == "synthetic")
            return QString("Synthetic | profile=%1 | samples=%2 | t=[%3,%4] | truth_k=0.08")
                .arg(syntheticProfile).arg(sampleCount).arg(tStart, 0, 'g', 8).arg(tEnd, 0, 'g', 8);
        if (source == "csv")
            return QString("CSV | path=%1 | x=%2 | y=%3 | header=%4")
                .arg(csvPath).arg(csvXColumn).arg(csvYColumn).arg(csvHasHeader ? "yes" : "no");
        return QString("Hydro Package | path=%1 | catchment=%2 | profile=%3")
            .arg(hydroPackagePath, hydroCatchmentId, hydroPackageProfile);
    }

    QStringList adaptiveArgs(const QString& hydroBatch, const QString& outputRoot) const
    {
        return QStringList{
            "--hydrobatch", hydroBatch,
            "--output-root", outputRoot,
            "--data-source", source,
            "--synthetic-profile", syntheticProfile,
            "--synthetic-truth-k", "0.08",
            "--sample-count", QString::number(sampleCount),
            "--t-start", QString::number(tStart, 'g', 17),
            "--t-end", QString::number(tEnd, 'g', 17),
            "--csv-path", csvPath,
            "--csv-x-column", QString::number(csvXColumn),
            "--csv-y-column", QString::number(csvYColumn),
            "--csv-has-header", csvHasHeader ? "true" : "false",
            "--hydro-package-path", hydroPackagePath,
            "--hydro-catchment-id", hydroCatchmentId,
            "--hydro-package-profile", hydroPackageProfile
        };
    }
};

template <typename T>
T* requiredSourceWidget(QMainWindow* window, const char* objectName, QString& error)
{
    T* widget = window ? window->findChild<T*>(objectName) : nullptr;
    if (!widget && error.isEmpty())
        error = QString("Adaptive pipeline could not locate GUI data-source widget '%1'. Rebuild HydroPINN from current source.").arg(objectName);
    return widget;
}

bool snapshotGuiSource(QMainWindow* window, PipelineSourceSnapshot& s, QString& error)
{
    QComboBox* source = requiredSourceWidget<QComboBox>(window, "HydroDataSourceCombo", error);
    QComboBox* profile = requiredSourceWidget<QComboBox>(window, "HydroSyntheticProfileCombo", error);
    QSpinBox* samples = requiredSourceWidget<QSpinBox>(window, "HydroSyntheticSampleCount", error);
    QDoubleSpinBox* tStart = requiredSourceWidget<QDoubleSpinBox>(window, "HydroSyntheticTStart", error);
    QDoubleSpinBox* tEnd = requiredSourceWidget<QDoubleSpinBox>(window, "HydroSyntheticTEnd", error);
    QLineEdit* csvPath = requiredSourceWidget<QLineEdit>(window, "HydroCsvPathEdit", error);
    QSpinBox* csvX = requiredSourceWidget<QSpinBox>(window, "HydroCsvXColumn", error);
    QSpinBox* csvY = requiredSourceWidget<QSpinBox>(window, "HydroCsvYColumn", error);
    QCheckBox* csvHeader = requiredSourceWidget<QCheckBox>(window, "HydroCsvHeaderCheck", error);
    QLineEdit* hydroPath = requiredSourceWidget<QLineEdit>(window, "HydroPackagePathEdit", error);
    QLineEdit* catchment = requiredSourceWidget<QLineEdit>(window, "HydroCatchmentIdEdit", error);
    QComboBox* hydroProfile = requiredSourceWidget<QComboBox>(window, "HydroPackageProfileCombo", error);
    if (!error.isEmpty()) return false;

    s.sourceDisplay = source->currentText();
    if (s.sourceDisplay == "Synthetic") s.source = "synthetic";
    else if (s.sourceDisplay == "CSV File") s.source = "csv";
    else if (s.sourceDisplay == "Hydro Package") s.source = "hydro";
    else { error = "Unknown GUI data source: " + s.sourceDisplay; return false; }

    s.syntheticProfile = profile->currentText().trimmed();
    s.sampleCount = samples->value();
    s.tStart = tStart->value();
    s.tEnd = tEnd->value();
    s.csvPath = csvPath->text().trimmed();
    s.csvXColumn = csvX->value();
    s.csvYColumn = csvY->value();
    s.csvHasHeader = csvHeader->isChecked();
    s.hydroPackagePath = hydroPath->text().trimmed();
    s.hydroCatchmentId = catchment->text().trimmed();
    s.hydroPackageProfile = hydroProfile->currentText().trimmed();

    if (s.source == "synthetic") {
        if (s.syntheticProfile != "reduced_reservoir") {
            error = "The five-method paper pipeline requires Synthetic profile 'reduced_reservoir'.";
            return false;
        }
        if (s.sampleCount < 32 || !(s.tEnd > s.tStart)) {
            error = "Synthetic pipeline requires at least 32 samples and t_end > t_start.";
            return false;
        }
    } else if (s.source == "csv") {
        if (s.csvPath.isEmpty()) { error = "CSV File is selected but the path is empty."; return false; }
        if (s.csvXColumn != 0 || s.csvYColumn < 3) {
            error = "Reduced-reservoir CSV physics requires column 0=time, 1=P, 2=PET, runoff target >=3.";
            return false;
        }
    } else if (s.hydroPackagePath.isEmpty()) {
        error = "Hydro Package is selected but the package path is empty.";
        return false;
    }
    return true;
}

void runFullPipeline(QMainWindow* window)
{
    const QString root = repoRoot();
    if (root.isEmpty()) {
        QMessageBox::critical(window, "Adaptive Tuning Pipeline", "Unable to locate the PyTorchCPP repository root.");
        return;
    }
    const QString batch = hydroBatchExecutable(root);
    if (batch.isEmpty()) {
        QMessageBox::critical(window, "Adaptive Tuning Pipeline", "HydroBatch was not found. Build HydroBatch first.");
        return;
    }
    const QString script = root + "/Hydro/experiments/gistohq_sligo/run_adaptive_full_pipeline.py";
    if (!QFileInfo::exists(script)) {
        QMessageBox::critical(window, "Adaptive Tuning Pipeline", "Adaptive pipeline script was not found. Pull the current repository and rebuild.");
        return;
    }

    PipelineSourceSnapshot source;
    QString sourceError;
    if (!snapshotGuiSource(window, source, sourceError)) {
        QMessageBox::critical(window, "Adaptive Tuning Pipeline - Data Source", sourceError);
        return;
    }

    const auto answer = QMessageBox::question(
        window,
        "Adaptive Tuning Pipeline - Confirm Data Source",
        "The paper-grade adaptive pipeline will use:\n\n" + source.description() +
        "\n\nStages inherit validation-selected winners; test metrics are not used for tuning. Continue?",
        QMessageBox::Yes | QMessageBox::No,
        QMessageBox::Yes);
    if (answer != QMessageBox::Yes) return;

    const QString defaultRoot = root + "/Hydro/experiments/gistohq_sligo/batch_outputs";
    QDir().mkpath(defaultRoot);
    const QString parent = QFileDialog::getExistingDirectory(window, "Choose Adaptive Pipeline Output Parent", defaultRoot);
    if (parent.isEmpty()) return;
    const QString outputRoot = parent + "/adaptive_pipeline_" + QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss");
    QDir().mkpath(outputRoot);

    auto* dialog = new QDialog(window);
    dialog->setAttribute(Qt::WA_DeleteOnClose);
    dialog->setWindowTitle("HydroPINN Adaptive Paper Tuning Pipeline");
    dialog->resize(980, 720);
    auto* layout = new QVBoxLayout(dialog);
    auto* status = new QLabel("Starting adaptive paper-grade tuning...", dialog);
    status->setWordWrap(true);
    auto* log = new QTextEdit(dialog);
    log->setReadOnly(true);
    auto* buttons = new QHBoxLayout();
    auto* stop = new QPushButton("Stop Pipeline", dialog);
    auto* close = new QPushButton("Close", dialog);
    close->setEnabled(false);
    buttons->addStretch(1); buttons->addWidget(stop); buttons->addWidget(close);
    layout->addWidget(status); layout->addWidget(log, 1); layout->addLayout(buttons);
    QObject::connect(close, &QPushButton::clicked, dialog, &QDialog::close);

    log->append("Adaptive paper pipeline:");
    log->append("  1. Supervised architecture + real FFN memory-horizon tuning");
    log->append("  2. Physics tuning inheriting Stage-1 architectures");
    log->append("  3. Per-method optimizer tuning inheriting Stage-1/2 winners");
    log->append("  4. Multi-seed robustness using frozen Stage-3 settings");
    log->append("  5. Paper-ready summaries + frozen configs");
    log->append("\nData source snapshot: " + source.description());
    log->append("Output root: " + outputRoot + "\n");

    auto* process = new QProcess(dialog);
    process->setProcessChannelMode(QProcess::MergedChannels);
    process->setWorkingDirectory(root);
    QObject::connect(process, &QProcess::readyReadStandardOutput, dialog, [process, log]() {
        log->moveCursor(QTextCursor::End);
        log->insertPlainText(QString::fromLocal8Bit(process->readAllStandardOutput()));
        log->moveCursor(QTextCursor::End);
    });
    QObject::connect(process, qOverload<int,QProcess::ExitStatus>(&QProcess::finished), dialog,
                     [status, log, stop, close, outputRoot](int code, QProcess::ExitStatus st) {
        stop->setEnabled(false); close->setEnabled(true);
        if (st == QProcess::NormalExit && code == 0) {
            status->setText("Adaptive paper pipeline completed: " + outputRoot);
            log->append("\n[adaptive-gui] COMPLETE\nPaper summaries and frozen configs are under:\n" + outputRoot);
        } else {
            status->setText("Adaptive paper pipeline stopped or failed. See log for the last completed stage.");
            log->append(QString("\n[adaptive-gui] FAILED exit_code=%1").arg(code));
        }
    });
    QObject::connect(stop, &QPushButton::clicked, dialog, [process, stop, status]() {
        stop->setEnabled(false); status->setText("Stopping adaptive pipeline...");
        if (process->state() != QProcess::NotRunning) {
            process->terminate();
            QTimer::singleShot(5000, process, [process]() { if (process->state() != QProcess::NotRunning) process->kill(); });
        }
    });

    dialog->show();
    process->start("python3", QStringList{script} + source.adaptiveArgs(batch, outputRoot));
    if (!process->waitForStarted(5000)) {
        status->setText("Unable to start adaptive pipeline.");
        stop->setEnabled(false); close->setEnabled(true);
    }
}

void install()
{
    QMainWindow* window = nullptr;
    for (QWidget* w : QApplication::topLevelWidgets()) if ((window = qobject_cast<QMainWindow*>(w))) break;
    if (!window) { QTimer::singleShot(100, [](){ install(); }); return; }
    QMenu* menu = batchMenu(window);
    if (!menu) { QTimer::singleShot(100, [](){ install(); }); return; }
    if (menu->findChild<QAction*>("HydroFullTuningPipelineAction")) return;

    auto* action = new QAction("Run Adaptive Paper Tuning Pipeline...", menu);
    action->setObjectName("HydroFullTuningPipelineAction");
    action->setToolTip("Run validation-selected adaptive tuning and produce paper-ready summaries using the current GUI data source.");
    QAction* before = menu->actions().isEmpty() ? nullptr : menu->actions().first();
    menu->insertAction(before, action); menu->insertSeparator(action);
    QObject::connect(action, &QAction::triggered, window, [window]() { runFullPipeline(window); });

    if (QToolBar* toolbar = window->findChild<QToolBar*>("HydroBatchToolBar")) {
        auto* toolbarAction = new QAction("Run Adaptive Pipeline", toolbar);
        toolbarAction->setToolTip("Run paper-grade adaptive tuning using the current GUI data source.");
        toolbar->insertAction(toolbar->actions().isEmpty() ? nullptr : toolbar->actions().first(), toolbarAction);
        QObject::connect(toolbarAction, &QAction::triggered, window, [window]() { runFullPipeline(window); });
    }
}

void schedule() { QTimer::singleShot(0, [](){ install(); }); }
}

Q_COREAPP_STARTUP_FUNCTION(schedule)
