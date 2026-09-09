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
            return QString("Synthetic | profile=%1 | samples=%2 | t=[%3,%4]")
                .arg(syntheticProfile).arg(sampleCount).arg(tStart, 0, 'g', 8).arg(tEnd, 0, 'g', 8);
        if (source == "csv")
            return QString("CSV | path=%1 | x=%2 | y=%3 | header=%4")
                .arg(csvPath).arg(csvXColumn).arg(csvYColumn).arg(csvHasHeader ? "yes" : "no");
        return QString("Hydro Package | path=%1 | catchment=%2 | profile=%3")
            .arg(hydroPackagePath, hydroCatchmentId, hydroPackageProfile);
    }

    QStringList fullPaperArgs(const QString& hydroBatch, const QString& outputRoot) const
    {
        return QStringList{
            "--hydrobatch", hydroBatch,
            "--output-root", outputRoot,
            "--data-source", source,
            "--csv-path", csvPath,
            "--csv-x-column", QString::number(csvXColumn),
            "--csv-y-column", QString::number(csvYColumn),
            "--csv-has-header", csvHasHeader ? "true" : "false",
            "--hydro-package-path", hydroPackagePath,
            "--hydro-catchment-id", hydroCatchmentId,
            "--hydro-package-profile", hydroPackageProfile,
            "--synthetic-sample-count", QString::number(sampleCount),
            "--synthetic-t-start", QString::number(tStart, 'g', 17),
            "--synthetic-t-end", QString::number(tEnd, 'g', 17),
            "--synthetic-truth-k", "0.08"
        };
    }
};

template <typename T>
T* requiredSourceWidget(QMainWindow* window, const char* objectName, QString& error)
{
    T* widget = window ? window->findChild<T*>(objectName) : nullptr;
    if (!widget && error.isEmpty())
        error = QString("Full paper pipeline could not locate GUI data-source widget '%1'. Rebuild HydroPINN from current source.").arg(objectName);
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

    // The full paper pipeline always runs its own controlled synthetic study.
    // The GUI source must therefore identify the real-data experiment.
    if (s.source == "synthetic") {
        error = "Full Paper Pipeline automatically runs the controlled synthetic verification. Select Hydro Package (or CSV File) in Data Source for the real-data comparison, then run the button again.";
        return false;
    }
    if (s.sampleCount < 32 || !(s.tEnd > s.tStart)) {
        error = "Controlled synthetic verification requires at least 32 samples and t_end > t_start.";
        return false;
    }
    if (s.source == "csv") {
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
        QMessageBox::critical(window, "Full Paper Pipeline", "Unable to locate the PyTorchCPP repository root.");
        return;
    }
    const QString batch = hydroBatchExecutable(root);
    if (batch.isEmpty()) {
        QMessageBox::critical(window, "Full Paper Pipeline", "HydroBatch was not found. Rebuild HydroBatch first so it includes the current common-domain data fix.");
        return;
    }
    const QString script = root + "/Hydro/experiments/gistohq_sligo/run_full_paper_pipeline.py";
    if (!QFileInfo::exists(script)) {
        QMessageBox::critical(window, "Full Paper Pipeline", "Full paper pipeline script was not found. Pull the current repository and rebuild HydroPINN.");
        return;
    }

    PipelineSourceSnapshot source;
    QString sourceError;
    if (!snapshotGuiSource(window, source, sourceError)) {
        QMessageBox::critical(window, "Full Paper Pipeline - Data Source", sourceError);
        return;
    }

    const auto answer = QMessageBox::question(
        window,
        "Full Paper Pipeline - Confirm",
        "One click will run:\n\n"
        "1. Controlled reduced-reservoir synthetic verification\n"
        "2. Five-method adaptive real-data study\n"
        "3. Five-seed robustness\n"
        "4. KGE/degeneracy diagnostics\n"
        "5. Final paper tables\n"
        "6. Publication figures (600-dpi PNG + PDF + SVG)\n\n"
        "Real-data source:\n" + source.description() +
        "\n\nAll five GIStoOHQ methods use one common contiguous hourly domain. Real-data model selection uses validation KGE/NSE/RMSE only; held-out test metrics are not used for tuning. Continue?",
        QMessageBox::Yes | QMessageBox::No,
        QMessageBox::Yes);
    if (answer != QMessageBox::Yes) return;

    const QString defaultRoot = root + "/Hydro/experiments/gistohq_sligo/batch_outputs";
    QDir().mkpath(defaultRoot);
    const QString parent = QFileDialog::getExistingDirectory(window, "Choose Full Paper Pipeline Output Parent", defaultRoot);
    if (parent.isEmpty()) return;
    const QString outputRoot = parent + "/paper_run_" + QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss");
    QDir().mkpath(outputRoot);

    auto* dialog = new QDialog(window);
    dialog->setAttribute(Qt::WA_DeleteOnClose);
    dialog->setWindowTitle("HydroPINN Full Paper Pipeline");
    dialog->resize(1020, 760);
    auto* layout = new QVBoxLayout(dialog);
    auto* status = new QLabel("Starting complete paper workflow...", dialog);
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

    log->append("Full paper pipeline:");
    log->append("  1. Controlled synthetic five-method verification + k recovery");
    log->append("  2. Real-data supervised architecture/memory tuning");
    log->append("  3. Hybrid physics tuning with weak-to-moderate physics weights");
    log->append("  4. Hydrologic validation selection (KGE -> NSE -> RMSE; non-degenerate)");
    log->append("  5. Per-method optimizer tuning");
    log->append("  6. Five-seed robustness");
    log->append("  7. Diagnostics, final tables, and publication figures");
    log->append("\nReal-data source: " + source.description());
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
            status->setText("Full paper pipeline completed: " + outputRoot);
            log->append("\n[full-paper-gui] COMPLETE\nTables, diagnostics, frozen configs, and PNG/PDF/SVG figures are under:\n" + outputRoot);
        } else {
            status->setText("Full paper pipeline stopped or failed. See log for the last completed stage.");
            log->append(QString("\n[full-paper-gui] FAILED exit_code=%1").arg(code));
        }
    });
    QObject::connect(stop, &QPushButton::clicked, dialog, [process, stop, status]() {
        stop->setEnabled(false); status->setText("Stopping full paper pipeline...");
        if (process->state() != QProcess::NotRunning) {
            process->terminate();
            QTimer::singleShot(5000, process, [process]() { if (process->state() != QProcess::NotRunning) process->kill(); });
        }
    });

    dialog->show();
    process->start("python3", QStringList{script} + source.fullPaperArgs(batch, outputRoot));
    if (!process->waitForStarted(5000)) {
        status->setText("Unable to start full paper pipeline.");
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

    auto* action = new QAction("Run Full Paper Pipeline...", menu);
    action->setObjectName("HydroFullTuningPipelineAction");
    action->setToolTip("Run controlled verification, adaptive real-data tuning, robustness, diagnostics, final tables, and publication figures.");
    QAction* before = menu->actions().isEmpty() ? nullptr : menu->actions().first();
    menu->insertAction(before, action); menu->insertSeparator(action);
    QObject::connect(action, &QAction::triggered, window, [window]() { runFullPipeline(window); });

    if (QToolBar* toolbar = window->findChild<QToolBar*>("HydroBatchToolBar")) {
        auto* toolbarAction = new QAction("Full Paper Pipeline", toolbar);
        toolbarAction->setToolTip("Run the complete paper experiment and artifact workflow using the current real-data source.");
        toolbar->insertAction(toolbar->actions().isEmpty() ? nullptr : toolbar->actions().first(), toolbarAction);
        QObject::connect(toolbarAction, &QAction::triggered, window, [window]() { runFullPipeline(window); });
    }
}

void schedule() { QTimer::singleShot(0, [](){ install(); }); }
}

Q_COREAPP_STARTUP_FUNCTION(schedule)
