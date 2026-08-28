#include <QAction>
#include <QApplication>
#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QMainWindow>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QProcess>
#include <QTimer>
#include <QToolBar>

namespace {
QString locateRepositoryRoot()
{
    const QStringList starts = {QDir::currentPath(), QCoreApplication::applicationDirPath()};
    for (const QString& start : starts) {
        QDir dir(start);
        for (int depth = 0; depth < 8; ++depth) {
            if (QFileInfo::exists(dir.filePath("HydroPINN.pro")) ||
                QFileInfo::exists(dir.filePath("HydroBatch.pro"))) {
                return dir.absolutePath();
            }
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

bool runGenerator(QWidget* parent, const QStringList& generatorArgs, const QString& description)
{
    const QString repoRoot = locateRepositoryRoot();
    if (repoRoot.isEmpty()) {
        QMessageBox::critical(parent, "Sweep Preset", "Unable to locate the PyTorchCPP repository root.");
        return false;
    }
    const QString experimentDir = repoRoot + "/Hydro/experiments/gistohq_sligo";
    const QString generator = experimentDir + "/generate_hyperparameter_sweep.py";
    if (!QFileInfo::exists(generator)) {
        QMessageBox::critical(parent, "Sweep Preset", "Sweep generator not found:\n" + generator);
        return false;
    }

    QStringList args;
    args << generator;
    args << generatorArgs;

    QProcess process;
    process.setWorkingDirectory(experimentDir);
    process.setProcessChannelMode(QProcess::MergedChannels);
    process.start("python3", args);
    if (!process.waitForStarted(5000)) {
        QMessageBox::critical(parent, "Sweep Preset", "Unable to start python3 sweep generator.");
        return false;
    }
    process.waitForFinished(-1);
    const QString output = QString::fromLocal8Bit(process.readAll());
    if (process.exitStatus() != QProcess::NormalExit || process.exitCode() != 0) {
        QMessageBox::critical(parent, "Sweep Preset", "Sweep generation failed:\n\n" + output);
        return false;
    }

    QMessageBox::information(
        parent,
        "Sweep Preset Generated",
        description + "\n\n" + output +
            "\nUse Batch > Run Config Batch... and select:\n" +
            experimentDir + "/hyperparameter_stage1.batch");
    return true;
}

void installPresetActions()
{
    QMainWindow* window = nullptr;
    for (QWidget* widget : QApplication::topLevelWidgets()) {
        window = qobject_cast<QMainWindow*>(widget);
        if (window) break;
    }
    if (!window) {
        QTimer::singleShot(100, []() { installPresetActions(); });
        return;
    }

    QMenu* batchMenu = findBatchMenu(window);
    if (!batchMenu) {
        QTimer::singleShot(100, []() { installPresetActions(); });
        return;
    }
    if (batchMenu->findChild<QMenu*>("HydroSweepPresetsMenu")) return;

    QMenu* presets = batchMenu->insertMenu(batchMenu->actions().isEmpty() ? nullptr : batchMenu->actions().first(),
                                           "Sweep Presets");
    presets->setObjectName("HydroSweepPresetsMenu");

    QAction* stage1 = presets->addAction("Stage 1 Architecture/Activation");
    stage1->setToolTip("Generate the curated Stage-1 FFN/LSTM tuning sweep with the established defaults.");
    QObject::connect(stage1, &QAction::triggered, window, [window]() {
        runGenerator(window, {},
                     "Stage 1 preset restored: FFN 6 h architecture/activation sweep plus LSTM 12/24 h architecture sweep.");
    });

    QAction* sigmoid = presets->addAction("Sigmoid-only FFN");
    sigmoid->setToolTip("Generate only the nine missing FFN sigmoid architecture cases.");
    QObject::connect(sigmoid, &QAction::triggered, window, [window]() {
        runGenerator(window,
                     {"--ffn-only", "--activations", "sigmoid",
                      "--ffn-architectures", "16;24;32;48;16,16;24,24;32,16;32,32;48,24"},
                     "Sigmoid-only FFN preset: nine new cases, without repeating tanh/ReLU or LSTM runs.");
    });

    presets->addSeparator();

    QAction* memorySweep = presets->addAction("Existing Supervised Memory Sweep");
    memorySweep->setToolTip("Keep the original supervised_sweep.batch workflow visible and available.");
    QObject::connect(memorySweep, &QAction::triggered, window, [window]() {
        const QString repoRoot = locateRepositoryRoot();
        const QString path = repoRoot + "/Hydro/experiments/gistohq_sligo/supervised_sweep.batch";
        QMessageBox::information(window, "Existing Memory Sweep",
                                 "The original FFN/LSTM memory sweep is unchanged.\n\nUse Batch > Run Config Batch... and select:\n" + path);
    });

    QAction* lstmMemorySweep = presets->addAction("Existing LSTM Memory Sweep");
    lstmMemorySweep->setToolTip("Keep the original lstm_sweep.batch workflow visible and available.");
    QObject::connect(lstmMemorySweep, &QAction::triggered, window, [window]() {
        const QString repoRoot = locateRepositoryRoot();
        const QString path = repoRoot + "/Hydro/experiments/gistohq_sligo/lstm_sweep.batch";
        QMessageBox::information(window, "Existing LSTM Memory Sweep",
                                 "The original LSTM sequence-length sweep is unchanged.\n\nUse Batch > Run Config Batch... and select:\n" + path);
    });

    if (QToolBar* toolbar = window->findChild<QToolBar*>("HydroBatchToolBar")) {
        QAction* presetToolbarAction = new QAction("Sweep Presets", toolbar);
        presetToolbarAction->setMenu(presets);
        toolbar->insertAction(toolbar->actions().isEmpty() ? nullptr : toolbar->actions().first(), presetToolbarAction);
    }
}

void schedulePresetInstall()
{
    QTimer::singleShot(0, []() { installPresetActions(); });
}
}

Q_COREAPP_STARTUP_FUNCTION(schedulePresetInstall)
