#include <QAction>
#include <QApplication>
#include <QCoreApplication>
#include <QDateTime>
#include <QDialog>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QLabel>
#include <QMainWindow>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QProcess>
#include <QPushButton>
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

struct PipelineStage {
    QString label;
    QString folder;
    QStringList generatorArgs;
};

QList<PipelineStage> stages()
{
    return {
        {
            "Stage 1 - Supervised architecture / memory",
            "01_stage1_supervised",
            {"--methods","ffn,lstm",
             "--ffn-architectures","16;24;32;48;16,16;24,24;32,16;32,32;48,24",
             "--ffn-activations","tanh,relu",
             "--ffn-lags","1,2,3,4,5,6",
             "--lstm-architectures","16;24;32;48;24,24;32,32",
             "--lstm-sequences","12,24",
             "--learning-rates","0.003","--batch-sizes","32","--seeds","42"}
        },
        {
            "Stage 2 - Physics tuning",
            "02_stage2_physics",
            {"--methods","ffn_pinn,lstm_pinn,pinn",
             "--ffn-architectures","16,16","--ffn-activations","relu",
             "--lstm-architectures","32","--lstm-sequences","12",
             "--pinn-architectures","24,24",
             "--learning-rates","0.003","--batch-sizes","32","--seeds","42",
             "--physics-weights","0.001,0.005,0.01,0.025,0.05,0.1",
             "--recession-k","0.01,0.02,0.04,0.08,0.16"}
        },
        {
            "Stage 3 - Learning rate / batch-size tuning",
            "03_stage3_optimizer",
            {"--methods","ffn,ffn_pinn,lstm,lstm_pinn,pinn",
             "--ffn-architectures","16,16","--ffn-activations","relu","--ffn-lags","1,2,3,4,5,6",
             "--lstm-architectures","32","--lstm-sequences","12","--pinn-architectures","24,24",
             "--learning-rates","0.001,0.003,0.005","--batch-sizes","16,32,64","--seeds","42",
             "--physics-weights","0.01","--recession-k","0.04"}
        },
        {
            "Stage 4 - Multi-seed robustness",
            "04_stage4_robustness",
            {"--methods","ffn,ffn_pinn,lstm,lstm_pinn,pinn",
             "--ffn-architectures","16,16","--ffn-activations","relu","--ffn-lags","1,2,3,4,5,6",
             "--lstm-architectures","32","--lstm-sequences","12","--pinn-architectures","24,24",
             "--learning-rates","0.003","--batch-sizes","32",
             "--seeds","42,123,2026,31415,27182",
             "--physics-weights","0.01","--recession-k","0.04"}
        }
    };
}

class PipelineController : public QObject
{
public:
    PipelineController(QMainWindow* window,
                       const QString& root,
                       const QString& outputRoot,
                       QDialog* dialog,
                       QLabel* status,
                       QTextEdit* log,
                       QPushButton* stop,
                       QPushButton* close)
        : QObject(dialog), window_(window), root_(root), outputRoot_(outputRoot), dialog_(dialog),
          status_(status), log_(log), stop_(stop), close_(close), stages_(stages())
    {
        process_ = new QProcess(this);
        process_->setProcessChannelMode(QProcess::MergedChannels);
        QObject::connect(process_, &QProcess::readyReadStandardOutput, this, [this]() {
            log_->moveCursor(QTextCursor::End);
            log_->insertPlainText(QString::fromLocal8Bit(process_->readAllStandardOutput()));
            log_->moveCursor(QTextCursor::End);
        });
        QObject::connect(process_, qOverload<int,QProcess::ExitStatus>(&QProcess::finished), this,
                         [this](int code, QProcess::ExitStatus st) { onFinished(code, st); });
        QObject::connect(stop_, &QPushButton::clicked, this, [this]() {
            cancelled_ = true;
            stop_->setEnabled(false);
            status_->setText("Stopping full pipeline...");
            if (process_->state() != QProcess::NotRunning) {
                process_->terminate();
                QTimer::singleShot(5000, process_, [this]() {
                    if (process_->state() != QProcess::NotRunning) process_->kill();
                });
            }
        });
    }

    void start()
    {
        runGenerator();
    }

private:
    void appendHeader(const QString& text)
    {
        log_->append("\n============================================================");
        log_->append(text);
        log_->append("============================================================\n");
    }

    void runGenerator()
    {
        if (cancelled_) return finishCancelled();
        if (stageIndex_ >= stages_.size()) return finishSuccess();
        const PipelineStage& stage = stages_[stageIndex_];
        appendHeader(QString("%1/%2  %3").arg(stageIndex_ + 1).arg(stages_.size()).arg(stage.label));
        status_->setText(QString("Generating %1/%2: %3").arg(stageIndex_ + 1).arg(stages_.size()).arg(stage.label));

        const QString work = root_ + "/Hydro/experiments/gistohq_sligo";
        QStringList args{work + "/generate_unified_sweep.py"};
        args << stage.generatorArgs;
        currentStep_ = Generate;
        process_->setWorkingDirectory(work);
        process_->start("python3", args);
        if (!process_->waitForStarted(5000)) fail("Unable to start Python sweep generator.");
    }

    void runBatch()
    {
        const QString exe = hydroBatchExecutable(root_);
        if (exe.isEmpty()) return fail("HydroBatch executable was not found. Build HydroBatch first.");
        const PipelineStage& stage = stages_[stageIndex_];
        const QString batch = root_ + "/Hydro/experiments/gistohq_sligo/unified_sweep.batch";
        const QString out = outputRoot_ + "/" + stage.folder;
        QDir().mkpath(out);
        status_->setText(QString("Running %1/%2: %3").arg(stageIndex_ + 1).arg(stages_.size()).arg(stage.label));
        log_->append("Batch output: " + out + "\n");
        currentStep_ = Batch;
        process_->setWorkingDirectory(root_);
        process_->start(exe, {batch, out});
        if (!process_->waitForStarted(5000)) fail("Unable to start HydroBatch.");
    }

    void onFinished(int code, QProcess::ExitStatus st)
    {
        if (cancelled_) return finishCancelled();
        if (st != QProcess::NormalExit || code != 0) {
            const QString step = currentStep_ == Generate ? "generation" : "batch execution";
            return fail(QString("Pipeline stopped: %1 failed during %2.").arg(stages_[stageIndex_].label, step));
        }
        if (currentStep_ == Generate) {
            runBatch();
        } else {
            log_->append("[pipeline] completed: " + stages_[stageIndex_].label + "\n");
            ++stageIndex_;
            runGenerator();
        }
    }

    void fail(const QString& message)
    {
        status_->setText(message);
        log_->append("[pipeline] ERROR: " + message);
        stop_->setEnabled(false);
        close_->setEnabled(true);
    }

    void finishSuccess()
    {
        status_->setText("Full tuning pipeline completed. Results: " + outputRoot_);
        appendHeader("FULL PIPELINE COMPLETE");
        log_->append("Each stage has its own batch_summary.csv under:\n" + outputRoot_);
        stop_->setEnabled(false);
        close_->setEnabled(true);
    }

    void finishCancelled()
    {
        status_->setText("Full tuning pipeline stopped by user.");
        log_->append("[pipeline] stopped by user.");
        stop_->setEnabled(false);
        close_->setEnabled(true);
    }

    enum Step { Generate, Batch };
    QMainWindow* window_{};
    QString root_;
    QString outputRoot_;
    QDialog* dialog_{};
    QLabel* status_{};
    QTextEdit* log_{};
    QPushButton* stop_{};
    QPushButton* close_{};
    QProcess* process_{};
    QList<PipelineStage> stages_;
    int stageIndex_ = 0;
    Step currentStep_ = Generate;
    bool cancelled_ = false;
};

void runFullPipeline(QMainWindow* window)
{
    const QString root = repoRoot();
    if (root.isEmpty()) {
        QMessageBox::critical(window, "Full Tuning Pipeline", "Unable to locate the PyTorchCPP repository root.");
        return;
    }
    if (hydroBatchExecutable(root).isEmpty()) {
        QMessageBox::critical(window, "Full Tuning Pipeline", "HydroBatch was not found. Build HydroBatch first.");
        return;
    }

    const QString defaultRoot = root + "/Hydro/experiments/gistohq_sligo/batch_outputs";
    QDir().mkpath(defaultRoot);
    const QString parent = QFileDialog::getExistingDirectory(window, "Choose Full Pipeline Output Parent", defaultRoot);
    if (parent.isEmpty()) return;
    const QString outputRoot = parent + "/full_pipeline_" + QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss");
    QDir().mkpath(outputRoot);

    auto* dialog = new QDialog(window);
    dialog->setAttribute(Qt::WA_DeleteOnClose);
    dialog->setWindowTitle("HydroPINN Full Tuning Pipeline");
    dialog->resize(940, 680);
    auto* layout = new QVBoxLayout(dialog);
    auto* status = new QLabel("Preparing full tuning pipeline...", dialog);
    status->setWordWrap(true);
    auto* log = new QTextEdit(dialog);
    log->setReadOnly(true);
    auto* buttons = new QHBoxLayout();
    auto* stop = new QPushButton("Stop Pipeline", dialog);
    auto* close = new QPushButton("Close", dialog);
    close->setEnabled(false);
    buttons->addStretch(1);
    buttons->addWidget(stop);
    buttons->addWidget(close);
    layout->addWidget(status);
    layout->addWidget(log, 1);
    layout->addLayout(buttons);
    QObject::connect(close, &QPushButton::clicked, dialog, &QDialog::close);

    log->append("One-click pipeline stages:");
    int i = 0;
    for (const PipelineStage& s : stages()) log->append(QString("  %1. %2").arg(++i).arg(s.label));
    log->append("\nPipeline root: " + outputRoot + "\n");

    auto* controller = new PipelineController(window, root, outputRoot, dialog, status, log, stop, close);
    dialog->show();
    QTimer::singleShot(0, controller, [controller]() { controller->start(); });
}

void install()
{
    QMainWindow* window = nullptr;
    for (QWidget* w : QApplication::topLevelWidgets()) if ((window = qobject_cast<QMainWindow*>(w))) break;
    if (!window) { QTimer::singleShot(100, [](){ install(); }); return; }
    QMenu* menu = batchMenu(window);
    if (!menu) { QTimer::singleShot(100, [](){ install(); }); return; }
    if (menu->findChild<QAction*>("HydroFullTuningPipelineAction")) return;

    auto* action = new QAction("Run Full Tuning Pipeline...", menu);
    action->setObjectName("HydroFullTuningPipelineAction");
    action->setToolTip("Generate and run all established tuning stages sequentially with one click.");
    QAction* before = menu->actions().isEmpty() ? nullptr : menu->actions().first();
    menu->insertAction(before, action);
    menu->insertSeparator(action);
    QObject::connect(action, &QAction::triggered, window, [window]() { runFullPipeline(window); });

    if (QToolBar* toolbar = window->findChild<QToolBar*>("HydroBatchToolBar")) {
        auto* toolbarAction = new QAction("Run Full Pipeline", toolbar);
        toolbarAction->setToolTip("Run all tuning stages sequentially.");
        toolbar->insertAction(toolbar->actions().isEmpty() ? nullptr : toolbar->actions().first(), toolbarAction);
        QObject::connect(toolbarAction, &QAction::triggered, window, [window]() { runFullPipeline(window); });
    }
}

void schedule() { QTimer::singleShot(0, [](){ install(); }); }
}

Q_COREAPP_STARTUP_FUNCTION(schedule)
