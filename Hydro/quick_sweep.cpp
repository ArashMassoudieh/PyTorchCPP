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
#include <QRadioButton>
#include <QTextEdit>
#include <QTimer>
#include <QVBoxLayout>

namespace {
QString repoRoot()
{
    const QStringList starts = {QDir::currentPath(), QCoreApplication::applicationDirPath()};
    for (const QString& start : starts) {
        QDir d(start);
        for (int i = 0; i < 10; ++i) {
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

QString batchExecutable(const QString& root)
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

bool generate(const QString& root, const QStringList& args, QString* output)
{
    const QString work = root + "/Hydro/experiments/gistohq_sligo";
    QProcess p;
    p.setWorkingDirectory(work);
    p.setProcessChannelMode(QProcess::MergedChannels);
    QStringList a{work + "/generate_unified_sweep.py"};
    a << args;
    p.start("python3", a);
    if (!p.waitForStarted(5000)) return false;
    p.waitForFinished(-1);
    if (output) *output = QString::fromLocal8Bit(p.readAll());
    return p.exitStatus() == QProcess::NormalExit && p.exitCode() == 0;
}

void runBatch(QMainWindow* w, const QString& root, const QString& label)
{
    const QString exe = batchExecutable(root);
    if (exe.isEmpty()) {
        QMessageBox::critical(w, "Quick Sweep", "HydroBatch was not found. Build HydroBatch first.");
        return;
    }
    const QString base = root + "/Hydro/experiments/gistohq_sligo/batch_outputs";
    QDir().mkpath(base);
    const QString parent = QFileDialog::getExistingDirectory(w, "Choose output folder", base);
    if (parent.isEmpty()) return;
    const QString out = parent + "/" + label + "_" + QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss");
    QDir().mkpath(out);
    const QString batch = root + "/Hydro/experiments/gistohq_sligo/unified_sweep.batch";

    auto* dlg = new QDialog(w);
    dlg->setAttribute(Qt::WA_DeleteOnClose);
    dlg->setWindowTitle("Quick Sweep - Running");
    dlg->resize(880, 620);
    auto* lay = new QVBoxLayout(dlg);
    auto* status = new QLabel("Running " + label + "...", dlg);
    auto* log = new QTextEdit(dlg); log->setReadOnly(true);
    auto* row = new QHBoxLayout();
    auto* stop = new QPushButton("Stop", dlg);
    auto* close = new QPushButton("Close", dlg); close->setEnabled(false);
    row->addStretch(); row->addWidget(stop); row->addWidget(close);
    lay->addWidget(status); lay->addWidget(log, 1); lay->addLayout(row);

    auto* p = new QProcess(dlg);
    p->setWorkingDirectory(root);
    p->setProcessChannelMode(QProcess::MergedChannels);
    QObject::connect(p, &QProcess::readyReadStandardOutput, dlg, [p, log]() {
        log->moveCursor(QTextCursor::End);
        log->insertPlainText(QString::fromLocal8Bit(p->readAllStandardOutput()));
        log->moveCursor(QTextCursor::End);
    });
    QObject::connect(stop, &QPushButton::clicked, dlg, [p, stop, status]() {
        if (p->state() == QProcess::NotRunning) return;
        stop->setEnabled(false); status->setText("Stopping..."); p->terminate();
        QTimer::singleShot(5000, p, [p]() { if (p->state() != QProcess::NotRunning) p->kill(); });
    });
    QObject::connect(close, &QPushButton::clicked, dlg, &QDialog::close);
    QObject::connect(p, qOverload<int,QProcess::ExitStatus>(&QProcess::finished), dlg,
                     [status, stop, close, out](int code, QProcess::ExitStatus st) {
        stop->setEnabled(false); close->setEnabled(true);
        status->setText(st == QProcess::NormalExit && code == 0
            ? "Completed. Results: " + out + "/batch_summary.csv"
            : "Finished with failures. Check the log and " + out + "/batch_summary.csv");
    });
    log->append("Output: " + out + "\n");
    p->start(exe, {batch, out});
    if (!p->waitForStarted(5000)) {
        status->setText("Could not start HydroBatch."); stop->setEnabled(false); close->setEnabled(true);
    }
    dlg->show();
}

void showQuickSweep(QMainWindow* w)
{
    QDialog d(w);
    d.setWindowTitle("HydroPINN Quick Sweep");
    d.resize(680, 500);
    auto* root = new QVBoxLayout(&d);
    auto* intro = new QLabel(
        "Choose what you want to do. Recommended settings are filled automatically from the current Sligo Creek tuning results. "
        "Use Sweep Manager only when you want to edit the full grid.", &d);
    intro->setWordWrap(true); root->addWidget(intro);

    auto* baseline = new QRadioButton("1. Compare all five methods (5 runs)", &d);
    auto* physics = new QRadioButton("2. Tune physics parameters (65 runs)", &d);
    auto* optimizer = new QRadioButton("3. Tune learning rate and batch size (45 runs)", &d);
    auto* robust = new QRadioButton("4. Check robustness across 5 seeds (25 runs)", &d);
    baseline->setChecked(true);
    root->addWidget(baseline); root->addWidget(physics); root->addWidget(optimizer); root->addWidget(robust);

    auto* help = new QLabel(&d); help->setWordWrap(true); root->addWidget(help);
    auto updateHelp = [&]() {
        if (baseline->isChecked()) help->setText("Uses the current finalists: FFN 16x16 ReLU with 6 h memory, LSTM H=32 with 12 h memory, plus matching hybrid PINNs and a 24x24 standalone PINN.");
        else if (physics->isChecked()) help->setText("Sweeps physics weight {0.001, 0.005, 0.01, 0.025, 0.05, 0.1} and latent recession k {0.01, 0.02, 0.04, 0.08, 0.16}. Standalone PINN varies only k.");
        else if (optimizer->isChecked()) help->setText("Keeps the selected architectures fixed and sweeps learning rate {0.001, 0.003, 0.005} x batch size {16, 32, 64} for all five methods.");
        else help->setText("Keeps the current finalists fixed and repeats all five methods with seeds 42, 123, 2026, 31415, and 27182.");
    };
    for (QRadioButton* r : {baseline, physics, optimizer, robust}) QObject::connect(r, &QRadioButton::toggled, &d, [&](bool){ updateHelp(); });
    updateHelp();

    auto* note = new QLabel("Sigmoid is intentionally excluded from recommended sweeps because the completed Stage-1 run showed consistently poor generalization. Advanced Sweep Manager still allows custom activation choices.", &d);
    note->setWordWrap(true); root->addWidget(note);
    root->addStretch();

    auto* row = new QHBoxLayout();
    auto* advanced = new QPushButton("Open Advanced Sweep Manager", &d);
    auto* cancel = new QPushButton("Cancel", &d);
    auto* run = new QPushButton("Generate & Run", &d);
    row->addWidget(advanced); row->addStretch(); row->addWidget(cancel); row->addWidget(run); root->addLayout(row);
    QObject::connect(cancel, &QPushButton::clicked, &d, &QDialog::reject);
    QObject::connect(advanced, &QPushButton::clicked, &d, [w, &d]() {
        d.accept();
        if (QMenu* menu = batchMenu(w)) {
            for (QAction* a : menu->actions()) if (a && a->text().contains("Sweep Manager") && !a->text().contains("Quick")) { a->trigger(); return; }
        }
    });
    QObject::connect(run, &QPushButton::clicked, &d, [=, &d]() {
        const QString rootPath = repoRoot();
        if (rootPath.isEmpty()) { QMessageBox::critical(&d, "Quick Sweep", "Repository root not found."); return; }
        QStringList args;
        QString label;
        if (baseline->isChecked()) {
            label = "five_method_baseline";
            args = {"--methods","ffn,ffn_pinn,lstm,lstm_pinn,pinn","--ffn-architectures","16,16","--ffn-activations","relu","--ffn-lags","1,2,3,4,5,6","--lstm-architectures","32","--lstm-sequences","12","--pinn-architectures","24,24","--learning-rates","0.003","--batch-sizes","32","--seeds","42","--physics-weights","0.05","--recession-k","0.08"};
        } else if (physics->isChecked()) {
            label = "physics_stage1";
            args = {"--methods","ffn_pinn,lstm_pinn,pinn","--ffn-architectures","16,16","--ffn-activations","relu","--lstm-architectures","32","--lstm-sequences","12","--pinn-architectures","24,24","--learning-rates","0.003","--batch-sizes","32","--seeds","42","--physics-weights","0.001,0.005,0.01,0.025,0.05,0.1","--recession-k","0.01,0.02,0.04,0.08,0.16"};
        } else if (optimizer->isChecked()) {
            label = "optimizer_stage2";
            args = {"--methods","ffn,ffn_pinn,lstm,lstm_pinn,pinn","--ffn-architectures","16,16","--ffn-activations","relu","--ffn-lags","1,2,3,4,5,6","--lstm-architectures","32","--lstm-sequences","12","--pinn-architectures","24,24","--learning-rates","0.001,0.003,0.005","--batch-sizes","16,32,64","--seeds","42","--physics-weights","0.01","--recession-k","0.04"};
        } else {
            label = "robustness_stage3";
            args = {"--methods","ffn,ffn_pinn,lstm,lstm_pinn,pinn","--ffn-architectures","16,16","--ffn-activations","relu","--ffn-lags","1,2,3,4,5,6","--lstm-architectures","32","--lstm-sequences","12","--pinn-architectures","24,24","--learning-rates","0.003","--batch-sizes","32","--seeds","42,123,2026,31415,27182","--physics-weights","0.01","--recession-k","0.04"};
        }
        QString output;
        if (!generate(rootPath, args, &output)) { QMessageBox::critical(&d, "Quick Sweep", "Generation failed:\n\n" + output); return; }
        d.accept();
        runBatch(w, rootPath, label);
    });
    d.exec();
}

void install()
{
    QMainWindow* w = nullptr;
    for (QWidget* x : QApplication::topLevelWidgets()) if ((w = qobject_cast<QMainWindow*>(x))) break;
    if (!w) { QTimer::singleShot(100, [](){ install(); }); return; }
    QMenu* menu = batchMenu(w);
    if (!menu) { QTimer::singleShot(100, [](){ install(); }); return; }
    if (menu->findChild<QAction*>("HydroQuickSweepAction")) return;
    auto* a = new QAction("Quick Sweep...", menu);
    a->setObjectName("HydroQuickSweepAction");
    a->setToolTip("Guided one-click sweeps for all five HydroPINN methods.");
    menu->insertAction(menu->actions().isEmpty() ? nullptr : menu->actions().first(), a);
    QObject::connect(a, &QAction::triggered, w, [w](){ showQuickSweep(w); });
}

void schedule() { QTimer::singleShot(0, [](){ install(); }); }
}
Q_COREAPP_STARTUP_FUNCTION(schedule)
