#include <QAction>
#include <QApplication>
#include <QCheckBox>
#include <QCoreApplication>
#include <QDialog>
#include <QDir>
#include <QDoubleSpinBox>
#include <QFileInfo>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMainWindow>
#include <QMenu>
#include <QMessageBox>
#include <QProcess>
#include <QPushButton>
#include <QSpinBox>
#include <QTimer>
#include <QVBoxLayout>

namespace {
QString repoRoot()
{
    const QStringList starts = {QDir::currentPath(), QCoreApplication::applicationDirPath()};
    for (const QString& start : starts) {
        QDir dir(start);
        for (int depth = 0; depth < 8; ++depth) {
            if (QFileInfo::exists(dir.filePath("HydroPINN.pro")) || QFileInfo::exists(dir.filePath("HydroBatch.pro")))
                return dir.absolutePath();
            if (!dir.cdUp()) break;
        }
    }
    return {};
}

bool runGenerator(QWidget* parent, const QString& script, const QStringList& args, const QString& batchFile)
{
    const QString root = repoRoot();
    if (root.isEmpty()) {
        QMessageBox::critical(parent, "Hydro Sweep", "Unable to locate the PyTorchCPP repository root.");
        return false;
    }
    const QString dir = root + "/Hydro/experiments/gistohq_sligo";
    const QString path = dir + "/" + script;
    if (!QFileInfo::exists(path)) {
        QMessageBox::critical(parent, "Hydro Sweep", "Generator not found:\n" + path);
        return false;
    }

    QStringList processArgs{path};
    processArgs << args;
    QProcess process;
    process.setWorkingDirectory(dir);
    process.setProcessChannelMode(QProcess::MergedChannels);
    process.start("python3", processArgs);
    if (!process.waitForStarted(5000)) {
        QMessageBox::critical(parent, "Hydro Sweep", "Unable to start python3.");
        return false;
    }
    process.waitForFinished(-1);
    const QString output = QString::fromLocal8Bit(process.readAll());
    if (process.exitStatus() != QProcess::NormalExit || process.exitCode() != 0) {
        QMessageBox::critical(parent, "Hydro Sweep", "Sweep generation failed:\n\n" + output);
        return false;
    }
    QMessageBox::information(parent, "Hydro Sweep Generated",
                             output + "\nRun with Batch > Run Config Batch...:\n" + dir + "/" + batchFile);
    return true;
}

void showStage2Dialog(QMainWindow* window)
{
    QDialog dialog(window);
    dialog.setWindowTitle("Stage 2 - Optimizer Tuning");
    dialog.resize(650, 540);
    auto* root = new QVBoxLayout(&dialog);

    auto* intro = new QLabel(
        "Stage 2 freezes the selected architecture/memory settings and sweeps learning rate x batch size. "
        "Use the sigmoid switch only if Stage 1B shows that a sigmoid FFN should advance.", &dialog);
    intro->setWordWrap(true);
    root->addWidget(intro);

    auto* ffnBox = new QGroupBox("FFN finalist", &dialog);
    auto* ffnForm = new QFormLayout(ffnBox);
    auto* reluCheck = new QCheckBox("Include ReLU finalist", ffnBox);
    auto* sigmoidCheck = new QCheckBox("Include sigmoid finalist", ffnBox);
    reluCheck->setChecked(true);
    sigmoidCheck->setChecked(false);
    auto* reluHidden = new QLineEdit("16,16", ffnBox);
    auto* sigmoidHidden = new QLineEdit("16,16", ffnBox);
    sigmoidHidden->setEnabled(false);
    ffnForm->addRow(reluCheck);
    ffnForm->addRow("ReLU hidden layers", reluHidden);
    ffnForm->addRow(sigmoidCheck);
    ffnForm->addRow("Sigmoid hidden layers", sigmoidHidden);
    QObject::connect(sigmoidCheck, &QCheckBox::toggled, sigmoidHidden, &QWidget::setEnabled);
    root->addWidget(ffnBox);

    auto* lstmBox = new QGroupBox("LSTM finalists", &dialog);
    auto* lstmForm = new QFormLayout(lstmBox);
    auto* lstm12 = new QLineEdit("32", lstmBox);
    auto* lstm24deep = new QLineEdit("24,24", lstmBox);
    auto* lstm24 = new QLineEdit("32", lstmBox);
    lstmForm->addRow("12 h hidden layers", lstm12);
    lstmForm->addRow("24 h deep hidden layers", lstm24deep);
    lstmForm->addRow("24 h hidden layers", lstm24);
    root->addWidget(lstmBox);

    auto* tuningBox = new QGroupBox("Training grid", &dialog);
    auto* tuningForm = new QFormLayout(tuningBox);
    auto* lrs = new QLineEdit("0.001,0.003,0.005", tuningBox);
    auto* batches = new QLineEdit("16,32,64", tuningBox);
    auto* epochs = new QSpinBox(tuningBox);
    epochs->setRange(1, 100000); epochs->setValue(150);
    auto* seed = new QSpinBox(tuningBox);
    seed->setRange(0, 2147483647); seed->setValue(42);
    tuningForm->addRow("Learning rates", lrs);
    tuningForm->addRow("Batch sizes", batches);
    tuningForm->addRow("Epochs", epochs);
    tuningForm->addRow("Fixed seed", seed);
    root->addWidget(tuningBox);

    auto* buttons = new QHBoxLayout();
    auto* cancel = new QPushButton("Cancel", &dialog);
    auto* generate = new QPushButton("Generate Stage 2", &dialog);
    buttons->addStretch(1); buttons->addWidget(cancel); buttons->addWidget(generate);
    root->addLayout(buttons);
    QObject::connect(cancel, &QPushButton::clicked, &dialog, &QDialog::reject);
    QObject::connect(generate, &QPushButton::clicked, &dialog, [&]() {
        QStringList activations;
        if (reluCheck->isChecked()) activations << "relu";
        if (sigmoidCheck->isChecked()) activations << "sigmoid";
        if (activations.isEmpty()) {
            QMessageBox::warning(&dialog, "Stage 2", "Select at least one FFN activation finalist.");
            return;
        }
        QStringList args{
            "--ffn-activations", activations.join(','),
            "--ffn-hidden", reluHidden->text().trimmed(),
            "--sigmoid-hidden", sigmoidHidden->text().trimmed(),
            "--lstm12-hidden", lstm12->text().trimmed(),
            "--lstm24-deep-hidden", lstm24deep->text().trimmed(),
            "--lstm24-hidden", lstm24->text().trimmed(),
            "--learning-rates", lrs->text().trimmed(),
            "--batch-sizes", batches->text().trimmed(),
            "--epochs", QString::number(epochs->value()),
            "--seed", QString::number(seed->value())
        };
        if (runGenerator(&dialog, "generate_stage2_sweep.py", args, "hyperparameter_stage2.batch"))
            dialog.accept();
    });
    dialog.exec();
}

struct WinnerWidgets {
    QCheckBox* enabled{};
    QLineEdit* hidden{};
    QDoubleSpinBox* lr{};
    QSpinBox* batch{};
    QSpinBox* sequence{};
};

void showStage3Dialog(QMainWindow* window)
{
    QDialog dialog(window);
    dialog.setWindowTitle("Stage 3 - Multi-seed Robustness");
    dialog.resize(700, 650);
    auto* root = new QVBoxLayout(&dialog);
    auto* intro = new QLabel(
        "Enter the actual Stage-2 winners. Stage 3 freezes those hyperparameters and varies only random seed. "
        "The default seed set is 42, 123, 2026, 31415, 27182.", &dialog);
    intro->setWordWrap(true); root->addWidget(intro);

    auto makeWinner = [&](const QString& title, const QString& hidden, int sequence) {
        WinnerWidgets w;
        auto* box = new QGroupBox(title, &dialog);
        auto* form = new QFormLayout(box);
        w.enabled = new QCheckBox("Include finalist", box); w.enabled->setChecked(true);
        w.hidden = new QLineEdit(hidden, box);
        w.lr = new QDoubleSpinBox(box); w.lr->setDecimals(6); w.lr->setRange(0.000001, 1.0); w.lr->setValue(0.003);
        w.batch = new QSpinBox(box); w.batch->setRange(1, 100000); w.batch->setValue(32);
        w.sequence = new QSpinBox(box); w.sequence->setRange(1, 10000); w.sequence->setValue(sequence);
        form->addRow(w.enabled); form->addRow("Hidden layers", w.hidden); form->addRow("Learning rate", w.lr);
        form->addRow("Batch size", w.batch);
        if (sequence > 0) form->addRow("Sequence length (h)", w.sequence); else w.sequence->hide();
        root->addWidget(box);
        return w;
    };

    WinnerWidgets ffn = makeWinner("FFN winner", "16,16", 0);
    auto* activation = new QLineEdit("relu", &dialog);
    qobject_cast<QFormLayout*>(ffn.enabled->parentWidget()->layout())->addRow("Activation", activation);
    WinnerWidgets lstm1 = makeWinner("LSTM winner 1", "32", 12);
    WinnerWidgets lstm2 = makeWinner("LSTM winner 2", "32", 24);

    auto* globalBox = new QGroupBox("Robustness settings", &dialog);
    auto* globalForm = new QFormLayout(globalBox);
    auto* seeds = new QLineEdit("42,123,2026,31415,27182", globalBox);
    auto* epochs = new QSpinBox(globalBox); epochs->setRange(1, 100000); epochs->setValue(150);
    globalForm->addRow("Seeds", seeds); globalForm->addRow("Epochs", epochs); root->addWidget(globalBox);

    auto* buttons = new QHBoxLayout();
    auto* cancel = new QPushButton("Cancel", &dialog);
    auto* generate = new QPushButton("Generate Stage 3", &dialog);
    buttons->addStretch(1); buttons->addWidget(cancel); buttons->addWidget(generate); root->addLayout(buttons);
    QObject::connect(cancel, &QPushButton::clicked, &dialog, &QDialog::reject);
    QObject::connect(generate, &QPushButton::clicked, &dialog, [&]() {
        QStringList args{
            "--seeds", seeds->text().trimmed(), "--epochs", QString::number(epochs->value()),
            ffn.enabled->isChecked() ? "--ffn-enabled" : "--no-ffn-enabled",
            "--ffn-hidden", ffn.hidden->text().trimmed(), "--ffn-activation", activation->text().trimmed(),
            "--ffn-lr", QString::number(ffn.lr->value(), 'g', 12), "--ffn-batch", QString::number(ffn.batch->value()),
            lstm1.enabled->isChecked() ? "--lstm1-enabled" : "--no-lstm1-enabled",
            "--lstm1-sequence", QString::number(lstm1.sequence->value()), "--lstm1-hidden", lstm1.hidden->text().trimmed(),
            "--lstm1-lr", QString::number(lstm1.lr->value(), 'g', 12), "--lstm1-batch", QString::number(lstm1.batch->value()),
            lstm2.enabled->isChecked() ? "--lstm2-enabled" : "--no-lstm2-enabled",
            "--lstm2-sequence", QString::number(lstm2.sequence->value()), "--lstm2-hidden", lstm2.hidden->text().trimmed(),
            "--lstm2-lr", QString::number(lstm2.lr->value(), 'g', 12), "--lstm2-batch", QString::number(lstm2.batch->value())
        };
        if (runGenerator(&dialog, "generate_stage3_sweep.py", args, "hyperparameter_stage3.batch")) dialog.accept();
    });
    dialog.exec();
}

void installDialogs()
{
    QMainWindow* window = nullptr;
    for (QWidget* widget : QApplication::topLevelWidgets()) {
        if ((window = qobject_cast<QMainWindow*>(widget))) break;
    }
    if (!window) { QTimer::singleShot(100, [](){ installDialogs(); }); return; }
    QMenu* presets = window->findChild<QMenu*>("HydroSweepPresetsMenu");
    if (!presets) { QTimer::singleShot(100, [](){ installDialogs(); }); return; }
    if (presets->findChild<QAction*>("HydroStage2ConfigureAction")) return;

    QAction* stage2 = new QAction("Stage 2 Configure...", presets);
    stage2->setObjectName("HydroStage2ConfigureAction");
    stage2->setToolTip("Configure Stage-2 learning-rate/batch-size tuning, including an optional sigmoid FFN finalist.");
    QAction* stage3 = new QAction("Stage 3 Multi-seed Robustness...", presets);
    stage3->setObjectName("HydroStage3ConfigureAction");
    stage3->setToolTip("Enter Stage-2 winners and generate the multi-seed robustness sweep.");

    const QList<QAction*> actions = presets->actions();
    QAction* before = actions.isEmpty() ? nullptr : actions.last();
    presets->insertAction(before, stage2);
    presets->insertAction(before, stage3);
    QObject::connect(stage2, &QAction::triggered, window, [window](){ showStage2Dialog(window); });
    QObject::connect(stage3, &QAction::triggered, window, [window](){ showStage3Dialog(window); });
}

void scheduleInstall() { QTimer::singleShot(0, [](){ installDialogs(); }); }
}

Q_COREAPP_STARTUP_FUNCTION(scheduleInstall)
