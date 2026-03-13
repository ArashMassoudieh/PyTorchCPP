#include "hydropinnwindow.h"

#include "models/ffn_wrapper.h"
#include "models/ffn_pinn_wrapper.h"
#include "models/lstm_wrapper.h"
#include "models/lstm_pinn_wrapper.h"

#include <QCheckBox>
#include <QComboBox>
#include <QCoreApplication>
#include <QDateTime>
#include <QDoubleSpinBox>
#include <QElapsedTimer>
#include <QFileDialog>
#include <QFormLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPushButton>
#include <QSpinBox>
#include <QSplitter>
#include <QTabWidget>
#include <QTextEdit>
#include <QVBoxLayout>
#include <QWidget>

#include <QtCharts/QChart>
#include <QtCharts/QChartView>
#include <QtCharts/QLegend>
#include <QtCharts/QLineSeries>
#include <QtCharts/QValueAxis>

#include <algorithm>
#include <exception>

HydroPINNWindow::HydroPINNWindow(QWidget* parent)
    : QMainWindow(parent), statusLabel_(new QLabel(this)), modeCombo_(new QComboBox(this)),
      runButton_(new QPushButton("Run Selected", this)), runAllButton_(new QPushButton("Run All", this)),
      runFFNButton_(new QPushButton("Run FFN", this)), runFFNPINNButton_(new QPushButton("Run FFN_PINN", this)),
      runLSTMButton_(new QPushButton("Run LSTM", this)), runLSTMPINNButton_(new QPushButton("Run LSTM_PINN", this)),
      logText_(new QTextEdit(this)), chartView_(new QChartView(this)),
      epochsSpin_(new QSpinBox(this)), batchSpin_(new QSpinBox(this)), lrSpin_(new QDoubleSpinBox(this)),
      lambdaSpin_(new QDoubleSpinBox(this)), dataWeightSpin_(new QDoubleSpinBox(this)),
      physicsWeightSpin_(new QDoubleSpinBox(this)), hiddenLayersEdit_(new QLineEdit(this)),
      activationCombo_(new QComboBox(this)), evalCheck_(new QCheckBox("Evaluate test metrics", this)),
      dataSourceCombo_(new QComboBox(this)), csvPathEdit_(new QLineEdit(this)),
      browseCsvButton_(new QPushButton("Browse...", this)), csvXColSpin_(new QSpinBox(this)),
      csvYColSpin_(new QSpinBox(this)), csvHeaderCheck_(new QCheckBox("CSV has header row", this)),
      sampleCountSpin_(new QSpinBox(this)), tStartSpin_(new QDoubleSpinBox(this)),
      tEndSpin_(new QDoubleSpinBox(this)), profileCombo_(new QComboBox(this)) {
    setWindowTitle("HydroPINN - Experiment Runner");
    resize(1200, 760);

    auto* central = new QWidget(this);
    auto* root = new QVBoxLayout(central);

    auto* title = new QLabel("HydroPINN Modes", central);
    title->setStyleSheet("font-size: 18px; font-weight: bold;");

    modeCombo_->addItems({"ffn", "ffn_pinn", "lstm", "lstm_pinn"});
    activationCombo_->addItems({"relu", "tanh", "sigmoid"});
    dataSourceCombo_->addItems({"Synthetic", "CSV File"});
    profileCombo_->addItems({"exp_decay", "damped_sine", "mixed_wave"});

    auto* tabs = new QTabWidget(central);

    auto* trainTab = new QWidget(tabs);
    auto* trainForm = new QFormLayout(trainTab);
    epochsSpin_->setRange(1, 20000);
    epochsSpin_->setValue(180);
    batchSpin_->setRange(1, 4096);
    batchSpin_->setValue(32);
    lrSpin_->setDecimals(6);
    lrSpin_->setRange(1e-6, 1.0);
    lrSpin_->setSingleStep(0.001);
    lrSpin_->setValue(0.002);
    lambdaSpin_->setDecimals(4);
    lambdaSpin_->setRange(0.0, 100.0);
    lambdaSpin_->setValue(0.8);
    dataWeightSpin_->setDecimals(4);
    dataWeightSpin_->setRange(0.0, 100.0);
    dataWeightSpin_->setValue(1.0);
    physicsWeightSpin_->setDecimals(4);
    physicsWeightSpin_->setRange(0.0, 100.0);
    physicsWeightSpin_->setValue(0.2);
    hiddenLayersEdit_->setText("24,24");
    evalCheck_->setChecked(true);
    trainForm->addRow("Epochs", epochsSpin_);
    trainForm->addRow("Batch size", batchSpin_);
    trainForm->addRow("Learning rate", lrSpin_);
    trainForm->addRow("Lambda (PINN)", lambdaSpin_);
    trainForm->addRow("Data loss weight", dataWeightSpin_);
    trainForm->addRow("Physics loss weight", physicsWeightSpin_);
    trainForm->addRow("Hidden layers (csv)", hiddenLayersEdit_);
    trainForm->addRow("Activation", activationCombo_);
    trainForm->addRow(evalCheck_);
    tabs->addTab(trainTab, "Training");

    auto* dataTab = new QWidget(tabs);
    auto* dataForm = new QFormLayout(dataTab);
    sampleCountSpin_->setRange(32, 20000);
    sampleCountSpin_->setValue(240);
    tStartSpin_->setDecimals(3);
    tStartSpin_->setRange(-1000.0, 1000.0);
    tStartSpin_->setValue(0.0);
    tEndSpin_->setDecimals(3);
    tEndSpin_->setRange(-1000.0, 1000.0);
    tEndSpin_->setValue(5.0);

    csvPathEdit_->setPlaceholderText("Select CSV path with x/y columns");
    auto* csvPathRow = new QWidget(dataTab);
    auto* csvPathLayout = new QHBoxLayout(csvPathRow);
    csvPathLayout->setContentsMargins(0, 0, 0, 0);
    csvPathLayout->addWidget(csvPathEdit_, 1);
    csvPathLayout->addWidget(browseCsvButton_);

    csvXColSpin_->setRange(0, 100);
    csvYColSpin_->setRange(0, 100);
    csvXColSpin_->setValue(0);
    csvYColSpin_->setValue(1);
    csvHeaderCheck_->setChecked(true);

    dataForm->addRow("Data source", dataSourceCombo_);
    dataForm->addRow("Synthetic profile", profileCombo_);
    dataForm->addRow("Synthetic sample count", sampleCountSpin_);
    dataForm->addRow("Synthetic t_start", tStartSpin_);
    dataForm->addRow("Synthetic t_end", tEndSpin_);
    dataForm->addRow("CSV file", csvPathRow);
    dataForm->addRow("CSV x column (0-based)", csvXColSpin_);
    dataForm->addRow("CSV y column (0-based)", csvYColSpin_);
    dataForm->addRow(csvHeaderCheck_);
    tabs->addTab(dataTab, "Data");

    // Button panel
    auto* btnBox = new QGroupBox("Actions", central);
    auto* btnGrid = new QGridLayout(btnBox);
    btnGrid->addWidget(runButton_, 0, 0);
    btnGrid->addWidget(runAllButton_, 0, 1);
    btnGrid->addWidget(runFFNButton_, 1, 0);
    btnGrid->addWidget(runFFNPINNButton_, 1, 1);
    btnGrid->addWidget(runLSTMButton_, 2, 0);
    btnGrid->addWidget(runLSTMPINNButton_, 2, 1);

    auto* topRow = new QHBoxLayout();
    topRow->addWidget(modeCombo_, 1);

    logText_->setReadOnly(true);
    logText_->setPlaceholderText("Run logs will appear here...");

    auto* chart = new QChart();
    chart->setTitle("Prediction vs Target (Test Set)");
    chartView_->setChart(chart);
    chartView_->setRenderHint(QPainter::Antialiasing);
    chartView_->setMinimumHeight(260);

    auto* splitter = new QSplitter(Qt::Vertical, central);
    auto* lower = new QWidget(splitter);
    auto* lowerLayout = new QVBoxLayout(lower);
    lowerLayout->addWidget(statusLabel_);
    lowerLayout->addWidget(logText_);
    splitter->addWidget(chartView_);
    splitter->addWidget(lower);
    splitter->setStretchFactor(0, 2);
    splitter->setStretchFactor(1, 1);

    root->addWidget(title);
    root->addLayout(topRow);
    root->addWidget(tabs);
    root->addWidget(btnBox);
    root->addWidget(splitter, 1);

    setCentralWidget(central);

    connect(runButton_, &QPushButton::clicked, this, &HydroPINNWindow::runSelectedMode);
    connect(runAllButton_, &QPushButton::clicked, this, &HydroPINNWindow::runAllModes);
    connect(runFFNButton_, &QPushButton::clicked, this, [this]() { runMode("ffn"); });
    connect(runFFNPINNButton_, &QPushButton::clicked, this, [this]() { runMode("ffn_pinn"); });
    connect(runLSTMButton_, &QPushButton::clicked, this, [this]() { runMode("lstm"); });
    connect(runLSTMPINNButton_, &QPushButton::clicked, this, [this]() { runMode("lstm_pinn"); });
    connect(modeCombo_, &QComboBox::currentTextChanged, this, [this](const QString&) { updateStatus(); });
    connect(dataSourceCombo_, &QComboBox::currentTextChanged, this, [this](const QString&) {
        updateDataSourceUiState();
        updateStatus();
    });
    connect(browseCsvButton_, &QPushButton::clicked, this, &HydroPINNWindow::browseCsv);

    updateDataSourceUiState();
    updateStatus();
    appendLog("HydroPINN ready.");
    appendLog("Use Data tab: choose Synthetic generator options or provide a CSV file path.");
}

HydroRunConfig HydroPINNWindow::currentConfig() const {
    HydroRunConfig cfg;
    cfg.epochs = epochsSpin_->value();
    cfg.batch_size = batchSpin_->value();
    cfg.learning_rate = lrSpin_->value();
    cfg.lambda_decay = lambdaSpin_->value();
    cfg.data_weight = dataWeightSpin_->value();
    cfg.physics_weight = physicsWeightSpin_->value();
    cfg.use_csv_data = (dataSourceCombo_->currentText() == "CSV File");
    cfg.csv_path = csvPathEdit_->text().toStdString();
    cfg.csv_x_column = csvXColSpin_->value();
    cfg.csv_y_column = csvYColSpin_->value();
    cfg.csv_has_header = csvHeaderCheck_->isChecked();
    cfg.sample_count = sampleCountSpin_->value();
    cfg.t_start = tStartSpin_->value();
    cfg.t_end = tEndSpin_->value();
    cfg.hidden_layers_csv = hiddenLayersEdit_->text().toStdString();
    cfg.activation = activationCombo_->currentText().toStdString();
    cfg.synthetic_profile = profileCombo_->currentText().toStdString();
    cfg.evaluate_metrics = evalCheck_->isChecked();
    return cfg;
}

void HydroPINNWindow::setRunningUiState(bool running) {
    runButton_->setEnabled(!running);
    runAllButton_->setEnabled(!running);
    runFFNButton_->setEnabled(!running);
    runFFNPINNButton_->setEnabled(!running);
    runLSTMButton_->setEnabled(!running);
    runLSTMPINNButton_->setEnabled(!running);
    dataSourceCombo_->setEnabled(!running);
    browseCsvButton_->setEnabled(!running && dataSourceCombo_->currentText() == "CSV File");
    runButton_->setText(running ? "Running..." : "Run Selected");
}

void HydroPINNWindow::updateDataSourceUiState() {
    const bool useCsv = (dataSourceCombo_->currentText() == "CSV File");
    csvPathEdit_->setEnabled(useCsv);
    browseCsvButton_->setEnabled(useCsv);
    csvXColSpin_->setEnabled(useCsv);
    csvYColSpin_->setEnabled(useCsv);
    csvHeaderCheck_->setEnabled(useCsv);

    profileCombo_->setEnabled(!useCsv);
    sampleCountSpin_->setEnabled(!useCsv);
    tStartSpin_->setEnabled(!useCsv);
    tEndSpin_->setEnabled(!useCsv);
}

void HydroPINNWindow::browseCsv() {
    const QString path = QFileDialog::getOpenFileName(this,
                                                      "Select input CSV",
                                                      QString(),
                                                      "CSV Files (*.csv);;All Files (*)");
    if (!path.isEmpty()) {
        csvPathEdit_->setText(path);
    }
}

void HydroPINNWindow::appendLog(const QString& line) {
    const QString ts = QDateTime::currentDateTime().toString("hh:mm:ss");
    logText_->append(QString("[%1] %2").arg(ts, line));
}

void HydroPINNWindow::updateStatus() {
    const QString source = (dataSourceCombo_->currentText() == "CSV File") ? "CSV" : "Synthetic";
    statusLabel_->setText(QString("Ready: mode=%1, data=%2").arg(modeCombo_->currentText(), source));
}

void HydroPINNWindow::runSelectedMode() {
    runMode(modeCombo_->currentText());
}

void HydroPINNWindow::runAllModes() {
    const QStringList modes = {"ffn", "ffn_pinn", "lstm", "lstm_pinn"};
    for (const QString& m : modes) {
        runMode(m);
    }
}

void HydroPINNWindow::updatePlot(const QString& mode, const HydroRunResult& result) {
    if (result.x.empty() || result.y_true.empty() || result.y_pred.empty()) {
        return;
    }

    auto* chart = chartView_->chart();
    chart->removeAllSeries();

    auto* trueSeries = new QLineSeries(chart);
    trueSeries->setName("Target");
    auto* predSeries = new QLineSeries(chart);
    predSeries->setName("Prediction");

    const size_t n = std::min(result.x.size(), std::min(result.y_true.size(), result.y_pred.size()));
    for (size_t i = 0; i < n; ++i) {
        trueSeries->append(result.x[i], result.y_true[i]);
        predSeries->append(result.x[i], result.y_pred[i]);
    }

    chart->addSeries(trueSeries);
    chart->addSeries(predSeries);

    auto* axisX = new QValueAxis(chart);
    axisX->setTitleText("t");
    auto* axisY = new QValueAxis(chart);
    axisY->setTitleText("y");

    chart->addAxis(axisX, Qt::AlignBottom);
    chart->addAxis(axisY, Qt::AlignLeft);
    trueSeries->attachAxis(axisX);
    trueSeries->attachAxis(axisY);
    predSeries->attachAxis(axisX);
    predSeries->attachAxis(axisY);

    chart->setTitle(QString("Prediction vs Target - %1").arg(mode));
    chart->legend()->setVisible(true);
}

void HydroPINNWindow::runMode(const QString& mode) {
    appendLog(QString("Starting mode: %1").arg(mode));
    setRunningUiState(true);
    statusLabel_->setText(QString("Running mode: %1 ...").arg(mode));
    appendLog("Dispatch started.");

    HydroRunConfig cfg = currentConfig();
    if (cfg.use_csv_data) {
        appendLog(QString("Using CSV data: %1 (x_col=%2, y_col=%3, header=%4)")
                      .arg(QString::fromStdString(cfg.csv_path))
                      .arg(cfg.csv_x_column)
                      .arg(cfg.csv_y_column)
                      .arg(cfg.csv_has_header ? "yes" : "no"));
    } else {
        appendLog(QString("Using synthetic generator: profile=%1, samples=%2, range=[%3,%4]")
                      .arg(QString::fromStdString(cfg.synthetic_profile))
                      .arg(cfg.sample_count)
                      .arg(cfg.t_start, 0, 'g', 6)
                      .arg(cfg.t_end, 0, 'g', 6));
    }

    QCoreApplication::processEvents();

    QElapsedTimer timer;
    timer.start();

    HydroRunResult result;
    QString errorDetails;

    try {
        if (mode == "ffn") {
            FFNWrapper runner;
            result = runner.train(cfg);
        } else if (mode == "ffn_pinn") {
            FFNPINNWrapper runner;
            result = runner.train(cfg);
        } else if (mode == "lstm") {
            LSTMWrapper runner;
            result = runner.train(cfg);
        } else if (mode == "lstm_pinn") {
            LSTMPINNWrapper runner;
            result = runner.train(cfg);
        } else {
            result.success = false;
            errorDetails = QString("Unknown mode selected: %1").arg(mode);
        }
    } catch (const std::exception& e) {
        result.success = false;
        errorDetails = QString("Exception: %1").arg(e.what());
    } catch (...) {
        result.success = false;
        errorDetails = "Unknown non-std exception during mode execution.";
    }

    const qint64 elapsedMs = timer.elapsed();
    if (result.success) {
        statusLabel_->setText(QString("Completed mode: %1 (%2 ms)").arg(mode).arg(elapsedMs));
        appendLog(QString("Mode '%1' finished successfully in %2 ms.").arg(mode).arg(elapsedMs));
        appendLog(QString("  final_loss=%1, mse=%2, msg=%3")
                      .arg(result.final_loss, 0, 'g', 8)
                      .arg(result.mse, 0, 'g', 8)
                      .arg(QString::fromStdString(result.message)));
        updatePlot(mode, result);
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

    setRunningUiState(false);
}
