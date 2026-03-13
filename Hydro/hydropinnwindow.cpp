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
#include <QTabWidget>
#include <QTextBrowser>
#include <QTextEdit>
#include <QVBoxLayout>
#include <QWidget>

#include <QtCharts/QChart>
#include <QtCharts/QChartView>
#include <QtCharts/QLegend>
#include <QtCharts/QLineSeries>
#include <QtCharts/QValueAxis>

#include <algorithm>
#include <cmath>
#include <exception>
#include <fstream>

HydroPINNWindow::HydroPINNWindow(QWidget* parent)
    : QMainWindow(parent), statusLabel_(new QLabel(this)), modeCombo_(new QComboBox(this)),
      runButton_(new QPushButton("Run Selected", this)), runAllButton_(new QPushButton("Run All", this)),
      runFFNButton_(new QPushButton("Run FFN", this)), runFFNPINNButton_(new QPushButton("Run FFN_PINN", this)),
      runLSTMButton_(new QPushButton("Run LSTM", this)), runLSTMPINNButton_(new QPushButton("Run LSTM_PINN", this)),
      logText_(new QTextEdit(this)), chartView_(new QChartView(this)), perfSummaryText_(new QTextBrowser(this)),
      epochsSpin_(new QSpinBox(this)), batchSpin_(new QSpinBox(this)), lrSpin_(new QDoubleSpinBox(this)),
      lambdaSpin_(new QDoubleSpinBox(this)), dataWeightSpin_(new QDoubleSpinBox(this)),
      physicsWeightSpin_(new QDoubleSpinBox(this)), hiddenLayersEdit_(new QLineEdit(this)),
      activationCombo_(new QComboBox(this)), evalCheck_(new QCheckBox("Evaluate test metrics", this)),
      dataSourceCombo_(new QComboBox(this)), csvPathEdit_(new QLineEdit(this)),
      browseCsvButton_(new QPushButton("Browse...", this)), csvXColSpin_(new QSpinBox(this)),
      csvYColSpin_(new QSpinBox(this)), csvHeaderCheck_(new QCheckBox("CSV has header row", this)),
      sampleCountSpin_(new QSpinBox(this)), tStartSpin_(new QDoubleSpinBox(this)),
      tEndSpin_(new QDoubleSpinBox(this)), profileCombo_(new QComboBox(this)),
      generateSyntheticButton_(new QPushButton("Generate Synthetic Data", this)),
      syntheticExportPathEdit_(new QLineEdit(this)),
      browseSyntheticExportButton_(new QPushButton("Browse...", this)),
      runPredictionButton_(new QPushButton("Run Selected", this)), runAllPredictionButton_(new QPushButton("Run All", this)),
      runPredictionFFNButton_(new QPushButton("Run FFN", this)), runPredictionFFNPINNButton_(new QPushButton("Run FFN_PINN", this)),
      runPredictionLSTMButton_(new QPushButton("Run LSTM", this)), runPredictionLSTMPINNButton_(new QPushButton("Run LSTM_PINN", this)),
      configureGAButton_(new QPushButton("Configure GA", this)), startGAButton_(new QPushButton("Start GA", this)),
      stopGAButton_(new QPushButton("Stop GA", this)), refreshPerformanceButton_(new QPushButton("Refresh Assessment", this)),
      clearPlotButton_(new QPushButton("Clear Plot", this)) {
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

    syntheticExportPathEdit_->setPlaceholderText("Optional export path for generated synthetic CSV");
    auto* syntheticExportRow = new QWidget(dataTab);
    auto* syntheticExportLayout = new QHBoxLayout(syntheticExportRow);
    syntheticExportLayout->setContentsMargins(0, 0, 0, 0);
    syntheticExportLayout->addWidget(syntheticExportPathEdit_, 1);
    syntheticExportLayout->addWidget(browseSyntheticExportButton_);

    dataForm->addRow("Synthetic export", syntheticExportRow);
    dataForm->addRow(generateSyntheticButton_);
    tabs->addTab(dataTab, "Data");

    auto* networkTab = new QWidget(tabs);
    auto* networkForm = new QFormLayout(networkTab);
    hiddenLayersEdit_->setText("24,24");
    networkForm->addRow("Hidden layers (csv)", hiddenLayersEdit_);
    networkForm->addRow("Activation", activationCombo_);
    tabs->addTab(networkTab, "Network Structure");

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
    evalCheck_->setChecked(true);
    trainForm->addRow("Epochs", epochsSpin_);
    trainForm->addRow("Batch size", batchSpin_);
    trainForm->addRow("Learning rate", lrSpin_);
    trainForm->addRow("Lambda (PINN)", lambdaSpin_);
    trainForm->addRow("Data loss weight", dataWeightSpin_);
    trainForm->addRow("Physics loss weight", physicsWeightSpin_);
    tabs->addTab(trainTab, "Training");

    auto* predictionTab = new QWidget(tabs);
    auto* predictionLayout = new QVBoxLayout(predictionTab);
    auto* predictionButtons = new QGroupBox("Prediction Actions", predictionTab);
    auto* predictionGrid = new QGridLayout(predictionButtons);
    predictionGrid->addWidget(runPredictionButton_, 0, 0);
    predictionGrid->addWidget(runAllPredictionButton_, 0, 1);
    predictionGrid->addWidget(runPredictionFFNButton_, 1, 0);
    predictionGrid->addWidget(runPredictionFFNPINNButton_, 1, 1);
    predictionGrid->addWidget(runPredictionLSTMButton_, 2, 0);
    predictionGrid->addWidget(runPredictionLSTMPINNButton_, 2, 1);
    predictionLayout->addWidget(predictionButtons);
    predictionLayout->addStretch(1);
    tabs->addTab(predictionTab, "Prediction");

    auto* gaTab = new QWidget(tabs);
    auto* gaLayout = new QVBoxLayout(gaTab);
    auto* gaBox = new QGroupBox("Genetic Algorithm (workflow-compatible)", gaTab);
    auto* gaButtonLayout = new QHBoxLayout(gaBox);
    gaButtonLayout->addWidget(configureGAButton_);
    gaButtonLayout->addWidget(startGAButton_);
    gaButtonLayout->addWidget(stopGAButton_);
    stopGAButton_->setEnabled(false);
    gaLayout->addWidget(gaBox);
    gaLayout->addWidget(new QLabel("Hydro modes keep current 4-mode training flow; GA controls are prepared for future Hydro-specific optimization hooks.", gaTab));
    gaLayout->addStretch(1);
    tabs->addTab(gaTab, "GA");

    auto* performanceTab = new QWidget(tabs);
    auto* performanceLayout = new QVBoxLayout(performanceTab);
    performanceLayout->addWidget(evalCheck_);
    performanceLayout->addWidget(refreshPerformanceButton_);
    perfSummaryText_->setPlaceholderText("Performance assessment summary appears here after runs.");
    performanceLayout->addWidget(perfSummaryText_, 1);
    tabs->addTab(performanceTab, "Performance Assessment");

    auto* plotTab = new QWidget(tabs);
    auto* plotLayout = new QVBoxLayout(plotTab);
    plotLayout->addWidget(clearPlotButton_, 0);
    plotLayout->addWidget(chartView_, 1);
    tabs->addTab(plotTab, "Plot");

    auto* logTab = new QWidget(tabs);
    auto* logLayout = new QVBoxLayout(logTab);
    logText_->setReadOnly(true);
    logText_->setPlaceholderText("Run logs will appear here...");
    logLayout->addWidget(logText_, 1);
    tabs->addTab(logTab, "Logs");

    auto* topRow = new QHBoxLayout();
    topRow->addWidget(modeCombo_, 1);

    auto* chart = new QChart();
    chart->setTitle("Prediction vs Target (Test Set)");
    chartView_->setChart(chart);
    chartView_->setRenderHint(QPainter::Antialiasing);
    chartView_->setMinimumHeight(260);

    root->addWidget(title);
    root->addLayout(topRow);
    root->addWidget(tabs);
    root->addWidget(statusLabel_);

    setCentralWidget(central);

    connect(runButton_, &QPushButton::clicked, this, &HydroPINNWindow::runSelectedMode);
    connect(runAllButton_, &QPushButton::clicked, this, &HydroPINNWindow::runAllModes);
    connect(runFFNButton_, &QPushButton::clicked, this, [this]() { runMode("ffn"); });
    connect(runFFNPINNButton_, &QPushButton::clicked, this, [this]() { runMode("ffn_pinn"); });
    connect(runLSTMButton_, &QPushButton::clicked, this, [this]() { runMode("lstm"); });
    connect(runLSTMPINNButton_, &QPushButton::clicked, this, [this]() { runMode("lstm_pinn"); });
    connect(runPredictionButton_, &QPushButton::clicked, this, &HydroPINNWindow::runSelectedMode);
    connect(runAllPredictionButton_, &QPushButton::clicked, this, &HydroPINNWindow::runAllModes);
    connect(runPredictionFFNButton_, &QPushButton::clicked, this, [this]() { runMode("ffn"); });
    connect(runPredictionFFNPINNButton_, &QPushButton::clicked, this, [this]() { runMode("ffn_pinn"); });
    connect(runPredictionLSTMButton_, &QPushButton::clicked, this, [this]() { runMode("lstm"); });
    connect(runPredictionLSTMPINNButton_, &QPushButton::clicked, this, [this]() { runMode("lstm_pinn"); });
    connect(modeCombo_, &QComboBox::currentTextChanged, this, [this](const QString&) { updateStatus(); });
    connect(dataSourceCombo_, &QComboBox::currentTextChanged, this, [this](const QString&) {
        updateDataSourceUiState();
        updateStatus();
    });
    connect(browseCsvButton_, &QPushButton::clicked, this, &HydroPINNWindow::browseCsv);
    connect(browseSyntheticExportButton_, &QPushButton::clicked, this, &HydroPINNWindow::browseSyntheticExportPath);
    connect(generateSyntheticButton_, &QPushButton::clicked, this, &HydroPINNWindow::generateSyntheticDataPreview);
    connect(configureGAButton_, &QPushButton::clicked, this, &HydroPINNWindow::configureGAPlaceholder);
    connect(startGAButton_, &QPushButton::clicked, this, &HydroPINNWindow::startGAPlaceholder);
    connect(stopGAButton_, &QPushButton::clicked, this, &HydroPINNWindow::stopGAPlaceholder);
    connect(refreshPerformanceButton_, &QPushButton::clicked, this, &HydroPINNWindow::refreshPerformanceAssessment);
    connect(clearPlotButton_, &QPushButton::clicked, this, &HydroPINNWindow::clearPlot);

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
    generateSyntheticButton_->setEnabled(!running && dataSourceCombo_->currentText() != "CSV File");
    syntheticExportPathEdit_->setEnabled(!running && dataSourceCombo_->currentText() != "CSV File");
    browseSyntheticExportButton_->setEnabled(!running && dataSourceCombo_->currentText() != "CSV File");
    runButton_->setText(running ? "Running..." : "Run Selected");
    runPredictionButton_->setEnabled(!running);
    runAllPredictionButton_->setEnabled(!running);
    runPredictionFFNButton_->setEnabled(!running);
    runPredictionFFNPINNButton_->setEnabled(!running);
    runPredictionLSTMButton_->setEnabled(!running);
    runPredictionLSTMPINNButton_->setEnabled(!running);
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
    generateSyntheticButton_->setEnabled(!useCsv);
    syntheticExportPathEdit_->setEnabled(!useCsv);
    browseSyntheticExportButton_->setEnabled(!useCsv);
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


void HydroPINNWindow::browseSyntheticExportPath() {
    const QString path = QFileDialog::getSaveFileName(this,
                                                      "Save generated synthetic CSV",
                                                      QString(),
                                                      "CSV Files (*.csv);;All Files (*)");
    if (!path.isEmpty()) {
        syntheticExportPathEdit_->setText(path);
    }
}

void HydroPINNWindow::generateSyntheticDataPreview() {
    if (dataSourceCombo_->currentText() == "CSV File") {
        QMessageBox::information(this, "HydroPINN", "Synthetic preview is available only when Data source is set to Synthetic.");
        return;
    }

    const int samples = sampleCountSpin_->value();
    const double tStart = tStartSpin_->value();
    const double tEnd = tEndSpin_->value();
    const QString profile = profileCombo_->currentText();

    if (samples < 2 || tEnd <= tStart) {
        QMessageBox::warning(this, "HydroPINN", "Invalid synthetic range/sample settings.");
        return;
    }

    std::vector<double> xs;
    std::vector<double> ys;
    xs.reserve(static_cast<size_t>(samples));
    ys.reserve(static_cast<size_t>(samples));

    for (int i = 0; i < samples; ++i) {
        const double r = static_cast<double>(i) / static_cast<double>(samples - 1);
        const double t = tStart + (tEnd - tStart) * r;
        double y = 0.0;
        if (profile == "damped_sine") {
            y = std::sin(t) * std::exp(-0.15 * t);
        } else if (profile == "mixed_wave") {
            y = 0.7 * std::sin(1.5 * t) + 0.3 * std::cos(0.5 * t);
        } else {
            y = std::exp(-0.8 * t);
        }
        xs.push_back(t);
        ys.push_back(y);
    }

    HydroRunResult preview;
    preview.success = true;
    preview.message = "Synthetic preview generated.";
    preview.x = xs;
    preview.y_true = ys;
    preview.y_pred = ys;
    updatePlot("synthetic_preview", preview);

    appendLog(QString("Generated synthetic preview: profile=%1, samples=%2, range=[%3,%4]")
                  .arg(profile)
                  .arg(samples)
                  .arg(tStart, 0, 'g', 6)
                  .arg(tEnd, 0, 'g', 6));

    const QString outPath = syntheticExportPathEdit_->text().trimmed();
    if (!outPath.isEmpty()) {
        std::ofstream out(outPath.toStdString());
        if (!out.is_open()) {
            QMessageBox::warning(this, "HydroPINN", QString("Failed to open export file: %1").arg(outPath));
            return;
        }
        out << "t,y\n";
        for (int i = 0; i < samples; ++i) {
            out << xs[static_cast<size_t>(i)] << "," << ys[static_cast<size_t>(i)] << "\n";
        }
        out.close();
        appendLog(QString("Synthetic data exported to: %1").arg(outPath));
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

void HydroPINNWindow::configureGAPlaceholder() {
    appendLog("GA configuration requested (Hydro GA backend is not implemented yet).");
    QMessageBox::information(this,
                             "HydroPINN GA",
                             "GA controls are available in the workflow, but Hydro-specific GA optimization is not wired yet.");
}

void HydroPINNWindow::startGAPlaceholder() {
    appendLog("GA start requested (placeholder).");
    startGAButton_->setEnabled(false);
    stopGAButton_->setEnabled(true);
    statusLabel_->setText("GA placeholder run started (no backend yet).");
}

void HydroPINNWindow::stopGAPlaceholder() {
    appendLog("GA stop requested (placeholder).");
    startGAButton_->setEnabled(true);
    stopGAButton_->setEnabled(false);
    updateStatus();
}

void HydroPINNWindow::refreshPerformanceAssessment() {
    const HydroRunConfig cfg = currentConfig();
    const QString summary = QString(
                                "<b>Performance Assessment Snapshot</b><br/>"
                                "Mode: %1<br/>"
                                "Data source: %2<br/>"
                                "Evaluate metrics: %3<br/>"
                                "Training: epochs=%4, batch=%5, lr=%6<br/>"
                                "PINN: lambda=%7, data_w=%8, physics_w=%9<br/>"
                                "Network: layers=%10, activation=%11")
                                .arg(modeCombo_->currentText())
                                .arg(cfg.use_csv_data ? "CSV" : "Synthetic")
                                .arg(cfg.evaluate_metrics ? "yes" : "no")
                                .arg(cfg.epochs)
                                .arg(cfg.batch_size)
                                .arg(cfg.learning_rate, 0, 'g', 6)
                                .arg(cfg.lambda_decay, 0, 'g', 6)
                                .arg(cfg.data_weight, 0, 'g', 6)
                                .arg(cfg.physics_weight, 0, 'g', 6)
                                .arg(QString::fromStdString(cfg.hidden_layers_csv))
                                .arg(QString::fromStdString(cfg.activation));
    perfSummaryText_->setHtml(summary);
    appendLog("Performance assessment snapshot refreshed.");
}

void HydroPINNWindow::clearPlot() {
    auto* chart = chartView_->chart();
    chart->removeAllSeries();
    chart->setTitle("Prediction vs Target (Test Set)");
    appendLog("Plot cleared.");
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
