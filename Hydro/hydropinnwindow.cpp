#include "hydropinnwindow.h"

#include "models/ffn_wrapper.h"
#include "models/ffn_pinn_wrapper.h"
#include "models/pinn_wrapper.h"
#include "models/lstm_wrapper.h"
#include "models/lstm_pinn_wrapper.h"
#include "evaluation/experiment_exporter.h"
#include "evaluation/experiment_loader.h"
#include "evaluation/hydro_metrics.h"
#include "dataset/hydro_tensor_builder.h"
#include "dataset/lagged_tensor_builder.h"
#include "dataset/csv_tensor_builder.h"

#include <QCheckBox>
#include <QComboBox>
#include <QCoreApplication>
#include <QDateTime>
#include <QDialog>
#include <QDoubleSpinBox>
#include <QElapsedTimer>
#include <QFileDialog>
#include <QFormLayout>
#include <QFrame>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMessageBox>
#include <QPen>
#include <QPushButton>
#include <QScrollArea>
#include <QStringList>
#include <QSpinBox>
#include <QTabWidget>
#include <QTextBrowser>
#include <QTextDocument>
#include <QTextEdit>
#include <QVBoxLayout>
#include <QWidget>

#include <QtCharts/QChart>
#include <QtCharts/QAbstractSeries>
#include <QtCharts/QChartView>
#include <QtCharts/QLegend>
#include <QtCharts/QLineSeries>
#include <QtCharts/QScatterSeries>
#include <QtCharts/QAbstractAxis>
#include <QtCharts/QValueAxis>
#include <QtCharts/QXYSeries>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <functional>
#include <limits>
#include <random>
#include <set>

namespace {
QString parseLayerActivationText(const QString& layerText) {
    const int comma = layerText.lastIndexOf(',');
    if (comma < 0) return QString();
    return layerText.mid(comma + 1).trimmed();
}

QString modeDisplayName(const QString& mode) {
    if (mode == "ffn") return "FFN";
    if (mode == "ffn_pinn") return "FFN + PINN";
    if (mode == "pinn") return "PINN";
    if (mode == "lstm") return "LSTM";
    if (mode == "lstm_pinn") return "LSTM + PINN";
    return mode;
}
}

HydroPINNWindow::HydroPINNWindow(QWidget* parent)
    : QMainWindow(parent), statusLabel_(new QLabel(this)), modeCombo_(new QComboBox(this)),
      logText_(new QTextEdit(this)), chartView_(new QChartView(this)), perfSummaryText_(new QTextBrowser(this)),
      epochsSpin_(new QSpinBox(this)), batchSpin_(new QSpinBox(this)), lrSpin_(new QDoubleSpinBox(this)),
      lambdaSpin_(new QDoubleSpinBox(this)), dataWeightSpin_(new QDoubleSpinBox(this)),
      physicsWeightSpin_(new QDoubleSpinBox(this)), pinnPhysicsProfileCombo_(new QComboBox(this)),
      forcingGainSpin_(new QDoubleSpinBox(this)), pinnCollocationSpin_(new QSpinBox(this)), hiddenLayersEdit_(new QLineEdit(this)),
      inputLagsEdit_(new QLineEdit(this)), useTimeLaggedFFNCheck_(new QCheckBox("Use time-lagged FFN inputs", this)),
      activationCombo_(new QComboBox(this)), layerSizeSpin_(new QSpinBox(this)), layerActivationCombo_(new QComboBox(this)),
      addLayerButton_(new QPushButton("Add Layer", this)), removeLayerButton_(new QPushButton("Remove Selected", this)),
      layersList_(new QListWidget(this)), outputActivationCombo_(new QComboBox(this)),
      evalCheck_(new QCheckBox("Evaluate test metrics", this)), splitRatioSpin_(new QDoubleSpinBox(this)),
      shuffleCheck_(new QCheckBox("Shuffle training", this)), seedSpin_(new QSpinBox(this)), optimizerCombo_(new QComboBox(this)),
      weightDecaySpin_(new QDoubleSpinBox(this)), momentumSpin_(new QDoubleSpinBox(this)), normalizationCombo_(new QComboBox(this)),
      incrementalCheck_(new QCheckBox("Use incremental / rolling-window training", this)), windowSizeSpin_(new QDoubleSpinBox(this)),
      windowStepSpin_(new QDoubleSpinBox(this)), epochsPerWindowSpin_(new QSpinBox(this)),
      resetOptimizerWindowCheck_(new QCheckBox("Reset optimizer on new window", this)),
      dataSourceCombo_(new QComboBox(this)), csvPathEdit_(new QLineEdit(this)),
      browseCsvButton_(new QPushButton("Browse...", this)), csvXColSpin_(new QSpinBox(this)),
      csvYColSpin_(new QSpinBox(this)), csvHeaderCheck_(new QCheckBox("CSV has header row", this)),
      useNeuroforgeCsvPresetButton_(new QPushButton("Use NeuroForge CSV Preset (x=t, y=target)", this)),
      hydroPackagePathEdit_(new QLineEdit(this)), browseHydroPackageButton_(new QPushButton("Browse...", this)),
      hydroCatchmentIdEdit_(new QLineEdit(this)), hydroPackageProfileCombo_(new QComboBox(this)),
      hydroForecastFeatureCheck_(new QCheckBox("Append leakage-safe forecast feature", this)),
      hydroForecastVariableEdit_(new QLineEdit(this)), hydroForecastLeadSpin_(new QDoubleSpinBox(this)),
      hydroForecastEnsembleEdit_(new QLineEdit(this)),
      sampleCountSpin_(new QSpinBox(this)), tStartSpin_(new QDoubleSpinBox(this)),
      tEndSpin_(new QDoubleSpinBox(this)), profileCombo_(new QComboBox(this)),
      generateSyntheticButton_(new QPushButton("Generate Synthetic Data", this)),
      syntheticExportPathEdit_(new QLineEdit(this)),
      browseSyntheticExportButton_(new QPushButton("Browse...", this)),
      runPredictionButton_(new QPushButton("Run Selected", this)), runAllPredictionButton_(new QPushButton("Run All", this)),
      runPredictionFFNButton_(new QPushButton("Run FFN", this)), runPredictionFFNPINNButton_(new QPushButton("Run FFN + PINN", this)),
      runPredictionPINNButton_(new QPushButton("Run PINN", this)),
      runPredictionLSTMButton_(new QPushButton("Run LSTM", this)), runPredictionLSTMPINNButton_(new QPushButton("Run LSTM + PINN", this)),
      loadInferenceArtifactsButton_(new QPushButton("Load Inference Artifacts...", this)),
      predictionUseCurrentDataCheck_(new QCheckBox("Run loaded checkpoint on current Data source", this)),
      runTrainingButton_(new QPushButton("Train Selected", this)), runAllTrainingButton_(new QPushButton("Train All", this)),
      runTrainingFFNButton_(new QPushButton("Train FFN", this)), runTrainingFFNPINNButton_(new QPushButton("Train FFN + PINN", this)),
      runTrainingPINNButton_(new QPushButton("Train PINN", this)),
      runTrainingLSTMButton_(new QPushButton("Train LSTM", this)), runTrainingLSTMPINNButton_(new QPushButton("Train LSTM + PINN", this)),
      gaLagCandidatesSpin_(new QSpinBox(this)), gaMaxLagSpin_(new QSpinBox(this)), configureGAButton_(new QPushButton("Configure GA", this)), startGAButton_(new QPushButton("Start GA", this)),
      stopGAButton_(new QPushButton("Stop GA", this)), refreshPerformanceButton_(new QPushButton("Refresh Assessment", this)),
      exportExperimentButton_(new QPushButton("Export Experiment...", this)),
      loadExperimentConfigButton_(new QPushButton("Load Experiment Config...", this)),
      clearPlotButton_(new QPushButton("Clear Plot", this)), showInputsOutputsButton_(new QPushButton("Show Inputs + Output", this)),
      zoomInPlotButton_(new QPushButton("Zoom In", this)), zoomOutPlotButton_(new QPushButton("Zoom Out", this)),
      fitPlotButton_(new QPushButton("Fit Axes", this)),
      plotAllTargetPredButton_(new QPushButton("Target vs Predicted (All)", this)),
      plotOneToOneButton_(new QPushButton("1:1 Target vs Predicted (All)", this)),
      plotTaylorButton_(new QPushButton("Taylor Diagram (All)", this)),
      plotSubplotsButton_(new QPushButton("Show Approach Subplots (Same Plot)", this)),
      plotResidualsButton_(new QPushButton("Residuals vs t (All)", this)),
      plotErrorCdfButton_(new QPushButton("|Error| CDF (All)", this)),
      plotFlowDurationButton_(new QPushButton("Flow Duration (All)", this)),
      plotCumulativeResidualButton_(new QPushButton("Cumulative Physics Residual", this)) {
    setWindowTitle("HydroPINN - Experiment Runner");
    resize(1024, 700);

    auto* central = new QWidget(this);
    auto* scrollArea = new QScrollArea(this);
    scrollArea->setWidgetResizable(true);
    scrollArea->setFrameShape(QFrame::NoFrame);
    scrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    scrollArea->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    auto* root = new QVBoxLayout(central);

    auto* title = new QLabel("HydroPINN Approaches", central);
    title->setStyleSheet("font-size: 18px; font-weight: bold;");

    modeCombo_->addItem("FFN (Hydro baseline)", "ffn");
    modeCombo_->addItem("FFN + PINN (Hydro baseline + physics)", "ffn_pinn");
    modeCombo_->addItem("LSTM", "lstm");
    modeCombo_->addItem("LSTM + PINN", "lstm_pinn");
    modeCombo_->addItem("PINN (physics-first)", "pinn");
    activationCombo_->addItems({"relu", "tanh", "sigmoid"});
    dataSourceCombo_->addItems({"Synthetic", "CSV File", "Hydro Package"});
    hydroPackageProfileCombo_->addItems({"rainfall-runoff", "water-balance"});
    hydroForecastVariableEdit_->setText("precipitation");
    hydroForecastLeadSpin_->setRange(0.0, 1000.0);
    hydroForecastLeadSpin_->setDecimals(3);
    hydroForecastLeadSpin_->setSuffix(" h");
    hydroForecastEnsembleEdit_->setPlaceholderText("Optional ensemble_member");
    profileCombo_->addItems({"watershed_balance", "rainfall_runoff", "neuroforge_inputs_target", "exp_decay", "damped_sine", "mixed_wave"});

    auto* tabs = new QTabWidget(central);
    tabs->setUsesScrollButtons(true);
    tabs->setElideMode(Qt::ElideRight);

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
    dataForm->addRow(useNeuroforgeCsvPresetButton_);

    hydroPackagePathEdit_->setPlaceholderText("Select generic Hydro package directory");
    auto* packageRow = new QWidget(dataTab);
    auto* packageLayout = new QHBoxLayout(packageRow);
    packageLayout->setContentsMargins(0, 0, 0, 0);
    packageLayout->addWidget(hydroPackagePathEdit_, 1);
    packageLayout->addWidget(browseHydroPackageButton_);
    hydroCatchmentIdEdit_->setPlaceholderText("Stable catchment_id from package");
    dataForm->addRow("Hydro package", packageRow);
    dataForm->addRow("Package catchment ID", hydroCatchmentIdEdit_);
    dataForm->addRow("Package profile", hydroPackageProfileCombo_);
    dataForm->addRow(hydroForecastFeatureCheck_);
    dataForm->addRow("Forecast variable", hydroForecastVariableEdit_);
    dataForm->addRow("Required forecast lead", hydroForecastLeadSpin_);
    dataForm->addRow("Forecast ensemble", hydroForecastEnsembleEdit_);

    syntheticExportPathEdit_->setPlaceholderText("Optional export path for generated synthetic CSV");
    auto* syntheticExportRow = new QWidget(dataTab);
    auto* syntheticExportLayout = new QHBoxLayout(syntheticExportRow);
    syntheticExportLayout->setContentsMargins(0, 0, 0, 0);
    syntheticExportLayout->addWidget(syntheticExportPathEdit_, 1);
    syntheticExportLayout->addWidget(browseSyntheticExportButton_);

    dataForm->addRow("Synthetic export", syntheticExportRow);
    dataForm->addRow(generateSyntheticButton_);
    dataForm->addRow(new QLabel("Tip: for neuroforge_inputs_target export, set CSV x column=0 (t) and y column=6 (target).", dataTab));
    tabs->addTab(dataTab, "Data");

    auto* workflowTab = new QWidget(tabs);
    auto* workflowLayout = new QVBoxLayout(workflowTab);
    auto* workflowGuide = new QTextBrowser(workflowTab);
    workflowGuide->setOpenExternalLinks(false);
    workflowGuide->setHtml(QStringLiteral(
        "<h2>HydroPINN workflow</h2>"
        "<ol>"
        "<li><b>Choose data.</b> Start with <code>watershed_balance</code> or <code>rainfall_runoff</code> for the app's primary hydrology workflow, "
        "or switch to CSV and select x/y columns for an observed hydrograph.</li>"
        "<li><b>Set the model family.</b> Use FFN and LSTM as supervised baselines, "
        "then compare FFN + PINN, LSTM + PINN, and standalone PINN.</li>"
        "<li><b>Tune physics.</b> Pick a PINN physics profile, set data/physics "
        "loss weights, and add collocation points when the residual should be "
        "evaluated away from supervised samples.</li>"
        "<li><b>Train and compare.</b> Use <i>Train All</i> to populate the "
        "performance table and all comparison plots from one configuration.</li>"
        "<li><b>Refine.</b> Use GA lag search for FFN-family approaches, then rerun "
        "the selected modes with updated lag groups.</li>"
        "</ol>"
        "<h3>Recommended forward path</h3>"
        "<ul>"
        "<li>Establish FFN and LSTM supervised baselines on the same chronological train/validation/test split.</li>"
        "<li>Add PINN residuals with a modest physics weight, then increase only if "
        "test error, mass-balance residuals, and hydrograph diagnostics remain stable.</li>"
        "<li>Use rainfall-runoff or water-balance profiles for hydrology-specific "
        "experiments instead of the exponential-decay smoke-test profile.</li>"
        "<li>Export synthetic data when a run should be reproducible outside the GUI.</li>"
        "</ul>"));
    workflowLayout->addWidget(workflowGuide, 1);
    tabs->addTab(workflowTab, "Hydro Workflow");

    auto* networkTab = new QWidget(tabs);
    auto* networkLayout = new QVBoxLayout(networkTab);
    auto* networkTopForm = new QFormLayout();
    hiddenLayersEdit_->setText("24,24");
    inputLagsEdit_->setText("1");
    inputLagsEdit_->setPlaceholderText("Lag steps, e.g. 1,2;1;1,3");
    activationCombo_->setCurrentText("tanh");
    networkTopForm->addRow("Hidden layers (csv)", hiddenLayersEdit_);
    networkTopForm->addRow("Input lag steps (groups by ';')", inputLagsEdit_);
    networkTopForm->addRow(useTimeLaggedFFNCheck_);
    networkTopForm->addRow("Default activation", activationCombo_);

    auto* layerBuilderGroup = new QGroupBox("Layer Builder (NeuroForge-style)", networkTab);
    auto* layerBuilderForm = new QFormLayout(layerBuilderGroup);
    layerSizeSpin_->setRange(1, 2000);
    layerSizeSpin_->setValue(24);
    layerActivationCombo_->addItems({"relu", "tanh", "sigmoid", "linear"});
    layerActivationCombo_->setCurrentText("tanh");
    outputActivationCombo_->addItems({"linear", "relu", "tanh", "sigmoid"});
    outputActivationCombo_->setCurrentText("linear");

    auto* layerButtons = new QWidget(layerBuilderGroup);
    auto* layerButtonsLayout = new QHBoxLayout(layerButtons);
    layerButtonsLayout->setContentsMargins(0, 0, 0, 0);
    layerButtonsLayout->addWidget(addLayerButton_);
    layerButtonsLayout->addWidget(removeLayerButton_);

    layersList_->addItem("Layer 1: 24 nodes, tanh");
    layersList_->addItem("Layer 2: 24 nodes, tanh");

    layerBuilderForm->addRow("Layer size", layerSizeSpin_);
    layerBuilderForm->addRow("Layer activation", layerActivationCombo_);
    layerBuilderForm->addRow(layerButtons);
    layerBuilderForm->addRow("Configured layers", layersList_);
    layerBuilderForm->addRow("Output activation", outputActivationCombo_);

    auto* lagsGroup = new QGroupBox("Lag Configuration", networkTab);
    auto* lagsLayout = new QVBoxLayout(lagsGroup);
    lagsLayout->addWidget(new QLabel("Lag syntax: separate features with ';', and lags within each feature with ','.\nIf fewer groups than features are supplied, the first group is reused.", lagsGroup));

    networkLayout->addLayout(networkTopForm);
    networkLayout->addWidget(layerBuilderGroup);
    networkLayout->addWidget(lagsGroup);
    networkLayout->addStretch(1);
    tabs->addTab(networkTab, "Network");

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
    physicsWeightSpin_->setValue(0.05);
    pinnPhysicsProfileCombo_->addItems({"water_balance", "linear_reservoir", "cstr_first_order", "exp_decay"});
    forcingGainSpin_->setDecimals(4);
    forcingGainSpin_->setRange(0.0, 100.0);
    forcingGainSpin_->setValue(1.0);
    pinnCollocationSpin_->setRange(0, 50000);
    pinnCollocationSpin_->setValue(0);
    evalCheck_->setChecked(true);
    trainForm->addRow("Epochs", epochsSpin_);
    trainForm->addRow("Batch size", batchSpin_);
    trainForm->addRow("Learning rate", lrSpin_);
    trainForm->addRow("Lambda (PINN)", lambdaSpin_);
    trainForm->addRow("Data loss weight", dataWeightSpin_);
    trainForm->addRow("Physics loss weight", physicsWeightSpin_);
    trainForm->addRow("PINN physics profile", pinnPhysicsProfileCombo_);
    trainForm->addRow("PINN forcing gain", forcingGainSpin_);
    trainForm->addRow("PINN collocation points", pinnCollocationSpin_);

    trainForm->addRow(new QLabel("PINN water-domain hints: use exp_decay for pure decay; use linear_reservoir/cstr_first_order for forcing-driven dynamics; use water_balance with watershed_balance or rainfall_runoff for watershed mass-balance training. Collocation adds Raissi-style physics points.", trainTab));

    splitRatioSpin_->setDecimals(3);
    splitRatioSpin_->setRange(0.1, 0.85);
    splitRatioSpin_->setSingleStep(0.05);
    splitRatioSpin_->setValue(0.8);
    shuffleCheck_->setChecked(true);
    seedSpin_->setRange(0, 1000000000);
    seedSpin_->setValue(42);
    optimizerCombo_->addItems({"adam", "sgd", "rmsprop"});
    weightDecaySpin_->setDecimals(6);
    weightDecaySpin_->setRange(0.0, 1.0);
    weightDecaySpin_->setValue(0.0);
    momentumSpin_->setDecimals(4);
    momentumSpin_->setRange(0.0, 0.9999);
    momentumSpin_->setValue(0.9);
    normalizationCombo_->addItems({"none", "standardize", "minmax"});

    incrementalCheck_->setChecked(false);
    windowSizeSpin_->setDecimals(3);
    windowSizeSpin_->setRange(0.01, 10000.0);
    windowSizeSpin_->setValue(1.0);
    windowStepSpin_->setDecimals(3);
    windowStepSpin_->setRange(0.001, 10000.0);
    windowStepSpin_->setValue(0.5);
    epochsPerWindowSpin_->setRange(1, 10000);
    epochsPerWindowSpin_->setValue(25);
    resetOptimizerWindowCheck_->setChecked(false);

    trainForm->addRow("Train fraction (validation=0.10)", splitRatioSpin_);
    trainForm->addRow(shuffleCheck_);
    trainForm->addRow("Random seed", seedSpin_);
    trainForm->addRow("Optimizer", optimizerCombo_);
    trainForm->addRow("Weight decay", weightDecaySpin_);
    trainForm->addRow("Momentum", momentumSpin_);
    trainForm->addRow("Normalization", normalizationCombo_);
    trainForm->addRow(incrementalCheck_);
    trainForm->addRow("Window size", windowSizeSpin_);
    trainForm->addRow("Window step", windowStepSpin_);
    trainForm->addRow("Epochs/window", epochsPerWindowSpin_);
    trainForm->addRow(resetOptimizerWindowCheck_);
    trainForm->addRow(new QLabel("Note: optimizer/normalization/incremental options are exposed for NeuroForge parity; current Hydro backend applies core training settings first.", trainTab));

    auto* trainingButtons = new QWidget(trainTab);
    auto* trainingGrid = new QGridLayout(trainingButtons);
    trainingGrid->setContentsMargins(0, 0, 0, 0);
    trainingGrid->addWidget(runTrainingButton_, 0, 0);
    trainingGrid->addWidget(runAllTrainingButton_, 0, 1);
    trainingGrid->addWidget(runTrainingFFNButton_, 1, 0);
    trainingGrid->addWidget(runTrainingFFNPINNButton_, 1, 1);
    trainingGrid->addWidget(runTrainingLSTMButton_, 1, 2);
    trainingGrid->addWidget(runTrainingLSTMPINNButton_, 1, 3);
    trainingGrid->addWidget(runTrainingPINNButton_, 1, 4);
    for (int col = 0; col < 5; ++col) {
        trainingGrid->setColumnStretch(col, 1);
    }
    trainForm->addRow(trainingButtons);
    tabs->addTab(trainTab, "Training");

    auto* predictionTab = new QWidget(tabs);
    auto* predictionLayout = new QVBoxLayout(predictionTab);
    predictionLayout->addWidget(new QLabel("NeuroForge-like flow: first train approach(es) in Training tab, then review stored prediction curves here.", predictionTab));
    auto* predictionButtons = new QGroupBox("Prediction Plot Actions (from last successful runs)", predictionTab);
    auto* predictionGrid = new QGridLayout(predictionButtons);
    predictionGrid->addWidget(runPredictionButton_, 0, 0);
    predictionGrid->addWidget(runAllPredictionButton_, 0, 1);
    predictionGrid->addWidget(runPredictionFFNButton_, 1, 0);
    predictionGrid->addWidget(runPredictionFFNPINNButton_, 1, 1);
    predictionGrid->addWidget(runPredictionLSTMButton_, 1, 2);
    predictionGrid->addWidget(runPredictionLSTMPINNButton_, 1, 3);
    predictionGrid->addWidget(runPredictionPINNButton_, 1, 4);
    for (int col = 0; col < 5; ++col) {
        predictionGrid->setColumnStretch(col, 1);
    }

    runPredictionButton_->setText("Show Selected");
    runAllPredictionButton_->setText("Show All");
    runPredictionFFNButton_->setText("Show FFN");
    runPredictionFFNPINNButton_->setText("Show FFN + PINN");
    runPredictionPINNButton_->setText("Show PINN");
    runPredictionLSTMButton_->setText("Show LSTM");
    runPredictionLSTMPINNButton_->setText("Show LSTM + PINN");
    predictionUseCurrentDataCheck_->setChecked(false);
    predictionLayout->addWidget(loadInferenceArtifactsButton_);
    predictionLayout->addWidget(predictionUseCurrentDataCheck_);
    predictionLayout->addWidget(predictionButtons);
    predictionLayout->addStretch(1);
    tabs->addTab(predictionTab, "Prediction");

    auto* gaTab = new QWidget(tabs);
    auto* gaLayout = new QVBoxLayout(gaTab);
    auto* gaBox = new QGroupBox("GA Lag Optimization (FFN / FFN + PINN)", gaTab);
    auto* gaForm = new QFormLayout(gaBox);
    gaLagCandidatesSpin_->setRange(2, 10000);
    gaLagCandidatesSpin_->setValue(60);
    gaLagCandidatesSpin_->setToolTip("Evaluation budget for lag-set candidates; higher values improve search coverage but take longer.");
    gaMaxLagSpin_->setRange(1, 1000);
    gaMaxLagSpin_->setValue(5);
    gaForm->addRow("Candidate lag-set budget", gaLagCandidatesSpin_);
    gaForm->addRow("Maximum lag step", gaMaxLagSpin_);
    auto* gaButtonRow = new QWidget(gaBox);
    auto* gaButtonLayout = new QHBoxLayout(gaButtonRow);
    gaButtonLayout->setContentsMargins(0, 0, 0, 0);
    gaButtonLayout->addWidget(configureGAButton_);
    gaButtonLayout->addWidget(startGAButton_);
    gaButtonLayout->addWidget(stopGAButton_);
    gaForm->addRow(gaButtonRow);
    stopGAButton_->setEnabled(false);
    gaLayout->addWidget(gaBox);
    gaLayout->addWidget(new QLabel("This runs a lightweight evolutionary lag-structure search for the selected FFN / FFN + PINN mode, then writes the best lag-step groups back to Network Structure.", gaTab));
    gaLayout->addStretch(1);
    tabs->addTab(gaTab, "GA");

    auto* performanceTab = new QWidget(tabs);
    auto* performanceLayout = new QVBoxLayout(performanceTab);
    performanceLayout->addWidget(evalCheck_);
    performanceLayout->addWidget(refreshPerformanceButton_);
    performanceLayout->addWidget(exportExperimentButton_);
    performanceLayout->addWidget(loadExperimentConfigButton_);
    perfSummaryText_->setPlaceholderText("Performance assessment summary appears here after runs.");
    performanceLayout->addWidget(perfSummaryText_, 1);
    tabs->addTab(performanceTab, "Performance");

    auto* plotTab = new QWidget(tabs);
    auto* plotLayout = new QVBoxLayout(plotTab);
    auto* plotButtons = new QWidget(plotTab);
    auto* plotButtonsLayout = new QGridLayout(plotButtons);
    plotButtonsLayout->setContentsMargins(0, 0, 0, 0);
    plotButtonsLayout->addWidget(showInputsOutputsButton_, 0, 0);
    plotButtonsLayout->addWidget(plotAllTargetPredButton_, 0, 1);
    plotButtonsLayout->addWidget(plotOneToOneButton_, 0, 2);
    plotButtonsLayout->addWidget(plotSubplotsButton_, 0, 3);
    plotButtonsLayout->addWidget(plotResidualsButton_, 1, 0);
    plotButtonsLayout->addWidget(plotErrorCdfButton_, 1, 1);
    plotButtonsLayout->addWidget(plotTaylorButton_, 1, 2);
    plotButtonsLayout->addWidget(plotFlowDurationButton_, 1, 3);
    plotButtonsLayout->addWidget(zoomInPlotButton_, 2, 0);
    plotButtonsLayout->addWidget(zoomOutPlotButton_, 2, 1);
    plotButtonsLayout->addWidget(fitPlotButton_, 2, 2);
    plotButtonsLayout->addWidget(clearPlotButton_, 2, 3);
    plotButtonsLayout->addWidget(plotCumulativeResidualButton_, 3, 0, 1, 2);
    for (int col = 0; col < 4; ++col) {
        plotButtonsLayout->setColumnStretch(col, 1);
    }
    plotLayout->addWidget(chartView_, 1);
    plotLayout->addWidget(plotButtons, 0);
    tabs->addTab(plotTab, "Plot");


    auto* resultsRoadmapTab = new QWidget(tabs);
    auto* resultsRoadmapLayout = new QVBoxLayout(resultsRoadmapTab);
    auto* resultsRoadmap = new QTextBrowser(resultsRoadmapTab);
    resultsRoadmap->setOpenExternalLinks(false);
    resultsRoadmap->setHtml(QStringLiteral(
        "<h2>Suggested watershed results tabs</h2>"
        "<p>HydroPINN should prioritize plots and result summaries that hydrologists use "
        "to judge both predictive skill and water-balance credibility.</p>"
        "<ul>"
        "<li><b>Hydrograph + hyetograph:</b> target/predicted runoff with rainfall or "
        "effective precipitation bars on a shared event timeline.</li>"
        "<li><b>Mass-balance residuals:</b> plot <code>P - ET - Q - dS/dt</code> through "
        "time and summarize mean bias, RMSE, and signed cumulative residual.</li>"
        "<li><b>Cumulative water balance:</b> cumulative precipitation, ET, runoff, and "
        "storage change to expose long-term drift.</li>"
        "<li><b>Flow-duration and peak-flow diagnostics:</b> compare high-flow, low-flow, "
        "timing-to-peak, and volume errors across FFN/LSTM/PINN variants.</li>"
        "<li><b>Regime-conditioned errors:</b> split metrics by wet/dry periods, soil "
        "storage state, groundwater state, and impervious quickflow dominance.</li>"
        "<li><b>Experiment table export:</b> one row per approach with data loss, physics "
        "loss, NSE/KGE/RMSE/MAE/bias, peak timing error, and run configuration.</li>"
        "</ul>"));
    resultsRoadmapLayout->addWidget(resultsRoadmap, 1);
    tabs->addTab(resultsRoadmapTab, "Results Plan");

    auto* logTab = new QWidget(tabs);
    auto* logLayout = new QVBoxLayout(logTab);
    logText_->setReadOnly(true);
    logText_->document()->setMaximumBlockCount(1200);
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
    chartView_->setRubberBand(QChartView::RectangleRubberBand);

    root->addWidget(title);
    root->addLayout(topRow);
    root->addWidget(tabs);
    root->addWidget(statusLabel_);
    auto* modeInfo = new QLabel(QStringLiteral("Hydro provides 5 local approaches: FFN, FFN + PINN, PINN, LSTM, and LSTM + PINN.\n"
                                              "NeuroForge naming/workflow is used for UI parity only (not inherited model code).\n"
                                              "LSTM approaches use the LibTorch LSTM backend."),
                                central);
    modeInfo->setWordWrap(true);
    root->addWidget(modeInfo);

    scrollArea->setWidget(central);
    setCentralWidget(scrollArea);

    connect(runTrainingButton_, &QPushButton::clicked, this, &HydroPINNWindow::runSelectedMode);
    connect(runAllTrainingButton_, &QPushButton::clicked, this, &HydroPINNWindow::runAllModes);
    connect(runTrainingFFNButton_, &QPushButton::clicked, this, [this]() { runMode("ffn"); });
    connect(runTrainingFFNPINNButton_, &QPushButton::clicked, this, [this]() { runMode("ffn_pinn"); });
    connect(runTrainingPINNButton_, &QPushButton::clicked, this, [this]() { runMode("pinn"); });
    connect(runTrainingLSTMButton_, &QPushButton::clicked, this, [this]() { runMode("lstm"); });
    connect(runTrainingLSTMPINNButton_, &QPushButton::clicked, this, [this]() { runMode("lstm_pinn"); });
    connect(runPredictionButton_, &QPushButton::clicked, this, &HydroPINNWindow::showSelectedPrediction);
    connect(runAllPredictionButton_, &QPushButton::clicked, this, &HydroPINNWindow::showAllPredictions);
    connect(runPredictionFFNButton_, &QPushButton::clicked, this, [this]() { showPredictionForMode("ffn"); });
    connect(runPredictionFFNPINNButton_, &QPushButton::clicked, this, [this]() { showPredictionForMode("ffn_pinn"); });
    connect(runPredictionPINNButton_, &QPushButton::clicked, this, [this]() { showPredictionForMode("pinn"); });
    connect(runPredictionLSTMButton_, &QPushButton::clicked, this, [this]() { showPredictionForMode("lstm"); });
    connect(runPredictionLSTMPINNButton_, &QPushButton::clicked, this, [this]() { showPredictionForMode("lstm_pinn"); });
    connect(loadInferenceArtifactsButton_, &QPushButton::clicked, this, &HydroPINNWindow::loadInferenceArtifacts);
    connect(useNeuroforgeCsvPresetButton_, &QPushButton::clicked, this, &HydroPINNWindow::applyNeuroforgeCsvPreset);
    connect(modeCombo_, &QComboBox::currentTextChanged, this, [this](const QString&) {
        updateFfnLagUiState();
        updateStatus();
    });
    connect(useTimeLaggedFFNCheck_, &QCheckBox::toggled, this, [this](bool) { updateFfnLagUiState(); });
    connect(addLayerButton_, &QPushButton::clicked, this, [this]() {
        const int layerSize = layerSizeSpin_->value();
        const QString layerAct = layerActivationCombo_->currentText();
        layersList_->addItem(QString("Layer %1: %2 nodes, %3")
                                 .arg(layersList_->count() + 1)
                                 .arg(layerSize)
                                 .arg(layerAct));
        syncNetworkCsvFromLayerList();
    });
    connect(removeLayerButton_, &QPushButton::clicked, this, [this]() {
        const int row = layersList_->currentRow();
        if (row >= 0) {
            delete layersList_->takeItem(row);
            syncNetworkCsvFromLayerList();
        }
    });
    connect(hiddenLayersEdit_, &QLineEdit::editingFinished, this, [this]() {
        if (!hiddenLayersEdit_->text().trimmed().isEmpty()) {
            layersList_->clear();
            const QStringList parts = hiddenLayersEdit_->text().split(',', Qt::SkipEmptyParts);
            for (int i = 0; i < parts.size(); ++i) {
                const QString p = parts[i].trimmed();
                bool ok = false;
                const int n = p.toInt(&ok);
                if (ok && n > 0) {
                    layersList_->addItem(QString("Layer %1: %2 nodes, %3")
                                             .arg(i + 1)
                                             .arg(n)
                                             .arg(activationCombo_->currentText()));
                }
            }
        }
    });
    connect(dataSourceCombo_, &QComboBox::currentTextChanged, this, [this](const QString&) {
        updateDataSourceUiState();
        updateStatus();
    });
    connect(hydroForecastFeatureCheck_, &QCheckBox::toggled, this, [this](bool) {
        updateDataSourceUiState();
    });
    connect(browseCsvButton_, &QPushButton::clicked, this, &HydroPINNWindow::browseCsv);
    connect(browseHydroPackageButton_, &QPushButton::clicked, this, &HydroPINNWindow::browseHydroPackage);
    connect(browseSyntheticExportButton_, &QPushButton::clicked, this, &HydroPINNWindow::browseSyntheticExportPath);
    connect(generateSyntheticButton_, &QPushButton::clicked, this, &HydroPINNWindow::generateSyntheticDataPreview);
    connect(configureGAButton_, &QPushButton::clicked, this, &HydroPINNWindow::configureGAPlaceholder);
    connect(startGAButton_, &QPushButton::clicked, this, &HydroPINNWindow::startGAPlaceholder);
    connect(stopGAButton_, &QPushButton::clicked, this, &HydroPINNWindow::stopGAPlaceholder);
    connect(refreshPerformanceButton_, &QPushButton::clicked, this, &HydroPINNWindow::refreshPerformanceAssessment);
    connect(exportExperimentButton_, &QPushButton::clicked, this, &HydroPINNWindow::exportExperimentArtifacts);
    connect(loadExperimentConfigButton_, &QPushButton::clicked, this, &HydroPINNWindow::loadExperimentConfiguration);
    connect(clearPlotButton_, &QPushButton::clicked, this, &HydroPINNWindow::clearPlot);
    connect(showInputsOutputsButton_, &QPushButton::clicked, this, &HydroPINNWindow::showSyntheticInputsOutputs);
    connect(plotAllTargetPredButton_, &QPushButton::clicked, this, &HydroPINNWindow::plotAllTargetVsPredicted);
    connect(plotOneToOneButton_, &QPushButton::clicked, this, &HydroPINNWindow::plotOneToOneAllModes);
    connect(plotTaylorButton_, &QPushButton::clicked, this, &HydroPINNWindow::plotTaylorDiagramAllModes);
    connect(plotSubplotsButton_, &QPushButton::clicked, this, &HydroPINNWindow::showModeSubplots);
    connect(plotResidualsButton_, &QPushButton::clicked, this, &HydroPINNWindow::plotResidualsAllModes);
    connect(plotErrorCdfButton_, &QPushButton::clicked, this, &HydroPINNWindow::plotErrorCdfAllModes);
    connect(plotFlowDurationButton_, &QPushButton::clicked, this, &HydroPINNWindow::plotFlowDurationAllModes);
    connect(plotCumulativeResidualButton_, &QPushButton::clicked, this, &HydroPINNWindow::plotCumulativeResidualAllModes);
    connect(zoomInPlotButton_, &QPushButton::clicked, this, &HydroPINNWindow::zoomInPlot);
    connect(zoomOutPlotButton_, &QPushButton::clicked, this, &HydroPINNWindow::zoomOutPlot);
    connect(fitPlotButton_, &QPushButton::clicked, this, &HydroPINNWindow::fitPlotAxes);

    updateDataSourceUiState();
    updateFfnLagUiState();
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
    cfg.pinn_physics_profile = pinnPhysicsProfileCombo_->currentText().toStdString();
    cfg.forcing_gain = forcingGainSpin_->value();
    cfg.pinn_collocation_points = pinnCollocationSpin_->value();
    cfg.use_csv_data = (dataSourceCombo_->currentText() == "CSV File");
    cfg.use_hydro_package = (dataSourceCombo_->currentText() == "Hydro Package");
    cfg.hydro_package_path = hydroPackagePathEdit_->text().toStdString();
    cfg.hydro_catchment_id = hydroCatchmentIdEdit_->text().toStdString();
    cfg.hydro_package_profile = hydroPackageProfileCombo_->currentText().toStdString();
    cfg.use_hydro_forecast_feature = hydroForecastFeatureCheck_->isChecked();
    cfg.hydro_forecast_variable = hydroForecastVariableEdit_->text().toStdString();
    cfg.hydro_forecast_lead_hours = hydroForecastLeadSpin_->value();
    cfg.hydro_forecast_ensemble_member = hydroForecastEnsembleEdit_->text().toStdString();
    cfg.csv_path = csvPathEdit_->text().toStdString();
    cfg.csv_x_column = csvXColSpin_->value();
    cfg.csv_y_column = csvYColSpin_->value();
    cfg.csv_has_header = csvHeaderCheck_->isChecked();
    cfg.sample_count = sampleCountSpin_->value();
    cfg.t_start = tStartSpin_->value();
    cfg.t_end = tEndSpin_->value();
    cfg.hidden_layers_csv = hiddenLayersEdit_->text().toStdString();
    cfg.input_lags_csv = inputLagsEdit_->text().toStdString();
    cfg.use_time_lagged_ffn = useTimeLaggedFFNCheck_->isChecked();
    const std::vector<QString> layerActs = configuredLayerActivations();
    if (!layerActs.empty()) {
        cfg.activation = layerActs.front().toStdString();
    } else {
        cfg.activation = activationCombo_->currentText().toStdString();
    }
    cfg.synthetic_profile = profileCombo_->currentText().toStdString();
    cfg.evaluate_metrics = evalCheck_->isChecked();
    cfg.train_split_ratio = splitRatioSpin_->value();
    cfg.shuffle_training = shuffleCheck_->isChecked();
    cfg.random_seed = seedSpin_->value();
    cfg.optimizer = optimizerCombo_->currentText().toStdString();
    cfg.weight_decay = weightDecaySpin_->value();
    cfg.momentum = momentumSpin_->value();
    cfg.normalization = normalizationCombo_->currentText().toStdString();
    cfg.use_incremental_training = incrementalCheck_->isChecked();
    cfg.window_size = windowSizeSpin_->value();
    cfg.window_step = windowStepSpin_->value();
    cfg.epochs_per_window = epochsPerWindowSpin_->value();
    cfg.reset_optimizer_on_new_window = resetOptimizerWindowCheck_->isChecked();
    return cfg;
}

void HydroPINNWindow::setRunningUiState(bool running) {
    dataSourceCombo_->setEnabled(!running);
    browseCsvButton_->setEnabled(!running && dataSourceCombo_->currentText() == "CSV File");
    browseHydroPackageButton_->setEnabled(!running && dataSourceCombo_->currentText() == "Hydro Package");
    hydroPackagePathEdit_->setEnabled(!running && dataSourceCombo_->currentText() == "Hydro Package");
    hydroCatchmentIdEdit_->setEnabled(!running && dataSourceCombo_->currentText() == "Hydro Package");
    hydroPackageProfileCombo_->setEnabled(!running && dataSourceCombo_->currentText() == "Hydro Package");
    hydroForecastFeatureCheck_->setEnabled(!running && dataSourceCombo_->currentText() == "Hydro Package");
    const bool forecastEnabled = !running && dataSourceCombo_->currentText() == "Hydro Package" && hydroForecastFeatureCheck_->isChecked();
    hydroForecastVariableEdit_->setEnabled(forecastEnabled);
    hydroForecastLeadSpin_->setEnabled(forecastEnabled);
    hydroForecastEnsembleEdit_->setEnabled(forecastEnabled);
    generateSyntheticButton_->setEnabled(!running && dataSourceCombo_->currentText() == "Synthetic");
    syntheticExportPathEdit_->setEnabled(!running && dataSourceCombo_->currentText() == "Synthetic");
    browseSyntheticExportButton_->setEnabled(!running && dataSourceCombo_->currentText() == "Synthetic");
    runPredictionButton_->setText(running ? "Running..." : "Show Selected");
    runPredictionButton_->setEnabled(!running);
    runAllPredictionButton_->setEnabled(!running);
    runPredictionFFNButton_->setEnabled(!running);
    runPredictionFFNPINNButton_->setEnabled(!running);
    runPredictionPINNButton_->setEnabled(!running);
    runPredictionLSTMButton_->setEnabled(!running);
    runPredictionLSTMPINNButton_->setEnabled(!running);
    loadInferenceArtifactsButton_->setEnabled(!running);
    predictionUseCurrentDataCheck_->setEnabled(!running);
    useTimeLaggedFFNCheck_->setEnabled(!running && (selectedModeKey() == "ffn" || selectedModeKey() == "ffn_pinn"));
    inputLagsEdit_->setEnabled(!running && (selectedModeKey() == "ffn" || selectedModeKey() == "ffn_pinn") && useTimeLaggedFFNCheck_->isChecked());
    runTrainingButton_->setEnabled(!running);
    runAllTrainingButton_->setEnabled(!running);
    runTrainingFFNButton_->setEnabled(!running);
    runTrainingFFNPINNButton_->setEnabled(!running);
    runTrainingPINNButton_->setEnabled(!running);
    runTrainingLSTMButton_->setEnabled(!running);
    runTrainingLSTMPINNButton_->setEnabled(!running);
    exportExperimentButton_->setEnabled(!running && !lastModeResults_.empty());
    loadExperimentConfigButton_->setEnabled(!running);
    plotAllTargetPredButton_->setEnabled(!running);
    plotOneToOneButton_->setEnabled(!running);
    plotTaylorButton_->setEnabled(!running);
    plotSubplotsButton_->setEnabled(!running);
    plotResidualsButton_->setEnabled(!running);
    plotErrorCdfButton_->setEnabled(!running);
    plotFlowDurationButton_->setEnabled(!running);
    plotCumulativeResidualButton_->setEnabled(!running);
    useNeuroforgeCsvPresetButton_->setEnabled(!running && dataSourceCombo_->currentText() == "CSV File");
}

QString HydroPINNWindow::selectedModeKey() const {
    return modeCombo_->currentData().toString();
}

std::vector<QString> HydroPINNWindow::configuredLayerActivations() const {
    std::vector<QString> acts;
    acts.reserve(static_cast<size_t>(layersList_->count()));
    for (int i = 0; i < layersList_->count(); ++i) {
        const QString act = parseLayerActivationText(layersList_->item(i)->text());
        if (!act.isEmpty()) acts.push_back(act);
    }
    return acts;
}

void HydroPINNWindow::syncNetworkCsvFromLayerList() {
    QStringList layerSizes;
    for (int i = 0; i < layersList_->count(); ++i) {
        const QString txt = layersList_->item(i)->text();
        const int colon = txt.indexOf(':');
        const int nodesPos = txt.indexOf("nodes");
        if (colon < 0 || nodesPos < 0) continue;
        const QString between = txt.mid(colon + 1, nodesPos - (colon + 1)).trimmed();
        const QString firstNumber = between.split(' ', Qt::SkipEmptyParts).value(0);
        bool ok = false;
        const int n = firstNumber.toInt(&ok);
        if (ok && n > 0) layerSizes << QString::number(n);
    }
    if (!layerSizes.isEmpty()) {
        hiddenLayersEdit_->setText(layerSizes.join(','));
    }
}

void HydroPINNWindow::updateFfnLagUiState() {
    const bool ffnMode = (selectedModeKey() == "ffn" || selectedModeKey() == "ffn_pinn");
    useTimeLaggedFFNCheck_->setEnabled(ffnMode);
    inputLagsEdit_->setEnabled(ffnMode && useTimeLaggedFFNCheck_->isChecked());
}

void HydroPINNWindow::updateDataSourceUiState() {
    const bool useCsv = (dataSourceCombo_->currentText() == "CSV File");
    const bool usePackage = (dataSourceCombo_->currentText() == "Hydro Package");
    csvPathEdit_->setEnabled(useCsv);
    browseCsvButton_->setEnabled(useCsv);
    csvXColSpin_->setEnabled(useCsv);
    csvYColSpin_->setEnabled(useCsv);
    csvHeaderCheck_->setEnabled(useCsv);
    useNeuroforgeCsvPresetButton_->setEnabled(useCsv);
    hydroPackagePathEdit_->setEnabled(usePackage);
    browseHydroPackageButton_->setEnabled(usePackage);
    hydroCatchmentIdEdit_->setEnabled(usePackage);
    hydroPackageProfileCombo_->setEnabled(usePackage);
    hydroForecastFeatureCheck_->setEnabled(usePackage);
    const bool forecastEnabled = usePackage && hydroForecastFeatureCheck_->isChecked();
    hydroForecastVariableEdit_->setEnabled(forecastEnabled);
    hydroForecastLeadSpin_->setEnabled(forecastEnabled);
    hydroForecastEnsembleEdit_->setEnabled(forecastEnabled);

    const bool useSynthetic = !useCsv && !usePackage;
    profileCombo_->setEnabled(useSynthetic);
    sampleCountSpin_->setEnabled(useSynthetic);
    tStartSpin_->setEnabled(useSynthetic);
    tEndSpin_->setEnabled(useSynthetic);
    generateSyntheticButton_->setEnabled(useSynthetic);
    syntheticExportPathEdit_->setEnabled(useSynthetic);
    browseSyntheticExportButton_->setEnabled(useSynthetic);
}

void HydroPINNWindow::applyNeuroforgeCsvPreset() {
    csvXColSpin_->setValue(0);
    csvYColSpin_->setValue(6);
    csvHeaderCheck_->setChecked(true);
    appendLog("Applied NeuroForge CSV preset: x=0 (t), y=6 (target), header=yes.");
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

void HydroPINNWindow::browseHydroPackage() {
    const QString path = QFileDialog::getExistingDirectory(this, "Select Hydro package directory");
    if (!path.isEmpty()) hydroPackagePathEdit_->setText(path);
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
    std::vector<double> temperature;
    std::vector<double> pressure;
    std::vector<double> flowRate;
    std::vector<double> concentration;
    std::vector<double> velocity;
    std::vector<double> rainfall;
    std::vector<double> evapotranspiration;
    std::vector<double> soilStorage;
    std::vector<double> groundwaterStorage;
    std::vector<double> imperviousFraction;
    xs.reserve(static_cast<size_t>(samples));
    ys.reserve(static_cast<size_t>(samples));
    temperature.reserve(static_cast<size_t>(samples));
    pressure.reserve(static_cast<size_t>(samples));
    flowRate.reserve(static_cast<size_t>(samples));
    concentration.reserve(static_cast<size_t>(samples));
    velocity.reserve(static_cast<size_t>(samples));
    rainfall.reserve(static_cast<size_t>(samples));
    evapotranspiration.reserve(static_cast<size_t>(samples));
    soilStorage.reserve(static_cast<size_t>(samples));
    groundwaterStorage.reserve(static_cast<size_t>(samples));
    imperviousFraction.reserve(static_cast<size_t>(samples));

    if (profile == "neuroforge_inputs_target") {
        std::srand(42);
        const double dt = (tEnd - tStart) / static_cast<double>(samples - 1);
        const double bufferStart = tStart - 1.0;
        std::vector<double> allT;
        std::vector<double> allTemp;
        std::vector<double> allPress;
        std::vector<double> allFlow;
        std::vector<double> allConc;
        std::vector<double> allVel;

        double x0 = 0.0;
        double x1 = 0.0;
        double x2 = 1.0;
        double x3 = 0.0;
        double x4 = 0.0;

        const int totalSteps = static_cast<int>(std::floor((tEnd - bufferStart) / dt)) + 1;
        allT.reserve(static_cast<size_t>(totalSteps));
        allTemp.reserve(static_cast<size_t>(totalSteps));
        allPress.reserve(static_cast<size_t>(totalSteps));
        allFlow.reserve(static_cast<size_t>(totalSteps));
        allConc.reserve(static_cast<size_t>(totalSteps));
        allVel.reserve(static_cast<size_t>(totalSteps));

        auto uniformNoise = []() {
            return (static_cast<double>(std::rand()) / RAND_MAX - 0.5) * 2.0;
        };

        for (int i = 0; i < totalSteps; ++i) {
            const double t = bufferStart + dt * static_cast<double>(i);

            x0 = x0 + 0.5 * (0.0 - x0) * dt + 1.5 * std::sqrt(dt) * uniformNoise();
            x1 = x1 + 1.0 * (0.0 - x1) * dt + 1.2 * std::sqrt(dt) * uniformNoise();
            x2 = x2 + 2.0 * (1.0 - x2) * dt + 0.8 * std::sqrt(dt) * uniformNoise();
            x3 = x3 + 0.3 * (0.0 - x3) * dt + 1.0 * std::sqrt(dt) * uniformNoise();
            x4 = x4 + 0.8 * (0.0 - x4) * dt + 1.8 * std::sqrt(dt) * uniformNoise();

            allT.push_back(t);
            allTemp.push_back(x0);
            allPress.push_back(x1);
            allFlow.push_back(x2);
            allConc.push_back(x3);
            allVel.push_back(x4);
        }

        auto interpol = [&](const std::vector<double>& vals, double tq) {
            if (tq <= allT.front()) return vals.front();
            if (tq >= allT.back()) return vals.back();
            const auto it = std::lower_bound(allT.begin(), allT.end(), tq);
            const size_t hi = static_cast<size_t>(it - allT.begin());
            const size_t lo = hi - 1;
            const double t0 = allT[lo];
            const double t1 = allT[hi];
            const double r = (tq - t0) / (t1 - t0);
            return vals[lo] * (1.0 - r) + vals[hi] * r;
        };

        for (int i = 0; i < samples; ++i) {
            const double t = tStart + dt * static_cast<double>(i);
            const double temp = interpol(allTemp, t);
            const double press = interpol(allPress, t);
            const double flow = interpol(allFlow, t);
            const double conc = interpol(allConc, t);
            const double vel = interpol(allVel, t);
            const double target = 0.4 * interpol(allTemp, t - 0.1) +
                                  0.3 * interpol(allPress, t - 0.3) +
                                  0.2 * interpol(allConc, t - 0.2) +
                                  0.1 * interpol(allVel, t - 0.5) +
                                  0.05 * (static_cast<double>(std::rand()) / RAND_MAX - 0.5);

            xs.push_back(t);
            temperature.push_back(temp);
            pressure.push_back(press);
            flowRate.push_back(flow);
            concentration.push_back(conc);
            velocity.push_back(vel);
            ys.push_back(target);
        }
    } else if (profile == "watershed_balance") {
        const double dt = 1.0 / static_cast<double>(samples - 1);
        constexpr double kPi = 3.14159265358979323846;
        double soil = 12.0;
        double groundwater = 18.0;
        for (int i = 0; i < samples; ++i) {
            const double r = static_cast<double>(i) / static_cast<double>(samples - 1);
            const double t = tStart + (tEnd - tStart) * r;
            const double stormA = 16.0 * std::exp(-0.5 * std::pow((t - (tStart + 0.18 * (tEnd - tStart))) / std::max(0.05, 0.035 * (tEnd - tStart)), 2.0));
            const double stormB = 10.0 * std::exp(-0.5 * std::pow((t - (tStart + 0.46 * (tEnd - tStart))) / std::max(0.05, 0.055 * (tEnd - tStart)), 2.0));
            const double stormC = 7.0 * std::exp(-0.5 * std::pow((t - (tStart + 0.78 * (tEnd - tStart))) / std::max(0.05, 0.08 * (tEnd - tStart)), 2.0));
            const double rain = stormA + stormB + stormC + 1.5 * std::max(0.0, std::sin(2.0 * kPi * r * 4.0));
            const double temp = 4.0 + 16.0 * std::sin(kPi * r - 0.25);
            const double snowpackFactor = std::max(0.0, 1.0 - temp / 4.0);
            const double snowmelt = std::max(0.0, temp - 1.0) * (0.12 + 0.18 * snowpackFactor);
            const double et = std::max(0.0, 0.06 * (temp + 3.0) * (0.6 + 0.4 * std::sin(kPi * r)));
            const double impervious = 0.12 + 0.10 * std::sin(2.0 * kPi * r + 0.5);
            const double effectivePrecip = rain + snowmelt;
            const double perviousFraction = std::max(0.0, 1.0 - impervious);
            const double infiltrationCapacity = effectivePrecip * perviousFraction * (0.55 + 0.20 * std::sin(2.0 * kPi * r - 0.3));
            const double infiltration = std::min(infiltrationCapacity, std::max(0.0, 30.0 - soil));
            const double quickRunoff = std::max(0.0, effectivePrecip - infiltration);
            const double recharge = 0.10 * soil;
            const double baseflow = 0.045 * groundwater;
            const double lateralFlow = 0.035 * soil;
            const double runoff = quickRunoff + lateralFlow + baseflow;
            soil = std::max(0.0, soil + (infiltration - et - recharge - lateralFlow) * dt);
            groundwater = std::max(0.0, groundwater + (recharge - baseflow) * dt);

            // The water-balance PINN residual uses total watershed storage
            // (soil + groundwater), so runoff is generated conservatively as
            // P - ET - dS_total/dt.
            xs.push_back(t);
            rainfall.push_back(effectivePrecip);
            evapotranspiration.push_back(et);
            temperature.push_back(temp);
            soilStorage.push_back(soil);
            groundwaterStorage.push_back(groundwater);
            imperviousFraction.push_back(impervious);
            ys.push_back(runoff);
        }
    } else if (profile == "rainfall_runoff") {
        // Use normalized simulation time for storage dynamics so changing the displayed t-range
        // does not change runoff magnitude or destabilize training.
        const double dt = 1.0 / static_cast<double>(samples - 1);
        constexpr double kPi = 3.14159265358979323846;
        double storage = 8.0;
        for (int i = 0; i < samples; ++i) {
            const double r = static_cast<double>(i) / static_cast<double>(samples - 1);
            const double t = tStart + (tEnd - tStart) * r;
            const double storm1 = 18.0 * std::exp(-0.5 * std::pow((t - (tStart + 0.22 * (tEnd - tStart))) / std::max(0.05, 0.04 * (tEnd - tStart)), 2.0));
            const double storm2 = 12.0 * std::exp(-0.5 * std::pow((t - (tStart + 0.58 * (tEnd - tStart))) / std::max(0.05, 0.07 * (tEnd - tStart)), 2.0));
            const double seasonalRain = 2.0 * std::max(0.0, std::sin(2.0 * kPi * r * 3.0));
            const double rain = storm1 + storm2 + seasonalRain;
            const double temp = 12.0 + 10.0 * std::sin(2.0 * kPi * r - 0.4);
            const double et = std::max(0.0, 0.08 * (temp + 5.0));
            const double quickflow = 0.35 * rain;
            const double baseflow = 0.08 * storage;
            const double runoff = quickflow + baseflow;
            storage = std::max(0.0, storage + (rain - et - runoff) * dt);

            xs.push_back(t);
            rainfall.push_back(rain);
            evapotranspiration.push_back(et);
            temperature.push_back(temp);
            soilStorage.push_back(storage);
            ys.push_back(runoff);
        }
    } else {
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
    }

    lastSyntheticX_ = xs;
    lastSyntheticTarget_ = ys;
    lastSyntheticInputs_.clear();
    if (profile == "neuroforge_inputs_target") {
        lastSyntheticInputs_["temperature"] = temperature;
        lastSyntheticInputs_["pressure"] = pressure;
        lastSyntheticInputs_["flow_rate"] = flowRate;
        lastSyntheticInputs_["concentration"] = concentration;
        lastSyntheticInputs_["velocity"] = velocity;
    } else if (profile == "watershed_balance" || profile == "rainfall_runoff") {
        lastSyntheticInputs_["effective_precipitation"] = rainfall;
        lastSyntheticInputs_["evapotranspiration"] = evapotranspiration;
        lastSyntheticInputs_["temperature"] = temperature;
        lastSyntheticInputs_["soil_storage"] = soilStorage;
        if (profile == "watershed_balance") {
            lastSyntheticInputs_["groundwater_storage"] = groundwaterStorage;
            std::vector<double> totalStorage;
            totalStorage.reserve(soilStorage.size());
            for (size_t i = 0; i < soilStorage.size(); ++i) {
                totalStorage.push_back(soilStorage[i] + groundwaterStorage[i]);
            }
            lastSyntheticInputs_["total_storage"] = totalStorage;
            lastSyntheticInputs_["impervious_fraction"] = imperviousFraction;
        }
    } else {
        lastSyntheticInputs_["synthetic_input"] = ys;
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
        if (profile == "neuroforge_inputs_target") {
            out << "t,temperature,pressure,flow_rate,concentration,velocity,target\n";
            for (int i = 0; i < samples; ++i) {
                const size_t k = static_cast<size_t>(i);
                out << xs[k] << ","
                    << temperature[k] << ","
                    << pressure[k] << ","
                    << flowRate[k] << ","
                    << concentration[k] << ","
                    << velocity[k] << ","
                    << ys[k] << "\n";
            }
        } else if (profile == "watershed_balance" || profile == "rainfall_runoff") {
            if (profile == "watershed_balance") {
                out << "t,effective_precipitation,evapotranspiration,temperature,soil_storage,groundwater_storage,total_storage,impervious_fraction,runoff\n";
            } else {
                out << "t,rainfall,evapotranspiration,temperature,soil_storage,runoff\n";
            }
            for (int i = 0; i < samples; ++i) {
                const size_t k = static_cast<size_t>(i);
                out << xs[k] << ","
                    << rainfall[k] << ","
                    << evapotranspiration[k] << ","
                    << temperature[k] << ","
                    << soilStorage[k] << ",";
                if (profile == "watershed_balance") {
                    out << groundwaterStorage[k] << ","
                        << (soilStorage[k] + groundwaterStorage[k]) << ","
                        << imperviousFraction[k] << ",";
                }
                out << ys[k] << "\n";
            }
        } else {
            out << "t,y\n";
            for (int i = 0; i < samples; ++i) {
                out << xs[static_cast<size_t>(i)] << "," << ys[static_cast<size_t>(i)] << "\n";
            }
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
    appendLog("GA lag optimization configuration opened.");
    QMessageBox::information(this,
                             "HydroPINN GA",
                             "GA lag optimization samples unique candidate per-input lag-step groups for FFN / FFN + PINN, "
                             "trains each candidate briefly, and writes the best lag-step structure back to the Network Structure tab.");
}

void HydroPINNWindow::startGAPlaceholder() {
    runLagOptimizationSearch();
}

int HydroPINNWindow::estimatedFfnInputCountForLagSearch(const HydroRunConfig& cfg, const QString& mode) const {
    if (cfg.synthetic_profile == "neuroforge_inputs_target") {
        return (mode == "ffn_pinn") ? 6 : 5;
    }
    if (cfg.synthetic_profile == "watershed_balance") {
        return (mode == "ffn_pinn") ? 8 : 7;
    }
    if (cfg.synthetic_profile == "rainfall_runoff") {
        return 5;
    }
    if (cfg.synthetic_profile == "watershed_balance") {
        return 7;
    }
    if (mode == "ffn_pinn" &&
        (cfg.pinn_physics_profile == "linear_reservoir" ||
         cfg.pinn_physics_profile == "cstr_first_order" ||
         cfg.pinn_physics_profile == "water_balance")) {
        return 2;
    }
    return 1;
}

void HydroPINNWindow::runLagOptimizationSearch() {
    const QString mode = selectedModeKey();
    if (mode != "ffn" && mode != "ffn_pinn") {
        QMessageBox::information(this,
                                 "HydroPINN GA",
                                 "Lag optimization is only available for FFN and FFN + PINN approaches.");
        appendLog("GA lag optimization skipped: selected approach is not FFN / FFN + PINN.");
        return;
    }

    gaStopRequested_ = false;
    appendLog(QString("Starting GA-style lag optimization for %1.").arg(modeDisplayName(mode)));
    startGAButton_->setEnabled(false);
    stopGAButton_->setEnabled(true);
    statusLabel_->setText("Running GA-style lag optimization...");
    QCoreApplication::processEvents();

    HydroRunConfig baseCfg = currentConfig();
    baseCfg.use_time_lagged_ffn = true;
    baseCfg.epochs = std::max(1, std::min(baseCfg.epochs, epochsPerWindowSpin_->value()));

    const int candidateCount = gaLagCandidatesSpin_->value();
    const int maxLag = gaMaxLagSpin_->value();
    const int inputGroups = estimatedFfnInputCountForLagSearch(baseCfg, mode);
    std::set<QString> testedSpecs;
    std::mt19937 rng(static_cast<uint32_t>(std::max(0, baseCfg.random_seed)));
    std::uniform_int_distribution<int> lagDist(1, maxLag);
    std::uniform_int_distribution<int> countDist(1, std::min(3, maxLag));

    struct LagCandidateSummary {
        QString spec;
        double score = std::numeric_limits<double>::infinity();
        double loss = std::numeric_limits<double>::infinity();
    };
    std::vector<LagCandidateSummary> successfulCandidates;

    double bestMse = std::numeric_limits<double>::infinity();
    double bestLoss = std::numeric_limits<double>::infinity();
    QString bestSpec;
    HydroRunResult bestResult;

    auto normalizeGroup = [](std::vector<int>& lags) {
        std::sort(lags.begin(), lags.end());
        lags.erase(std::unique(lags.begin(), lags.end()), lags.end());
        if (lags.empty()) {
            lags.push_back(1);
        }
    };

    auto candidateToSpec = [&normalizeGroup](std::vector<std::vector<int>> candidate) {
        QStringList groups;
        for (auto& lags : candidate) {
            normalizeGroup(lags);
            QStringList lagTokens;
            for (const int lag : lags) lagTokens << QString::number(lag);
            groups << lagTokens.join(',');
        }
        return groups.join(';');
    };

    auto randomCandidate = [&]() {
        std::vector<std::vector<int>> candidate(static_cast<size_t>(inputGroups));
        for (auto& lags : candidate) {
            const int n = countDist(rng);
            for (int i = 0; i < n; ++i) {
                lags.push_back(lagDist(rng));
            }
            normalizeGroup(lags);
        }
        return candidate;
    };

    auto parseCandidate = [&](const QString& spec) {
        std::vector<std::vector<int>> candidate;
        const QStringList groups = spec.split(';', Qt::SkipEmptyParts);
        for (const QString& group : groups) {
            std::vector<int> lags;
            const QStringList tokens = group.split(',', Qt::SkipEmptyParts);
            for (const QString& token : tokens) {
                bool ok = false;
                const int lag = token.trimmed().toInt(&ok);
                if (ok && lag >= 1 && lag <= maxLag) {
                    lags.push_back(lag);
                }
            }
            normalizeGroup(lags);
            candidate.push_back(lags);
        }
        while (static_cast<int>(candidate.size()) < inputGroups) {
            candidate.push_back(candidate.empty() ? std::vector<int>{1} : candidate.front());
        }
        candidate.resize(static_cast<size_t>(inputGroups));
        return candidate;
    };

    auto mutateCandidate = [&](std::vector<std::vector<int>> candidate) {
        if (candidate.empty()) {
            return randomCandidate();
        }
        std::uniform_int_distribution<int> groupDist(0, static_cast<int>(candidate.size()) - 1);
        auto& lags = candidate[static_cast<size_t>(groupDist(rng))];
        std::uniform_int_distribution<int> opDist(0, 2);
        const int op = opDist(rng);
        if (op == 0 || lags.empty()) {
            lags.push_back(lagDist(rng));
        } else if (op == 1 && lags.size() > 1) {
            std::uniform_int_distribution<int> removeDist(0, static_cast<int>(lags.size()) - 1);
            lags.erase(lags.begin() + removeDist(rng));
        } else {
            std::uniform_int_distribution<int> replaceDist(0, static_cast<int>(lags.size()) - 1);
            lags[static_cast<size_t>(replaceDist(rng))] = lagDist(rng);
        }
        normalizeGroup(lags);
        return candidate;
    };

    int evaluated = 0;
    int attempts = 0;
    const int maxAttempts = std::max(candidateCount * 20, candidateCount + 10);
    while (evaluated < candidateCount && attempts < maxAttempts && !gaStopRequested_) {
        ++attempts;
        HydroRunConfig trialCfg = baseCfg;
        QString candidateSpec;
        if (attempts == 1 && !inputLagsEdit_->text().trimmed().isEmpty()) {
            candidateSpec = candidateToSpec(parseCandidate(inputLagsEdit_->text().trimmed()));
        } else if (attempts <= inputGroups + 1) {
            const int lag = std::min(maxLag, attempts - 1);
            candidateSpec = candidateToSpec(std::vector<std::vector<int>>(static_cast<size_t>(inputGroups), std::vector<int>{std::max(1, lag)}));
        } else if (!successfulCandidates.empty() && evaluated >= std::max(4, candidateCount / 4)) {
            const int eliteCount = std::max(1, std::min<int>(static_cast<int>(successfulCandidates.size()), std::max(2, candidateCount / 10)));
            std::sort(successfulCandidates.begin(), successfulCandidates.end(), [](const LagCandidateSummary& a, const LagCandidateSummary& b) {
                return a.score < b.score;
            });
            std::uniform_int_distribution<int> eliteDist(0, eliteCount - 1);
            candidateSpec = candidateToSpec(mutateCandidate(parseCandidate(successfulCandidates[static_cast<size_t>(eliteDist(rng))].spec)));
        } else {
            candidateSpec = candidateToSpec(randomCandidate());
        }
        if (testedSpecs.find(candidateSpec) != testedSpecs.end()) {
            continue;
        }
        testedSpecs.insert(candidateSpec);
        trialCfg.input_lags_csv = candidateSpec.toStdString();
        try {
            HydroRunResult trial;
            if (mode == "ffn") {
                FFNWrapper runner;
                trial = runner.train(trialCfg);
            } else {
                FFNPINNWrapper runner;
                trial = runner.train(trialCfg);
            }

            const double score = std::isfinite(trial.validation_mse) ? trial.validation_mse : trial.final_loss;
            if (trial.success && std::isfinite(score)) {
                successfulCandidates.push_back({QString::fromStdString(trialCfg.input_lags_csv), score, trial.final_loss});
                const bool improved = score < bestMse;
                if (improved) {
                    bestMse = score;
                    bestLoss = trial.final_loss;
                    bestSpec = QString::fromStdString(trialCfg.input_lags_csv);
                    bestResult = trial;
                }
                const bool progressCheckpoint = ((evaluated + 1) % 10 == 0) || (evaluated + 1 == candidateCount);
                if (improved || progressCheckpoint) {
                    appendLog(QString("GA lag candidate %1/%2%3: lag_steps=%4, validation_mse=%5, loss=%6")
                                  .arg(evaluated + 1)
                                  .arg(candidateCount)
                                  .arg(improved ? " (new best)" : "")
                                  .arg(QString::fromStdString(trialCfg.input_lags_csv))
                                  .arg(trial.validation_mse, 0, 'g', 8)
                                  .arg(trial.final_loss, 0, 'g', 8));
                }
            }
        } catch (const std::exception& e) {
            appendLog(QString("GA lag candidate %1/%2 failed: %3")
                          .arg(evaluated + 1)
                          .arg(candidateCount)
                          .arg(e.what()));
        }
        ++evaluated;
        QCoreApplication::processEvents();
    }

    if (gaStopRequested_) {
        appendLog(QString("GA lag optimization cancelled after %1 evaluated candidate(s).").arg(evaluated));
        statusLabel_->setText("GA lag optimization cancelled.");
        stopGAButton_->setEnabled(false);
        startGAButton_->setEnabled(true);
        updateFfnLagUiState();
        return;
    }

    if (evaluated < candidateCount) {
        appendLog(QString("GA lag optimization evaluated %1/%2 unique candidates; search space was exhausted or duplicate-heavy.")
                      .arg(evaluated)
                      .arg(candidateCount));
    }

    if (bestSpec.isEmpty()) {
        appendLog("GA lag optimization finished without a valid candidate.");
        statusLabel_->setText("GA lag optimization failed.");
    } else {
        std::sort(successfulCandidates.begin(), successfulCandidates.end(), [](const LagCandidateSummary& a, const LagCandidateSummary& b) {
            return a.score < b.score;
        });

        const int confirmCount = std::min<int>(5, static_cast<int>(successfulCandidates.size()));
        HydroRunConfig confirmBaseCfg = currentConfig();
        confirmBaseCfg.use_time_lagged_ffn = true;
        double confirmedScore = std::numeric_limits<double>::infinity();
        double confirmedLoss = std::numeric_limits<double>::infinity();
        QString confirmedSpec = bestSpec;
        HydroRunResult confirmedResult = bestResult;

        appendLog(QString("GA lag optimization confirming top %1 candidate(s) with full epoch count=%2.")
                      .arg(confirmCount)
                      .arg(confirmBaseCfg.epochs));
        for (int i = 0; i < confirmCount && !gaStopRequested_; ++i) {
            HydroRunConfig confirmCfg = confirmBaseCfg;
            confirmCfg.input_lags_csv = successfulCandidates[static_cast<size_t>(i)].spec.toStdString();
            try {
                HydroRunResult confirmTrial;
                if (mode == "ffn") {
                    FFNWrapper runner;
                    confirmTrial = runner.train(confirmCfg);
                } else {
                    FFNPINNWrapper runner;
                    confirmTrial = runner.train(confirmCfg);
                }
                const double score = std::isfinite(confirmTrial.validation_mse) ? confirmTrial.validation_mse : confirmTrial.final_loss;
                appendLog(QString("GA confirmation %1/%2: lag_steps=%3, validation_mse=%4, loss=%5")
                              .arg(i + 1)
                              .arg(confirmCount)
                              .arg(successfulCandidates[static_cast<size_t>(i)].spec)
                              .arg(confirmTrial.validation_mse, 0, 'g', 8)
                              .arg(confirmTrial.final_loss, 0, 'g', 8));
                if (confirmTrial.success && std::isfinite(score) && score < confirmedScore) {
                    confirmedScore = score;
                    confirmedLoss = confirmTrial.final_loss;
                    confirmedSpec = successfulCandidates[static_cast<size_t>(i)].spec;
                    confirmedResult = confirmTrial;
                }
            } catch (const std::exception& e) {
                appendLog(QString("GA confirmation %1/%2 failed: %3")
                              .arg(i + 1)
                              .arg(confirmCount)
                              .arg(e.what()));
            }
            QCoreApplication::processEvents();
        }

        if (gaStopRequested_) {
            appendLog("GA lag optimization cancelled during full-epoch candidate confirmation.");
            statusLabel_->setText("GA lag optimization cancelled.");
            stopGAButton_->setEnabled(false);
            startGAButton_->setEnabled(true);
            updateFfnLagUiState();
            return;
        }

        if (std::isfinite(confirmedScore)) {
            bestSpec = confirmedSpec;
            bestMse = confirmedScore;
            bestLoss = confirmedLoss;
            bestResult = confirmedResult;
        }

        useTimeLaggedFFNCheck_->setChecked(true);
        inputLagsEdit_->setText(bestSpec);
        lastModeResults_[mode] = bestResult;
        updatePlot(mode, bestResult);
        appendLog(QString("GA lag optimization selected lag_steps=%1 (selection_metric=confirmed_validation_mse, score=%2, loss=%3).")
                      .arg(bestSpec)
                      .arg(bestMse, 0, 'g', 8)
                      .arg(bestLoss, 0, 'g', 8));
        statusLabel_->setText(QString("GA lag optimization complete: %1").arg(bestSpec));
        refreshPerformanceAssessment();
    }

    stopGAButton_->setEnabled(false);
    startGAButton_->setEnabled(true);
    updateFfnLagUiState();
}

void HydroPINNWindow::stopGAPlaceholder() {
    gaStopRequested_ = true;
    appendLog("GA stop requested; cancellation will occur after the active training trial finishes.");
    statusLabel_->setText("Stopping GA lag optimization...");
    stopGAButton_->setEnabled(false);
}

void HydroPINNWindow::refreshPerformanceAssessment() {
    const HydroRunConfig cfg = currentConfig();
    const QString backendInfo = (selectedModeKey() == "ffn")
                                    ? "Hydro FFN baseline"
                                    : (selectedModeKey() == "ffn_pinn")
                                          ? "Hydro FFN + PINN residual"
                                          : (selectedModeKey() == "pinn")
                                                ? "Hydro standalone PINN residual"
                                                : (selectedModeKey() == "lstm")
                                                      ? "Hydro LibTorch LSTM"
                                                      : (selectedModeKey() == "lstm_pinn")
                                                            ? "Hydro LibTorch LSTM + PINN residual"
                                                            : "Unknown";

    QString summary = QString(
                          "<b>Performance Assessment Snapshot</b><br/>"
                          "Approach: %1<br/>"
                          "Backend implementation: %2<br/>"
                          "Data source: %3<br/>"
                          "Evaluate metrics: %4<br/>"
                          "Training: epochs=%5, batch=%6, lr=%7<br/>"
                          "PINN: lambda=%8, data_w=%9, physics_w=%10, profile=%11, forcing_gain=%12, collocation=%13<br/>"
                          "Network: layers=%14, lag_steps=%15, FFN input style=%16, activation=%17<br/>"
                          "Split/shuffle: train=%18, validation=0.10, test=remainder, shuffle=%19, seed=%20<br/>"
                          "Optimizer: %21, weight_decay=%22, momentum=%23<br/>"
                          "Normalization: %24<br/>"
                          "Incremental: enabled=%25, window_size=%26, window_step=%27, epochs/window=%28, reset_opt=%29")
                          .arg(modeCombo_->currentText())
                          .arg(backendInfo)
                          .arg(cfg.use_hydro_package ? "Hydro Package" : (cfg.use_csv_data ? "CSV" : "Synthetic"))
                          .arg(cfg.evaluate_metrics ? "yes" : "no")
                          .arg(cfg.epochs)
                          .arg(cfg.batch_size)
                          .arg(cfg.learning_rate, 0, 'g', 6)
                          .arg(cfg.lambda_decay, 0, 'g', 6)
                          .arg(cfg.data_weight, 0, 'g', 6)
                          .arg(cfg.physics_weight, 0, 'g', 6)
                          .arg(QString::fromStdString(cfg.pinn_physics_profile))
                          .arg(cfg.forcing_gain, 0, 'g', 6)
                          .arg(cfg.pinn_collocation_points)
                          .arg(QString::fromStdString(cfg.hidden_layers_csv))
                          .arg(QString::fromStdString(cfg.input_lags_csv))
                          .arg((selectedModeKey() == "ffn" || selectedModeKey() == "ffn_pinn") ? (cfg.use_time_lagged_ffn ? "time-lagged" : "basic") : (selectedModeKey() == "pinn" ? (cfg.pinn_physics_profile == "water_balance" ? "water-balance feature input" : "physics-coordinate input") : "ignored for LSTM"))
                          .arg(QString::fromStdString(cfg.activation))
                          .arg(cfg.train_split_ratio, 0, 'g', 4)
                          .arg(cfg.shuffle_training ? "yes" : "no")
                          .arg(cfg.random_seed)
                          .arg(QString::fromStdString(cfg.optimizer))
                          .arg(cfg.weight_decay, 0, 'g', 6)
                          .arg(cfg.momentum, 0, 'g', 6)
                          .arg(QString::fromStdString(cfg.normalization))
                          .arg(cfg.use_incremental_training ? "yes" : "no")
                          .arg(cfg.window_size, 0, 'g', 6)
                          .arg(cfg.window_step, 0, 'g', 6)
                          .arg(cfg.epochs_per_window)
                          .arg(cfg.reset_optimizer_on_new_window ? "yes" : "no");

    summary += "<br/><br/><b>Latest Approach Results</b><br/>";
    const QStringList orderedModes = {"ffn", "ffn_pinn", "lstm", "lstm_pinn", "pinn"};
    bool hasAnyModeResult = false;
    for (const QString& mode : orderedModes) {
        auto it = lastModeResults_.find(mode);
        if (it == lastModeResults_.end()) {
            summary += QString("%1: no run yet.<br/>").arg(modeDisplayName(mode));
            continue;
        }

        hasAnyModeResult = true;
        const HydroRunResult& r = it->second;
        summary += QString("%1: %2, final_loss=%3")
                       .arg(modeDisplayName(mode))
                       .arg(r.success ? "success" : "failed")
                       .arg(r.final_loss, 0, 'g', 8);

        summary += QString(", validation_mse=%1, test_mse=%2, rmse=%3, mae=%4, nse=%5, kge=%6, r=%7, pbias=%8, volume_error=%9, peak_timing_error=%10, peak_magnitude_error_percent=%11, high_flow_rmse=%12, low_flow_rmse=%13, physics_residual_mean=%14, physics_residual_rmse=%15, cumulative_physics_residual=%16, physics_loss=%17")
                       .arg(r.validation_mse, 0, 'g', 8)
                       .arg(r.mse, 0, 'g', 8)
                       .arg(r.rmse, 0, 'g', 8)
                       .arg(r.mae, 0, 'g', 8)
                       .arg(r.nse, 0, 'g', 8)
                       .arg(r.kge, 0, 'g', 8)
                       .arg(r.correlation, 0, 'g', 8)
                       .arg(r.pbias, 0, 'g', 8)
                       .arg(r.volume_error_percent, 0, 'g', 8)
                       .arg(r.peak_timing_error, 0, 'g', 8)
                       .arg(r.peak_magnitude_error_percent, 0, 'g', 8)
                       .arg(r.high_flow_rmse, 0, 'g', 8)
                       .arg(r.low_flow_rmse, 0, 'g', 8)
                       .arg(r.physics_residual_mean, 0, 'g', 8)
                       .arg(r.physics_residual_rmse, 0, 'g', 8)
                       .arg(r.cumulative_physics_residual, 0, 'g', 8)
                       .arg(r.physics_loss, 0, 'g', 8);

        if (!r.message.empty()) {
            summary += QString(", msg=%1").arg(QString::fromStdString(r.message).toHtmlEscaped());
        }
        summary += "<br/>";
    }

    if (!hasAnyModeResult) {
        summary += "No approach runs have been recorded yet.<br/>";
    }

    perfSummaryText_->setHtml(summary);
    appendLog("Performance assessment snapshot refreshed.");
}

void HydroPINNWindow::exportExperimentArtifacts() {
    if (lastModeResults_.empty()) {
        QMessageBox::information(this, "HydroPINN Export", "Run at least one approach before exporting.");
        return;
    }
    const QString directory = QFileDialog::getExistingDirectory(this, "Select experiment output directory");
    if (directory.isEmpty()) return;
    const QString experimentId = QString("hydro_%1").arg(QDateTime::currentDateTimeUtc().toString("yyyyMMdd_HHmmss_zzz"));
    std::map<std::string, HydroRunResult> results;
    for (const auto& entry : lastModeResults_) results.emplace(entry.first.toStdString(), entry.second);
    try {
        HydroExperimentExporter().exportRun(directory.toStdString(), experimentId.toStdString(), currentConfig(), results);
        appendLog(QString("Experiment artifacts exported to %1/%2").arg(directory, experimentId));
        QMessageBox::information(this, "HydroPINN Export", QString("Exported experiment:\n%1/%2").arg(directory, experimentId));
    } catch (const std::exception& error) {
        appendLog(QString("Experiment export failed: %1").arg(error.what()));
        QMessageBox::critical(this, "HydroPINN Export", QString::fromUtf8(error.what()));
    }
}

void HydroPINNWindow::loadInferenceArtifacts() {
    const QString directory = QFileDialog::getExistingDirectory(
        this, "Load Hydro inference artifact directory");
    if (directory.isEmpty()) return;

    try {
        auto artifacts = HydroArtifactLoader().loadForInference(directory.toStdString());
        QStringList approaches;
        std::map<QString, std::unique_ptr<HydroInferenceSession>> preparedSessions;
        for (const auto& entry : artifacts.models) {
            const QString approach = QString::fromStdString(entry.first);
            approaches.push_back(approach);
            preparedSessions.emplace(
                approach, std::make_unique<HydroInferenceSession>(artifacts, entry.first));
        }
        const QString experimentId = QString::fromStdString(artifacts.experiment.experiment_id);
        auto storedResults = HydroArtifactLoader().loadPredictions(directory.toStdString());
        const auto storedMetrics = HydroArtifactLoader().loadMetrics(directory.toStdString());
        for (auto& entry : storedResults) {
            const auto metric = storedMetrics.find(entry.first);
            if (metric == storedMetrics.end()) throw std::runtime_error("Exported metrics are missing approach: " + entry.first);
            const auto consistent = [](const double recomputed, const double exported) {
                return !std::isfinite(exported) ||
                       (std::isfinite(recomputed) &&
                        std::abs(recomputed - exported) <= 1.0e-10 * std::max({1.0, std::abs(recomputed), std::abs(exported)}));
            };
            if (!consistent(entry.second.mse, metric->second.mse) || !consistent(entry.second.rmse, metric->second.rmse) ||
                !consistent(entry.second.mae, metric->second.mae)) {
                throw std::runtime_error("Exported summary metrics do not match predictions for: " + entry.first);
            }
            entry.second.final_loss = metric->second.final_loss;
            entry.second.validation_mse = metric->second.validation_mse;
            entry.second.physics_loss = metric->second.physics_loss;
        }
        if (storedMetrics.size() != storedResults.size()) throw std::runtime_error("Exported metrics and predictions have different approach sets.");
        const auto storedResiduals = HydroArtifactLoader().loadPhysicsResiduals(directory.toStdString());
        for (const auto& entry : storedResiduals) {
            const auto result = storedResults.find(entry.first);
            if (result == storedResults.end() || entry.second.values.size() != result->second.x.size() ||
                entry.second.split != result->second.split) {
                throw std::runtime_error("Exported physics residuals do not align with predictions for: " + entry.first);
            }
            for (std::size_t i = 0; i < entry.second.x.size(); ++i) {
                const double tolerance = 1.0e-10 * std::max({1.0, std::abs(entry.second.x[i]), std::abs(result->second.x[i])});
                if (std::abs(entry.second.x[i] - result->second.x[i]) > tolerance) {
                    throw std::runtime_error("Exported physics residual timestamps do not match predictions for: " + entry.first);
                }
            }
            result->second.physics_residual = entry.second.values;
            populateHydroPhysicsResidualMetrics(result->second);
        }
        std::map<QString, HydroRunResult> restoredResults;
        for (const auto& entry : storedResults) {
            if (artifacts.models.find(entry.first) == artifacts.models.end()) {
                throw std::runtime_error("Exported predictions have no matching checkpoint: " + entry.first);
            }
            restoredResults.emplace(QString::fromStdString(entry.first), entry.second);
        }
        lastModeResults_ = std::move(restoredResults);
        inferenceSessions_ = std::move(preparedSessions);
        loadedInferenceArtifacts_ = std::make_unique<HydroInferenceArtifacts>(std::move(artifacts));
        appendLog(QString("Loaded and prepared inference artifacts for experiment '%1': %2.")
                      .arg(experimentId, approaches.join(", ")));
        QMessageBox::information(
            this, "HydroPINN Inference Artifacts",
            QString("Loaded experiment '%1'.\nAvailable checkpoints: %2")
                .arg(experimentId, approaches.join(", ")));
        refreshPerformanceAssessment();
    } catch (const std::exception& error) {
        loadedInferenceArtifacts_.reset();
        inferenceSessions_.clear();
        appendLog(QString("Inference artifact load failed: %1").arg(error.what()));
        QMessageBox::critical(this, "HydroPINN Inference Artifacts", QString::fromUtf8(error.what()));
    }
}

void HydroPINNWindow::loadExperimentConfiguration() {
    const QString path = QFileDialog::getOpenFileName(
        this, "Load Hydro experiment configuration", QString(), "Hydro experiment config (experiment_config.json);;JSON files (*.json)");
    if (path.isEmpty()) return;
    try {
        const auto loaded = HydroExperimentLoader().loadConfig(path.toStdString());
        const auto& cfg = loaded.config;
        auto selectText = [](QComboBox* combo, const std::string& value) {
            const int index = combo->findText(QString::fromStdString(value), Qt::MatchFixedString);
            if (index < 0) throw std::runtime_error("Configuration value is not supported by this GUI: " + value);
            combo->setCurrentIndex(index);
        };
        epochsSpin_->setValue(cfg.epochs);
        batchSpin_->setValue(cfg.batch_size);
        lrSpin_->setValue(cfg.learning_rate);
        lambdaSpin_->setValue(cfg.lambda_decay);
        dataWeightSpin_->setValue(cfg.data_weight);
        physicsWeightSpin_->setValue(cfg.physics_weight);
        forcingGainSpin_->setValue(cfg.forcing_gain);
        pinnCollocationSpin_->setValue(cfg.pinn_collocation_points);
        selectText(pinnPhysicsProfileCombo_, cfg.pinn_physics_profile);
        hiddenLayersEdit_->setText(QString::fromStdString(cfg.hidden_layers_csv));
        inputLagsEdit_->setText(QString::fromStdString(cfg.input_lags_csv));
        useTimeLaggedFFNCheck_->setChecked(cfg.use_time_lagged_ffn);
        selectText(activationCombo_, cfg.activation);
        evalCheck_->setChecked(cfg.evaluate_metrics);
        splitRatioSpin_->setValue(cfg.train_split_ratio);
        shuffleCheck_->setChecked(cfg.shuffle_training);
        seedSpin_->setValue(cfg.random_seed);
        selectText(optimizerCombo_, cfg.optimizer);
        weightDecaySpin_->setValue(cfg.weight_decay);
        momentumSpin_->setValue(cfg.momentum);
        selectText(normalizationCombo_, cfg.normalization);
        dataSourceCombo_->setCurrentText(cfg.use_hydro_package ? "Hydro Package" : (cfg.use_csv_data ? "CSV File" : "Synthetic"));
        csvPathEdit_->setText(QString::fromStdString(cfg.csv_path));
        csvXColSpin_->setValue(cfg.csv_x_column);
        csvYColSpin_->setValue(cfg.csv_y_column);
        csvHeaderCheck_->setChecked(cfg.csv_has_header);
        selectText(profileCombo_, cfg.synthetic_profile);
        sampleCountSpin_->setValue(cfg.sample_count);
        tStartSpin_->setValue(cfg.t_start);
        tEndSpin_->setValue(cfg.t_end);
        hydroPackagePathEdit_->setText(QString::fromStdString(cfg.hydro_package_path));
        hydroCatchmentIdEdit_->setText(QString::fromStdString(cfg.hydro_catchment_id));
        selectText(hydroPackageProfileCombo_, cfg.hydro_package_profile);
        hydroForecastFeatureCheck_->setChecked(cfg.use_hydro_forecast_feature);
        hydroForecastVariableEdit_->setText(QString::fromStdString(cfg.hydro_forecast_variable));
        hydroForecastLeadSpin_->setValue(cfg.hydro_forecast_lead_hours);
        hydroForecastEnsembleEdit_->setText(QString::fromStdString(cfg.hydro_forecast_ensemble_member));
        updateDataSourceUiState();
        updateFfnLagUiState();
        appendLog(QString("Loaded experiment configuration '%1' from %2")
                      .arg(QString::fromStdString(loaded.experiment_id), path));
        QMessageBox::information(this, "HydroPINN", "Experiment configuration loaded. Review paths before running.");
    } catch (const std::exception& error) {
        appendLog(QString("Experiment configuration load failed: %1").arg(error.what()));
        QMessageBox::critical(this, "HydroPINN", QString::fromUtf8(error.what()));
    }
}

void HydroPINNWindow::clearPlot() {
    auto* chart = chartView_->chart();
    chart->removeAllSeries();

    const auto existingAxes = chart->axes();
    for (QAbstractAxis* axis : existingAxes) {
        chart->removeAxis(axis);
        delete axis;
    }

    chart->setTitle("Prediction vs Target (Test Set)");
    appendLog("Plot cleared.");
}

void HydroPINNWindow::zoomInPlot() {
    chartView_->chart()->zoomIn();
    appendLog("Plot zoomed in.");
}

void HydroPINNWindow::zoomOutPlot() {
    chartView_->chart()->zoomOut();
    appendLog("Plot zoomed out.");
}

void HydroPINNWindow::fitPlotAxesInternal(bool logMessage) {
    auto* chart = chartView_->chart();
    chart->zoomReset();

    double minX = std::numeric_limits<double>::infinity();
    double maxX = -std::numeric_limits<double>::infinity();
    double minY = std::numeric_limits<double>::infinity();
    double maxY = -std::numeric_limits<double>::infinity();

    bool hasPoints = false;
    for (QAbstractSeries* baseSeries : chart->series()) {
        auto* xy = qobject_cast<QXYSeries*>(baseSeries);
        if (!xy) continue;
        const auto points = xy->pointsVector();
        for (const QPointF& p : points) {
            minX = std::min(minX, p.x());
            maxX = std::max(maxX, p.x());
            minY = std::min(minY, p.y());
            maxY = std::max(maxY, p.y());
            hasPoints = true;
        }
    }

    if (!hasPoints) {
        if (logMessage) appendLog("Fit Axes: no plottable points found.");
        return;
    }

    const double dx = std::max(1e-9, maxX - minX);
    const double dy = std::max(1e-9, maxY - minY);
    const double padX = 0.05 * dx;
    const double padY = 0.05 * dy;

    for (QAbstractAxis* axisBase : chart->axes(Qt::Horizontal)) {
        auto* axis = qobject_cast<QValueAxis*>(axisBase);
        if (axis) axis->setRange(minX - padX, maxX + padX);
    }
    for (QAbstractAxis* axisBase : chart->axes(Qt::Vertical)) {
        auto* axis = qobject_cast<QValueAxis*>(axisBase);
        if (axis) axis->setRange(minY - padY, maxY + padY);
    }

    if (logMessage) appendLog("Plot axes fit to current data extents.");
}

void HydroPINNWindow::fitPlotAxes() {
    fitPlotAxesInternal(true);
}

void HydroPINNWindow::showSyntheticInputsOutputs() {
    if (lastSyntheticX_.empty() || lastSyntheticTarget_.empty()) {
        QMessageBox::information(this,
                                 "HydroPINN Plot",
                                 "No synthetic input/output data available yet. Generate synthetic data first.");
        return;
    }

    auto* chart = chartView_->chart();
    chart->removeAllSeries();
    const auto existingAxes = chart->axes();
    for (QAbstractAxis* axis : existingAxes) {
        chart->removeAxis(axis);
        delete axis;
    }

    const size_t n = lastSyntheticX_.size();
    for (const auto& kv : lastSyntheticInputs_) {
        auto* s = new QLineSeries(chart);
        s->setName(kv.first + " (input)");
        const auto& vals = kv.second;
        const size_t m = std::min(n, vals.size());
        for (size_t i = 0; i < m; ++i) s->append(lastSyntheticX_[i], vals[i]);
        chart->addSeries(s);
    }

    auto* targetSeries = new QLineSeries(chart);
    targetSeries->setName("target (output)");
    for (size_t i = 0; i < n; ++i) targetSeries->append(lastSyntheticX_[i], lastSyntheticTarget_[i]);
    chart->addSeries(targetSeries);

    auto* axisX = new QValueAxis(chart);
    axisX->setTitleText("t");
    auto* axisY = new QValueAxis(chart);
    axisY->setTitleText("value");
    chart->addAxis(axisX, Qt::AlignBottom);
    chart->addAxis(axisY, Qt::AlignLeft);
    for (QAbstractSeries* series : chart->series()) {
        series->attachAxis(axisX);
        series->attachAxis(axisY);
    }

    // Explicitly fit to min/max across all synthetic inputs + output.
    double minX = std::numeric_limits<double>::infinity();
    double maxX = -std::numeric_limits<double>::infinity();
    double minY = std::numeric_limits<double>::infinity();
    double maxY = -std::numeric_limits<double>::infinity();

    for (double x : lastSyntheticX_) {
        minX = std::min(minX, x);
        maxX = std::max(maxX, x);
    }
    for (const auto& kv : lastSyntheticInputs_) {
        for (double v : kv.second) {
            minY = std::min(minY, v);
            maxY = std::max(maxY, v);
        }
    }
    for (double y : lastSyntheticTarget_) {
        minY = std::min(minY, y);
        maxY = std::max(maxY, y);
    }

    if (std::isfinite(minX) && std::isfinite(maxX) && std::isfinite(minY) && std::isfinite(maxY)) {
        const double dx = std::max(1e-9, maxX - minX);
        const double dy = std::max(1e-9, maxY - minY);
        axisX->setRange(minX - 0.03 * dx, maxX + 0.03 * dx);
        axisY->setRange(minY - 0.05 * dy, maxY + 0.05 * dy);
    }

    chart->setTitle("Synthetic Inputs + Output (NeuroForge-style)");
    chart->legend()->setVisible(true);
    appendLog("Displayed synthetic inputs and output target on plot (fit by combined min/max of all series).");
}

void HydroPINNWindow::runSelectedMode() {
    runMode(selectedModeKey());
}

void HydroPINNWindow::runAllModes() {
    const QStringList modes = {"ffn", "ffn_pinn", "lstm", "lstm_pinn", "pinn"};
    for (const QString& m : modes) {
        runMode(m);
    }
}

void HydroPINNWindow::showPredictionForMode(const QString& mode) {
    if (predictionUseCurrentDataCheck_->isChecked()) {
        if (!runLoadedInferenceForMode(mode)) {
            QMessageBox::warning(this, "HydroPINN Inference",
                                 "Loaded checkpoint inference failed. Review the Logs tab for details.");
        }
        return;
    }

    const auto it = lastModeResults_.find(mode);
    if (it == lastModeResults_.end()) {
        appendLog(QString("No stored prediction available for approach '%1'. Run training first.").arg(modeDisplayName(mode)));
        QMessageBox::information(this,
                                 "HydroPINN Prediction",
                                 QString("No stored result for approach '%1'.\nTrain it from the Training tab first.").arg(modeDisplayName(mode)));
        return;
    }
    updatePlot(mode, it->second);
    appendLog(QString("Displayed stored target vs prediction for approach '%1'.").arg(modeDisplayName(mode)));
}

bool HydroPINNWindow::runLoadedInferenceForMode(const QString& mode) {
    const auto session = inferenceSessions_.find(mode);
    if (session == inferenceSessions_.end()) {
        appendLog(QString("No loaded checkpoint session is available for '%1'.").arg(modeDisplayName(mode)));
        return false;
    }
    const HydroRunConfig current = currentConfig();
    if (!current.use_hydro_package && !current.use_csv_data) {
        appendLog("Loaded checkpoint inference currently requires Hydro Package or CSV as the Data tab source.");
        return false;
    }
    if (!loadedInferenceArtifacts_) {
        appendLog("Loaded checkpoint inference has no active experiment configuration.");
        return false;
    }
    try {
        const HydroRunConfig& exported = loadedInferenceArtifacts_->experiment.config;
        if (current.use_hydro_package != exported.use_hydro_package ||
            current.use_csv_data != exported.use_csv_data) {
            throw std::runtime_error("Current data-source type does not match the loaded experiment.");
        }
        if (current.use_hydro_package && (current.hydro_package_profile != exported.hydro_package_profile ||
            current.use_hydro_forecast_feature != exported.use_hydro_forecast_feature ||
            (exported.use_hydro_forecast_feature &&
             (current.hydro_forecast_variable != exported.hydro_forecast_variable ||
              std::abs(current.hydro_forecast_lead_hours - exported.hydro_forecast_lead_hours) > 1.0e-9 ||
              current.hydro_forecast_ensemble_member != exported.hydro_forecast_ensemble_member)))) {
            throw std::runtime_error(
                "Current Hydro Package feature settings do not match the loaded experiment configuration.");
        }
        if (current.use_csv_data &&
            (current.csv_x_column != exported.csv_x_column || current.csv_y_column != exported.csv_y_column ||
             current.csv_has_header != exported.csv_has_header || current.synthetic_profile != exported.synthetic_profile)) {
            throw std::runtime_error("Current CSV column/profile settings do not match the loaded experiment.");
        }
        HydroRunConfig config = exported;
        config.hydro_package_path = current.hydro_package_path;
        config.hydro_catchment_id = current.hydro_catchment_id;
        config.csv_path = current.csv_path;
        torch::Tensor inputs;
        torch::Tensor targets;
        torch::Tensor plotX;
        if (config.use_hydro_package) {
            if (!loadHydroPackageTensors(config, inputs, targets, plotX))
                throw std::runtime_error("Unable to build tensors from the selected Hydro package.");
        } else loadHydroCsvTensors(config, inputs, targets, plotX);
        if ((mode == "ffn" || mode == "ffn_pinn") && exported.use_time_lagged_ffn) {
            const auto lagged = buildHydroLaggedTensor(inputs, exported.input_lags_csv);
            inputs = lagged.inputs;
            targets = targets.slice(0, lagged.leading_rows, targets.size(0)).contiguous();
            plotX = plotX.slice(0, lagged.leading_rows, plotX.size(0)).contiguous();
        }
        const torch::Tensor predictions = session->second->predictSeries(inputs).to(torch::kCPU).contiguous();
        const int64_t offset = inputs.size(0) - predictions.size(0);
        if (offset < 0 || offset + predictions.size(0) > targets.size(0)) {
            throw std::runtime_error("Inference predictions do not align with the selected package.");
        }
        const auto alignedTargets = targets.slice(0, offset, offset + predictions.size(0)).to(torch::kCPU).contiguous();
        const auto alignedX = plotX.slice(0, offset, offset + predictions.size(0)).to(torch::kCPU).contiguous();
        HydroRunResult result;
        result.success = true;
        result.message = "Loaded checkpoint inference on current Hydro package.";
        result.x.reserve(static_cast<std::size_t>(predictions.size(0)));
        result.y_true.reserve(result.x.capacity());
        result.y_pred.reserve(result.x.capacity());
        result.split.assign(static_cast<std::size_t>(predictions.size(0)), "test");
        for (int64_t i = 0; i < predictions.size(0); ++i) {
            result.x.push_back(alignedX[i].item<double>());
            result.y_true.push_back(alignedTargets[i].item<double>());
            result.y_pred.push_back(predictions[i].item<double>());
        }
        populateHydroMetrics(result, result.y_true, result.y_pred);
        populateHydroPeakMetrics(result);
        lastModeResults_[mode] = std::move(result);
        updatePlot(mode, lastModeResults_.at(mode));
        appendLog(QString("Ran loaded '%1' checkpoint on %2 current package samples.")
                      .arg(modeDisplayName(mode)).arg(predictions.size(0)));
        return true;
    } catch (const std::exception& error) {
        appendLog(QString("Loaded checkpoint inference failed for '%1': %2")
                      .arg(modeDisplayName(mode), QString::fromUtf8(error.what())));
        return false;
    }
}

void HydroPINNWindow::showSelectedPrediction() {
    showPredictionForMode(selectedModeKey());
}

void HydroPINNWindow::showAllPredictions() {
    if (predictionUseCurrentDataCheck_->isChecked()) {
        for (const auto& entry : inferenceSessions_) runLoadedInferenceForMode(entry.first);
        return;
    }

    const QStringList modes = {"ffn", "ffn_pinn", "lstm", "lstm_pinn", "pinn"};
    for (const QString& mode : modes) {
        if (lastModeResults_.find(mode) == lastModeResults_.end()) {
            appendLog(QString("Skipping approach '%1' (no stored prediction yet).").arg(modeDisplayName(mode)));
            continue;
        }
        updatePlot(mode, lastModeResults_[mode]);
    }
}

void HydroPINNWindow::updatePlot(const QString& mode, const HydroRunResult& result) {
    if (result.x.empty() || result.y_true.empty() || result.y_pred.empty()) {
        return;
    }

    auto* chart = chartView_->chart();
    chart->removeAllSeries();

    const auto existingAxes = chart->axes();
    for (QAbstractAxis* axis : existingAxes) {
        chart->removeAxis(axis);
        delete axis;
    }

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

    chart->setTitle(QString("Prediction vs Target - %1").arg(modeDisplayName(mode)));
    chart->legend()->setVisible(true);
    fitPlotAxesInternal(false);
}



void HydroPINNWindow::plotAllTargetVsPredicted() {
    const QStringList modes = {"ffn", "ffn_pinn", "lstm", "lstm_pinn", "pinn"};
    auto* chart = chartView_->chart();
    chart->removeAllSeries();
    const auto existingAxes = chart->axes();
    for (QAbstractAxis* axis : existingAxes) { chart->removeAxis(axis); delete axis; }

    bool addedAny = false;
    bool targetAdded = false;

    const QList<QColor> modeColors = {QColor(30, 144, 255), QColor(220, 20, 60), QColor(86, 180, 233), QColor(46, 139, 87), QColor(255, 140, 0)};
    const QList<Qt::PenStyle> modeStyles = {Qt::SolidLine, Qt::DashLine, Qt::DotLine, Qt::DashDotLine, Qt::DashDotDotLine};

    int modeIdx = 0;
    for (const QString& mode : modes) {
        auto it = lastModeResults_.find(mode);
        if (it == lastModeResults_.end()) continue;
        const HydroRunResult& r = it->second;
        const size_t n = std::min(r.x.size(), std::min(r.y_true.size(), r.y_pred.size()));
        if (n == 0) continue;

        if (!targetAdded) {
            auto* target = new QLineSeries(chart);
            target->setName("Target");
            QPen targetPen(QColor(60, 60, 60));
            targetPen.setWidth(2);
            target->setPen(targetPen);
            for (size_t i = 0; i < n; ++i) target->append(r.x[i], r.y_true[i]);
            chart->addSeries(target);
            targetAdded = true;
            addedAny = true;
        }

        auto* pred = new QLineSeries(chart);
        pred->setName(QString("Prediction (%1)").arg(modeDisplayName(mode)));
        QPen predPen(modeColors[modeIdx % modeColors.size()]);
        predPen.setWidth(2 + (modeIdx % 2));
        predPen.setStyle(modeStyles[modeIdx % modeStyles.size()]);
        pred->setPen(predPen);

        for (size_t i = 0; i < n; ++i) pred->append(r.x[i], r.y_pred[i]);
        chart->addSeries(pred);
        addedAny = true;
        ++modeIdx;
    }

    if (!addedAny) {
        appendLog("No stored approach results available for all-mode target/predicted plot.");
        return;
    }

    auto* axisX = new QValueAxis(chart);
    axisX->setTitleText("t");
    auto* axisY = new QValueAxis(chart);
    axisY->setTitleText("y");
    chart->addAxis(axisX, Qt::AlignBottom);
    chart->addAxis(axisY, Qt::AlignLeft);
    for (auto* series : chart->series()) {
        series->attachAxis(axisX);
        series->attachAxis(axisY);
    }
    chart->setTitle("Target vs Predicted (All Approaches)");
    chart->legend()->setVisible(true);
    fitPlotAxesInternal(false);
    appendLog("Displayed target vs predicted curves for all stored approaches (styled per mode for overlap visibility).");
}

void HydroPINNWindow::plotOneToOneAllModes() {
    const QStringList modes = {"ffn", "ffn_pinn", "lstm", "lstm_pinn", "pinn"};
    const QList<QColor> modeColors = {
        QColor(0, 114, 178),  // blue
        QColor(213, 94, 0),   // vermillion
        QColor(86, 180, 233), // sky blue
        QColor(0, 158, 115),  // bluish green
        QColor(204, 121, 167) // purple
    };
    auto* chart = chartView_->chart();
    chart->removeAllSeries();
    const auto existingAxes = chart->axes();
    for (QAbstractAxis* axis : existingAxes) { chart->removeAxis(axis); delete axis; }

    bool addedAny = false;
    double minV = std::numeric_limits<double>::infinity();
    double maxV = -std::numeric_limits<double>::infinity();

    int modeIdx = 0;
    for (const QString& mode : modes) {
        auto it = lastModeResults_.find(mode);
        if (it == lastModeResults_.end()) continue;
        const HydroRunResult& r = it->second;
        const size_t n = std::min(r.y_true.size(), r.y_pred.size());
        if (n == 0) continue;

        double meanY = 0.0;
        for (size_t i = 0; i < n; ++i) meanY += r.y_true[i];
        meanY /= static_cast<double>(n);
        double ssRes = 0.0;
        double ssTot = 0.0;

        auto* pts = new QScatterSeries(chart);
        pts->setMarkerSize(7.0);
        pts->setColor(modeColors[modeIdx % modeColors.size()]);
        for (size_t i = 0; i < n; ++i) {
            pts->append(r.y_true[i], r.y_pred[i]);
            minV = std::min(minV, std::min(r.y_true[i], r.y_pred[i]));
            maxV = std::max(maxV, std::max(r.y_true[i], r.y_pred[i]));
            const double e = r.y_true[i] - r.y_pred[i];
            ssRes += e * e;
            const double d = r.y_true[i] - meanY;
            ssTot += d * d;
        }
        const double r2 = (ssTot > 1e-12) ? (1.0 - ssRes / ssTot) : 0.0;
        pts->setName(QString("%1 (R²=%2)").arg(modeDisplayName(mode)).arg(r2, 0, 'f', 3));
        chart->addSeries(pts);
        addedAny = true;
        ++modeIdx;
    }

    if (!addedAny) {
        appendLog("No stored approach results available for 1:1 target/predicted plot.");
        return;
    }

    if (!std::isfinite(minV) || !std::isfinite(maxV) || minV == maxV) {
        minV = -1.0;
        maxV = 1.0;
    }

    auto* identity = new QLineSeries(chart);
    identity->setName("y = x");
    identity->append(minV, minV);
    identity->append(maxV, maxV);
    chart->addSeries(identity);

    auto* axisX = new QValueAxis(chart);
    axisX->setTitleText("Target");
    auto* axisY = new QValueAxis(chart);
    axisY->setTitleText("Predicted");
    chart->addAxis(axisX, Qt::AlignBottom);
    chart->addAxis(axisY, Qt::AlignLeft);
    for (auto* series : chart->series()) {
        series->attachAxis(axisX);
        series->attachAxis(axisY);
    }
    chart->setTitle("1:1 Target vs Predicted (All Approaches)");
    chart->legend()->setVisible(true);
    fitPlotAxesInternal(false);
    appendLog("Displayed 1:1 target vs predicted scatter plot for all stored approaches (R² shown in legend).");
}

void HydroPINNWindow::plotTaylorDiagramAllModes() {
    const QStringList modes = {"ffn", "ffn_pinn", "lstm", "lstm_pinn", "pinn"};
    const QList<QColor> modeColors = {
        QColor(0, 114, 178),
        QColor(213, 94, 0),
        QColor(86, 180, 233),
        QColor(0, 158, 115),
        QColor(204, 121, 167)
    };
    auto* chart = chartView_->chart();
    chart->removeAllSeries();
    const auto existingAxes = chart->axes();
    for (QAbstractAxis* axis : existingAxes) { chart->removeAxis(axis); delete axis; }

    bool haveReference = false;
    double refStd = 0.0;
    double maxRadius = 1.0;

    struct TaylorPoint {
        QString mode;
        double corr;
        double stddev;
        double x;
        double y;
    };
    std::vector<TaylorPoint> points;

    auto meanStd = [](const std::vector<double>& v) {
        double m = 0.0;
        for (double x : v) m += x;
        m /= static_cast<double>(v.size());
        double var = 0.0;
        for (double x : v) { const double d = x - m; var += d * d; }
        var /= static_cast<double>(v.size());
        return std::pair<double, double>(m, std::sqrt(std::max(0.0, var)));
    };

    for (const QString& mode : modes) {
        auto it = lastModeResults_.find(mode);
        if (it == lastModeResults_.end()) continue;
        const HydroRunResult& r = it->second;
        const size_t n = std::min(r.y_true.size(), r.y_pred.size());
        if (n < 2) continue;

        std::vector<double> yt(r.y_true.begin(), r.y_true.begin() + static_cast<long>(n));
        std::vector<double> yp(r.y_pred.begin(), r.y_pred.begin() + static_cast<long>(n));

        auto [mt, st] = meanStd(yt);
        auto [mp, sp] = meanStd(yp);
        if (!haveReference) {
            refStd = st;
            haveReference = true;
        }

        double cov = 0.0;
        for (size_t i = 0; i < n; ++i) cov += (yt[i] - mt) * (yp[i] - mp);
        cov /= static_cast<double>(n);
        double corr = (st > 0.0 && sp > 0.0) ? (cov / (st * sp)) : 0.0;
        corr = std::max(-1.0, std::min(1.0, corr));

        const double theta = std::acos(corr);
        const double x = sp * std::cos(theta);
        const double y = sp * std::sin(theta);

        points.push_back({mode, corr, sp, x, y});
        maxRadius = std::max(maxRadius, std::hypot(x, y));
    }

    if (!haveReference || points.empty()) {
        appendLog("No stored approach results available for Taylor diagram.");
        return;
    }

    maxRadius = std::max(maxRadius, std::abs(refStd));
    const double axisMax = std::max(1e-6, maxRadius * 1.1);

    // Draw standard-deviation circles (quarter arcs) with adaptive count.
    const int arcSegments = 120;
    const int ringCount = std::max(3, std::min(8, static_cast<int>(std::ceil(axisMax / std::max(1e-6, refStd * 0.5)))));
    for (int ring = 1; ring <= ringCount; ++ring) {
        const double radius = axisMax * static_cast<double>(ring) / static_cast<double>(ringCount);
        auto* arc = new QLineSeries(chart);
        arc->setName(QString("σ=%1").arg(radius, 0, 'g', 3));
        QPen arcPen(QColor(190, 190, 190));
        arcPen.setStyle(Qt::DashLine);
        arc->setPen(arcPen);
        for (int i = 0; i <= arcSegments; ++i) {
            const double halfPi = 1.5707963267948966;
            const double angle = (halfPi * static_cast<double>(i)) / static_cast<double>(arcSegments);
            arc->append(radius * std::cos(angle), radius * std::sin(angle));
        }
        chart->addSeries(arc);
    }

    // Draw correlation rays.
    const std::vector<double> corrGuides = {0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0};
    for (double c : corrGuides) {
        const double theta = std::acos(std::max(-1.0, std::min(1.0, c)));
        auto* ray = new QLineSeries(chart);
        ray->setName(QString("r=%1").arg(c, 0, 'g', 3));
        QPen rayPen(QColor(210, 210, 210));
        rayPen.setStyle(Qt::DotLine);
        ray->setPen(rayPen);
        ray->append(0.0, 0.0);
        ray->append(axisMax * std::cos(theta), axisMax * std::sin(theta));
        chart->addSeries(ray);
    }

    // Make overlapping points visible by deterministic radial offset.
    std::map<QString, int> overlapCounter;
    int pointIdx = 0;
    for (const TaylorPoint& pnt : points) {
        const QString key = QString("%1|%2")
                                .arg(std::round(pnt.x * 1000.0) / 1000.0, 0, 'f', 3)
                                .arg(std::round(pnt.y * 1000.0) / 1000.0, 0, 'f', 3);
        const int overlapIdx = overlapCounter[key]++;
        const double norm = std::max(1e-12, std::hypot(pnt.x, pnt.y));
        const double offset = axisMax * 0.02 * static_cast<double>(overlapIdx);
        const double x = pnt.x + offset * (pnt.x / norm);
        const double y = pnt.y + offset * (pnt.y / norm);

        auto* point = new QScatterSeries(chart);
        point->setName(QString("%1 (r=%2, σ=%3)")
                           .arg(modeDisplayName(pnt.mode))
                           .arg(pnt.corr, 0, 'f', 2)
                           .arg(pnt.stddev, 0, 'g', 4));
        point->setMarkerSize(11.0);
        point->setColor(modeColors[pointIdx % modeColors.size()]);
        point->append(x, y);
        chart->addSeries(point);
        ++pointIdx;
    }

    auto* ref = new QScatterSeries(chart);
    ref->setName("Reference target");
    ref->setMarkerSize(13.0);
    ref->setColor(QColor(80, 80, 80));
    ref->append(refStd, 0.0);
    chart->addSeries(ref);

    auto* axisX = new QValueAxis(chart);
    axisX->setTitleText("σ cos(θ)");
    axisX->setRange(0.0, axisMax);
    auto* axisY = new QValueAxis(chart);
    axisY->setTitleText("σ sin(θ)");
    axisY->setRange(0.0, axisMax);
    chart->addAxis(axisX, Qt::AlignBottom);
    chart->addAxis(axisY, Qt::AlignLeft);
    for (auto* series : chart->series()) {
        series->attachAxis(axisX);
        series->attachAxis(axisY);
    }

    chart->setTitle("Taylor Diagram (All Approaches)");
    chart->legend()->setVisible(true);
    fitPlotAxesInternal(false);
    appendLog(QString("Displayed Taylor diagram for all stored approaches (adaptive σ-circles=%1, high-contrast colors).").arg(ringCount));
}
void HydroPINNWindow::showModeSubplots() {
    const QStringList modes = {"ffn", "ffn_pinn", "lstm", "lstm_pinn", "pinn"};
    const QStringList titles = {"FFN", "FFN + PINN", "LSTM", "LSTM + PINN", "PINN"};
    const QList<QColor> modeColors = {
        QColor(0, 114, 178),
        QColor(213, 94, 0),
        QColor(86, 180, 233),
        QColor(0, 158, 115),
        QColor(204, 121, 167)
    };

    auto* chart = chartView_->chart();
    chart->removeAllSeries();
    const auto existingAxes = chart->axes();
    for (QAbstractAxis* axis : existingAxes) { chart->removeAxis(axis); delete axis; }

    const int subplotColumns = 2;
    const int subplotRows = (modes.size() + subplotColumns - 1) / subplotColumns;

    int plotted = 0;
    for (int i = 0; i < modes.size(); ++i) {
        auto it = lastModeResults_.find(modes[i]);
        if (it == lastModeResults_.end()) continue;
        const HydroRunResult& r = it->second;
        const size_t n = std::min(r.y_true.size(), r.y_pred.size());
        if (n == 0) continue;

        double minV = std::numeric_limits<double>::infinity();
        double maxV = -std::numeric_limits<double>::infinity();
        double meanY = 0.0;
        for (size_t k = 0; k < n; ++k) {
            meanY += r.y_true[k];
            minV = std::min(minV, std::min(r.y_true[k], r.y_pred[k]));
            maxV = std::max(maxV, std::max(r.y_true[k], r.y_pred[k]));
        }
        meanY /= static_cast<double>(n);

        double ssRes = 0.0;
        double ssTot = 0.0;
        for (size_t k = 0; k < n; ++k) {
            const double e = r.y_true[k] - r.y_pred[k];
            ssRes += e * e;
            const double d = r.y_true[k] - meanY;
            ssTot += d * d;
        }
        const double r2 = (ssTot > 1e-12) ? (1.0 - ssRes / ssTot) : 0.0;

        if (!std::isfinite(minV) || !std::isfinite(maxV) || minV == maxV) {
            minV = -1.0;
            maxV = 1.0;
        }
        const double span = std::max(1e-9, maxV - minV);

        const int row = i / subplotColumns;
        const int col = i % subplotColumns;
        const double panelX0 = static_cast<double>(col);
        const double panelY0 = static_cast<double>(subplotRows - 1 - row); // top row has highest y band

        auto mapCoord = [&](double v) {
            const double nv = (v - minV) / span;
            return std::max(0.0, std::min(1.0, nv));
        };

        auto* pts = new QScatterSeries(chart);
        pts->setName(QString("%1 (R²=%2)").arg(titles[i]).arg(r2, 0, 'f', 3));
        pts->setMarkerSize(5.5);
        pts->setColor(modeColors[i % modeColors.size()]);
        for (size_t k = 0; k < n; ++k) {
            pts->append(panelX0 + mapCoord(r.y_true[k]), panelY0 + mapCoord(r.y_pred[k]));
        }
        chart->addSeries(pts);

        auto* identity = new QLineSeries(chart);
        identity->setName(QString("%1: y=x").arg(titles[i]));
        QPen idPen(QColor(90, 90, 90));
        idPen.setStyle(Qt::DashLine);
        identity->setPen(idPen);
        identity->append(panelX0, panelY0);
        identity->append(panelX0 + 1.0, panelY0 + 1.0);
        chart->addSeries(identity);

        // Panel frame (hidden from legend)
        auto* frame = new QLineSeries(chart);
        frame->setName(QString("__frame_%1").arg(i));
        frame->append(panelX0, panelY0);
        frame->append(panelX0 + 1.0, panelY0);
        frame->append(panelX0 + 1.0, panelY0 + 1.0);
        frame->append(panelX0, panelY0 + 1.0);
        frame->append(panelX0, panelY0);
        QPen framePen(QColor(170, 170, 170));
        frame->setPen(framePen);
        chart->addSeries(frame);
        frame->setVisible(true);

        ++plotted;
    }

    if (plotted == 0) {
        appendLog("No stored approach results available for approach subplots.");
        return;
    }

    QPen splitPen(QColor(140, 140, 140));
    splitPen.setWidth(2);

    for (int col = 1; col < subplotColumns; ++col) {
        auto* splitV = new QLineSeries(chart);
        splitV->setName(QString("__split_v_%1").arg(col));
        splitV->append(static_cast<double>(col), 0.0);
        splitV->append(static_cast<double>(col), static_cast<double>(subplotRows));
        splitV->setPen(splitPen);
        chart->addSeries(splitV);
    }
    for (int row = 1; row < subplotRows; ++row) {
        auto* splitH = new QLineSeries(chart);
        splitH->setName(QString("__split_h_%1").arg(row));
        splitH->append(0.0, static_cast<double>(row));
        splitH->append(static_cast<double>(subplotColumns), static_cast<double>(row));
        splitH->setPen(splitPen);
        chart->addSeries(splitH);
    }

    auto* axisX = new QValueAxis(chart);
    axisX->setTitleText("Subplot canvas (Target → within each panel)");
    axisX->setRange(0.0, static_cast<double>(subplotColumns));
    axisX->setTickCount(subplotColumns + 1);
    auto* axisY = new QValueAxis(chart);
    axisY->setTitleText("Subplot canvas (Predicted → within each panel)");
    axisY->setRange(0.0, static_cast<double>(subplotRows));
    axisY->setTickCount(subplotRows + 1);

    chart->addAxis(axisX, Qt::AlignBottom);
    chart->addAxis(axisY, Qt::AlignLeft);
    for (auto* series : chart->series()) {
        series->attachAxis(axisX);
        series->attachAxis(axisY);
        if (series->name().startsWith("__")) {
            series->setName(QString());
        }
    }

    chart->setTitle("Approach Subplots (1:1 Target vs Predicted, same Plot tab)");
    chart->legend()->setVisible(true);
    appendLog(QString("Displayed approach subplots on main Plot tab (%1/%2 approaches with data).")
                  .arg(plotted)
                  .arg(modes.size()));
}

void HydroPINNWindow::plotResidualsAllModes() {
    const QStringList modes = {"ffn", "ffn_pinn", "lstm", "lstm_pinn", "pinn"};
    const QList<QColor> modeColors = {QColor(0, 114, 178), QColor(213, 94, 0), QColor(86, 180, 233), QColor(0, 158, 115), QColor(204, 121, 167)};
    const QList<Qt::PenStyle> modeStyles = {Qt::SolidLine, Qt::DashLine, Qt::DotLine, Qt::DashDotLine, Qt::DashDotDotLine};

    auto* chart = chartView_->chart();
    chart->removeAllSeries();
    const auto existingAxes = chart->axes();
    for (QAbstractAxis* axis : existingAxes) { chart->removeAxis(axis); delete axis; }

    bool addedAny = false;
    int modeIdx = 0;
    for (const QString& mode : modes) {
        auto it = lastModeResults_.find(mode);
        if (it == lastModeResults_.end()) continue;
        const HydroRunResult& r = it->second;
        const size_t n = std::min(r.x.size(), std::min(r.y_true.size(), r.y_pred.size()));
        if (n == 0) continue;

        auto* residual = new QLineSeries(chart);
        residual->setName(QString("Residual (%1)").arg(modeDisplayName(mode)));
        QPen pen(modeColors[modeIdx % modeColors.size()]);
        pen.setWidth(2);
        pen.setStyle(modeStyles[modeIdx % modeStyles.size()]);
        residual->setPen(pen);
        for (size_t i = 0; i < n; ++i) {
            residual->append(r.x[i], r.y_pred[i] - r.y_true[i]);
        }
        chart->addSeries(residual);
        addedAny = true;
        ++modeIdx;
    }

    if (!addedAny) {
        appendLog("No stored approach results available for residual plot.");
        return;
    }

    auto* zero = new QLineSeries(chart);
    zero->setName("Zero error");
    QPen zeroPen(QColor(80, 80, 80));
    zeroPen.setStyle(Qt::DashLine);
    zero->setPen(zeroPen);
    zero->append(0.0, 0.0);
    zero->append(1.0, 0.0);
    chart->addSeries(zero);

    auto* axisX = new QValueAxis(chart);
    axisX->setTitleText("t");
    auto* axisY = new QValueAxis(chart);
    axisY->setTitleText("Residual (pred - target)");
    chart->addAxis(axisX, Qt::AlignBottom);
    chart->addAxis(axisY, Qt::AlignLeft);
    for (auto* series : chart->series()) {
        series->attachAxis(axisX);
        series->attachAxis(axisY);
    }
    chart->setTitle("Residuals vs t (All Approaches)");
    chart->legend()->setVisible(true);
    fitPlotAxesInternal(false);

    zero->clear();
    zero->append(axisX->min(), 0.0);
    zero->append(axisX->max(), 0.0);

    appendLog("Displayed residual-vs-time plot for all stored approaches.");
}

void HydroPINNWindow::plotErrorCdfAllModes() {
    const QStringList modes = {"ffn", "ffn_pinn", "lstm", "lstm_pinn", "pinn"};
    const QList<QColor> modeColors = {QColor(0, 114, 178), QColor(213, 94, 0), QColor(86, 180, 233), QColor(0, 158, 115), QColor(204, 121, 167)};
    const QList<Qt::PenStyle> modeStyles = {Qt::SolidLine, Qt::DashLine, Qt::DotLine, Qt::DashDotLine, Qt::DashDotDotLine};

    auto* chart = chartView_->chart();
    chart->removeAllSeries();
    const auto existingAxes = chart->axes();
    for (QAbstractAxis* axis : existingAxes) { chart->removeAxis(axis); delete axis; }

    bool addedAny = false;
    int modeIdx = 0;
    for (const QString& mode : modes) {
        auto it = lastModeResults_.find(mode);
        if (it == lastModeResults_.end()) continue;
        const HydroRunResult& r = it->second;
        const size_t n = std::min(r.y_true.size(), r.y_pred.size());
        if (n == 0) continue;

        std::vector<double> absErr;
        absErr.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            absErr.push_back(std::abs(r.y_pred[i] - r.y_true[i]));
        }
        std::sort(absErr.begin(), absErr.end());

        auto* cdf = new QLineSeries(chart);
        cdf->setName(QString("|Error| CDF (%1)").arg(modeDisplayName(mode)));
        QPen pen(modeColors[modeIdx % modeColors.size()]);
        pen.setWidth(2);
        pen.setStyle(modeStyles[modeIdx % modeStyles.size()]);
        cdf->setPen(pen);
        for (size_t i = 0; i < absErr.size(); ++i) {
            const double p = static_cast<double>(i + 1) / static_cast<double>(absErr.size());
            cdf->append(absErr[i], p);
        }
        chart->addSeries(cdf);
        addedAny = true;
        ++modeIdx;
    }

    if (!addedAny) {
        appendLog("No stored approach results available for |error| CDF plot.");
        return;
    }

    auto* axisX = new QValueAxis(chart);
    axisX->setTitleText("|Prediction error|");
    auto* axisY = new QValueAxis(chart);
    axisY->setTitleText("Cumulative probability");
    axisY->setRange(0.0, 1.0);
    chart->addAxis(axisX, Qt::AlignBottom);
    chart->addAxis(axisY, Qt::AlignLeft);
    for (auto* series : chart->series()) {
        series->attachAxis(axisX);
        series->attachAxis(axisY);
    }

    chart->setTitle("Absolute Error CDF (All Approaches)");
    chart->legend()->setVisible(true);
    fitPlotAxesInternal(false);
    appendLog("Displayed |error| CDF plot for all stored approaches.");
}

void HydroPINNWindow::plotFlowDurationAllModes() {
    const QStringList modes = {"ffn", "ffn_pinn", "lstm", "lstm_pinn", "pinn"};
    const QList<QColor> colors = {QColor(0, 114, 178), QColor(213, 94, 0), QColor(86, 180, 233),
                                  QColor(0, 158, 115), QColor(204, 121, 167)};
    auto* chart = chartView_->chart();
    chart->removeAllSeries();
    const auto existingAxes = chart->axes();
    for (QAbstractAxis* axis : existingAxes) { chart->removeAxis(axis); delete axis; }
    bool addedAny = false;
    int colorIndex = 0;
    for (const QString& mode : modes) {
        const auto found = lastModeResults_.find(mode);
        if (found == lastModeResults_.end()) continue;
        const auto& result = found->second;
        const size_t n = std::min(result.y_true.size(), result.y_pred.size());
        std::vector<double> observed;
        std::vector<double> predicted;
        for (size_t i = 0; i < n; ++i) {
            if (i < result.split.size() && result.split[i] != "test") continue;
            observed.push_back(result.y_true[i]);
            predicted.push_back(result.y_pred[i]);
        }
        if (observed.empty()) continue;
        std::sort(observed.begin(), observed.end(), std::greater<double>());
        std::sort(predicted.begin(), predicted.end(), std::greater<double>());
        auto* observedSeries = new QLineSeries(chart);
        auto* predictedSeries = new QLineSeries(chart);
        observedSeries->setName(QString("Observed FDC (%1)").arg(modeDisplayName(mode)));
        predictedSeries->setName(QString("Predicted FDC (%1)").arg(modeDisplayName(mode)));
        QPen observedPen(colors[colorIndex % colors.size()]);
        observedPen.setWidth(2);
        QPen predictedPen = observedPen;
        predictedPen.setStyle(Qt::DashLine);
        observedSeries->setPen(observedPen);
        predictedSeries->setPen(predictedPen);
        for (size_t rank = 0; rank < observed.size(); ++rank) {
            const double exceedance = 100.0 * static_cast<double>(rank + 1) /
                                      static_cast<double>(observed.size() + 1);
            observedSeries->append(exceedance, observed[rank]);
            predictedSeries->append(exceedance, predicted[rank]);
        }
        chart->addSeries(observedSeries);
        chart->addSeries(predictedSeries);
        addedAny = true;
        ++colorIndex;
    }
    if (!addedAny) {
        appendLog("No held-out test results are available for a flow-duration plot.");
        return;
    }
    auto* axisX = new QValueAxis(chart);
    axisX->setTitleText("Exceedance probability (%)");
    axisX->setRange(0.0, 100.0);
    auto* axisY = new QValueAxis(chart);
    axisY->setTitleText("Runoff / discharge");
    chart->addAxis(axisX, Qt::AlignBottom);
    chart->addAxis(axisY, Qt::AlignLeft);
    for (auto* series : chart->series()) { series->attachAxis(axisX); series->attachAxis(axisY); }
    chart->setTitle("Flow-Duration Curves (Held-Out Test Data)");
    chart->legend()->setVisible(true);
    fitPlotAxesInternal(false);
    appendLog("Displayed observed and predicted flow-duration curves for held-out test data.");
}

void HydroPINNWindow::plotCumulativeResidualAllModes() {
    const QStringList modes = {"ffn_pinn", "lstm_pinn", "pinn"};
    const QList<QColor> colors = {QColor(213, 94, 0), QColor(0, 158, 115), QColor(204, 121, 167)};
    auto* chart = chartView_->chart();
    chart->removeAllSeries();
    const auto existingAxes = chart->axes();
    for (QAbstractAxis* axis : existingAxes) { chart->removeAxis(axis); delete axis; }
    bool addedAny = false;
    int colorIndex = 0;
    for (const QString& mode : modes) {
        const auto found = lastModeResults_.find(mode);
        if (found == lastModeResults_.end()) continue;
        const auto& result = found->second;
        const size_t n = std::min(result.x.size(), result.physics_residual.size());
        if (n < 2) continue;
        auto* series = new QLineSeries(chart);
        series->setName(QString("Cumulative residual (%1)").arg(modeDisplayName(mode)));
        QPen pen(colors[colorIndex % colors.size()]);
        pen.setWidth(2);
        series->setPen(pen);
        double cumulative = 0.0;
        bool hasFiniteResidual = false;
        for (size_t i = 1; i < n; ++i) {
            const double residual = result.physics_residual[i];
            const double dt = result.x[i] - result.x[i - 1];
            if (!std::isfinite(residual) || !std::isfinite(dt) || dt <= 0.0) continue;
            cumulative += residual * dt;
            series->append(result.x[i], cumulative);
            hasFiniteResidual = true;
        }
        if (!hasFiniteResidual) { delete series; continue; }
        chart->addSeries(series);
        addedAny = true;
        ++colorIndex;
    }
    if (!addedAny) {
        appendLog("No finite PINN physics residuals are available for cumulative plotting.");
        return;
    }
    auto* axisX = new QValueAxis(chart);
    axisX->setTitleText("Time");
    auto* axisY = new QValueAxis(chart);
    axisY->setTitleText("Integrated signed physics residual");
    chart->addAxis(axisX, Qt::AlignBottom);
    chart->addAxis(axisY, Qt::AlignLeft);
    for (auto* series : chart->series()) { series->attachAxis(axisX); series->attachAxis(axisY); }
    chart->setTitle("Cumulative Physics Residual (PINN Approaches)");
    chart->legend()->setVisible(true);
    fitPlotAxesInternal(false);
    appendLog("Displayed timestep-integrated cumulative physics residuals for PINN approaches.");
}

void HydroPINNWindow::runMode(const QString& mode) {
    appendLog(QString("Starting approach: %1").arg(modeDisplayName(mode)));
    static bool modeImplementationNoteLogged = false;
    if (!modeImplementationNoteLogged) {
        appendLog("Approach implementation note: Hydro wrappers are local implementations; NeuroForge labels are workflow-compatible naming.");
        modeImplementationNoteLogged = true;
    }
    setRunningUiState(true);
    statusLabel_->setText(QString("Running approach: %1 ...").arg(modeDisplayName(mode)));
    appendLog("Dispatch started.");

    HydroRunConfig cfg = currentConfig();
    const std::vector<QString> layerActs = configuredLayerActivations();
    if (!layerActs.empty()) {
        bool mixedActivations = false;
        for (size_t i = 1; i < layerActs.size(); ++i) {
            if (layerActs[i].compare(layerActs[0], Qt::CaseInsensitive) != 0) {
                mixedActivations = true;
                break;
            }
        }
        if (mixedActivations) {
            appendLog(QString("Mixed layer activations configured (%1). Current backend supports a single activation; using first layer activation: %2")
                          .arg(QString::number(layerActs.size()))
                          .arg(QString::fromStdString(cfg.activation)));
        } else {
            appendLog(QString("Using activation from Network Builder: %1").arg(QString::fromStdString(cfg.activation)));
        }
    }
    if (cfg.use_hydro_package) {
        appendLog(QString("Using Hydro package: path=%1, catchment=%2, profile=%3")
                      .arg(QString::fromStdString(cfg.hydro_package_path))
                      .arg(QString::fromStdString(cfg.hydro_catchment_id))
                      .arg(QString::fromStdString(cfg.hydro_package_profile)));
    } else if (cfg.use_csv_data) {
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
        if (mode == "ffn_pinn" || mode == "pinn" || mode == "lstm_pinn") {
            if (cfg.pinn_physics_profile == "exp_decay" && cfg.synthetic_profile != "exp_decay") {
                appendLog("Note: selected PINN profile is exp_decay; non-exp synthetic targets may reduce physics consistency.");
            }
            if ((cfg.pinn_physics_profile == "linear_reservoir" || cfg.pinn_physics_profile == "cstr_first_order" || cfg.pinn_physics_profile == "water_balance") &&
                (cfg.synthetic_profile != "neuroforge_inputs_target" && cfg.synthetic_profile != "watershed_balance" && cfg.synthetic_profile != "rainfall_runoff")) {
                appendLog("Note: forcing/water-balance PINN profiles work best with multi-feature inputs (CSV, neuroforge_inputs_target, watershed_balance, or rainfall_runoff synthetic profile).");
            }
        }
    }
    appendLog(QString("Extra options => split=%1, optimizer=%2, normalization=%3, incremental=%4")
                  .arg(cfg.train_split_ratio, 0, 'g', 4)
                  .arg(QString::fromStdString(cfg.optimizer))
                  .arg(QString::fromStdString(cfg.normalization))
                  .arg(cfg.use_incremental_training ? "yes" : "no"));
    const bool ffnStyleApplies = (mode == "ffn" || mode == "ffn_pinn");
    const QString inputStyle = ffnStyleApplies
        ? (cfg.use_time_lagged_ffn ? "time-lagged" : "basic")
        : (mode == "pinn" ? (cfg.pinn_physics_profile == "water_balance" ? "water-balance feature input" : "physics-coordinate input") : "ignored for LSTM");
    appendLog(QString("Network options => hidden_layers=%1, input_lag_steps=%2, input_style=%3, activation=%4")
                  .arg(QString::fromStdString(cfg.hidden_layers_csv))
                  .arg(QString::fromStdString(cfg.input_lags_csv))
                  .arg(inputStyle)
                  .arg(QString::fromStdString(cfg.activation)));
    if (mode == "lstm" || mode == "lstm_pinn") {
        appendLog(QString("LSTM sequence length=%1 (independent of FFN lag-search settings).")
                      .arg(cfg.lstm_sequence_length));
    }
    if (mode == "ffn_pinn" || mode == "pinn" || mode == "lstm_pinn") {
        appendLog(QString("PINN physics => profile=%1, forcing_gain=%2, collocation=%3")
                      .arg(QString::fromStdString(cfg.pinn_physics_profile))
                      .arg(cfg.forcing_gain, 0, 'g', 6)
                      .arg(cfg.pinn_collocation_points));
    }

    if (mode == "ffn_pinn" && cfg.pinn_physics_profile == "water_balance" && cfg.use_time_lagged_ffn) {
        cfg.use_time_lagged_ffn = false;
        appendLog("Water-balance FFN + PINN uses same-time P/ET/storage features; time-lagged FFN inputs are disabled for the PINN run to avoid inconsistent residuals.");
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
        } else if (mode == "pinn") {
            appendLog(cfg.pinn_physics_profile == "water_balance"
                          ? "Standalone PINN uses physics-only water-balance loss with P/ET/total-storage features."
                          : "Standalone PINN uses the explicit physics-only wrapper (data_weight=0).");
            PINNWrapper runner;
            result = runner.train(cfg);
        } else if (mode == "lstm") {
            LSTMWrapper runner;
            result = runner.train(cfg);
        } else if (mode == "lstm_pinn") {
            LSTMPINNWrapper runner;
            result = runner.train(cfg);
        } else {
            result.success = false;
            errorDetails = QString("Unknown approach selected: %1").arg(modeDisplayName(mode));
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
        statusLabel_->setText(QString("Completed approach: %1 (%2 ms)").arg(modeDisplayName(mode)).arg(elapsedMs));
        appendLog(QString("Approach '%1' finished successfully in %2 ms.").arg(modeDisplayName(mode)).arg(elapsedMs));
        appendLog(QString("  final_loss=%1, validation_mse=%2, test_mse=%3, rmse=%4, mae=%5, nse=%6, pbias=%7, physics_loss=%8, msg=%9")
                      .arg(result.final_loss, 0, 'g', 8)
                      .arg(result.validation_mse, 0, 'g', 8)
                      .arg(result.mse, 0, 'g', 8)
                      .arg(result.rmse, 0, 'g', 8)
                      .arg(result.mae, 0, 'g', 8)
                      .arg(result.nse, 0, 'g', 8)
                      .arg(result.pbias, 0, 'g', 8)
                      .arg(result.physics_loss, 0, 'g', 8)
                      .arg(QString::fromStdString(result.message)));
        lastModeResults_[mode] = result;
        updatePlot(mode, result);
        refreshPerformanceAssessment();
    } else {
        statusLabel_->setText(QString("Approach failed: %1").arg(modeDisplayName(mode)));
        appendLog(QString("Approach '%1' failed.").arg(modeDisplayName(mode)));
        if (!errorDetails.isEmpty()) {
            appendLog(QString("Failure details: %1").arg(errorDetails));
        }
        QMessageBox::warning(this, "HydroPINN",
                             QString("Approach '%1' failed.%2")
                                 .arg(modeDisplayName(mode))
                                 .arg(errorDetails.isEmpty() ? "" : QString("\n\n%1").arg(errorDetails)));
    }

    setRunningUiState(false);
}
