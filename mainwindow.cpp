#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QMenuBar>
#include <QMessageBox>
#include <QFileDialog>
#include <QDateTime>
#include "gasettingsdialog.h"
#include "DataLoadDialog.h"
#include "chartwindow.h"
#include "syntheticdatadialog.h"
#include "networkarchitecturedialog.h"
#include "incrementaltrainingdialog.h"
#include <QJsonDocument>
#include <QJsonObject>
#include <QFile>
#include <QSettings>
#include <QTimer>
#include "lagconfigdialog.h"
#include "DataPlotDialog.h"
#include <QInputDialog>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , gaRunning_(false)
{
    ui->setupUi(this);

    setWindowTitle("NeuroForge - Neural Network GA Optimizer");
    resize(1000, 700);

    setupUI();
    setupMenus();
    connectSignals();

    logMessage("Application started. Ready to load data and configure optimization.");

    // Load last project if available
    QTimer::singleShot(100, this, &MainWindow::loadLastProject);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::setupUI()
{
    // Create central widget
    QWidget *centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);
    
    // Main layout
    QVBoxLayout *mainLayout = new QVBoxLayout(centralWidget);
    
    // Status label
    statusLabel = new QLabel("Status: Ready", this);
    statusLabel->setStyleSheet("QLabel { font-weight: bold; color: #2c3e50; }");
    mainLayout->addWidget(statusLabel);
    
    // Log output area
    logOutput = new QTextEdit(this);
    logOutput->setReadOnly(true);
    logOutput->setStyleSheet("QTextEdit { background-color: #ecf0f1; font-family: 'Courier New'; }");
    mainLayout->addWidget(logOutput);
    
    // Progress bar
    progressBar = new QProgressBar(this);
    progressBar->setRange(0, 100);
    progressBar->setValue(0);
    progressBar->setTextVisible(true);
    mainLayout->addWidget(progressBar);
    
    // Button layout
    QHBoxLayout *buttonLayout = new QHBoxLayout();
    
    startButton = new QPushButton("Start Optimization", this);
    startButton->setStyleSheet("QPushButton { background-color: #27ae60; color: white; padding: 10px; font-weight: bold; }");
    startButton->setEnabled(false);  // Enable after loading data
    buttonLayout->addWidget(startButton);
    
    stopButton = new QPushButton("Stop", this);
    stopButton->setStyleSheet("QPushButton { background-color: #e74c3c; color: white; padding: 10px; font-weight: bold; }");
    stopButton->setEnabled(false);
    buttonLayout->addWidget(stopButton);
    
    mainLayout->addLayout(buttonLayout);
    
    // Status bar
    statusBar()->showMessage("Ready");
}

void MainWindow::setupMenus()
{
    // File menu
    QMenu* fileMenu = menuBar()->addMenu("File");

    QAction* loadDataAction = new QAction("Load Data...", this);
    loadDataAction->setShortcut(QKeySequence::Open);
    connect(loadDataAction, &QAction::triggered, this, &MainWindow::onLoadData);
    fileMenu->addAction(loadDataAction);

    QAction* generateDataAction = new QAction("Generate Synthetic Data...", this);
    connect(generateDataAction, &QAction::triggered, this, &MainWindow::onGenerateSyntheticData);
    fileMenu->addAction(generateDataAction);

    fileMenu->addSeparator();

    QAction* newProjectAction = new QAction("New Project", this);
    newProjectAction->setShortcut(QKeySequence::New);
    connect(newProjectAction, &QAction::triggered, this, &MainWindow::onNewProject);
    fileMenu->addAction(newProjectAction);

    QAction* loadProjectAction = new QAction("Load Project...", this);
    loadProjectAction->setShortcut(QKeySequence("Ctrl+Shift+O"));
    connect(loadProjectAction, &QAction::triggered, this, &MainWindow::onLoadProject);
    fileMenu->addAction(loadProjectAction);

    QAction* saveProjectAction = new QAction("Save Project", this);
    saveProjectAction->setShortcut(QKeySequence::Save);
    connect(saveProjectAction, &QAction::triggered, this, &MainWindow::onSaveProject);
    fileMenu->addAction(saveProjectAction);

    QAction* saveProjectAsAction = new QAction("Save Project As...", this);
    saveProjectAsAction->setShortcut(QKeySequence::SaveAs);
    connect(saveProjectAsAction, &QAction::triggered, this, &MainWindow::onSaveProjectAs);
    fileMenu->addAction(saveProjectAsAction);

    fileMenu->addSeparator();

    QAction* exitAction = new QAction("Exit", this);
    exitAction->setShortcut(QKeySequence::Quit);
    connect(exitAction, &QAction::triggered, this, &MainWindow::onExit);
    fileMenu->addAction(exitAction);

    // Configuration menu
    QMenu* configMenu = menuBar()->addMenu("Configuration");

    QAction* configLagsAction = new QAction("Configure Time Lags...", this);
    configLagsAction->setShortcut(QKeySequence("Ctrl+L"));
    configLagsAction->setToolTip("Define which past time steps to use as input features");
    connect(configLagsAction, &QAction::triggered, this, &MainWindow::onConfigureLags);
    configMenu->addAction(configLagsAction);

    QAction* configNetworkAction = new QAction("Configure Network Architecture...", this);
    configNetworkAction->setToolTip("Set hidden layers and activation functions");
    connect(configNetworkAction, &QAction::triggered, this, &MainWindow::onConfigureNetwork);
    configMenu->addAction(configNetworkAction);

    configMenu->addSeparator();

    QAction* configGAAction = new QAction("Configure GA Settings...", this);
    configGAAction->setToolTip("Set genetic algorithm parameters");
    connect(configGAAction, &QAction::triggered, this, &MainWindow::onConfigureGA);
    configMenu->addAction(configGAAction);

    QAction* configIncrementalAction = new QAction("Configure Incremental Training...", this);
    configIncrementalAction->setToolTip("Set rolling window training parameters");
    connect(configIncrementalAction, &QAction::triggered, this, &MainWindow::onConfigureIncrementalTraining);
    configMenu->addAction(configIncrementalAction);

    // Run menu
    QMenu* runMenu = menuBar()->addMenu("Run");

    startGAAction_ = new QAction("Start GA Optimization", this);
    startGAAction_->setShortcut(QKeySequence("Ctrl+G"));
    startGAAction_->setEnabled(false);
    connect(startGAAction_, &QAction::triggered, this, &MainWindow::onStartGA);
    runMenu->addAction(startGAAction_);

    stopGAAction_ = new QAction("Stop GA", this);
    stopGAAction_->setEnabled(false);
    connect(stopGAAction_, &QAction::triggered, this, &MainWindow::onStopGA);
    runMenu->addAction(stopGAAction_);

    runMenu->addSeparator();

    QAction* startIncrementalAction = new QAction("Start Incremental Training", this);
    startIncrementalAction->setShortcut(QKeySequence("Ctrl+T"));
    startIncrementalAction->setToolTip("Train network using rolling windows");
    connect(startIncrementalAction, &QAction::triggered, this, &MainWindow::onStartIncrementalTraining);
    runMenu->addAction(startIncrementalAction);

    QAction* trainWindowAction = new QAction("Train on Latest Window", this);
    trainWindowAction->setShortcut(QKeySequence("Ctrl+W"));
    trainWindowAction->setToolTip("Train on the most recent data window (online learning)");
    connect(trainWindowAction, &QAction::triggered, this, &MainWindow::onTrainOnLatestWindow);
    runMenu->addAction(trainWindowAction);

    // View menu
    QMenu* viewMenu = menuBar()->addMenu("View");

    QAction* plotDataAction = new QAction("Plot Raw Data...", this);
    plotDataAction->setShortcut(QKeySequence("Ctrl+D"));
    plotDataAction->setToolTip("Visualize input and target data");
    connect(plotDataAction, &QAction::triggered, this, &MainWindow::onPlotData);
    viewMenu->addAction(plotDataAction);

    viewMenu->addSeparator();

    QAction* plotResultsAction = new QAction("Plot Prediction Results", this);
    plotResultsAction->setShortcut(QKeySequence("Ctrl+P"));
    plotResultsAction->setToolTip("Plot model predictions vs actual values");
    connect(plotResultsAction, &QAction::triggered, this, &MainWindow::onPlotResults);
    viewMenu->addAction(plotResultsAction);

    // Help menu
    QMenu* helpMenu = menuBar()->addMenu("Help");

    QAction* aboutAction = new QAction("About", this);
    connect(aboutAction, &QAction::triggered, this, &MainWindow::onAbout);
    helpMenu->addAction(aboutAction);
}


void MainWindow::connectSignals()
{
    // Connect button signals
    // connect(startButton, &QPushButton::clicked, this, &MainWindow::onStartOptimization);
    // connect(stopButton, &QPushButton::clicked, this, &MainWindow::onStopOptimization);
    
    // You'll add these connections as you implement the functionality
}

void MainWindow::logMessage(const QString &message) const
{
    QString timestamp = QDateTime::currentDateTime().toString("hh:mm:ss");
    logOutput->append(QString("[%1] %2").arg(timestamp).arg(message));
    
    // Auto-scroll to bottom
    QTextCursor cursor = logOutput->textCursor();
    cursor.movePosition(QTextCursor::End);
    logOutput->setTextCursor(cursor);
}

void MainWindow::updateProgress(int value)
{
    progressBar->setValue(value);
    
    if (value == 0) {
        progressBar->setFormat("Ready");
    } else if (value == 100) {
        progressBar->setFormat("Complete - 100%");
    } else {
        progressBar->setFormat(QString("%1%").arg(value));
    }
}

void MainWindow::onAbout()
{
    QMessageBox::about(this, "About NeuroForge",
        "<h2>NeuroForge</h2>"
        "<p><i>Neural Network GA Optimizer</i></p>"
        "<p>Version 1.0</p>"
        "<p>A genetic algorithm-based neural network optimizer for time series prediction.</p>"
        "<p><b>Features:</b></p>"
        "<ul>"
        "<li>Time series data loading and preprocessing</li>"
        "<li>Genetic algorithm optimization</li>"
        "<li>Neural network training with PyTorch</li>"
        "<li>Model evaluation and export</li>"
        "</ul>"
        "<p>Built with Qt and LibTorch.</p>");
}

void MainWindow::onExit()
{
    QMessageBox::StandardButton reply;
    reply = QMessageBox::question(this, "Exit", 
                                   "Are you sure you want to exit?",
                                   QMessageBox::Yes | QMessageBox::No);
    
    if (reply == QMessageBox::Yes) {
        qApp->quit();
    }
}

void MainWindow::onConfigureGA()
{
    GASettingsDialog dialog(&ga_, this);

    if (dialog.exec() == QDialog::Accepted) {
        logMessage("GA settings updated successfully");

        // Optional: Display current settings
        QString settingsInfo = QString("GA Configuration:\n"
                                       "  Population: %1\n"
                                       "  Generations: %2\n"
                                       "  Mutation Rate: %3")
                                   .arg(ga_.Settings.totalpopulation)
                                   .arg(ga_.Settings.generations)
                                   .arg(ga_.Settings.mutation_probability);
        logMessage(settingsInfo);
    }
}

void MainWindow::onStartGA()
{
    // ... existing validation code ...

    try {
        // Reset GA state
        ga_.Reset();

        ga_.model.clear();
        ga_.model.setTimeSeriesData(inputData, targetData);
        ga_.model.setTimeRange(TimeStart(DataType::Train), TimeEnd(DataType::Test), dt(), split_ratio);
        ga_.model.setAvailableSeriesCount(inputData.size());

        // Create progress window
        ProgressWindow* progressWindow = new ProgressWindow(this, "GA Optimization");
        progressWindow->SetPrimaryChartVisible(true);
        progressWindow->SetSecondaryChartVisible(true);
        progressWindow->SetPrimaryChartXRange(0, ga_.Settings.generations);
        progressWindow->SetSecondaryChartXRange(0, ga_.Settings.generations);
        progressWindow->SetPrimaryChartAutoScale(true);
        progressWindow->SetSecondaryChartAutoScale(true);
        progressWindow->show();
        QApplication::processEvents();

        ga_.setProgressWindow(progressWindow);

        logMessage("Initializing GA population...");
        ga_.Initialize();

        logMessage("Running optimization...");
        gaRunning_ = true;
        startGAAction_->setEnabled(false);
        stopGAAction_->setEnabled(true);

        NeuralNetworkWrapper& bestModel = ga_.Optimize();

        auto sorted_indices = ga_.getRanks();
        Individual bestIndividual = ga_.Individuals[sorted_indices[0]];

        logMessage(QString("Optimization complete! Best fitness: %1")
                       .arg(bestIndividual.fitness));

        // ===== STORE BEST MODEL FOR PLOTTING =====
        if (bestModel_) {
            delete bestModel_;
        }

        logMessage(QString("DEBUG: Original model state BEFORE copy:"));
        logMessage(QString("  - Initialized: %1").arg(bestModel.isInitialized()));
        logMessage(QString("  - Total parameters: %1").arg(bestModel.getTotalParameters()));

        bestModel_ = new NeuralNetworkWrapper(bestModel);

        logMessage(QString("DEBUG: Copied model state AFTER copy:"));
        logMessage(QString("  - Initialized: %1").arg(bestModel_->isInitialized()));
        logMessage(QString("  - Total parameters: %1").arg(bestModel_->getTotalParameters()));

        // Make predictions and compare
        try {
            logMessage("DEBUG: Testing original model predictions...");
            TimeSeriesSet<double> orig_pred = bestModel.predict(
                DataType::Test,
                TimeStart(DataType::Test),
                TimeEnd(DataType::Test),
                dt()
                );

            // Show first 5 prediction values
            QString orig_values = "DEBUG: Original predictions: ";
            for (size_t i = 0; i < std::min(size_t(5), orig_pred[0].size()); i++) {
                orig_values += QString::number(orig_pred[0][i].c, 'f', 4) + " ";
            }
            logMessage(orig_values);

            // Check if all predictions are the same
            bool all_same = true;
            double first_val = orig_pred[0][0].c;
            for (size_t i = 1; i < std::min(size_t(20), orig_pred[0].size()); i++) {
                if (std::abs(orig_pred[0][i].c - first_val) > 1e-6) {
                    all_same = false;
                    break;
                }
            }
            logMessage(QString("DEBUG: Original predictions constant? %1").arg(all_same ? "YES" : "NO"));

        } catch (const std::exception& e) {
            logMessage(QString("ERROR: Original model prediction failed: %1").arg(e.what()));
        }

        try {
            logMessage("DEBUG: Testing copied model predictions...");
            TimeSeriesSet<double> copy_pred = bestModel_->predict(
                DataType::Test,
                TimeStart(DataType::Test),
                TimeEnd(DataType::Test),
                dt()
                );

            // Show first 5 prediction values
            QString copy_values = "DEBUG: Copied predictions: ";
            for (size_t i = 0; i < std::min(size_t(5), copy_pred[0].size()); i++) {
                copy_values += QString::number(copy_pred[0][i].c, 'f', 4) + " ";
            }
            logMessage(copy_values);

            // Check if all predictions are the same
            bool all_same = true;
            double first_val = copy_pred[0][0].c;
            for (size_t i = 1; i < std::min(size_t(20), copy_pred[0].size()); i++) {
                if (std::abs(copy_pred[0][i].c - first_val) > 1e-6) {
                    all_same = false;
                    break;
                }
            }
            logMessage(QString("DEBUG: Copied predictions constant? %1").arg(all_same ? "YES" : "NO"));

        } catch (const std::exception& e) {
            logMessage(QString("ERROR: Copied model prediction failed: %1").arg(e.what()));
        }

        // Save model
        QString timestamp = QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss");
        QString modelPath = QString("best_model_%1.pt").arg(timestamp);

        //bestModel.saveModel(modelPath.toStdString());
        //logMessage(QString("Best model saved to: %1").arg(modelPath));

        ga_.setProgressWindow(nullptr);

        // ===== ASK USER IF THEY WANT TO SEE PLOTS =====
        QMessageBox::StandardButton reply = QMessageBox::question(
            this,
            "Optimization Complete",
            QString("GA optimization completed!\n\n"
                    "Best fitness: %1\n"
                    "Model saved to:\n%2\n\n"
                    "Would you like to view the prediction plots?")
                .arg(bestIndividual.fitness)
                .arg(modelPath),
            QMessageBox::Yes | QMessageBox::No
            );

        if (reply == QMessageBox::Yes) {
            onPlotResults();
        }

        progressWindow->close();
        progressWindow->deleteLater();

    } catch (const std::exception& e) {
        logMessage(QString("ERROR: %1").arg(e.what()));
        QMessageBox::critical(this, "Error",
                              QString("GA optimization failed:\n%1").arg(e.what()));
    }

    gaRunning_ = false;
    startGAAction_->setEnabled(true);
    stopGAAction_->setEnabled(false);

    logMessage("=== GA Optimization Finished ===");
}

void MainWindow::onStopGA()
{
    if (!gaRunning_) {
        return;
    }

    // For now, just log (full implementation would need thread handling)
    logMessage("Stop requested - GA will finish current generation");
    QMessageBox::information(this, "Stop Requested",
                             "GA will stop after completing the current generation.");

    // Note: Full stop implementation requires running GA in a separate thread
    // and checking a stop flag periodically
}

void MainWindow::onLoadData()
{
    DataLoadDialog dialog(this);

    if (dialog.exec() == QDialog::Accepted && dialog.isDataValid()) {
        // Get loaded data
        inputData = dialog.getInputData();
        targetData = dialog.getTargetData();

        currentProject_.inputDataPath = dialog.getInputFilePath();
        currentProject_.targetDataPath = dialog.getTargetFilePath();

        // Log success
        logMessage("=== Data Loaded Successfully ===");
        logMessage(QString("Input: %1 time series with %2 points each")
                       .arg(inputData.size())
                       .arg(inputData[0].size()));
        logMessage(QString("Target: %1 points").arg(targetData.size()));
        logMessage(QString("Time range: %1 to %2")
                       .arg(inputData[0].front().t)
                       .arg(inputData[0].back().t));

        // Enable optimization
        startButton->setEnabled(true);
        startGAAction_->setEnabled(true);

        statusBar()->showMessage("Data loaded successfully", 3000);
        statusLabel->setText("Status: Data loaded - Ready to optimize");

        QMessageBox::information(this, "Success",
                                 QString("Data loaded successfully!\n\n"
                                         "Input: %1 series\n"
                                         "Target: %2 points")
                                     .arg(inputData.size())
                                     .arg(targetData.size()));
    }
}


void MainWindow::plotPredictionsVsTime(NeuralNetworkWrapper& model, DataType data_type)
{
    try {
        QString dataTypeName = (data_type == DataType::Train) ? "Train" : "Test";

        // Get target data as tensor
        torch::Tensor targetTensor = model.getTargetData(data_type);

        // Get the actual number of points in tensor
        int64_t num_points = targetTensor.size(0);

        logMessage(QString("DEBUG: %1 tensor has %2 points").arg(dataTypeName).arg(num_points));

        // Get time range
        double data_start = TimeStart(data_type);
        double data_end = TimeEnd(data_type);
        double dt_val = dt();

        // The tensor size tells us the ACTUAL number of points
        // Adjust end time to match: end = start + (n-1) * dt
        double adjusted_end = data_start + (num_points - 1) * dt_val;

        logMessage(QString("DEBUG: Original range [%1, %2], adjusted to [%3, %4]")
                       .arg(data_start).arg(data_end)
                       .arg(data_start).arg(adjusted_end));

        // Convert tensor to TimeSeries using adjusted range
        TimeSeries<double> dataTarget = TimeSeries<double>::fromTensor(
            targetTensor,
            false,           // has_time = false
            data_start,      // time_offset
            dt_val           // time_step
            );

        if (dataTarget.empty()) {
            throw std::runtime_error(QString("%1 target data is empty after conversion")
                                         .arg(dataTypeName).toStdString());
        }

        logMessage(QString("DEBUG: %1 target TimeSeries: %2 points, range [%3, %4]")
                       .arg(dataTypeName)
                       .arg(dataTarget.size())
                       .arg(dataTarget.mint())
                       .arg(dataTarget.maxt()));

        // Make predictions using the actual bounds from TimeSeries
        double pred_start = dataTarget.mint();
        double pred_end = dataTarget.maxt();

        TimeSeriesSet<double> predictions = model.predict(data_type, pred_start, pred_end, dt_val);

        if (predictions.size() == 0) {
            throw std::runtime_error("No predictions generated");
        }

        TimeSeries<double> predSeries = predictions[0];

        logMessage(QString("DEBUG: Prediction TimeSeries: %1 points, range [%2, %3]")
                       .arg(predSeries.size())
                       .arg(predSeries.mint())
                       .arg(predSeries.maxt()));

        // Calculate R²
        double r2 = R2(dataTarget, predSeries);

        // Calculate MSE
        double mse = 0.0;
        int n = 0;
        for (double t = pred_start; t <= pred_end; t += dt_val) {
            double target_val = dataTarget.interpol(t);
            double pred_val = predSeries.interpol(t);
            double error = target_val - pred_val;
            mse += error * error;
            n++;
        }
        if (n > 0) {
            mse /= n;
        }

        // Create TimeSeriesSet for plotting
        TimeSeriesSet<double> plotData;

        dataTarget.setName("Target");
        plotData.append(dataTarget);

        predSeries.setName("Predicted");
        plotData.append(predSeries);

        // Show chart
        QString title = QString("Predictions vs Target over Time (%1 Data)\nR² = %2, MSE = %3")
                            .arg(dataTypeName)
                            .arg(r2, 0, 'f', 4)
                            .arg(mse, 0, 'f', 6);

        ChartWindow* chartWin = ChartWindow::showChart(plotData, title, this);
        chartWin->setAxisLabels("Time", "Value");
        chartWin->chartViewer()->setPlotMode(ChartViewer::Lines);

        logMessage(QString("Opened predictions vs time plot (%1: R² = %2, MSE = %3)")
                       .arg(dataTypeName)
                       .arg(r2, 0, 'f', 4)
                       .arg(mse, 0, 'f', 6));

    } catch (const std::exception& e) {
        logMessage(QString("Plot error: %1").arg(e.what()));
        QMessageBox::warning(this, "Plot Error",
                             QString("Failed to plot predictions vs time:\n%1").arg(e.what()));
    }
}

void MainWindow::plotPredictionsVsTarget(NeuralNetworkWrapper& model, DataType data_type)
{
    try {
        QString dataTypeName = (data_type == DataType::Train) ? "Train" : "Test";

        // Get target data as tensor
        torch::Tensor targetTensor = model.getTargetData(data_type);

        // Get actual number of points
        int64_t num_points = targetTensor.size(0);

        // Get time range
        double data_start = TimeStart(data_type);
        double dt_val = dt();

        // Adjust end time to match tensor size: end = start + (n-1) * dt
        double adjusted_end = data_start + (num_points - 1) * dt_val;

        // Convert tensor to TimeSeries
        TimeSeries<double> dataTarget = TimeSeries<double>::fromTensor(
            targetTensor,
            false,
            data_start,
            dt_val
            );

        if (dataTarget.empty()) {
            throw std::runtime_error(QString("%1 target data is empty after conversion")
                                         .arg(dataTypeName).toStdString());
        }

        // Use actual bounds from dataTarget
        double pred_start = dataTarget.mint();
        double pred_end = dataTarget.maxt();

        // Make predictions
        TimeSeriesSet<double> predictions = model.predict(data_type, pred_start, pred_end, dt_val);

        if (predictions.size() == 0) {
            throw std::runtime_error("No predictions generated");
        }

        TimeSeries<double> predSeries = predictions[0];

        // Calculate R²
        double r2 = R2(dataTarget, predSeries);

        // Calculate MSE
        double mse = 0.0;
        int n = 0;
        for (double t = pred_start; t <= pred_end; t += dt_val) {
            double target_val = dataTarget.interpol(t);
            double pred_val = predSeries.interpol(t);
            double error = target_val - pred_val;
            mse += error * error;
            n++;
        }
        if (n > 0) {
            mse /= n;
        }

        // Create scatter plot
        TimeSeriesSet<double> scatterData;

        TimeSeries<double> scatterSeries;
        scatterSeries.setName(QString("%1 Data").arg(dataTypeName).toStdString());

        // Sample points for scatter plot (don't plot every single point if too many)
        int sample_interval = std::max(1, static_cast<int>(num_points / 1000));

        for (int i = 0; i < num_points; i += sample_interval) {
            double t = pred_start + i * dt_val;
            if (t > pred_end) break;

            double target_val = dataTarget.interpol(t);
            double pred_val = predSeries.interpol(t);
            scatterSeries.append(target_val, pred_val);
        }
        scatterData.append(scatterSeries);

        // Add perfect prediction line (y = x)
        TimeSeries<double> perfectLine;
        perfectLine.setName("Perfect Prediction (y=x)");

        double minVal = dataTarget.minC();
        double maxVal = dataTarget.maxC();
        double range = maxVal - minVal;
        double margin = range * 0.05;

        perfectLine.append(minVal - margin, minVal - margin);
        perfectLine.append(maxVal + margin, maxVal + margin);
        scatterData.append(perfectLine);

        // Show chart
        QString title = QString("Predicted vs Target (%1 Data)\nR² = %2, MSE = %3, N = %4")
                            .arg(dataTypeName)
                            .arg(r2, 0, 'f', 4)
                            .arg(mse, 0, 'f', 6)
                            .arg(n);

        ChartWindow* chartWin = ChartWindow::showChart(scatterData, title, this);
        chartWin->setAxisLabels("Target", "Predicted");
        chartWin->chartViewer()->setPlotMode(ChartViewer::Symbols);

        logMessage(QString("Opened predicted vs target scatter plot (%1: R² = %2, MSE = %3)")
                       .arg(dataTypeName)
                       .arg(r2, 0, 'f', 4)
                       .arg(mse, 0, 'f', 6));

    } catch (const std::exception& e) {
        logMessage(QString("Plot error: %1").arg(e.what()));
        QMessageBox::warning(this, "Plot Error",
                             QString("Failed to plot predicted vs target:\n%1").arg(e.what()));
    }
}

void MainWindow::onPlotResults()
{
    if (!bestModel_ || !bestModel_->isInitialized()) {
        QMessageBox::information(this, "No Results",
                                 "Please run GA optimization first to generate results.");
        return;
    }

    // Show both plots
    plotPredictionsVsTime(*bestModel_, DataType::Test);
    plotPredictionsVsTarget(*bestModel_, DataType::Test);
}

double MainWindow::TimeStart(DataType data_type) const
{

    if (data_type == DataType::Train) {
        return targetData.mint();
    } else {
        // Test data starts after train data
        double full_range = targetData.maxt() - targetData.mint();
        return targetData.mint() + (split_ratio * full_range);
    }
}

double MainWindow::TimeEnd(DataType data_type) const
{
    if (data_type == DataType::Train) {
        // Train data ends at split point
        double full_range = targetData.maxt() - targetData.mint();
        return targetData.mint() + (split_ratio * full_range);
    } else {
        return targetData.maxt();
    }
}


double MainWindow::dt() const
{
    // Compute dt from actual target data
    if (targetData.size() == 0) {
        // No data loaded, return default
        return 0.1;
    }

    if (targetData.size() < 2) {
        // Not enough points to compute dt
        return 0.1;
    }

    // Compute dt as the difference between first two time points
    double dt = targetData.getTime(1) - targetData.getTime(0);

    // Validate that dt is positive
    if (dt <= 0) {
        logMessage("WARNING: Computed dt <= 0, using default 0.1");
        return 0.1;
    }

    return dt;
}

void MainWindow::setSplitRatio(double ratio)
{
    // Validate the split ratio
    if (ratio < 0.0 || ratio > 1.0) {
        logMessage(QString("WARNING: Invalid split ratio %1, must be between 0.0 and 1.0. "
                           "Clamping to valid range.").arg(ratio));
        ratio = std::max(0.0, std::min(1.0, ratio));
    }

    // Warn if using extreme values
    if (ratio < 0.1 || ratio > 0.9) {
        logMessage(QString("WARNING: Unusual split ratio %1. "
                           "Typically values between 0.5 and 0.9 are used.").arg(ratio));
    }

    split_ratio = ratio;

    logMessage(QString("Split ratio set to %1 (Train: %2%, Test: %3%)")
                   .arg(split_ratio)
                   .arg(split_ratio * 100, 0, 'f', 1)
                   .arg((1.0 - split_ratio) * 100, 0, 'f', 1));
}

std::pair<TimeSeriesSet<double>, TimeSeries<double>> createSyntheticGATestData(
    double t_start, double t_end, double dt, const std::string& output_path = "./") {

    // Seed for reproducibility
    srand(42);

    // Create 5 stationary input time series
    TimeSeriesSet<double> synthetic_input(5);
    double buffer_start = t_start - 1.0;  // Buffer for lag interpolation

    for (double t = buffer_start; t <= t_end; t += dt) {
        // Series 0: Sine wave
        synthetic_input[0].addPoint(t, 2.0 * std::sin(0.1 * t) + 0.5 * std::sin(0.3 * t));

        // Series 1: Cosine wave
        synthetic_input[1].addPoint(t, 1.5 * std::cos(0.15 * t) + 0.8 * std::cos(0.4 * t));

        // Series 2: White noise
        synthetic_input[2].addPoint(t, 1.0 + 0.5 * (static_cast<double>(rand()) / RAND_MAX - 0.5));

        // Series 3: AR(1) process
        static double prev_val = 0.0;
        double ar_val = 0.7 * prev_val + 0.3 * (static_cast<double>(rand()) / RAND_MAX - 0.5);
        synthetic_input[3].addPoint(t, ar_val);
        prev_val = ar_val;

        // Series 4: Multiple periodic components
        synthetic_input[4].addPoint(t, 1.2 * std::sin(0.2 * t) + 0.8 * std::cos(0.5 * t) + 0.4 * std::sin(0.8 * t));
    }

    synthetic_input.setSeriesName(0, "temperature");
    synthetic_input.setSeriesName(1, "pressure");
    synthetic_input.setSeriesName(2, "flow_rate");
    synthetic_input.setSeriesName(3, "concentration");
    synthetic_input.setSeriesName(4, "velocity");

    // Create target data (linear combination with lags)
    TimeSeries<double> synthetic_target;
    for (double t = t_start; t <= t_end; t += dt) {
        double target = 0.4 * synthetic_input[0].interpol(t - 0.1) +
                        0.3 * synthetic_input[1].interpol(t - 0.3) +
                        0.2 * synthetic_input[3].interpol(t - 0.2) +
                        0.1 * synthetic_input[4].interpol(t - 0.5) +
                        0.05 * (static_cast<double>(rand()) / RAND_MAX - 0.5);
        synthetic_target.addPoint(t, target);
    }

    // Save files
    std::string path_prefix = output_path;
    if (!output_path.empty() && output_path.back() != '/' && output_path.back() != '\\') {
        path_prefix += "/";
    }

    synthetic_input.write(path_prefix + "ga_test_input.csv");

    std::ofstream target_file(path_prefix + "ga_test_target.txt");
    if (target_file.is_open()) {
        for (size_t i = 0; i < synthetic_target.size(); ++i) {
            target_file << std::fixed << std::setprecision(6)
            << synthetic_target.getTime(i) << ","
            << synthetic_target.getValue(i) << std::endl;
        }
        target_file.close();
    }

    std::cout << "Synthetic data created: " << path_prefix + "ga_test_input.csv" << std::endl;
    std::cout << "Target combines: series 0 (lag1), series 1 (lag3), series 3 (lag2), series 4 (lag5)" << std::endl;

    return {synthetic_input, synthetic_target};
}

void MainWindow::onGenerateSyntheticData()
{
    SyntheticDataDialog dialog(this);

    if (dialog.exec() == QDialog::Accepted && dialog.dataGenerated()) {
        // Get generated data
        inputData = dialog.getInputData();
        targetData = dialog.getTargetData();

        // Log success
        logMessage("=== Synthetic Data Generated Successfully ===");
        logMessage(QString("Input: %1 time series with %2 points each")
                       .arg(inputData.size())
                       .arg(inputData[0].size()));
        logMessage(QString("Target: %1 points").arg(targetData.size()));
        logMessage(QString("Time range: %1 to %2")
                       .arg(inputData[0].front().t)
                       .arg(inputData[0].back().t));

        // Enable optimization
        startButton->setEnabled(true);
        startGAAction_->setEnabled(true);

        statusBar()->showMessage("Synthetic data generated successfully", 3000);
        statusLabel->setText("Status: Synthetic data loaded - Ready to optimize");
    }
}

void MainWindow::onConfigureNetwork()
{
    if (inputData.size() == 0 || targetData.size() == 0) {
        QMessageBox::warning(this, "No Data", "Please load or generate data first.");
        return;
    }

    // ... your debug code ...

    NetworkArchitectureDialog dialog(inputData, targetData,
                                     currentProject_.networkArchitecture, this);

    if (dialog.exec() == QDialog::Accepted && dialog.isModelValid()) {
        NeuralNetworkWrapper* configuredModel = dialog.getModel();

        if (configuredModel) {
            // Copy to our current model
            currentModel = *configuredModel;
            delete configuredModel;

            // IMPORTANT: Get architecture from dialog (which has correct activations)
            currentProject_.networkArchitecture = dialog.getArchitecture();

            logMessage("=== Network Architecture Configured ===");
            logMessage(QString("Network initialized with %1 parameters")
                           .arg(currentModel.getTotalParameters()));

            // DEBUG: Show saved activations
            logMessage("Saved activations:");
            for (size_t i = 0; i < currentProject_.networkArchitecture.activations.size(); ++i) {
                logMessage(QString("  Layer %1: %2")
                               .arg(i)
                               .arg(QString::fromStdString(currentProject_.networkArchitecture.activations[i])));
            }

            // Enable training
            startButton->setEnabled(true);

            statusBar()->showMessage("Network configured successfully", 3000);
            statusLabel->setText("Status: Network configured - Ready to train");
        }
    }
}

void MainWindow::onConfigureIncrementalTraining()
{
    IncrementalTrainingDialog dialog(incrementalParams_, this);

    if (dialog.exec() == QDialog::Accepted) {
        logMessage("=== Incremental Training Parameters Updated ===");
        logMessage(QString("Window size: %1, Step: %2")
                       .arg(incrementalParams_.windowSize)
                       .arg(incrementalParams_.windowStep));
        logMessage(QString("Epochs per window: %1, Batch size: %2")
                       .arg(incrementalParams_.epochsPerWindow)
                       .arg(incrementalParams_.batchSize));
        logMessage(QString("Learning rate: %1")
                       .arg(incrementalParams_.learningRate));

        if (incrementalParams_.useOverlap) {
            double overlap = incrementalParams_.windowSize - incrementalParams_.windowStep;
            logMessage(QString("Window overlap: %1 time units").arg(overlap));
        }
    }
}

void MainWindow::onStartIncrementalTraining()
{
    // Validate data is loaded
    if (inputData.size() == 0 || targetData.size() == 0) {
        QMessageBox::warning(this, "No Data",
                             "Please load data before starting incremental training.");
        return;
    }

    // Check if network is configured
    if (!currentProject_.networkArchitecture.isConfigured) {
        QMessageBox::warning(this, "Network Not Configured",
                             "Please configure the network architecture first.\n"
                             "(Training → Configure Network Architecture)");
        return;
    }

    try {
        logMessage("=== Starting Incremental Training ===");

        // Clear and configure the model
        manualModel_.clear();

        // Set architecture directly (not using CreateModel - that's for GA)
        logMessage("Configuring network architecture...");
        manualModel_.setLags(currentProject_.networkArchitecture.lags);
        manualModel_.setHiddenLayers(currentProject_.networkArchitecture.hiddenLayers);

        // Set original series names if available
        std::vector<std::string> series_names;
        for (int i = 0; i < inputData.size(); i++) {
            series_names.push_back(inputData[i].name());
        }
        manualModel_.setOriginalSeriesNames(series_names);

        // Calculate time parameters
        double t_start = TimeStart(DataType::Train);
        double t_end = TimeEnd(DataType::Test);
        double dt_val = dt();
        double ratio = getSplitRatio();
        double train_end = t_start + (ratio * (t_end - t_start));

        logMessage(QString("Time configuration:"));
        logMessage(QString("  Full range: %1 to %2").arg(t_start).arg(t_end));
        logMessage(QString("  Train: %1 to %2").arg(t_start).arg(train_end));
        logMessage(QString("  Test: %1 to %2").arg(train_end).arg(t_end));
        logMessage(QString("  dt: %1, split ratio: %2").arg(dt_val).arg(ratio));

        // Initialize the network structure
        logMessage("Initializing network...");
        int output_size = 1;  // Time series prediction typically has 1 output

        std::string activation = "relu";
        if (!currentProject_.networkArchitecture.activations.empty()) {
            activation = currentProject_.networkArchitecture.activations[0];
        }

        manualModel_.initializeNetwork(output_size, activation);

        if (!manualModel_.isInitialized()) {
            throw std::runtime_error("Network initialization failed");
        }

        logMessage(QString("Network initialized successfully"));
        logMessage(QString("  Total parameters: %1").arg(manualModel_.getTotalParameters()));
        logMessage(QString("  Hidden layers: %1").arg(currentProject_.networkArchitecture.hiddenLayers.size()));

        // Prepare training data
        logMessage("Preparing training data...");
        manualModel_.setInputData(DataType::Train, inputData, t_start, train_end, dt_val);
        manualModel_.setTargetData(DataType::Train, targetData, t_start, train_end, dt_val);

        logMessage("Preparing test data...");
        manualModel_.setInputData(DataType::Test, inputData, train_end, t_end, dt_val);
        manualModel_.setTargetData(DataType::Test, targetData, train_end, t_end, dt_val);

        logMessage("Data preparation complete");

        // Store configuration for incremental training
        // The trainIncremental method will need this
        manualModel_.setTimeSeriesData(inputData, targetData);
        manualModel_.setTimeRange(t_start, t_end, dt_val, ratio);
        manualModel_.setAvailableSeriesCount(inputData.size());

        // Create HyperParameters object and set it
        HyperParameters hyperparams;
        hyperparams.setLags(currentProject_.networkArchitecture.lags);
        hyperparams.setHiddenLayers(currentProject_.networkArchitecture.hiddenLayers);

        // Mark which series are selected (all of them for manual training)
        std::vector<int> selected_series;
        for (int i = 0; i < inputData.size(); i++) {
            selected_series.push_back(i);
        }
        hyperparams.setSelectedSeriesIds(selected_series);

        manualModel_.setHyperParameters(hyperparams);

        // Create progress window
        ProgressWindow* progressWindow = new ProgressWindow(this, "Incremental Training");
        progressWindow->SetPauseEnabled(true);
        progressWindow->show();
        QApplication::processEvents();

        // Disable main window controls
        startButton->setEnabled(false);
        startGAAction_->setEnabled(false);

        // Start incremental training with progress window
        logMessage("Starting incremental training...");
        std::vector<double> window_losses = manualModel_.trainIncremental(incrementalParams_, progressWindow);

        // Re-enable controls
        startButton->setEnabled(true);
        startGAAction_->setEnabled(true);

        // Check if cancelled
        if (progressWindow->IsCancelRequested()) {
            logMessage("Training was cancelled");
            statusLabel->setText("Status: Training cancelled");
            progressWindow->close();
            progressWindow->deleteLater();
            return;
        }

        // Report results
        logMessage(QString("\n=== Training Complete ==="));
        logMessage(QString("Trained on %1 windows").arg(window_losses.size()));

        if (!window_losses.empty()) {
            double avg_loss = std::accumulate(window_losses.begin(), window_losses.end(), 0.0) / window_losses.size();
            double min_loss = *std::min_element(window_losses.begin(), window_losses.end());
            double max_loss = *std::max_element(window_losses.begin(), window_losses.end());

            logMessage(QString("Average window loss: %1").arg(avg_loss, 0, 'f', 6));
            logMessage(QString("Best window loss: %1").arg(min_loss, 0, 'f', 6));
            logMessage(QString("Worst window loss: %1").arg(max_loss, 0, 'f', 6));
        }

        // Evaluate on full dataset
        logMessage("\n=== Evaluation ===");
        auto metrics = manualModel_.evaluate();

        // DEBUG: Print all available keys
        logMessage(QString("DEBUG: evaluate() returned %1 metrics:").arg(metrics.size()));
        for (const auto& pair : metrics) {
            logMessage(QString("  Key: '%1' = %2")
                           .arg(QString::fromStdString(pair.first))
                           .arg(pair.second, 0, 'f', 6));
        }

        // Now try to access the metrics
        if (metrics.count("MSE_Train_0") > 0) {
            logMessage(QString("Train - MSE: %1, R²: %2")
                           .arg(metrics["MSE_Train_0"], 0, 'f', 6)
                           .arg(metrics["R2_Train_0"], 0, 'f', 4));
        } else {
            logMessage("WARNING: MSE_Train_0 key not found in metrics");
        }

        if (metrics.count("MSE_Test_0") > 0) {
            logMessage(QString("Test  - MSE: %1, R²: %2")
                           .arg(metrics["MSE_Test_0"], 0, 'f', 6)
                           .arg(metrics["R2_Test_0"], 0, 'f', 4));
        } else {
            logMessage("WARNING: MSE_Test_0 key not found in metrics");
        }

        // Ask if user wants to see plots
        QMessageBox::StandardButton reply = QMessageBox::question(
            progressWindow,
            "Training Complete",
            QString("Incremental training completed!\n\n"
                    "Test R² = %1\n"
                    "Test MSE = %2\n\n"
                    "Would you like to view the prediction plots?")
                .arg(metrics["R2_Test_0"], 0, 'f', 4)
                .arg(metrics["MSE_Test_0"], 0, 'f', 6),
            QMessageBox::Yes | QMessageBox::No
            );

        if (reply == QMessageBox::Yes) {
            // Store as best model for plotting
            if (bestModel_) {
                delete bestModel_;
            }
            bestModel_ = new NeuralNetworkWrapper(manualModel_);

            onPlotResults();
        }

        progressWindow->close();
        progressWindow->deleteLater();

        statusLabel->setText("Status: Training complete");

    } catch (const std::exception& e) {
        logMessage(QString("ERROR: %1").arg(e.what()));
        QMessageBox::critical(this, "Training Error",
                              QString("Incremental training failed:\n%1").arg(e.what()));
        statusLabel->setText("Status: Training failed");
        startButton->setEnabled(true);
        startGAAction_->setEnabled(true);
    }
}

void MainWindow::onNewProject()
{
    if (maybeSaveProject()) {
        currentProject_ = ProjectConfig();
        currentProjectPath_.clear();
        inputData = TimeSeriesSet<double>();
        targetData = TimeSeries<double>();

        setWindowTitle("NeuroForge - New Project");
        logMessage("New project created");
    }
}

void MainWindow::onLoadProject()
{
    if (!maybeSaveProject()) {
        return;
    }

    QString filepath = QFileDialog::getOpenFileName(
        this,
        "Load Project",
        QDir::homePath(),
        "NeuroForge Projects (*.nfproj);;All Files (*)"
        );

    if (!filepath.isEmpty()) {
        if (loadProject(filepath)) {
            setCurrentProjectPath(filepath);
            saveLastProjectPath(filepath);
        }
    }
}

void MainWindow::onSaveProject()
{
    if (currentProjectPath_.isEmpty()) {
        onSaveProjectAs();
    } else {
        saveProject(currentProjectPath_);
    }
}

void MainWindow::onSaveProjectAs()
{
    QString filepath = QFileDialog::getSaveFileName(
        this,
        "Save Project As",
        QDir::homePath(),
        "NeuroForge Projects (*.nfproj);;All Files (*)"
        );

    if (!filepath.isEmpty()) {
        if (!filepath.endsWith(".nfproj", Qt::CaseInsensitive)) {
            filepath += ".nfproj";
        }

        if (saveProject(filepath)) {
            setCurrentProjectPath(filepath);
            saveLastProjectPath(filepath);
        }
    }
}

bool MainWindow::saveProject(const QString& filepath)
{
    updateProjectFromUI();

    QJsonObject json = currentProject_.toJson();
    QJsonDocument doc(json);

    QFile file(filepath);
    if (!file.open(QIODevice::WriteOnly)) {
        QMessageBox::critical(this, "Error",
                              QString("Could not save project:\n%1").arg(file.errorString()));
        return false;
    }

    file.write(doc.toJson());
    file.close();

    logMessage(QString("Project saved: %1").arg(filepath));
    statusBar()->showMessage("Project saved successfully", 3000);

    return true;
}

bool MainWindow::loadProject(const QString& filepath)
{
    QFile file(filepath);
    if (!file.open(QIODevice::ReadOnly)) {
        QMessageBox::critical(this, "Error",
                              QString("Could not load project:\n%1").arg(file.errorString()));
        return false;
    }

    QByteArray data = file.readAll();
    file.close();

    QJsonDocument doc = QJsonDocument::fromJson(data);
    if (doc.isNull() || !doc.isObject()) {
        QMessageBox::critical(this, "Error", "Invalid project file format");
        return false;
    }

    currentProject_.fromJson(doc.object());
    updateUIFromProject();

    logMessage(QString("Project loaded: %1").arg(filepath));
    statusBar()->showMessage("Project loaded successfully", 3000);

    return true;
}

void MainWindow::updateProjectFromUI()
{
    // Store current GA settings
    currentProject_.gaPopulation = ga_.Settings.totalpopulation;
    currentProject_.gaGenerations = ga_.Settings.generations;
    currentProject_.gaMutationRate = ga_.Settings.mutation_probability;
    currentProject_.gaOutputPath = QString::fromStdString(ga_.Settings.outputpath);

    // Store data settings
    currentProject_.splitRatio = split_ratio;

    if (currentModel.isInitialized() && !currentProject_.networkArchitecture.isConfigured) {
        // If model exists but architecture wasn't saved via dialog, mark as configured
        currentProject_.networkArchitecture.isConfigured = true;
    }

    // Store incremental training params
    currentProject_.incrementalParams = incrementalParams_;
}

void MainWindow::updateUIFromProject()
{
    // Restore GA settings
    ga_.Settings.totalpopulation = currentProject_.gaPopulation;
    ga_.Settings.generations = currentProject_.gaGenerations;
    ga_.Settings.mutation_probability = currentProject_.gaMutationRate;
    ga_.Settings.outputpath = currentProject_.gaOutputPath.toStdString();
    // Restore data settings
    split_ratio = currentProject_.splitRatio;

    // Restore incremental training params
    incrementalParams_ = currentProject_.incrementalParams;

    // Load data files if specified
    if (!currentProject_.inputDataPath.isEmpty() && !currentProject_.targetDataPath.isEmpty()) {
        try {
            // Load input data (TimeSeriesSet from CSV)
            inputData = TimeSeriesSet<double>(currentProject_.inputDataPath.toStdString(), true);

            // Load target data (TimeSeries from text file)
            std::ifstream targetFile(currentProject_.targetDataPath.toStdString());
            if (!targetFile.is_open()) {
                throw std::runtime_error("Could not open target file: " + currentProject_.targetDataPath.toStdString());
            }

            targetData = TimeSeries<double>();
            targetData.setName("target");

            std::string line;
            while (std::getline(targetFile, line)) {
                std::stringstream ss(line);
                std::string timeStr, valueStr;

                if (std::getline(ss, timeStr, ',') && std::getline(ss, valueStr)) {
                    double t = std::stod(timeStr);
                    double v = std::stod(valueStr);
                    targetData.addPoint(t, v);
                }
            }
            targetFile.close();

            logMessage(QString("Data loaded from project: %1 input series, %2 target points")
                           .arg(inputData.size())
                           .arg(targetData.size()));

            startButton->setEnabled(true);
            startGAAction_->setEnabled(true);
            statusLabel->setText("Status: Data loaded - Ready to optimize");

        } catch (const std::exception& e) {
            logMessage(QString("ERROR: Could not load data files: %1").arg(e.what()));
            QMessageBox::warning(this, "Data Load Error",
                                 QString("Could not load data files:\n%1").arg(e.what()));
        }
    }

    // Restore network architecture if configured
    if (currentProject_.networkArchitecture.isConfigured && inputData.size() > 0) {
        try {
            currentModel.clear();
            currentModel.setHiddenLayers(currentProject_.networkArchitecture.hiddenLayers);
            currentModel.setLags(currentProject_.networkArchitecture.lags);

            // Initialize network
            currentModel.initializeNetwork(1, "relu"); // 1 output

            logMessage(QString("Network architecture restored: %1 hidden layers, %2 parameters")
                           .arg(currentProject_.networkArchitecture.hiddenLayers.size())
                           .arg(currentModel.getTotalParameters()));

            logMessage(currentModel.ParametersToString().c_str());

        } catch (const std::exception& e) {
            logMessage(QString("ERROR: Could not restore network architecture: %1").arg(e.what()));
        }
    }

    logMessage("Project configuration restored");
}

void MainWindow::setCurrentProjectPath(const QString& path)
{
    currentProjectPath_ = path;
    QFileInfo fileInfo(path);
    setWindowTitle(QString("NeuroForge - %1").arg(fileInfo.fileName()));
}

bool MainWindow::maybeSaveProject()
{
    // For now, just ask without checking if modified
    // You could add a "modified" flag to track changes
    if (!currentProjectPath_.isEmpty()) {
        QMessageBox::StandardButton reply = QMessageBox::question(
            this,
            "Save Project?",
            "Do you want to save the current project?",
            QMessageBox::Yes | QMessageBox::No | QMessageBox::Cancel
            );

        if (reply == QMessageBox::Cancel) {
            return false;
        } else if (reply == QMessageBox::Yes) {
            return saveProject(currentProjectPath_);
        }
    }
    return true;
}

void MainWindow::loadLastProject()
{
    QString lastPath = getLastProjectPath();

    if (!lastPath.isEmpty() && QFile::exists(lastPath)) {
        QMessageBox::StandardButton reply = QMessageBox::question(
            this,
            "Load Last Project",
            QString("Load the last project?\n\n%1").arg(lastPath),
            QMessageBox::Yes | QMessageBox::No
            );

        if (reply == QMessageBox::Yes) {
            loadProject(lastPath);
            setCurrentProjectPath(lastPath);
        }
    }
}

void MainWindow::saveLastProjectPath(const QString& path)
{
    QSettings settings("NeuroForge", "NeuroForge");
    settings.setValue("lastProject", path);
}

QString MainWindow::getLastProjectPath() const
{
    QSettings settings("NeuroForge", "NeuroForge");
    return settings.value("lastProject", "").toString();
}

void MainWindow::onConfigureLags()
{
    if (inputData.size() == 0) {
        QMessageBox::warning(this, "No Data",
                             "Please load data before configuring lags.");
        return;
    }

    LagConfigDialog dialog(currentProject_.networkArchitecture, inputData, this);

    if (dialog.exec() == QDialog::Accepted) {
        logMessage("Lag configuration updated");

        // Log new configuration
        for (size_t i = 0; i < currentProject_.networkArchitecture.lags.size(); i++) {
            QString lag_str = QString("Series %1 lags: [").arg(i);
            for (size_t j = 0; j < currentProject_.networkArchitecture.lags[i].size(); j++) {
                lag_str += QString::number(currentProject_.networkArchitecture.lags[i][j]);
                if (j < currentProject_.networkArchitecture.lags[i].size() - 1) {
                    lag_str += ", ";
                }
            }
            lag_str += "]";
            logMessage(lag_str);
        }
    }
}

void MainWindow::onPlotData()
{
    // Check if data is loaded
    if (inputData.size() == 0 || targetData.size() == 0) {
        QMessageBox::information(this, "No Data",
                                 "No data loaded. Please load data first.\n\n"
                                 "Use File → Load Data or File → Generate Synthetic Data.");
        return;
    }

    // Show selection dialog
    DataPlotDialog dialog(inputData, targetData, this);

    if (dialog.exec() == QDialog::Accepted) {
        std::vector<int> selectedSeries = dialog.getSelectedInputSeries();
        bool plotTarget = dialog.shouldPlotTarget();

        // Create dataset for plotting
        TimeSeriesSet<double> plotData;

        // Add selected input series
        for (int idx : selectedSeries) {
            TimeSeries<double> series = inputData[idx];
            series.setName(QString("Input %1: %2")
                               .arg(idx)
                               .arg(QString::fromStdString(inputData[idx].name()))
                               .toStdString());
            plotData.append(series);
        }

        // Add target series
        if (plotTarget) {
            TimeSeries<double> target = targetData;
            target.setName(QString("Target: %1")
                               .arg(QString::fromStdString(targetData.name()))
                               .toStdString());
            plotData.append(target);
        }

        // Create chart title
        QString title = "Raw Data Visualization";
        if (selectedSeries.size() > 0 && plotTarget) {
            title += QString(" (%1 input series + target)").arg(selectedSeries.size());
        } else if (selectedSeries.size() > 0) {
            title += QString(" (%1 input series)").arg(selectedSeries.size());
        } else if (plotTarget) {
            title += " (target only)";
        }

        // Show chart
        ChartWindow* chartWin = ChartWindow::showChart(plotData, title, this);
        chartWin->setAxisLabels("Time", "Value");
        chartWin->chartViewer()->setPlotMode(ChartViewer::Lines);

        logMessage(QString("Plotted %1 series").arg(plotData.size()));
    }
}

bool MainWindow::extractLatestWindow(double window_size_time,
                                     torch::Tensor& window_input,
                                     torch::Tensor& window_target)
{
    if (inputData.size() == 0 || targetData.size() == 0) {
        logMessage("ERROR: No data loaded");
        return false;
    }

    // Get time range
    double t_min = targetData.mint();
    double t_max = targetData.maxt();
    double total_time = t_max - t_min;

    if (window_size_time > total_time) {
        logMessage(QString("ERROR: Window size (%1) exceeds available data (%2)")
                       .arg(window_size_time).arg(total_time));
        return false;
    }

    // Calculate window boundaries (last window)
    double window_start = t_max - window_size_time;
    double window_end = t_max;
    double dt_val = dt();

    logMessage(QString("Extracting window: [%1, %2] (size: %3 time units)")
                   .arg(window_start).arg(window_end).arg(window_size_time));

    // Create tensors for this window
    try {
        createTensorsFromTimeRange(inputData, targetData,
                                   window_start, window_end, dt_val,
                                   currentProject_.networkArchitecture.lags,
                                   window_input, window_target);

        logMessage(QString("Window extracted: %1 samples, %2 features")
                       .arg(window_input.size(0))
                       .arg(window_input.size(1)));

        return true;

    } catch (const std::exception& e) {
        logMessage(QString("ERROR extracting window: %1").arg(e.what()));
        return false;
    }
}

void MainWindow::createTensorsFromTimeRange(const TimeSeriesSet<double>& input_series,
                                            const TimeSeries<double>& target_series,
                                            double t_start, double t_end, double dt,
                                            const std::vector<std::vector<int>>& lags,
                                            torch::Tensor& output_input,
                                            torch::Tensor& output_target)
{
    // Calculate number of time steps
    int num_samples = static_cast<int>(std::round((t_end - t_start) / dt)) + 1;

    // Calculate total number of input features
    int total_features = 0;
    for (const auto& series_lags : lags) {
        total_features += series_lags.size();
    }

    if (total_features == 0) {
        throw std::runtime_error("No lags configured");
    }

    // Create input tensor
    output_input = torch::zeros({num_samples, total_features});

    // Fill input tensor with lagged values
    int feature_idx = 0;
    for (size_t series_idx = 0; series_idx < input_series.size(); series_idx++) {
        if (series_idx >= lags.size()) break;

        const TimeSeries<double>& series = input_series[series_idx];
        const std::vector<int>& series_lags = lags[series_idx];

        for (int lag : series_lags) {
            for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {
                double t = t_start + sample_idx * dt;
                double lagged_t = t - (lag * dt);

                // Get value at lagged time (or 0 if outside range)
                double value = 0.0;
                if (lagged_t >= series.mint() && lagged_t <= series.maxt()) {
                    value = series.interpol(lagged_t);
                }

                output_input[sample_idx][feature_idx] = value;
            }
            feature_idx++;
        }
    }

    // Create target tensor
    output_target = torch::zeros({num_samples, 1});

    for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {
        double t = t_start + sample_idx * dt;

        if (t >= target_series.mint() && t <= target_series.maxt()) {
            output_target[sample_idx][0] = target_series.interpol(t);
        }
    }
}

void MainWindow::onTrainOnLatestWindow()
{
    // Validate data and model
    if (inputData.size() == 0 || targetData.size() == 0) {
        QMessageBox::warning(this, "No Data",
                             "Please load data before training.");
        return;
    }

    if (!manualModel_.isInitialized()) {
        QMessageBox::StandardButton reply = QMessageBox::question(
            this,
            "Model Not Initialized",
            "The model has not been initialized yet.\n\n"
            "Would you like to initialize it now?",
            QMessageBox::Yes | QMessageBox::No
            );

        if (reply == QMessageBox::No) {
            return;
        }

        // Initialize model
        try {
            manualModel_.clear();
            manualModel_.setLags(currentProject_.networkArchitecture.lags);
            manualModel_.setHiddenLayers(currentProject_.networkArchitecture.hiddenLayers);

            int output_size = 1;
            std::string activation = "relu";
            if (!currentProject_.networkArchitecture.activations.empty()) {
                activation = currentProject_.networkArchitecture.activations[0];
            }

            manualModel_.initializeNetwork(output_size, activation);

            logMessage("Model initialized for online training");

        } catch (const std::exception& e) {
            QMessageBox::critical(this, "Initialization Error",
                                  QString("Failed to initialize model:\n%1").arg(e.what()));
            return;
        }
    }

    // Get parameters from user
    bool ok;

    double window_size = QInputDialog::getDouble(
        this,
        "Window Size",
        "Enter window size (in time units):",
        200.0,  // default
        10.0,   // min
        10000.0, // max
        1,      // decimals
        &ok
        );

    if (!ok) return;

    int epochs = QInputDialog::getInt(
        this,
        "Training Epochs",
        "Number of epochs to train:",
        50,   // default
        1,    // min
        1000, // max
        1,    // step
        &ok
        );

    if (!ok) return;

    int batch_size = QInputDialog::getInt(
        this,
        "Batch Size",
        "Batch size for training:",
        32,   // default
        1,    // min
        512,  // max
        1,    // step
        &ok
        );

    if (!ok) return;

    double learning_rate = QInputDialog::getDouble(
        this,
        "Learning Rate",
        "Learning rate:",
        0.001,   // default
        0.000001, // min
        0.1,     // max
        6,       // decimals
        &ok
        );

    if (!ok) return;

    // Extract latest window
    logMessage("=== Training on Latest Window ===");
    logMessage(QString("Window size: %1, Epochs: %2, Batch size: %3, LR: %4")
                   .arg(window_size).arg(epochs).arg(batch_size).arg(learning_rate));

    torch::Tensor window_input, window_target;

    if (!extractLatestWindow(window_size, window_input, window_target)) {
        QMessageBox::critical(this, "Error", "Failed to extract window from data");
        return;
    }

    // Train on this window
    try {
        statusLabel->setText("Status: Training on latest window...");
        progressBar->setRange(0, 0); // Indeterminate
        QApplication::processEvents();

        std::vector<double> loss_history = manualModel_.trainOnWindow(
            window_input, window_target,
            epochs, batch_size, learning_rate
            );

        progressBar->setRange(0, 100);
        progressBar->setValue(100);

        // Report results
        double initial_loss = loss_history.front();
        double final_loss = loss_history.back();
        double improvement = ((initial_loss - final_loss) / initial_loss) * 100.0;

        logMessage(QString("Training complete:"));
        logMessage(QString("  Initial loss: %1").arg(initial_loss, 0, 'f', 6));
        logMessage(QString("  Final loss: %2").arg(final_loss, 0, 'f', 6));
        logMessage(QString("  Improvement: %3%").arg(improvement, 0, 'f', 2));

        statusLabel->setText("Status: Window training complete");

        QMessageBox::information(
            this,
            "Training Complete",
            QString("Training on latest window complete!\n\n"
                    "Initial loss: %1\n"
                    "Final loss: %2\n"
                    "Improvement: %3%\n\n"
                    "The model has been updated with the latest data.")
                .arg(initial_loss, 0, 'f', 6)
                .arg(final_loss, 0, 'f', 6)
                .arg(improvement, 0, 'f', 2)
            );

        // Update best model for plotting
        if (bestModel_) {
            delete bestModel_;
        }
        bestModel_ = new NeuralNetworkWrapper(manualModel_);

    } catch (const std::exception& e) {
        progressBar->setRange(0, 100);
        progressBar->setValue(0);
        statusLabel->setText("Status: Training failed");

        logMessage(QString("ERROR: %1").arg(e.what()));
        QMessageBox::critical(this, "Training Error",
                              QString("Training failed:\n%1").arg(e.what()));
    }
}

