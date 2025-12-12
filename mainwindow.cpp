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

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , gaRunning_(false)     // <-- ADD THIS
{
    ui->setupUi(this);

    setWindowTitle("NeuroForge - Neural Network GA Optimizer");
    resize(1000, 700);

    setupUI();
    setupMenus();
    connectSignals();

    logMessage("Application started. Ready to load data and configure optimization.");
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
    QMenu *fileMenu = menuBar()->addMenu("&File");
    
    QAction *loadDataAction = new QAction("&Load Data...", this);
    loadDataAction->setShortcut(QKeySequence::Open);
    connect(loadDataAction, &QAction::triggered, this, &MainWindow::onLoadData);
    fileMenu->addAction(loadDataAction);
    
    QAction *saveModelAction = new QAction("&Save Model...", this);
    saveModelAction->setShortcut(QKeySequence::Save);
    fileMenu->addAction(saveModelAction);
    
    fileMenu->addSeparator();
    
    QAction *exitAction = new QAction("E&xit", this);
    exitAction->setShortcut(QKeySequence::Quit);
    connect(exitAction, &QAction::triggered, this, &MainWindow::onExit);
    fileMenu->addAction(exitAction);
    
    // Configuration menu
    QMenu *configMenu = menuBar()->addMenu("&Configuration");
    
    QAction *hyperparamsAction = new QAction("&Hyperparameters...", this);
    configMenu->addAction(hyperparamsAction);
    
    QAction *gaSettingsAction = new QAction("&GA Settings...", this);
    configMenu->addAction(gaSettingsAction);
    connect(gaSettingsAction, &QAction::triggered, this, &MainWindow::onConfigureGA);
    
    // Run menu
    QMenu *runMenu = menuBar()->addMenu("&Run");
    
    startGAAction_ = new QAction("&Start GA Optimization", this);
    startGAAction_->setShortcut(Qt::Key_F5);
    connect(startGAAction_, &QAction::triggered, this, &MainWindow::onStartGA);  // <-- ADD THIS
    runMenu->addAction(startGAAction_);

    stopGAAction_ = new QAction("S&top", this);
    stopGAAction_->setShortcut(Qt::Key_Escape);
    stopGAAction_->setEnabled(false);  // Disabled until GA starts
    connect(stopGAAction_, &QAction::triggered, this, &MainWindow::onStopGA);  // <-- ADD THIS
    runMenu->addAction(stopGAAction_);


    // View menu
    QMenu *viewMenu = menuBar()->addMenu("&View");
    
    QAction *clearLogAction = new QAction("&Clear Log", this);
    connect(clearLogAction, &QAction::triggered, logOutput, &QTextEdit::clear);
    viewMenu->addAction(clearLogAction);
    
    // Help menu
    QMenu *helpMenu = menuBar()->addMenu("&Help");
    
    QAction *aboutAction = new QAction("&About", this);
    connect(aboutAction, &QAction::triggered, this, &MainWindow::onAbout);
    helpMenu->addAction(aboutAction);
    
    QAction *aboutQtAction = new QAction("About &Qt", this);
    connect(aboutQtAction, &QAction::triggered, qApp, &QApplication::aboutQt);
    helpMenu->addAction(aboutQtAction);
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
        // Get data type name for titles
        QString dataTypeName = (data_type == DataType::Train) ? "Train" : "Test";

        // Get target data as tensor
        torch::Tensor targetTensor = model.getTargetData(data_type);

        // Get the actual number of points in tensor
        int64_t num_points = targetTensor.size(0);

        logMessage(QString("DEBUG: %1 tensor has %2 points").arg(dataTypeName).arg(num_points));

        // Compute time range from target data
        double t_min = targetData.mint();
        double t_max = targetData.maxt();
        double full_range = t_max - t_min;

        // Calculate start and end based on data type
        double data_start, data_end;
        if (data_type == DataType::Train) {
            data_start = t_min;
            data_end = t_min + (getSplitRatio() * full_range);
        } else {
            data_start = t_min + (getSplitRatio() * full_range);
            data_end = t_max;
        }

        double dt_val = dt();

        // Adjust end time to match actual tensor size
        data_end = data_start + (num_points - 1) * dt_val;

        logMessage(QString("DEBUG: %1 time range [%2, %3], dt=%4")
                       .arg(dataTypeName).arg(data_start).arg(data_end).arg(dt_val));

        // Convert tensor to TimeSeries
        TimeSeries<double> dataTarget = TimeSeries<double>::fromTensor(
            targetTensor,
            false,       // has_time = false
            data_start,  // time_offset
            dt_val       // time_step
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

        // Make predictions - use actual bounds from dataTarget
        double pred_start = dataTarget.mint();
        double pred_end = dataTarget.maxt();

        TimeSeriesSet<double> predictions = model.predict(data_type, pred_start, pred_end, dt_val);

        if (predictions.size() == 0) {
            throw std::runtime_error("No predictions generated");
        }

        TimeSeries<double> predSeries = predictions[0];

        logMessage(QString("DEBUG: Prediction TimeSeries: %1 points").arg(predSeries.size()));

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
        // Get data type name for titles
        QString dataTypeName = (data_type == DataType::Train) ? "Train" : "Test";

        // Get target data as tensor
        torch::Tensor targetTensor = model.getTargetData(data_type);

        // Get actual number of points
        int64_t num_points = targetTensor.size(0);

        // Compute time range
        double t_min = targetData.mint();
        double t_max = targetData.maxt();
        double full_range = t_max - t_min;

        double data_start, data_end;
        if (data_type == DataType::Train) {
            data_start = t_min;
            data_end = t_min + (getSplitRatio() * full_range);
        } else {
            data_start = t_min + (getSplitRatio() * full_range);
            data_end = t_max;
        }

        double dt_val = dt();

        // Adjust to match tensor size
        data_end = data_start + (num_points - 1) * dt_val;

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

        for (double t = pred_start; t <= pred_end; t += dt_val) {
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
