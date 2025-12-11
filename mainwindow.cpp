#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QMenuBar>
#include <QMessageBox>
#include <QFileDialog>
#include <QDateTime>
#include "gasettingsdialog.h"

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

void MainWindow::logMessage(const QString &message)
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
    // Check if data is loaded
    if (inputData.size() == 0 || targetData.size() == 0) {
        QMessageBox::warning(this, "No Data",
                             "Please load input and target data before starting GA optimization.");
        logMessage("ERROR: Cannot start GA - no data loaded");
        return;
    }

    // Check if already running
    if (gaRunning_) {
        QMessageBox::information(this, "Already Running",
                                 "GA optimization is already in progress.");
        return;
    }

    logMessage("=== Starting GA Optimization ===");
    statusLabel->setText("Status: Running GA Optimization...");

    // Disable start button, enable stop button
    startGAAction_->setEnabled(false);
    startButton->setEnabled(false);
    stopGAAction_->setEnabled(true);
    stopButton->setEnabled(true);
    gaRunning_ = true;

    progressBar->setValue(0);

    try {
        // Configure GA with current data
        logMessage(QString("Population: %1, Generations: %2")
                       .arg(ga_.Settings.totalpopulation)
                       .arg(ga_.Settings.generations));

        // Set up GA data interface
        double t_start = 0.0;
        double t_end = 100.0;
        double dt = 0.1;
        double split_ratio = 0.7;

        ga_.model.setTimeSeriesData(inputData, targetData);
        ga_.model.setTimeRange(t_start, t_end, dt, split_ratio);
        ga_.model.setAvailableSeriesCount(inputData.size());

        // Initialize GA
        logMessage("Initializing GA population...");
        ga_.Initialize();

        // Run optimization
        logMessage("Running optimization...");
        NeuralNetworkWrapper& bestModel = ga_.Optimize();

        // Get results
        auto ranks = ga_.getRanks();
        Individual bestIndividual = ga_.Individuals[ranks[0]];

        logMessage(QString("Optimization complete! Best fitness: %1")
                       .arg(bestIndividual.fitness));

        // Update progress
        progressBar->setValue(100);
        statusLabel->setText("Status: GA Optimization Complete");

        // Save results
        bestModel.saveModel("best_ga_model.pt");
        logMessage("Best model saved to: best_ga_model.pt");

        QMessageBox::information(this, "Complete",
                                 QString("GA optimization completed!\nBest fitness: %1")
                                     .arg(bestIndividual.fitness));

    } catch (const std::exception& e) {
        logMessage(QString("ERROR: %1").arg(e.what()));
        QMessageBox::critical(this, "Error",
                              QString("GA optimization failed:\n%1").arg(e.what()));
    }

    // Re-enable controls
    startGAAction_->setEnabled(true);
    startButton->setEnabled(true);
    stopGAAction_->setEnabled(false);
    stopButton->setEnabled(false);
    gaRunning_ = false;

    statusLabel->setText("Status: Ready");
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
