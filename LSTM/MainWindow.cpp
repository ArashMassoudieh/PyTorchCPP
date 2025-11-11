/**
 * @file MainWindow.cpp
 * @brief Implementation of main application window
 */

#include "MainWindow.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QLabel>
#include <QFileDialog>
#include <QMessageBox>
#include <QDateTime>

namespace lstm_predictor {

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent),
      device_(torch::kCPU),
      dataLoaded_(false),
      modelTrained_(false) {
    
    // Check for CUDA availability
    if (torch::cuda::is_available()) {
        device_ = torch::Device(torch::kCUDA);
        logMessage("CUDA is available! Using GPU.");
    } else {
        logMessage("CUDA not available. Using CPU.");
    }
    
    setupUI();
    
    // Initialize components
    preprocessor_ = std::make_unique<DataPreprocessor>();
    plotter_ = std::make_unique<ResultsPlotter>();
}

MainWindow::~MainWindow() = default;

void MainWindow::setupUI() {
    setWindowTitle("LSTM Time Series Predictor");
    resize(1200, 800);
    
    // Central widget
    QWidget* centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);
    
    QVBoxLayout* mainLayout = new QVBoxLayout(centralWidget);
    
    // Title
    QLabel* titleLabel = new QLabel("<h1>LSTM Time Series Predictor</h1>");
    titleLabel->setAlignment(Qt::AlignCenter);
    mainLayout->addWidget(titleLabel);
    
    // Data loading section
    QGroupBox* dataGroup = new QGroupBox("Data Loading");
    QHBoxLayout* dataLayout = new QHBoxLayout();
    
    filePathEdit_ = new QLineEdit();
    filePathEdit_->setPlaceholderText("Select CSV file...");
    
    loadDataButton_ = new QPushButton("Load Data");
    connect(loadDataButton_, &QPushButton::clicked, this, &MainWindow::onLoadData);
    
    dataLayout->addWidget(new QLabel("File:"));
    dataLayout->addWidget(filePathEdit_);
    dataLayout->addWidget(loadDataButton_);
    dataGroup->setLayout(dataLayout);
    mainLayout->addWidget(dataGroup);
    
    // Parameters section
    mainLayout->addWidget(createParameterControls());
    
    // Training controls
    QHBoxLayout* trainLayout = new QHBoxLayout();
    trainButton_ = new QPushButton("Train Model");
    trainButton_->setEnabled(false);
    connect(trainButton_, &QPushButton::clicked, this, &MainWindow::onTrain);
    trainLayout->addWidget(trainButton_);
    mainLayout->addLayout(trainLayout);
    
    // Progress bar
    progressBar_ = new QProgressBar();
    mainLayout->addWidget(progressBar_);
    
    // Console output
    QGroupBox* consoleGroup = new QGroupBox("Console Output");
    QVBoxLayout* consoleLayout = new QVBoxLayout();
    consoleOutput_ = new QTextEdit();
    consoleOutput_->setReadOnly(true);
    consoleOutput_->setMaximumHeight(150);
    consoleLayout->addWidget(consoleOutput_);
    consoleGroup->setLayout(consoleLayout);
    mainLayout->addWidget(consoleGroup);
    
    // Results tabs
    resultsTabWidget_ = new QTabWidget();
    mainLayout->addWidget(resultsTabWidget_);
}

QWidget* MainWindow::createParameterControls() {
    QGroupBox* paramGroup = new QGroupBox("Model Parameters");
    QVBoxLayout* paramLayout = new QVBoxLayout();
    
    // Create two rows of parameters
    QHBoxLayout* row1 = new QHBoxLayout();
    QHBoxLayout* row2 = new QHBoxLayout();
    
    // Row 1: Sequence, Hidden, Layers, Batch
    seqLengthSpin_ = new QSpinBox();
    seqLengthSpin_->setRange(1, 100);
    seqLengthSpin_->setValue(10);
    row1->addWidget(new QLabel("Seq Length:"));
    row1->addWidget(seqLengthSpin_);
    
    hiddenSizeSpin_ = new QSpinBox();
    hiddenSizeSpin_->setRange(8, 256);
    hiddenSizeSpin_->setValue(64);
    row1->addWidget(new QLabel("Hidden Size:"));
    row1->addWidget(hiddenSizeSpin_);
    
    numLayersSpin_ = new QSpinBox();
    numLayersSpin_->setRange(1, 5);
    numLayersSpin_->setValue(2);
    row1->addWidget(new QLabel("Layers:"));
    row1->addWidget(numLayersSpin_);
    
    batchSizeSpin_ = new QSpinBox();
    batchSizeSpin_->setRange(4, 128);
    batchSizeSpin_->setValue(32);
    row1->addWidget(new QLabel("Batch Size:"));
    row1->addWidget(batchSizeSpin_);
    
    // Row 2: Epochs, LR, Dropout, Weight Decay, Patience
    numEpochsSpin_ = new QSpinBox();
    numEpochsSpin_->setRange(1, 1000);
    numEpochsSpin_->setValue(200);
    row2->addWidget(new QLabel("Epochs:"));
    row2->addWidget(numEpochsSpin_);
    
    learningRateSpin_ = new QDoubleSpinBox();
    learningRateSpin_->setRange(0.0001, 0.1);
    learningRateSpin_->setDecimals(4);
    learningRateSpin_->setSingleStep(0.0001);
    learningRateSpin_->setValue(0.001);
    row2->addWidget(new QLabel("Learning Rate:"));
    row2->addWidget(learningRateSpin_);
    
    dropoutSpin_ = new QDoubleSpinBox();
    dropoutSpin_->setRange(0.0, 0.9);
    dropoutSpin_->setDecimals(2);
    dropoutSpin_->setSingleStep(0.05);
    dropoutSpin_->setValue(0.3);
    row2->addWidget(new QLabel("Dropout:"));
    row2->addWidget(dropoutSpin_);
    
    weightDecaySpin_ = new QDoubleSpinBox();
    weightDecaySpin_->setRange(0.0, 0.01);
    weightDecaySpin_->setDecimals(5);
    weightDecaySpin_->setSingleStep(0.0001);
    weightDecaySpin_->setValue(0.0001);
    row2->addWidget(new QLabel("Weight Decay:"));
    row2->addWidget(weightDecaySpin_);
    
    patienceSpin_ = new QSpinBox();
    patienceSpin_->setRange(5, 100);
    patienceSpin_->setValue(30);
    row2->addWidget(new QLabel("Patience:"));
    row2->addWidget(patienceSpin_);
    
    paramLayout->addLayout(row1);
    paramLayout->addLayout(row2);
    paramGroup->setLayout(paramLayout);
    
    return paramGroup;
}

void MainWindow::onLoadData() {
    QString fileName = QFileDialog::getOpenFileName(this,
        "Select CSV File", "", "CSV Files (*.csv *.txt)");
    
    if (fileName.isEmpty()) {
        return;
    }
    
    filePathEdit_->setText(fileName);
    logMessage("Loading data from: " + fileName);
    
    try {
        // Load and preprocess data
        preprocessor_ = std::make_unique<DataPreprocessor>(
            seqLengthSpin_->value(), 0.2);
        
        processedData_ = preprocessor_->loadAndPreprocess(fileName, true);
        
        logMessage(QString("Data loaded successfully!"));
        logMessage(QString("Training samples: %1").arg(processedData_.XTrain.size(0)));
        logMessage(QString("Test samples: %1").arg(processedData_.XTest.size(0)));
        
        dataLoaded_ = true;
        trainButton_->setEnabled(true);
        
    } catch (const std::exception& e) {
        QMessageBox::critical(this, "Error", QString("Failed to load data: %1")
                            .arg(e.what()));
        logMessage("ERROR: " + QString(e.what()));
    }
}

void MainWindow::onTrain() {
    if (!dataLoaded_) {
        QMessageBox::warning(this, "Warning", "Please load data first!");
        return;
    }
    
    logMessage("Starting model training...");
    trainButton_->setEnabled(false);
    
    try {
        // Create model
        int64_t inputSize = processedData_.XTrain.size(2);
        model_ = std::make_shared<LSTMModel>(
            inputSize,
            hiddenSizeSpin_->value(),
            numLayersSpin_->value(),
            1,  // Output size
            dropoutSpin_->value()
        );
        
        logMessage(QString("Model created with %1 parameters")
                  .arg(model_->getNumParameters()));
        
        // Create trainer
        trainer_ = std::make_unique<ModelTrainer>(*model_, device_, this);
        
        // Connect signals
        connect(trainer_.get(), &ModelTrainer::epochCompleted,
                this, &MainWindow::onEpochCompleted);
        connect(trainer_.get(), &ModelTrainer::trainingCompleted,
                this, &MainWindow::onTrainingCompleted);
        
        // Create datasets
        TimeSeriesDataset trainDataset(processedData_.XTrain, processedData_.yTrain);
        TimeSeriesDataset testDataset(processedData_.XTest, processedData_.yTest);
        
        // Train model
        auto history = trainer_->train(
            trainDataset,
            testDataset,
            numEpochsSpin_->value(),
            batchSizeSpin_->value(),
            learningRateSpin_->value(),
            weightDecaySpin_->value(),
            patienceSpin_->value()
        );
        
        // Create visualizations
        createResultsVisualization();
        
    } catch (const std::exception& e) {
        QMessageBox::critical(this, "Error", QString("Training failed: %1")
                            .arg(e.what()));
        logMessage("ERROR: " + QString(e.what()));
        trainButton_->setEnabled(true);
    }
}

void MainWindow::onEpochCompleted(int epoch, double trainLoss, double valLoss,
                                 double trainR2, double valR2) {
    progressBar_->setValue(epoch);
    
    if (epoch % 20 == 0) {
        logMessage(QString("Epoch %1: Train R²=%2, Val R²=%3")
                  .arg(epoch)
                  .arg(trainR2, 0, 'f', 4)
                  .arg(valR2, 0, 'f', 4));
    }
}

void MainWindow::onTrainingCompleted(bool success, QString message) {
    logMessage(message);
    trainButton_->setEnabled(true);
    
    if (success) {
        modelTrained_ = true;
        QMessageBox::information(this, "Success", "Training completed successfully!");
    }
}

void MainWindow::createResultsVisualization() {
    // Clear existing tabs
    resultsTabWidget_->clear();
    
    // Make predictions on all data
    auto predictions = trainer_->predict(processedData_.XAll);
    
    // Denormalize
    auto actualOrig = DataPreprocessor::denormalize(
        processedData_.yAll, processedData_.meanY, processedData_.stdY);
    auto predOrig = DataPreprocessor::denormalize(
        predictions, processedData_.meanY, processedData_.stdY);
    
    // Create time indices
    std::vector<int> timeIndices;
    for (int i = 0; i < processedData_.yAll.size(0); ++i) {
        timeIndices.push_back(seqLengthSpin_->value() + i);
    }
    
    // Calculate R² scores
    auto trainIndices = std::vector<int64_t>();
    auto testIndices = std::vector<int64_t>();
    for (size_t i = 0; i < processedData_.trainTestMask.size(); ++i) {
        if (processedData_.trainTestMask[i] == 0) {
            trainIndices.push_back(i);
        } else {
            testIndices.push_back(i);
        }
    }
    
    auto trainIdxTensor = torch::tensor(trainIndices);
    auto testIdxTensor = torch::tensor(testIndices);
    
    double trainR2 = ModelTrainer::calculateR2(
        actualOrig.index_select(0, trainIdxTensor),
        predOrig.index_select(0, trainIdxTensor)
    );
    
    double testR2 = ModelTrainer::calculateR2(
        actualOrig.index_select(0, testIdxTensor),
        predOrig.index_select(0, testIdxTensor)
    );
    
    logMessage(QString("Training R²: %1").arg(trainR2, 0, 'f', 6));
    logMessage(QString("Test R²: %1").arg(testR2, 0, 'f', 6));
    
    // Create plots
    auto trainPlot = plotter_->plotTimeSeries(timeIndices, actualOrig, predOrig,
                                             "Training Data", processedData_.trainTestMask, 0);
    auto testPlot = plotter_->plotTimeSeries(timeIndices, actualOrig, predOrig,
                                            "Test Data", processedData_.trainTestMask, 1);
    
    auto trainScatter = plotter_->plotScatter(
        actualOrig.index_select(0, trainIdxTensor),
        predOrig.index_select(0, trainIdxTensor),
        "Training Scatter", trainR2);
    
    auto testScatter = plotter_->plotScatter(
        actualOrig.index_select(0, testIdxTensor),
        predOrig.index_select(0, testIdxTensor),
        "Test Scatter", testR2);
    
    // Add to tabs
    resultsTabWidget_->addTab(trainPlot, "Training Time Series");
    resultsTabWidget_->addTab(testPlot, "Test Time Series");
    resultsTabWidget_->addTab(trainScatter, "Training Scatter");
    resultsTabWidget_->addTab(testScatter, "Test Scatter");
}

void MainWindow::logMessage(const QString& message) {
    QString timestamp = QDateTime::currentDateTime().toString("hh:mm:ss");
    consoleOutput_->append(QString("[%1] %2").arg(timestamp).arg(message));
}

} // namespace lstm_predictor
