/**
 * @file MainWindow.h
 * @brief Main application window for LSTM time series predictor
 * 
 * Provides GUI interface for:
 * - Loading data
 * - Configuring model parameters
 * - Training the model
 * - Visualizing results
 * 
 * @author Generated for Time Series Prediction
 * @date 2025
 */

#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include "LSTMModel.h"
#include "DataPreprocessor.h"
#include "ModelTrainer.h"
#include "ResultsPlotter.h"
#include <QMainWindow>
#include <QPushButton>
#include <QLineEdit>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QTextEdit>
#include <QProgressBar>
#include <QTabWidget>
#include <memory>

namespace lstm_predictor {

/**
 * @class MainWindow
 * @brief Main application window
 */
class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    /**
     * @brief Constructor
     * @param parent Qt parent widget
     */
    explicit MainWindow(QWidget* parent = nullptr);
    
    /**
     * @brief Destructor
     */
    ~MainWindow();

private slots:
    /**
     * @brief Handle load data button click
     */
    void onLoadData();

    /**
     * @brief Handle train button click
     */
    void onTrain();

    /**
     * @brief Handle epoch completion during training
     */
    void onEpochCompleted(int epoch, double trainLoss, double valLoss,
                         double trainR2, double valR2);

    /**
     * @brief Handle training completion
     */
    void onTrainingCompleted(bool success, QString message);

private:
    /**
     * @brief Setup the user interface
     */
    void setupUI();

    /**
     * @brief Create parameter controls
     * @return Widget containing parameter controls
     */
    QWidget* createParameterControls();

    /**
     * @brief Create results visualization
     */
    void createResultsVisualization();

    /**
     * @brief Log message to console
     * @param message Message to log
     */
    void logMessage(const QString& message);

    // UI Components
    QLineEdit* filePathEdit_;
    QPushButton* loadDataButton_;
    QPushButton* trainButton_;
    QTextEdit* consoleOutput_;
    QProgressBar* progressBar_;
    QTabWidget* resultsTabWidget_;
    
    // Model parameters
    QSpinBox* seqLengthSpin_;
    QSpinBox* hiddenSizeSpin_;
    QSpinBox* numLayersSpin_;
    QSpinBox* batchSizeSpin_;
    QSpinBox* numEpochsSpin_;
    QDoubleSpinBox* learningRateSpin_;
    QDoubleSpinBox* dropoutSpin_;
    QDoubleSpinBox* weightDecaySpin_;
    QSpinBox* patienceSpin_;
    
    // Data and model
    std::unique_ptr<DataPreprocessor> preprocessor_;
    ProcessedData processedData_;
    std::shared_ptr<LSTMModel> model_;
    std::unique_ptr<ModelTrainer> trainer_;
    std::unique_ptr<ResultsPlotter> plotter_;
    
    // Torch device
    torch::Device device_;
    
    // State
    bool dataLoaded_;
    bool modelTrained_;
};

} // namespace lstm_predictor

#endif // MAINWINDOW_H
