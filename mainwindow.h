#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTextEdit>
#include <QProgressBar>
#include <QPushButton>
#include <QLabel>
#include <QStatusBar>
#include <QThread>

#include "neuralnetworkwrapper.h"
#include "neuralnetworkfactory.h"
#include "hyperparameters.h"
#include "ga.h"
#include "TimeSeriesSet.h"
#include "TimeSeries.h"
#include "commontypes.h"

// IMPORTANT: QT_NO_KEYWORDS is REQUIRED when using Qt with LibTorch
// LibTorch has methods named "slots()" which conflicts with Qt's 'slots' macro
// Use Q_SLOTS, Q_SIGNALS, Q_EMIT instead of slots, signals, emit

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE



/**
 * @brief Main window for Neural Network GA Optimizer
 */
class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private Q_SLOTS:
    // Add your menu action slots here later
    void onAbout();
    void onExit();
    void onConfigureGA();
    void onStartGA();
    void onStopGA();
    void onLoadData();
    void onPlotResults();
    void onGenerateSyntheticData();
    void onConfigureNetwork();
    void onConfigureIncrementalTraining();
    void onStartIncrementalTraining();
    void onSaveProject();
    void onSaveProjectAs();
    void onLoadProject();
    void onNewProject();


private:
    Ui::MainWindow *ui;

    // Central widget components
    QTextEdit *logOutput;
    QProgressBar *progressBar;
    QPushButton *startButton;
    QPushButton *stopButton;
    QLabel *statusLabel;

    // Data members
    TimeSeriesSet<double> inputData;
    TimeSeries<double> targetData;
    NeuralNetworkWrapper currentModel;

    bool saveProject(const QString& filepath);
    bool loadProject(const QString& filepath);
    void updateProjectFromUI();
    void updateUIFromProject();
    void setCurrentProjectPath(const QString& path);
    bool maybeSaveProject();  // Ask to save if modified
    void loadLastProject();
    void saveLastProjectPath(const QString& path);
    QString getLastProjectPath() const;

    // Helper methods
    void setupUI();
    void setupMenus();
    void connectSignals();
    void logMessage(const QString &message) const;
    void updateProgress(int value);
    /**
     * @brief Plot predictions vs target scatter plot
     * @param model Neural network model to use for predictions
     * @param data_type Which dataset to plot (Train or Test)
     */
    void plotPredictionsVsTarget(NeuralNetworkWrapper& model, DataType data_type);

    /**
     * @brief Plot predictions vs target over time
     * @param model Neural network model to use for predictions
     * @param data_type Which dataset to plot (Train or Test)
     */
    void plotPredictionsVsTime(NeuralNetworkWrapper& model, DataType data_type);

    NeuralNetworkWrapper* bestModel_ = nullptr;
    NeuralNetworkWrapper manualModel_;
    IncrementalTrainingParams incrementalParams_;
    ProjectConfig currentProject_;                  ///< Current project configuration
    QString currentProjectPath_;                    ///< Path to current project file
    GeneticAlgorithm<NeuralNetworkWrapper> ga_;
    bool gaRunning_;
    QAction* startGAAction_;
    QAction* stopGAAction_;


    /**
     * @brief Get start time for given data type
     * @param data_type Train or Test
     * @return Start time
     */
    double TimeStart(DataType data_type) const;

    /**
     * @brief Get end time for given data type
     * @param data_type Train or Test
     * @return End time
     */
    double TimeEnd(DataType data_type) const;

    /**
     * @brief Get time delta (sampling interval)
     * @return Time delta
     */
    double dt() const;

    double split_ratio = 0.7;

    /**
     * @brief Get the train/test split ratio
     * @return Split ratio (0.0 to 1.0)
     */
    double getSplitRatio() const { return split_ratio; }

    /**
     * @brief Set the train/test split ratio
     * @param ratio Split ratio (0.0 to 1.0), where ratio is the fraction for training
     */
    void setSplitRatio(double ratio);




};

#endif // MAINWINDOW_H
