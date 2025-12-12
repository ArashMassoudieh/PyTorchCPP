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

    // Helper methods
    void setupUI();
    void setupMenus();
    void connectSignals();
    void logMessage(const QString &message);
    void updateProgress(int value);

    GeneticAlgorithm<NeuralNetworkWrapper> ga_;
    bool gaRunning_;
    QAction* startGAAction_;
    QAction* stopGAAction_;


};

#endif // MAINWINDOW_H
