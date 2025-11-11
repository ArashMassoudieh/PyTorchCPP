/**
 * @file main.cpp
 * @brief Entry point for LSTM Time Series Predictor application
 * 
 * This application provides a Qt-based GUI for training and visualizing
 * LSTM neural networks for time series prediction using LibTorch (PyTorch C++).
 * 
 * @author Generated for Time Series Prediction
 * @date 2025
 */

#include "MainWindow.h"
#include <QApplication>
#include <QStyleFactory>
#include <QDebug>
#include <torch/torch.h>

int main(int argc, char *argv[]) {
    // Initialize Qt Application
    QApplication app(argc, argv);
    
    // Set application metadata
    QApplication::setApplicationName("LSTM Time Series Predictor");
    QApplication::setApplicationVersion("1.0.0");
    QApplication::setOrganizationName("LSTM Predictor");
    
    // Set a modern style
    QApplication::setStyle(QStyleFactory::create("Fusion"));
    
    // Print LibTorch information
    qDebug() << "==============================================";
    qDebug() << "LSTM Time Series Predictor";
    qDebug() << "==============================================";
    qDebug() << "LibTorch version:" << TORCH_VERSION;
    qDebug() << "CUDA available:" << (torch::cuda::is_available() ? "Yes" : "No");
    
    if (torch::cuda::is_available()) {
        qDebug() << "CUDA devices:" << torch::cuda::device_count();
        for (int i = 0; i < torch::cuda::device_count(); ++i) {
            auto props = torch::cuda::getDeviceProperties(i);
            qDebug() << "  Device" << i << ":" << props.name;
        }
    }
    qDebug() << "==============================================\n";
    
    // Create and show main window
    lstm_predictor::MainWindow mainWindow;
    mainWindow.show();
    
    // Run application event loop
    return app.exec();
}
