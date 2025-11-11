/**
 * @file ResultsPlotter.h
 * @brief Visualization of training results and predictions
 * 
 * Uses Qt Charts to create plots for:
 * - Training/validation loss curves
 * - Training/validation R² curves
 * - Time series predictions vs actual
 * - Scatter plots
 * 
 * @author Generated for Time Series Prediction
 * @date 2025
 */

#ifndef RESULTSPLOTTER_H
#define RESULTSPLOTTER_H

#include "ModelTrainer.h"
#include "DataPreprocessor.h"
#include <QWidget>
#include <QtCharts/QChartView>
#include <QtCharts/QLineSeries>
#include <QtCharts/QScatterSeries>
#include <torch/torch.h>
#include <vector>

namespace lstm_predictor {

/**
 * @class ResultsPlotter
 * @brief Creates visualizations of model results
 */
class ResultsPlotter : public QWidget {
    Q_OBJECT

public:
    /**
     * @brief Constructor
     * @param parent Qt parent widget
     */
    explicit ResultsPlotter(QWidget* parent = nullptr);

    /**
     * @brief Plot training history (loss and R²)
     * 
     * @param history Training history data
     * @return Chart view widget
     */
    QChartView* plotTrainingHistory(const TrainingHistory& history);

    /**
     * @brief Plot time series comparison (measured vs modeled)
     * 
     * @param timeIndices Time step indices
     * @param actual Actual values
     * @param predicted Predicted values
     * @param title Chart title
     * @param trainTest Mask indicating train (0) or test (1)
     * @param showOnlyType 0=train only, 1=test only, -1=both
     * @return Chart view widget
     */
    QChartView* plotTimeSeries(const std::vector<int>& timeIndices,
                              const torch::Tensor& actual,
                              const torch::Tensor& predicted,
                              const QString& title,
                              const std::vector<int>& trainTest,
                              int showOnlyType = -1);

    /**
     * @brief Plot scatter plot (actual vs predicted)
     * 
     * @param actual Actual values
     * @param predicted Predicted values
     * @param title Chart title
     * @param r2Score R² score to display
     * @return Chart view widget
     */
    QChartView* plotScatter(const torch::Tensor& actual,
                           const torch::Tensor& predicted,
                           const QString& title,
                           double r2Score);

    /**
     * @brief Save chart to PNG file
     * 
     * @param chartView Chart to save
     * @param filename Output filename
     */
    static void saveChartToPNG(QChartView* chartView, const QString& filename);

private:
    /**
     * @brief Convert torch tensor to vector of doubles
     * 
     * @param tensor Input tensor
     * @return Vector of double values
     */
    static std::vector<double> tensorToVector(const torch::Tensor& tensor);
};

} // namespace lstm_predictor

#endif // RESULTSPLOTTER_H
