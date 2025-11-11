/**
 * @file ResultsPlotter.cpp
 * @brief Implementation of visualization components
 */

#include "ResultsPlotter.h"
#include <QtCharts/QChart>
#include <QtCharts/QValueAxis>
#include <QVBoxLayout>
#include <QPainter>

using namespace QtCharts;

namespace lstm_predictor {

ResultsPlotter::ResultsPlotter(QWidget* parent)
    : QWidget(parent) {
}

QChartView* ResultsPlotter::plotTrainingHistory(const TrainingHistory& history) {
    QChart* chart = new QChart();
    chart->setTitle("Training History");
    
    // Create series for losses
    QLineSeries* trainLossSeries = new QLineSeries();
    trainLossSeries->setName("Training Loss");
    
    QLineSeries* valLossSeries = new QLineSeries();
    valLossSeries->setName("Validation Loss");
    
    // Add data points
    for (size_t i = 0; i < history.trainLosses.size(); ++i) {
        trainLossSeries->append(i + 1, history.trainLosses[i]);
        valLossSeries->append(i + 1, history.valLosses[i]);
    }
    
    chart->addSeries(trainLossSeries);
    chart->addSeries(valLossSeries);
    
    // Create axes
    chart->createDefaultAxes();
    chart->axes(Qt::Horizontal).first()->setTitleText("Epoch");
    chart->axes(Qt::Vertical).first()->setTitleText("Loss");
    
    // Create chart view
    QChartView* chartView = new QChartView(chart);
    chartView->setRenderHint(QPainter::Antialiasing);
    
    return chartView;
}

QChartView* ResultsPlotter::plotTimeSeries(const std::vector<int>& timeIndices,
                                          const torch::Tensor& actual,
                                          const torch::Tensor& predicted,
                                          const QString& title,
                                          const std::vector<int>& trainTest,
                                          int showOnlyType) {
    QChart* chart = new QChart();
    chart->setTitle(title);
    
    // Convert tensors to vectors
    auto actualVec = tensorToVector(actual);
    auto predVec = tensorToVector(predicted);
    
    // Create series
    QLineSeries* actualSeries = new QLineSeries();
    actualSeries->setName("Measured");
    
    QLineSeries* predSeries = new QLineSeries();
    predSeries->setName("Modeled");
    
    // Add data points based on train/test filter
    for (size_t i = 0; i < timeIndices.size() && i < actualVec.size(); ++i) {
        if (showOnlyType == -1 || trainTest[i] == showOnlyType) {
            actualSeries->append(timeIndices[i], actualVec[i]);
            predSeries->append(timeIndices[i], predVec[i]);
        }
    }
    
    chart->addSeries(actualSeries);
    chart->addSeries(predSeries);
    
    // Create axes
    chart->createDefaultAxes();
    chart->axes(Qt::Horizontal).first()->setTitleText("Time Step");
    chart->axes(Qt::Vertical).first()->setTitleText("Concentration");
    
    QChartView* chartView = new QChartView(chart);
    chartView->setRenderHint(QPainter::Antialiasing);
    
    return chartView;
}

QChartView* ResultsPlotter::plotScatter(const torch::Tensor& actual,
                                       const torch::Tensor& predicted,
                                       const QString& title,
                                       double r2Score) {
    QChart* chart = new QChart();
    chart->setTitle(title + QString(" (R² = %1)").arg(r2Score, 0, 'f', 4));
    
    // Convert tensors to vectors
    auto actualVec = tensorToVector(actual);
    auto predVec = tensorToVector(predicted);
    
    // Create scatter series
    QScatterSeries* scatterSeries = new QScatterSeries();
    scatterSeries->setName("Predictions");
    scatterSeries->setMarkerSize(8.0);
    
    for (size_t i = 0; i < actualVec.size() && i < predVec.size(); ++i) {
        scatterSeries->append(actualVec[i], predVec[i]);
    }
    
    chart->addSeries(scatterSeries);
    
    // Add perfect prediction line
    if (!actualVec.empty()) {
        double minVal = *std::min_element(actualVec.begin(), actualVec.end());
        double maxVal = *std::max_element(actualVec.begin(), actualVec.end());
        
        QLineSeries* perfectLine = new QLineSeries();
        perfectLine->setName("Perfect Prediction");
        perfectLine->append(minVal, minVal);
        perfectLine->append(maxVal, maxVal);
        
        chart->addSeries(perfectLine);
    }
    
    // Create axes
    chart->createDefaultAxes();
    chart->axes(Qt::Horizontal).first()->setTitleText("Measured");
    chart->axes(Qt::Vertical).first()->setTitleText("Predicted");
    
    QChartView* chartView = new QChartView(chart);
    chartView->setRenderHint(QPainter::Antialiasing);
    
    return chartView;
}

void ResultsPlotter::saveChartToPNG(QChartView* chartView, const QString& filename) {
    QPixmap pixmap = chartView->grab();
    pixmap.save(filename, "PNG");
}

std::vector<double> ResultsPlotter::tensorToVector(const torch::Tensor& tensor) {
    auto tensorFlat = tensor.flatten().cpu();
    std::vector<double> vec;
    vec.reserve(tensorFlat.size(0));
    
    for (int64_t i = 0; i < tensorFlat.size(0); ++i) {
        vec.push_back(tensorFlat[i].item<double>());
    }
    
    return vec;
}

} // namespace lstm_predictor
