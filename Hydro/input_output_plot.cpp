#include "hydropinnwindow.h"

#include "dataset/csv_tensor_builder.h"
#include "dataset/gistohq_package_adapter.h"
#include "dataset/gistohq_tensor_builder.h"
#include "dataset/hydro_package_directory.h"

#include <QAbstractAxis>
#include <QChart>
#include <QChartView>
#include <QComboBox>
#include <QFileInfo>
#include <QLineSeries>
#include <QMessageBox>
#include <QValueAxis>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
QString csvColumnName(const std::vector<std::string>& header, int column) {
    if (column >= 0 && column < static_cast<int>(header.size()) && !header[static_cast<size_t>(column)].empty()) {
        return QString::fromStdString(header[static_cast<size_t>(column)]);
    }
    return QString("column_%1").arg(column);
}

std::vector<std::string> readCsvHeader(const HydroRunConfig& cfg) {
    if (!cfg.csv_has_header || cfg.csv_path.empty()) return {};
    std::ifstream in(cfg.csv_path);
    if (!in) throw std::runtime_error("Unable to open CSV file: " + cfg.csv_path);
    std::string line;
    if (!std::getline(in, line)) return {};
    return parseHydroCsvRow(line);
}

void resetChart(QChart* chart) {
    chart->removeAllSeries();
    const auto axes = chart->axes();
    for (QAbstractAxis* axis : axes) {
        chart->removeAxis(axis);
        delete axis;
    }
}
}

void HydroPINNWindow::showCurrentInputsOutputs() {
    const QString source = dataSourceCombo_->currentText();
    if (source == "Synthetic") {
        showSyntheticInputsOutputs();
        return;
    }

    try {
        std::vector<double> x;
        std::vector<QString> names;
        std::vector<std::vector<double>> inputs;
        std::vector<double> target;
        QString targetName = "Target";

        HydroRunConfig cfg = currentConfig();
        if (source == "CSV File") {
            torch::Tensor inputTensor;
            torch::Tensor targetTensor;
            torch::Tensor plotX;
            loadHydroCsvTensors(cfg, inputTensor, targetTensor, plotX);
            inputTensor = inputTensor.to(torch::kCPU).to(torch::kDouble).contiguous();
            targetTensor = targetTensor.to(torch::kCPU).to(torch::kDouble).contiguous();
            plotX = plotX.to(torch::kCPU).to(torch::kDouble).contiguous();

            const auto header = readCsvHeader(cfg);
            targetName = csvColumnName(header, cfg.csv_y_column);
            const int64_t rows = inputTensor.size(0);
            const int64_t columns = inputTensor.size(1);
            x.reserve(static_cast<size_t>(rows));
            target.reserve(static_cast<size_t>(rows));
            inputs.assign(static_cast<size_t>(columns), {});
            for (auto& values : inputs) values.reserve(static_cast<size_t>(rows));

            if (cfg.synthetic_profile == "neuroforge_inputs_target") {
                for (int column = 0; column < static_cast<int>(header.size()); ++column) {
                    if (column != cfg.csv_y_column) names.push_back(csvColumnName(header, column));
                }
                while (names.size() < static_cast<size_t>(columns)) {
                    names.push_back(QString("input_%1").arg(names.size()));
                }
            } else {
                names.push_back(csvColumnName(header, cfg.csv_x_column));
            }

            for (int64_t row = 0; row < rows; ++row) {
                x.push_back(plotX[row][0].item<double>());
                target.push_back(targetTensor[row][0].item<double>());
                for (int64_t column = 0; column < columns; ++column) {
                    inputs[static_cast<size_t>(column)].push_back(inputTensor[row][column].item<double>());
                }
            }
        } else if (source == "Hydro Package") {
            const auto packageRoot = resolveHydroPackageDirectory(cfg.hydro_package_path);
            if (!isGisToOhqHydroPinnExport(packageRoot)) {
                throw std::runtime_error(
                    "Inputs + Output currently supports GIStoOHQ HydroPINNExport packages. "
                    "Generic Hydro package plotting can be added when its feature-name contract is finalized.");
            }
            const auto prepared = prepareGisToOhqPackage(packageRoot, true);
            const auto table = buildGisToOhqTensorTable(prepared.model_rows);
            auto features = table.features.to(torch::kCPU).to(torch::kDouble).contiguous();
            auto targets = table.targets.to(torch::kCPU).to(torch::kDouble).contiguous();
            auto elapsed = table.elapsed_hours.to(torch::kCPU).to(torch::kDouble).contiguous();

            names = {"Precipitation", "Temperature", "Relative humidity", "Wind speed", "Solar radiation", "PET"};
            targetName = "Observed runoff";
            const int64_t rows = features.size(0);
            x.reserve(static_cast<size_t>(rows));
            target.reserve(static_cast<size_t>(rows));
            inputs.assign(6, {});
            for (auto& values : inputs) values.reserve(static_cast<size_t>(rows));
            for (int64_t row = 0; row < rows; ++row) {
                x.push_back(elapsed[row][0].item<double>());
                target.push_back(targets[row][0].item<double>());
                for (int column = 0; column < 6; ++column) {
                    inputs[static_cast<size_t>(column)].push_back(features[row][column].item<double>());
                }
            }
        } else {
            throw std::runtime_error("Unknown data source for Inputs + Output plot.");
        }

        if (x.empty() || target.empty() || inputs.empty()) {
            throw std::runtime_error("No input/output samples are available to plot.");
        }

        QChart* chart = chartView_->chart();
        resetChart(chart);

        auto* axisX = new QValueAxis(chart);
        axisX->setTitleText(source == "Hydro Package" ? "Elapsed time (h)" : "Configured x / time");
        auto* axisInputs = new QValueAxis(chart);
        axisInputs->setTitleText("Input values");
        auto* axisTarget = new QValueAxis(chart);
        axisTarget->setTitleText(targetName);

        chart->addAxis(axisX, Qt::AlignBottom);
        chart->addAxis(axisInputs, Qt::AlignLeft);
        chart->addAxis(axisTarget, Qt::AlignRight);

        double inputMin = std::numeric_limits<double>::infinity();
        double inputMax = -std::numeric_limits<double>::infinity();
        for (size_t column = 0; column < inputs.size(); ++column) {
            auto* series = new QLineSeries(chart);
            series->setName(column < names.size() ? names[column] : QString("Input %1").arg(column));
            const size_t count = std::min(x.size(), inputs[column].size());
            for (size_t row = 0; row < count; ++row) {
                const double value = inputs[column][row];
                if (!std::isfinite(value)) continue;
                series->append(x[row], value);
                inputMin = std::min(inputMin, value);
                inputMax = std::max(inputMax, value);
            }
            chart->addSeries(series);
            series->attachAxis(axisX);
            series->attachAxis(axisInputs);
        }

        auto* targetSeries = new QLineSeries(chart);
        targetSeries->setName(targetName + " (output)");
        double targetMin = std::numeric_limits<double>::infinity();
        double targetMax = -std::numeric_limits<double>::infinity();
        const size_t targetCount = std::min(x.size(), target.size());
        for (size_t row = 0; row < targetCount; ++row) {
            if (!std::isfinite(target[row])) continue;
            targetSeries->append(x[row], target[row]);
            targetMin = std::min(targetMin, target[row]);
            targetMax = std::max(targetMax, target[row]);
        }
        chart->addSeries(targetSeries);
        targetSeries->attachAxis(axisX);
        targetSeries->attachAxis(axisTarget);

        if (std::isfinite(inputMin) && std::isfinite(inputMax)) {
            const double span = std::max(1.0e-12, inputMax - inputMin);
            axisInputs->setRange(inputMin - 0.03 * span, inputMax + 0.03 * span);
        }
        if (std::isfinite(targetMin) && std::isfinite(targetMax)) {
            const double span = std::max(1.0e-12, targetMax - targetMin);
            axisTarget->setRange(targetMin - 0.05 * span, targetMax + 0.05 * span);
        }
        const auto [minX, maxX] = std::minmax_element(x.begin(), x.end());
        if (minX != x.end() && maxX != x.end()) {
            const double span = std::max(1.0e-12, *maxX - *minX);
            axisX->setRange(*minX - 0.01 * span, *maxX + 0.01 * span);
        }

        chart->setTitle(QString("Inputs + Output - %1").arg(source));
        chart->legend()->setVisible(true);
        appendLog(QString("Displayed %1 input column(s) plus output for %2 (%3 samples).")
                      .arg(inputs.size())
                      .arg(source)
                      .arg(x.size()));
    } catch (const std::exception& error) {
        appendLog(QString("Inputs + Output plot failed: %1").arg(error.what()));
        QMessageBox::warning(this, "HydroPINN Plot", QString::fromUtf8(error.what()));
    }
}
