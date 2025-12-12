#ifndef CHARTWINDOW_H
#define CHARTWINDOW_H

#include <QDialog>
#include <QPushButton>
#include <QVBoxLayout>
#include "chartviewer.h"
#include "TimeSeriesSet.h"

/**
 * @brief Pop-up window for displaying charts
 *
 * A dialog window that contains a ChartViewer widget, allowing
 * charts to be displayed in separate, independent windows.
 */
class ChartWindow : public QDialog
{
    Q_OBJECT

public:
    explicit ChartWindow(QWidget *parent = nullptr);
    explicit ChartWindow(const TimeSeriesSet<double>& data, QWidget *parent = nullptr);
    ~ChartWindow();

    /**
     * @brief Set the data to display
     * @param data TimeSeriesSet to visualize
     */
    void setData(const TimeSeriesSet<double>& data);

    /**
     * @brief Get the chart viewer widget
     * @return Pointer to ChartViewer
     */
    ChartViewer* chartViewer() { return chartViewer_; }

    /**
     * @brief Set window title
     * @param title Window title
     */
    void setWindowTitle(const QString& title);

    /**
     * @brief Set chart title
     * @param title Chart title
     */
    void setChartTitle(const QString& title);

    /**
     * @brief Set axis labels
     * @param xLabel X-axis label
     * @param yLabel Y-axis label
     */
    void setAxisLabels(const QString& xLabel, const QString& yLabel);

    /**
     * @brief Convenience method to show data in a new window
     * @param data TimeSeriesSet to display
     * @param title Window title
     * @param parent Parent widget
     * @return Pointer to the created window
     */
    static ChartWindow* showChart(const TimeSeriesSet<double>& data,
                                  const QString& title = QString(),
                                  QWidget* parent = nullptr);

private:
    void setupUI();

    ChartViewer* chartViewer_;
    QVBoxLayout* mainLayout_;
    QPushButton* closeButton_;
};

#endif // CHARTWINDOW_H
