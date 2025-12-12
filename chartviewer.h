#ifndef CHARTVIEWER_H
#define CHARTVIEWER_H

#include <QWidget>
#include <QChart>
#include <QChartView>
#include <QLineSeries>
#include <QScatterSeries>
#include <QValueAxis>
#include <QLogValueAxis>
#include <QToolBar>
#include <QCheckBox>
#include <QVBoxLayout>
#include <QMenu>
#include <QMap>
#include "TimeSeriesSet.h"
#include <QAreaSeries>


    /**
 * @brief Chart viewer for displaying TimeSeriesSet data with interactive features
 *
 * Features:
 * - Display multiple time series
 * - Toggle between line and scatter plot modes
 * - Log/linear scale for X and Y axes
 * - Show/hide individual series via legend
 * - Export to PNG
 * - Copy/paste data between viewers
 * - Interactive legend
 */
    class ChartViewer : public QWidget
{
    Q_OBJECT

public:
    enum PlotMode {
        Lines,          ///< Line plot
        Symbols,        ///< Scatter plot (symbols only)
        LinesAndSymbols, ///< Both lines and symbols
        Filled          ///< Filled area plot
    };

    explicit ChartViewer(QWidget *parent = nullptr);
    ~ChartViewer();

    /**
     * @brief Set the data to display
     * @param data TimeSeriesSet to visualize
     */
    void setData(const TimeSeriesSet<double>& data);

    /**
     * @brief Get current data
     * @return Current TimeSeriesSet
     */
    TimeSeriesSet<double> getData() const { return timeSeriesData_; }

    /**
     * @brief Set chart title
     * @param title Chart title
     */
    void setTitle(const QString& title);

    /**
     * @brief Set axis labels
     * @param xLabel X-axis label
     * @param yLabel Y-axis label
     */
    void setAxisLabels(const QString& xLabel, const QString& yLabel);

    /**
     * @brief Set plot mode (lines, symbols, or both)
     * @param mode Plot mode
     */
    void setPlotMode(PlotMode mode);

    /**
     * @brief Toggle X-axis scale (linear/log)
     * @param useLog true for logarithmic scale
     */
    void setXAxisLog(bool useLog);

    /**
     * @brief Toggle Y-axis scale (linear/log)
     * @param useLog true for logarithmic scale
     */
    void setYAxisLog(bool useLog);

    /**
     * @brief Show/hide a specific series
     * @param seriesName Name of the series
     * @param visible Visibility state
     */
    void setSeriesVisible(const QString& seriesName, bool visible);

    /**
     * @brief Export chart to PNG file
     * @param filename Output file path
     */
    void exportToPng(const QString& filename);

    /**
     * @brief Export data to CSV file
     * @param filename Output file path
     */
    void exportToCsv(const QString& filename);

    /**
     * @brief Reset zoom to show all data (zoom extents)
     */
    void zoomExtents();

    /**
     * @brief Enable/disable zoom and pan interactions
     * @param enabled true to enable zoom/pan
     */
    void setZoomEnabled(bool enabled);

    /**
    * @brief Force Y-axis to always start at zero (for positive-only data)
    * @param startAtZero true to force Y-axis minimum to 0
    */
    void setYAxisStartAtZero(bool startAtZero);

    /**
     * @brief Force X-axis to always start at zero (for positive-only data)
     * @param startAtZero true to force X-axis minimum to 0
     */
    void setXAxisStartAtZero(bool startAtZero);

public Q_SLOTS:
    void onCopy();
    void onPaste();
    void onExportPng();
    void onExportCsv();
    void onToggleXLog();
    void onToggleYLog();
    void onTogglePlotMode();
    void onLegendMarkerClicked();

Q_SIGNALS:
    void seriesVisibilityChanged(const QString& seriesName, bool visible);

private:
    void setupUI();
    void setupToolbar();
    void updateChart();
    void createSeries();
    void setupAxes();
    void updateAxesRanges();
    void applySeriesColors();
    void connectLegendMarkers();

    // UI Components
    QChart* chart_;
    QChartView* chartView_;
    QToolBar* toolbar_;
    QVBoxLayout* mainLayout_;

    // Actions
    QAction* copyAction_;
    QAction* pasteAction_;
    QAction* exportPngAction_;
    QAction* exportCsvAction_;
    QAction* xLogAction_;
    QAction* yLogAction_;
    QAction* plotModeAction_;

    // Data
    TimeSeriesSet<double> timeSeriesData_;
    TimeSeriesSet<double> clipboard_; // For copy/paste

    // Series management
    QMap<QString, QLineSeries*> lineSeries_;
    QMap<QString, QScatterSeries*> scatterSeries_;
    QMap<QString, QAreaSeries*> areaSeries_;
    QMap<QString, bool> seriesVisibility_;

    // Axes
    QAbstractAxis* xAxis_;
    QAbstractAxis* yAxis_;
    bool xAxisLog_;
    bool yAxisLog_;

    // Settings
    PlotMode plotMode_;
    QString xAxisLabel_;
    QString yAxisLabel_;

    // Colors for series
    QList<QColor> colorPalette_;
    void initializeColorPalette();

    bool yAxisStartAtZero_;  // Force Y-axis to start at zero
    bool xAxisStartAtZero_;  // Force X-axis to start at zero
};

#endif // CHARTVIEWER_H
