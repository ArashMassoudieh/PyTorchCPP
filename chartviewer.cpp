#include "chartviewer.h"
#include <QFileDialog>
#include <QMessageBox>
#include <QClipboard>
#include <QApplication>
#include <QLegendMarker>
#include <QPixmap>
#include <QDebug>

ChartViewer::ChartViewer(QWidget* parent)
    : QWidget(parent)
    , chart_(new QChart())
    , chartView_(new QChartView(chart_, this))
    , xAxis_(nullptr)
    , yAxis_(nullptr)
    , xAxisLog_(false)
    , yAxisLog_(false)
    , plotMode_(Lines)
    , xAxisLabel_("Time")
    , yAxisLabel_("Value")
    , yAxisStartAtZero_(false)
    , xAxisStartAtZero_(false)
{
    setupUI();
    initializeColorPalette();
}

ChartViewer::~ChartViewer()
{
    // QChart takes ownership of series and axes, so they're automatically deleted
}

void ChartViewer::setupUI()
{
    mainLayout_ = new QVBoxLayout(this);
    mainLayout_->setContentsMargins(0, 0, 0, 0);

    // Setup toolbar
    setupToolbar();
    mainLayout_->addWidget(toolbar_);

    // Setup chart view
    chartView_->setRenderHint(QPainter::Antialiasing);
    chartView_->setRubberBand(QChartView::RectangleRubberBand);
    chart_->legend()->setVisible(true);
    chart_->legend()->setAlignment(Qt::AlignRight);

    // Legend marker clicks will be connected after series are created

    mainLayout_->addWidget(chartView_);

    setLayout(mainLayout_);
}

void ChartViewer::setupToolbar()
{
    toolbar_ = new QToolBar(this);
    toolbar_->setIconSize(QSize(24, 24));

    // Copy action
    copyAction_ = toolbar_->addAction(QIcon(":/icons/copy.png"), "");
    copyAction_->setToolTip("Copy chart data to clipboard");
    connect(copyAction_, &QAction::triggered, this, &ChartViewer::onCopy);

    // Paste action
    pasteAction_ = toolbar_->addAction(QIcon(":/icons/paste.png"), "");
    pasteAction_->setToolTip("Paste chart data from clipboard");
    connect(pasteAction_, &QAction::triggered, this, &ChartViewer::onPaste);

    toolbar_->addSeparator();

    // Zoom extents action
    QAction* zoomExtentsAction = toolbar_->addAction(QIcon(":/icons/zoom-extents.png"), "");
    zoomExtentsAction->setToolTip("Zoom Extents");
    zoomExtentsAction->setStatusTip("Reset zoom to show all data");
    connect(zoomExtentsAction, &QAction::triggered, this, &ChartViewer::zoomExtents);

    toolbar_->addSeparator();

    // Export PNG action
    exportPngAction_ = toolbar_->addAction(QIcon(":/icons/export-png.png"), "");
    exportPngAction_->setToolTip("Export chart to PNG image");
    connect(exportPngAction_, &QAction::triggered, this, &ChartViewer::onExportPng);

    // Export CSV action
    exportCsvAction_ = toolbar_->addAction(QIcon(":/icons/export-csv.png"), "");
    exportCsvAction_->setToolTip("Export data to CSV file");
    connect(exportCsvAction_, &QAction::triggered, this, &ChartViewer::onExportCsv);

    toolbar_->addSeparator();

    // X-axis log toggle
    xLogAction_ = toolbar_->addAction("X: Linear");
    xLogAction_->setToolTip("Toggle X-axis between linear and logarithmic scale");
    xLogAction_->setCheckable(true);
    connect(xLogAction_, &QAction::triggered, this, &ChartViewer::onToggleXLog);

    // Y-axis log toggle
    yLogAction_ = toolbar_->addAction("Y: Linear");
    yLogAction_->setToolTip("Toggle Y-axis between linear and logarithmic scale");
    yLogAction_->setCheckable(true);
    connect(yLogAction_, &QAction::triggered, this, &ChartViewer::onToggleYLog);

    toolbar_->addSeparator();

    // Plot mode toggle
    plotModeAction_ = toolbar_->addAction("Lines");
    plotModeAction_->setToolTip("Toggle between lines, symbols, or both");
    connect(plotModeAction_, &QAction::triggered, this, &ChartViewer::onTogglePlotMode);
}

void ChartViewer::initializeColorPalette()
{
    // Define a nice color palette for series
    colorPalette_ << QColor(31, 119, 180)   // Blue
        << QColor(255, 127, 14)   // Orange
        << QColor(44, 160, 44)    // Green
        << QColor(214, 39, 40)    // Red
        << QColor(148, 103, 189)  // Purple
        << QColor(140, 86, 75)    // Brown
        << QColor(227, 119, 194)  // Pink
        << QColor(127, 127, 127)  // Gray
        << QColor(188, 189, 34)   // Olive
        << QColor(23, 190, 207);  // Cyan
}

void ChartViewer::setData(const TimeSeriesSet<double>& data)
{
    timeSeriesData_ = data;
    updateChart();
}

void ChartViewer::setTitle(const QString& title)
{
    chart_->setTitle(title);
}

void ChartViewer::setAxisLabels(const QString& xLabel, const QString& yLabel)
{
    xAxisLabel_ = xLabel;
    yAxisLabel_ = yLabel;

    if (xAxis_) xAxis_->setTitleText(xLabel);
    if (yAxis_) yAxis_->setTitleText(yLabel);
}

void ChartViewer::setPlotMode(PlotMode mode)
{
    plotMode_ = mode;
    updateChart();

    // Update button text
    switch (plotMode_) {
    case Lines:
        plotModeAction_->setText("Lines");
        break;
    case Symbols:
        plotModeAction_->setText("Symbols");
        break;
    case LinesAndSymbols:
        plotModeAction_->setText("Lines+Symbols");
        break;
    }
}

void ChartViewer::updateChart()
{
    // Clear existing series
    chart_->removeAllSeries();
    lineSeries_.clear();
    scatterSeries_.clear();

    if (timeSeriesData_.empty()) {
        return;
    }

    // Create series
    createSeries();

    // Setup axes
    setupAxes();

    // Update ranges
    updateAxesRanges();

    // Apply colors
    applySeriesColors();
}

void ChartViewer::createSeries()
{
    // Clear existing series
    chart_->removeAllSeries();
    lineSeries_.clear();
    scatterSeries_.clear();
    areaSeries_.clear();  // Clear area series

    if (timeSeriesData_.size() == 0) {
        return;
    }

    for (int i = 0; i < timeSeriesData_.size(); ++i) {
        QString seriesName = QString::fromStdString(timeSeriesData_.getSeriesName(i));

        // Create appropriate series based on plot mode
        if (plotMode_ == Filled) {
            // Create line series for upper bound
            QLineSeries* upperSeries = new QLineSeries();
            upperSeries->setName(seriesName);

            // Create line series for lower bound (at zero or y-min)
            QLineSeries* lowerSeries = new QLineSeries();

            // Add data points
            for (const auto& point : timeSeriesData_[i]) {
                upperSeries->append(point.t, point.c);
                lowerSeries->append(point.t, 0.0);  // Fill to zero
            }

            // Create area series
            QAreaSeries* area = new QAreaSeries(upperSeries, lowerSeries);
            area->setName(seriesName);
            area->setOpacity(0.5);  // Semi-transparent fill

            chart_->addSeries(area);
            areaSeries_[seriesName] = area;

            // Set visibility
            area->setVisible(seriesVisibility_.value(seriesName, true));
        }
        else if (plotMode_ == Lines || plotMode_ == LinesAndSymbols) {
            // Existing line series code
            QLineSeries* series = new QLineSeries();
            series->setName(seriesName);

            for (const auto& point : timeSeriesData_[i]) {
                series->append(point.t, point.c);
            }

            chart_->addSeries(series);
            lineSeries_[seriesName] = series;
            series->setVisible(seriesVisibility_.value(seriesName, true));
        }

        if (plotMode_ == Symbols || plotMode_ == LinesAndSymbols) {
            // Existing scatter series code
            QScatterSeries* scatter = new QScatterSeries();
            scatter->setName(seriesName);
            scatter->setMarkerSize(8.0);

            for (const auto& point : timeSeriesData_[i]) {
                scatter->append(point.t, point.c);
            }

            chart_->addSeries(scatter);
            scatterSeries_[seriesName] = scatter;
            scatter->setVisible(seriesVisibility_.value(seriesName, true));
        }
    }

    setupAxes();
    applySeriesColors();
    connectLegendMarkers();
    updateAxesRanges();
}

void ChartViewer::setupAxes()
{
    // Remove old axes
    if (xAxis_) {
        chart_->removeAxis(xAxis_);
        delete xAxis_;
    }
    if (yAxis_) {
        chart_->removeAxis(yAxis_);
        delete yAxis_;
    }

    // Create X-axis
    if (xAxisLog_) {
        QLogValueAxis* logAxis = new QLogValueAxis();
        logAxis->setLabelFormat("%g");
        logAxis->setBase(10.0);
        xAxis_ = logAxis;
    }
    else {
        QValueAxis* valueAxis = new QValueAxis();
        valueAxis->setLabelFormat("%g");
        xAxis_ = valueAxis;
    }
    xAxis_->setTitleText(xAxisLabel_);

    // Create Y-axis
    if (yAxisLog_) {
        QLogValueAxis* logAxis = new QLogValueAxis();
        logAxis->setLabelFormat("%g");
        logAxis->setBase(10.0);
        yAxis_ = logAxis;
    }
    else {
        QValueAxis* valueAxis = new QValueAxis();
        valueAxis->setLabelFormat("%g");
        yAxis_ = valueAxis;
    }
    yAxis_->setTitleText(yAxisLabel_);

    // Add axes to chart
    chart_->addAxis(xAxis_, Qt::AlignBottom);
    chart_->addAxis(yAxis_, Qt::AlignLeft);

    // Attach series to axes
    for (QLineSeries* series : lineSeries_.values()) {
        series->attachAxis(xAxis_);
        series->attachAxis(yAxis_);
    }
    for (QScatterSeries* series : scatterSeries_.values()) {
        series->attachAxis(xAxis_);
        series->attachAxis(yAxis_);
    }
    for (QAreaSeries* series : areaSeries_.values()) {
        series->attachAxis(xAxis_);
        series->attachAxis(yAxis_);
    }
}

void ChartViewer::updateAxesRanges()
{
    if (timeSeriesData_.size() == 0 || !xAxis_ || !yAxis_) {
        return;
    }

    // Calculate ranges from data
    double xMin = std::numeric_limits<double>::max();
    double xMax = std::numeric_limits<double>::lowest();
    double yMin = std::numeric_limits<double>::max();
    double yMax = std::numeric_limits<double>::lowest();

    for (int i = 0; i < timeSeriesData_.size(); ++i) {
        const QString seriesName = QString::fromStdString(timeSeriesData_.getSeriesName(i));
        // Skip if series is hidden
        if (!seriesVisibility_.value(seriesName, true)) {
            continue;
        }
        for (const auto& point : timeSeriesData_[i]) {
            xMin = std::min(xMin, point.t);
            xMax = std::max(xMax, point.t);
            yMin = std::min(yMin, point.c);
            yMax = std::max(yMax, point.c);
        }
    }

    // Handle case where all values are the same
    if (xMin == xMax) {
        // Add padding around single value
        double padding = (std::abs(xMin) > 1e-10) ? std::abs(xMin) * 0.1 : 1.0;
        xMin -= padding;
        xMax += padding;
    }
    if (yMin == yMax) {
        // Add padding around single value
        double padding = (std::abs(yMin) > 1e-10) ? std::abs(yMin) * 0.1 : 1.0;
        yMin -= padding;
        yMax += padding;
    }

    // Add margin (5% on each side)
    double xMargin = (xMax - xMin) * 0.05;
    double yMargin = (yMax - yMin) * 0.05;

    // Handle X-axis scaling
    if (xAxisLog_) {
        xMin = std::max(xMin - xMargin, 1e-10);
    }
    else {
        xMin -= xMargin;
        // Force X-axis to start at zero if enabled
        if (xAxisStartAtZero_) {
            xMin = std::max(xMin, 0.0);
        }
    }
    xMax += xMargin;

    // Handle Y-axis scaling
    if (yAxisLog_) {
        yMin = std::max(yMin - yMargin, 1e-10);
    }
    else {
        yMin -= yMargin;
        // Force Y-axis to start at zero if enabled
        if (yAxisStartAtZero_) {
            yMin = std::max(yMin, 0.0);
        }
    }
    yMax += yMargin;

    // Set ranges
    xAxis_->setRange(xMin, xMax);
    yAxis_->setRange(yMin, yMax);
}


void ChartViewer::applySeriesColors()
{
    int colorIndex = 0;

    for (auto it = lineSeries_.begin(); it != lineSeries_.end(); ++it) {
        QColor color = colorPalette_[colorIndex % colorPalette_.size()];
        it.value()->setColor(color);

        // If scatter series exists with same name, use same color
        if (scatterSeries_.contains(it.key())) {
            scatterSeries_[it.key()]->setColor(color);
            scatterSeries_[it.key()]->setBorderColor(color.darker(120));
        }

        colorIndex++;
    }

    // Apply colors to area series
    colorIndex = 0;
    for (auto it = areaSeries_.begin(); it != areaSeries_.end(); ++it) {
        QColor color = colorPalette_[colorIndex % colorPalette_.size()];

        // Set pen color for border
        QPen pen = it.value()->pen();
        pen.setColor(color);
        pen.setWidth(2);
        it.value()->setPen(pen);

        // Set brush color for fill (with transparency)
        QColor fillColor = color;
        fillColor.setAlpha(128);  // 50% transparency
        it.value()->setBrush(fillColor);

        colorIndex++;
    }

    // Connect legend marker signals after series are added
    connectLegendMarkers();
}

void ChartViewer::connectLegendMarkers()
{
    // Connect click signal for each legend marker
    const auto markers = chart_->legend()->markers();
    for (QLegendMarker* marker : markers) {
        connect(marker, &QLegendMarker::clicked,
            this, &ChartViewer::onLegendMarkerClicked);
    }
}

void ChartViewer::setXAxisLog(bool useLog)
{
    xAxisLog_ = useLog;
    updateChart();
    xLogAction_->setChecked(useLog);
    xLogAction_->setText(useLog ? "X: Log" : "X: Linear");
}

void ChartViewer::setYAxisLog(bool useLog)
{
    yAxisLog_ = useLog;
    updateChart();
    yLogAction_->setChecked(useLog);
    yLogAction_->setText(useLog ? "Y: Log" : "Y: Linear");
}

void ChartViewer::setSeriesVisible(const QString& seriesName, bool visible)
{
    seriesVisibility_[seriesName] = visible;

    // Update line series visibility
    if (lineSeries_.contains(seriesName)) {
        lineSeries_[seriesName]->setVisible(visible);
    }

    // Update scatter series visibility
    if (scatterSeries_.contains(seriesName)) {
        scatterSeries_[seriesName]->setVisible(visible);
    }

    // Update area series visibility
    if (areaSeries_.contains(seriesName)) {
        areaSeries_[seriesName]->setVisible(visible);
    }

    Q_EMIT seriesVisibilityChanged(seriesName, visible);
}

void ChartViewer::exportToPng(const QString& filename)
{
    QPixmap pixmap = chartView_->grab();
    if (!pixmap.save(filename, "PNG")) {
        QMessageBox::warning(this, "Export Failed",
            "Failed to export chart to PNG file.");
    }
}

void ChartViewer::exportToCsv(const QString& filename)
{
    if (timeSeriesData_.empty()) {
        QMessageBox::warning(this, "Export Failed",
            "No data to export.");
        return;
    }

    // Use TimeSeriesSet's write method
    timeSeriesData_.write(filename.toStdString(), ",");

    QMessageBox::information(this, "Export Successful",
        QString("Exported %1 time series to CSV file:\n%2")
        .arg(timeSeriesData_.size())
        .arg(filename));
}

void ChartViewer::onCopy()
{
    clipboard_ = timeSeriesData_;
    QMessageBox::information(this, "Copy",
        QString("Copied %1 time series to clipboard").arg(clipboard_.size()));
}

void ChartViewer::onPaste()
{
    if (clipboard_.empty()) {
        QMessageBox::warning(this, "Paste", "Clipboard is empty");
        return;
    }

    setData(clipboard_);
    QMessageBox::information(this, "Paste",
        QString("Pasted %1 time series from clipboard").arg(clipboard_.size()));
}

void ChartViewer::onExportPng()
{
    QString filename = QFileDialog::getSaveFileName(
        this,
        tr("Export Chart to PNG"),
        QString(),
        tr("PNG Images (*.png);;All Files (*)")
    );

    if (!filename.isEmpty()) {
        exportToPng(filename);
    }
}

void ChartViewer::onExportCsv()
{
    QString filename = QFileDialog::getSaveFileName(
        this,
        tr("Export Data to CSV"),
        QString(),
        tr("CSV Files (*.csv);;Text Files (*.txt);;All Files (*)")
    );

    if (!filename.isEmpty()) {
        exportToCsv(filename);
    }
}

void ChartViewer::onToggleXLog()
{
    setXAxisLog(!xAxisLog_);
}

void ChartViewer::onToggleYLog()
{
    setYAxisLog(!yAxisLog_);
}

void ChartViewer::onTogglePlotMode()
{
    // Cycle through modes
    switch (plotMode_) {
    case Lines:
        setPlotMode(Symbols);
        plotModeAction_->setText("Symbols");
        break;
    case Symbols:
        setPlotMode(LinesAndSymbols);
        plotModeAction_->setText("Lines+Symbols");
        break;
    case LinesAndSymbols:
        setPlotMode(Filled);
        plotModeAction_->setText("Filled");
        break;
    case Filled:
        setPlotMode(Lines);
        plotModeAction_->setText("Lines");
        break;
    }
}

void ChartViewer::onLegendMarkerClicked()
{
    QLegendMarker* marker = qobject_cast<QLegendMarker*>(sender());
    if (!marker) return;

    QString seriesName = marker->series()->name();

    // Toggle visibility
    bool currentlyVisible = marker->series()->isVisible();
    bool newVisible = !currentlyVisible;

    marker->series()->setVisible(newVisible);
    seriesVisibility_[seriesName] = newVisible;

    // Update marker appearance (dim when hidden)
    if (newVisible) {
        marker->setVisible(true);
        QFont font = marker->font();
        font.setStrikeOut(false);
        marker->setFont(font);
    }
    else {
        QFont font = marker->font();
        font.setStrikeOut(true);
        marker->setFont(font);
    }

    Q_EMIT seriesVisibilityChanged(seriesName, newVisible);
}

void ChartViewer::zoomExtents()
{
    if (!chart_) return;

    // Reset axes to show all data
    chart_->zoomReset();

    // Recalculate and set proper ranges
    updateAxesRanges();
}

void ChartViewer::setZoomEnabled(bool enabled)
{
    if (!chartView_) return;

    if (enabled) {
        chartView_->setRubberBand(QChartView::RectangleRubberBand);
    }
    else {
        chartView_->setRubberBand(QChartView::NoRubberBand);
    }
}

void ChartViewer::setYAxisStartAtZero(bool startAtZero)
{
    yAxisStartAtZero_ = startAtZero;
    updateAxesRanges();
}

void ChartViewer::setXAxisStartAtZero(bool startAtZero)
{
    xAxisStartAtZero_ = startAtZero;
    updateAxesRanges();  // Recalculate axis ranges with new constraint
}
