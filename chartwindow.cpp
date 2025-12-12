#include "chartwindow.h"
#include <QHBoxLayout>

ChartWindow::ChartWindow(QWidget *parent)
    : QDialog(parent)
    , chartViewer_(new ChartViewer(this))
{
    setupUI();
    resize(1000, 600);
}

ChartWindow::ChartWindow(const TimeSeriesSet<double>& data, QWidget *parent)
    : ChartWindow(parent)
{
    setData(data);
}

ChartWindow::~ChartWindow()
{
    // ChartViewer is deleted automatically as a child widget
}

void ChartWindow::setupUI()
{
    mainLayout_ = new QVBoxLayout(this);

    // Add chart viewer
    mainLayout_->addWidget(chartViewer_);

    // Add close button at bottom
    QHBoxLayout* buttonLayout = new QHBoxLayout();
    buttonLayout->addStretch();

    closeButton_ = new QPushButton("Close", this);
    closeButton_->setDefault(true);
    connect(closeButton_, &QPushButton::clicked, this, &QDialog::accept);

    buttonLayout->addWidget(closeButton_);
    mainLayout_->addLayout(buttonLayout);

    setLayout(mainLayout_);

    // Set window properties
    setWindowTitle("Chart Viewer");
    setAttribute(Qt::WA_DeleteOnClose, false); // Don't auto-delete when closed
}

void ChartWindow::setData(const TimeSeriesSet<double>& data)
{
    chartViewer_->setData(data);
}

void ChartWindow::setWindowTitle(const QString& title)
{
    QDialog::setWindowTitle(title);
}

void ChartWindow::setChartTitle(const QString& title)
{
    chartViewer_->setTitle(title);
}

void ChartWindow::setAxisLabels(const QString& xLabel, const QString& yLabel)
{
    chartViewer_->setAxisLabels(xLabel, yLabel);
}

ChartWindow* ChartWindow::showChart(const TimeSeriesSet<double>& data,
                                    const QString& title,
                                    QWidget* parent)
{
    ChartWindow* window = new ChartWindow(data, parent);

    if (!title.isEmpty()) {
        window->setWindowTitle(title);
        window->setChartTitle(title);
    }

    // Set default axis labels
    window->setAxisLabels("Time", "Value");

    // Show as a non-modal dialog (allows interaction with main window)
    window->show();
    window->raise();
    window->activateWindow();

    return window;
}

