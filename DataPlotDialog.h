#ifndef DATAPLOTDIALOG_H
#define DATAPLOTDIALOG_H

#include <QDialog>
#include <QListWidget>
#include <QCheckBox>
#include <QPushButton>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QLabel>
#include "TimeSeriesSet.h"
#include "TimeSeries.h"

/**
 * @brief Dialog for selecting which data series to plot
 */
class DataPlotDialog : public QDialog
{
    Q_OBJECT

public:
    explicit DataPlotDialog(const TimeSeriesSet<double>& inputData,
                            const TimeSeries<double>& targetData,
                            QWidget *parent = nullptr);

    /**
     * @brief Get selected input series indices
     */
    std::vector<int> getSelectedInputSeries() const;

    /**
     * @brief Check if target series should be plotted
     */
    bool shouldPlotTarget() const;

private Q_SLOTS:
    void onSelectAll();
    void onDeselectAll();
    void onAccept();

private:
    void setupUI();

    const TimeSeriesSet<double>& inputData_;
    const TimeSeries<double>& targetData_;

    QListWidget* seriesListWidget_;
    QCheckBox* targetCheckBox_;
    QPushButton* selectAllButton_;
    QPushButton* deselectAllButton_;
};

#endif // DATAPLOTDIALOG_H
