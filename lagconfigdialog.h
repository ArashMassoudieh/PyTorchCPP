#ifndef LAGCONFIGDIALOG_H
#define LAGCONFIGDIALOG_H

#include <QDialog>
#include <QSpinBox>
#include <QVBoxLayout>
#include <QFormLayout>
#include <QDialogButtonBox>
#include <QLabel>
#include <QPushButton>
#include <QScrollArea>
#include "TimeSeriesSet.h"
#include "commontypes.h"
#include <QGroupBox>

/**
 * @brief Dialog for configuring time lags for each input series
 */
class LagConfigDialog : public QDialog
{
    Q_OBJECT

public:
    explicit LagConfigDialog(NetworkArchitecture& architecture,
                             const TimeSeriesSet<double>& inputData,
                             QWidget *parent = nullptr);

private Q_SLOTS:
    void onAccept();
    void onUseDefaults();
    void onApplyToAll();

private:
    void setupUI();
    void loadCurrentLags();

    NetworkArchitecture& architecture_;
    const TimeSeriesSet<double>& inputData_;

    struct SeriesLagConfig {
        QGroupBox* groupBox;
        QSpinBox* minLagSpin;
        QSpinBox* maxLagSpin;
        QSpinBox* stepSpin;
        QLabel* resultLabel;
    };

    std::vector<SeriesLagConfig> seriesConfigs_;

    void updateResultLabel(int index);
};

#endif // LAGCONFIGDIALOG_H
