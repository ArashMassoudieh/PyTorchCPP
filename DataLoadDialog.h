#ifndef DATALOADDIALOG_H
#define DATALOADDIALOG_H

#include <QDialog>
#include <QLineEdit>
#include <QPushButton>
#include <QLabel>
#include <QCheckBox>
#include "TimeSeriesSet.h"
#include "TimeSeries.h"

class DataLoadDialog : public QDialog
{
    Q_OBJECT

public:
    explicit DataLoadDialog(QWidget* parent = nullptr);
    ~DataLoadDialog();

    // Getters for loaded data
    TimeSeriesSet<double> getInputData() const { return inputData_; }
    TimeSeries<double> getTargetData() const { return targetData_; }

    // Check if data was successfully loaded
    bool isDataValid() const { return dataValid_; }

private Q_SLOTS:
    void onBrowseInput();
    void onBrowseTarget();
    void onAccept();
    void onReject();
    void onPreviewData();

private:
    void setupUI();
    bool validateAndLoadData();
    bool loadInputData();
    bool loadTargetData();
    bool validateDataSizes();

    // UI Components
    QLineEdit* inputFileEdit_;
    QLineEdit* targetFileEdit_;
    QPushButton* browseInputBtn_;
    QPushButton* browseTargetBtn_;
    QPushButton* previewBtn_;
    QPushButton* okButton_;
    QPushButton* cancelButton_;
    QCheckBox* inputHasHeaderCheckBox_;
    QLabel* statusLabel_;

    // Data
    TimeSeriesSet<double> inputData_;
    TimeSeries<double> targetData_;
    bool dataValid_;
};

#endif // DATALOADDIALOG_H
