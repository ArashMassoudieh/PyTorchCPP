#include "DataPlotDialog.h"
#include <QMessageBox>
#include <QFormLayout>

DataPlotDialog::DataPlotDialog(const TimeSeriesSet<double>& inputData,
                               const TimeSeries<double>& targetData,
                               QWidget *parent)
    : QDialog(parent)
    , inputData_(inputData)
    , targetData_(targetData)
{
    setWindowTitle("Select Data Series to Plot");
    setModal(true);
    setupUI();
    resize(450, 500);
}

void DataPlotDialog::setupUI()
{
    QVBoxLayout* mainLayout = new QVBoxLayout(this);

    // Info label
    QLabel* infoLabel = new QLabel(
        "Select which data series you want to visualize.\n"
        "You can select multiple input series and the target series.");
    infoLabel->setWordWrap(true);
    infoLabel->setStyleSheet("QLabel { padding: 10px; background-color: #e8f4f8; border-radius: 5px; }");
    mainLayout->addWidget(infoLabel);

    // Input series group
    QGroupBox* inputGroupBox = new QGroupBox("Input Series");
    QVBoxLayout* inputLayout = new QVBoxLayout(inputGroupBox);

    // Buttons for select all/none
    QHBoxLayout* buttonLayout = new QHBoxLayout();
    selectAllButton_ = new QPushButton("Select All");
    deselectAllButton_ = new QPushButton("Deselect All");
    connect(selectAllButton_, &QPushButton::clicked, this, &DataPlotDialog::onSelectAll);
    connect(deselectAllButton_, &QPushButton::clicked, this, &DataPlotDialog::onDeselectAll);
    buttonLayout->addWidget(selectAllButton_);
    buttonLayout->addWidget(deselectAllButton_);
    buttonLayout->addStretch();
    inputLayout->addLayout(buttonLayout);

    // List of input series
    seriesListWidget_ = new QListWidget();
    seriesListWidget_->setSelectionMode(QAbstractItemView::MultiSelection);

    for (int i = 0; i < inputData_.size(); i++) {
        QString itemText = QString("Series %1: %2 (%3 points)")
        .arg(i)
            .arg(QString::fromStdString(inputData_[i].name()))
            .arg(inputData_[i].size());

        QListWidgetItem* item = new QListWidgetItem(itemText, seriesListWidget_);
        item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
        item->setCheckState(Qt::Unchecked);
    }

    inputLayout->addWidget(seriesListWidget_);
    mainLayout->addWidget(inputGroupBox);

    // Target series group
    QGroupBox* targetGroupBox = new QGroupBox("Target Series");
    QVBoxLayout* targetLayout = new QVBoxLayout(targetGroupBox);

    targetCheckBox_ = new QCheckBox(
        QString("Target: %1 (%2 points)")
            .arg(QString::fromStdString(targetData_.name()))
            .arg(targetData_.size())
        );
    targetCheckBox_->setChecked(true);  // Default to checked
    targetLayout->addWidget(targetCheckBox_);

    mainLayout->addWidget(targetGroupBox);

    // Data info
    QGroupBox* infoGroupBox = new QGroupBox("Data Information");
    QFormLayout* infoFormLayout = new QFormLayout(infoGroupBox);

    if (inputData_.size() > 0) {
        double t_min = inputData_[0].mint();
        double t_max = inputData_[0].maxt();

        infoFormLayout->addRow("Time range:",
                               new QLabel(QString("[%1, %2]").arg(t_min).arg(t_max)));
        infoFormLayout->addRow("Number of input series:",
                               new QLabel(QString::number(inputData_.size())));
        infoFormLayout->addRow("Points per series:",
                               new QLabel(QString::number(inputData_[0].size())));
    }

    mainLayout->addWidget(infoGroupBox);

    mainLayout->addStretch();

    // Dialog buttons
    QHBoxLayout* dialogButtonLayout = new QHBoxLayout();
    dialogButtonLayout->addStretch();

    QPushButton* plotButton = new QPushButton("Plot");
    plotButton->setDefault(true);
    connect(plotButton, &QPushButton::clicked, this, &DataPlotDialog::onAccept);
    dialogButtonLayout->addWidget(plotButton);

    QPushButton* cancelButton = new QPushButton("Cancel");
    connect(cancelButton, &QPushButton::clicked, this, &QDialog::reject);
    dialogButtonLayout->addWidget(cancelButton);

    mainLayout->addLayout(dialogButtonLayout);
}

void DataPlotDialog::onSelectAll()
{
    for (int i = 0; i < seriesListWidget_->count(); i++) {
        seriesListWidget_->item(i)->setCheckState(Qt::Checked);
    }
}

void DataPlotDialog::onDeselectAll()
{
    for (int i = 0; i < seriesListWidget_->count(); i++) {
        seriesListWidget_->item(i)->setCheckState(Qt::Unchecked);
    }
}

void DataPlotDialog::onAccept()
{
    // Check if at least one series is selected
    bool hasSelection = targetCheckBox_->isChecked();

    for (int i = 0; i < seriesListWidget_->count(); i++) {
        if (seriesListWidget_->item(i)->checkState() == Qt::Checked) {
            hasSelection = true;
            break;
        }
    }

    if (!hasSelection) {
        QMessageBox::warning(this, "No Selection",
                             "Please select at least one series to plot.");
        return;
    }

    accept();
}

std::vector<int> DataPlotDialog::getSelectedInputSeries() const
{
    std::vector<int> selected;

    for (int i = 0; i < seriesListWidget_->count(); i++) {
        if (seriesListWidget_->item(i)->checkState() == Qt::Checked) {
            selected.push_back(i);
        }
    }

    return selected;
}

bool DataPlotDialog::shouldPlotTarget() const
{
    return targetCheckBox_->isChecked();
}
