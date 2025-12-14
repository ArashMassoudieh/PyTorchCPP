#include "DataLoadDialog.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QGroupBox>
#include <QFileDialog>
#include <QMessageBox>
#include <QFileInfo>

DataLoadDialog::DataLoadDialog(QWidget* parent)
    : QDialog(parent)
    , dataValid_(false)
{
    setWindowTitle("Load Time Series Data");
    setMinimumWidth(600);

    setupUI();

    // Connect signals
    connect(browseInputBtn_, &QPushButton::clicked, this, &DataLoadDialog::onBrowseInput);
    connect(browseTargetBtn_, &QPushButton::clicked, this, &DataLoadDialog::onBrowseTarget);
    connect(previewBtn_, &QPushButton::clicked, this, &DataLoadDialog::onPreviewData);
    connect(okButton_, &QPushButton::clicked, this, &DataLoadDialog::onAccept);
    connect(cancelButton_, &QPushButton::clicked, this, &DataLoadDialog::onReject);
}

DataLoadDialog::~DataLoadDialog()
{
}

void DataLoadDialog::setupUI()
{
    QVBoxLayout* mainLayout = new QVBoxLayout(this);

    // Input file group
    QGroupBox* inputGroup = new QGroupBox("Input Time Series (Multiple Series)", this);
    QVBoxLayout* inputLayout = new QVBoxLayout(inputGroup);

    QHBoxLayout* inputPathLayout = new QHBoxLayout();
    QLabel* inputLabel = new QLabel("File:", this);
    inputFileEdit_ = new QLineEdit(this);
    inputFileEdit_->setPlaceholderText("Select CSV file with input time series...");
    browseInputBtn_ = new QPushButton("Browse...", this);

    inputPathLayout->addWidget(inputLabel);
    inputPathLayout->addWidget(inputFileEdit_);
    inputPathLayout->addWidget(browseInputBtn_);
    inputLayout->addLayout(inputPathLayout);

    inputHasHeaderCheckBox_ = new QCheckBox("File has header row", this);
    inputHasHeaderCheckBox_->setChecked(true);
    inputHasHeaderCheckBox_->setToolTip("Check if first row contains column names");
    inputLayout->addWidget(inputHasHeaderCheckBox_);

    QLabel* inputInfo = new QLabel("Format: CSV file with time in first column, followed by multiple series columns", this);
    inputInfo->setWordWrap(true);
    inputInfo->setStyleSheet("QLabel { color: gray; font-style: italic; font-size: 10pt; }");
    inputLayout->addWidget(inputInfo);

    mainLayout->addWidget(inputGroup);

    // Target file group
    QGroupBox* targetGroup = new QGroupBox("Target Time Series (Single Series)", this);
    QVBoxLayout* targetLayout = new QVBoxLayout(targetGroup);

    QHBoxLayout* targetPathLayout = new QHBoxLayout();
    QLabel* targetLabel = new QLabel("File:", this);
    targetFileEdit_ = new QLineEdit(this);
    targetFileEdit_->setPlaceholderText("Select text file with target time series...");
    browseTargetBtn_ = new QPushButton("Browse...", this);

    targetPathLayout->addWidget(targetLabel);
    targetPathLayout->addWidget(targetFileEdit_);
    targetPathLayout->addWidget(browseTargetBtn_);
    targetLayout->addLayout(targetPathLayout);

    QLabel* targetInfo = new QLabel("Format: Text file with two columns (time, value)", this);
    targetInfo->setWordWrap(true);
    targetInfo->setStyleSheet("QLabel { color: gray; font-style: italic; font-size: 10pt; }");
    targetLayout->addWidget(targetInfo);

    mainLayout->addWidget(targetGroup);

    // Status label
    statusLabel_ = new QLabel("Select input and target files to continue", this);
    statusLabel_->setStyleSheet("QLabel { padding: 10px; background-color: #f0f0f0; border-radius: 5px; }");
    statusLabel_->setWordWrap(true);
    mainLayout->addWidget(statusLabel_);

    // Buttons
    QHBoxLayout* buttonLayout = new QHBoxLayout();
    previewBtn_ = new QPushButton("Preview Data", this);
    previewBtn_->setEnabled(false);
    okButton_ = new QPushButton("Load Data", this);
    okButton_->setDefault(true);
    cancelButton_ = new QPushButton("Cancel", this);

    buttonLayout->addWidget(previewBtn_);
    buttonLayout->addStretch();
    buttonLayout->addWidget(okButton_);
    buttonLayout->addWidget(cancelButton_);

    mainLayout->addLayout(buttonLayout);
}

void DataLoadDialog::onBrowseInput()
{
    QString filename = QFileDialog::getOpenFileName(
        this,
        "Select Input Time Series File",
        inputFileEdit_->text().isEmpty() ? "" : QFileInfo(inputFileEdit_->text()).path(),
        "CSV Files (*.csv);;Text Files (*.txt);;All Files (*.*)"
        );

    if (!filename.isEmpty()) {
        inputFileEdit_->setText(filename);

        // Enable preview if both files selected
        if (!targetFileEdit_->text().isEmpty()) {
            previewBtn_->setEnabled(true);

        }
        inputFilePath_ = filename;
    }




}

void DataLoadDialog::onBrowseTarget()
{
    QString filename = QFileDialog::getOpenFileName(
        this,
        "Select Target Time Series File",
        targetFileEdit_->text().isEmpty() ? "" : QFileInfo(targetFileEdit_->text()).path(),
        "Text Files (*.txt);;CSV Files (*.csv);;All Files (*.*)"
        );

    if (!filename.isEmpty()) {
        targetFileEdit_->setText(filename);

        // Enable preview if both files selected
        if (!inputFileEdit_->text().isEmpty()) {
            previewBtn_->setEnabled(true);
        }

        targetFilePath_ = filename;
    }


}

void DataLoadDialog::onPreviewData()
{
    if (!loadInputData() || !loadTargetData()) {
        return;
    }

    // Show preview information
    QString previewText = QString(
                              "=== Data Preview ===\n\n"
                              "Input Data:\n"
                              "  Number of series: %1\n"
                              "  Data points per series: %2\n"
                              "  Time range: %3 to %4\n"
                              "  Time step (avg): %5\n\n"
                              "Target Data:\n"
                              "  Data points: %6\n"
                              "  Time range: %7 to %8\n"
                              "  Time step (avg): %9\n\n"
                              ).arg(inputData_.size())
                              .arg(inputData_[0].size())
                              .arg(inputData_[0].front().t)
                              .arg(inputData_[0].back().t)
                              .arg((inputData_[0].back().t - inputData_[0].front().t) / (inputData_[0].size() - 1))
                              .arg(targetData_.size())
                              .arg(targetData_.front().t)
                              .arg(targetData_.back().t)
                              .arg((targetData_.back().t - targetData_.front().t) / (targetData_.size() - 1));

    previewText += "Note: Input and target time series will be aligned automatically during training.\n";

    statusLabel_->setStyleSheet("QLabel { padding: 10px; background-color: #d4edda; border-radius: 5px; color: #155724; }");
    statusLabel_->setText("Data preview successful - ready to load");

    QMessageBox::information(this, "Data Preview", previewText);
}

bool DataLoadDialog::loadInputData()
{
    try {
        inputData_ = TimeSeriesSet<double>();

        bool hasHeader = inputHasHeaderCheckBox_->isChecked();
        inputData_.read(inputFileEdit_->text().toStdString(), hasHeader);

        if (inputData_.size() == 0) {
            throw std::runtime_error("No time series found in input file");
        }

        return true;

    } catch (const std::exception& e) {
        QMessageBox::critical(this, "Input Load Error",
                              QString("Failed to load input data:\n%1").arg(e.what()));
        statusLabel_->setStyleSheet("QLabel { padding: 10px; background-color: #f8d7da; border-radius: 5px; color: #721c24; }");
        statusLabel_->setText(QString("ERROR: %1").arg(e.what()));
        return false;
    }
}

bool DataLoadDialog::loadTargetData()
{
    try {
        targetData_ = TimeSeries<double>();
        targetData_.readfile(targetFileEdit_->text().toStdString());

        if (targetData_.size() == 0) {
            throw std::runtime_error("No data points found in target file");
        }

        return true;

    } catch (const std::exception& e) {
        QMessageBox::critical(this, "Target Load Error",
                              QString("Failed to load target data:\n%1").arg(e.what()));
        statusLabel_->setStyleSheet("QLabel { padding: 10px; background-color: #f8d7da; border-radius: 5px; color: #721c24; }");
        statusLabel_->setText(QString("ERROR: %1").arg(e.what()));
        return false;
    }
}

bool DataLoadDialog::validateDataSizes()
{
    if (inputData_.size() == 0 || targetData_.size() == 0) {
        return false;
    }

    // Just return true - sizes don't need to match
    // The neural network wrapper will handle interpolation/alignment
    return true;
}

bool DataLoadDialog::validateAndLoadData()
{
    // Check if files are selected
    if (inputFileEdit_->text().isEmpty()) {
        QMessageBox::warning(this, "Missing File", "Please select an input file.");
        return false;
    }

    if (targetFileEdit_->text().isEmpty()) {
        QMessageBox::warning(this, "Missing File", "Please select a target file.");
        return false;
    }

    // Load data
    if (!loadInputData()) {
        return false;
    }

    if (!loadTargetData()) {
        return false;
    }

    // Just verify data exists (no size matching required)
    if (inputData_.size() == 0 || targetData_.size() == 0) {
        QMessageBox::warning(this, "Invalid Data",
                             "Loaded data is empty. Please check your files.");
        return false;
    }

    dataValid_ = true;
    return true;
}

void DataLoadDialog::onAccept()
{
    if (validateAndLoadData()) {
        accept();
    }
}

void DataLoadDialog::onReject()
{
    reject();
}
