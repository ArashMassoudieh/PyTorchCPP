#include "syntheticdatadialog.h"
#include "chartwindow.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QGroupBox>
#include <QFileDialog>
#include <QMessageBox>
#include <fstream>
#include <cmath>
#include <cstdlib>

SyntheticDataDialog::SyntheticDataDialog(QWidget *parent)
    : QDialog(parent), dataGenerated_(false)
{
    setWindowTitle("Generate Synthetic Test Data");
    setModal(true);
    setupUI();
    connectSignals();
}

SyntheticDataDialog::~SyntheticDataDialog()
{
}

void SyntheticDataDialog::setupUI()
{
    QVBoxLayout *mainLayout = new QVBoxLayout(this);

    // Parameters group
    QGroupBox *paramsGroup = new QGroupBox("Time Range Parameters", this);
    QFormLayout *paramsLayout = new QFormLayout(paramsGroup);

    // Time start
    tStartSpin_ = new QDoubleSpinBox(this);
    tStartSpin_->setRange(-1000.0, 1000.0);
    tStartSpin_->setValue(0.0);
    tStartSpin_->setDecimals(2);
    paramsLayout->addRow("Start Time:", tStartSpin_);

    // Time end
    tEndSpin_ = new QDoubleSpinBox(this);
    tEndSpin_->setRange(-1000.0, 10000.0);
    tEndSpin_->setValue(100.0);
    tEndSpin_->setDecimals(2);
    paramsLayout->addRow("End Time:", tEndSpin_);

    // Time delta
    dtSpin_ = new QDoubleSpinBox(this);
    dtSpin_->setRange(0.001, 10.0);
    dtSpin_->setValue(0.1);
    dtSpin_->setDecimals(3);
    dtSpin_->setSingleStep(0.01);
    paramsLayout->addRow("Time Step (dt):", dtSpin_);

    mainLayout->addWidget(paramsGroup);

    // Output path group
    QGroupBox *pathGroup = new QGroupBox("Output Location", this);
    QHBoxLayout *pathLayout = new QHBoxLayout(pathGroup);

    outputPathEdit_ = new QLineEdit("./", this);
    pathLayout->addWidget(outputPathEdit_);

    browseButton_ = new QPushButton("Browse...", this);
    pathLayout->addWidget(browseButton_);

    mainLayout->addWidget(pathGroup);

    // Generate button
    generateButton_ = new QPushButton("Generate Data", this);
    generateButton_->setStyleSheet("QPushButton { font-weight: bold; padding: 8px; }");
    mainLayout->addWidget(generateButton_);

    // Status label
    statusLabel_ = new QLabel("Ready to generate data", this);
    statusLabel_->setStyleSheet("QLabel { color: gray; font-style: italic; }");
    mainLayout->addWidget(statusLabel_);

    // Plot buttons group
    QGroupBox *plotGroup = new QGroupBox("Visualization", this);
    QHBoxLayout *plotLayout = new QHBoxLayout(plotGroup);

    plotInputsButton_ = new QPushButton("Plot Input Series", this);
    plotInputsButton_->setEnabled(false);
    plotInputsButton_->setDefault(false);
    plotInputsButton_->setAutoDefault(false);
    plotLayout->addWidget(plotInputsButton_);

    plotTargetButton_ = new QPushButton("Plot Target Series", this);
    plotTargetButton_->setEnabled(false);
    plotLayout->addWidget(plotTargetButton_);

    mainLayout->addWidget(plotGroup);

    // Dialog buttons
    QHBoxLayout *buttonLayout = new QHBoxLayout();
    buttonLayout->addStretch();

    okButton_ = new QPushButton("OK", this);
    okButton_->setEnabled(false);
    buttonLayout->addWidget(okButton_);

    cancelButton_ = new QPushButton("Cancel", this);
    buttonLayout->addWidget(cancelButton_);

    mainLayout->addLayout(buttonLayout);

    setLayout(mainLayout);
    resize(500, 400);
}

void SyntheticDataDialog::connectSignals()
{
    connect(browseButton_, &QPushButton::clicked, this, [this]() {
        QString dir = QFileDialog::getExistingDirectory(this, "Select Output Directory",
                                                        outputPathEdit_->text());
        if (!dir.isEmpty()) {
            outputPathEdit_->setText(dir);
        }
    });

    connect(generateButton_, &QPushButton::clicked, this, &SyntheticDataDialog::onGenerate);
    connect(plotInputsButton_, &QPushButton::clicked, this, &SyntheticDataDialog::onPlotInputs);
    connect(plotTargetButton_, &QPushButton::clicked, this, &SyntheticDataDialog::onPlotTarget);
    connect(okButton_, &QPushButton::clicked, this, &QDialog::accept);
    connect(cancelButton_, &QPushButton::clicked, this, &QDialog::reject);
}

void SyntheticDataDialog::generateSyntheticData(double t_start, double t_end, double dt,
                                                const std::string& output_path)
{
    // Seed for reproducibility
    srand(42);

    // Create 5 stationary input time series using Ornstein-Uhlenbeck processes
    inputData_ = TimeSeriesSet<double>(5);
    double buffer_start = t_start - 1.0;  // Buffer for lag interpolation

    // OU process parameters: dX = theta * (mu - X) * dt + sigma * dW
    // theta: mean reversion speed, mu: long-term mean, sigma: volatility

    // Initialize OU processes
    double x0 = 0.0;  // temperature
    double x1 = 0.0;  // pressure
    double x2 = 1.0;  // flow_rate
    double x3 = 0.0;  // concentration
    double x4 = 0.0;  // velocity

    for (double t = buffer_start; t <= t_end; t += dt) {
        // Series 0: Temperature - slow mean reversion
        double theta0 = 0.5;
        double mu0 = 0.0;
        double sigma0 = 1.5;
        double dW0 = std::sqrt(dt) * (static_cast<double>(rand()) / RAND_MAX - 0.5) * 2.0;
        x0 = x0 + theta0 * (mu0 - x0) * dt + sigma0 * dW0;
        inputData_[0].addPoint(t, x0);

        // Series 1: Pressure - moderate mean reversion
        double theta1 = 1.0;
        double mu1 = 0.0;
        double sigma1 = 1.2;
        double dW1 = std::sqrt(dt) * (static_cast<double>(rand()) / RAND_MAX - 0.5) * 2.0;
        x1 = x1 + theta1 * (mu1 - x1) * dt + sigma1 * dW1;
        inputData_[1].addPoint(t, x1);

        // Series 2: Flow rate - fast mean reversion around mean=1
        double theta2 = 2.0;
        double mu2 = 1.0;
        double sigma2 = 0.8;
        double dW2 = std::sqrt(dt) * (static_cast<double>(rand()) / RAND_MAX - 0.5) * 2.0;
        x2 = x2 + theta2 * (mu2 - x2) * dt + sigma2 * dW2;
        inputData_[2].addPoint(t, x2);

        // Series 3: Concentration - very slow mean reversion (long memory)
        double theta3 = 0.3;
        double mu3 = 0.0;
        double sigma3 = 1.0;
        double dW3 = std::sqrt(dt) * (static_cast<double>(rand()) / RAND_MAX - 0.5) * 2.0;
        x3 = x3 + theta3 * (mu3 - x3) * dt + sigma3 * dW3;
        inputData_[3].addPoint(t, x3);

        // Series 4: Velocity - moderate mean reversion with higher volatility
        double theta4 = 0.8;
        double mu4 = 0.0;
        double sigma4 = 1.8;
        double dW4 = std::sqrt(dt) * (static_cast<double>(rand()) / RAND_MAX - 0.5) * 2.0;
        x4 = x4 + theta4 * (mu4 - x4) * dt + sigma4 * dW4;
        inputData_[4].addPoint(t, x4);
    }

    inputData_.setSeriesName(0, "temperature");
    inputData_.setSeriesName(1, "pressure");
    inputData_.setSeriesName(2, "flow_rate");
    inputData_.setSeriesName(3, "concentration");
    inputData_.setSeriesName(4, "velocity");

    // Create target data (linear combination with lags)
    targetData_ = TimeSeries<double>();
    for (double t = t_start; t <= t_end; t += dt) {
        double target = 0.4 * inputData_[0].interpol(t - 0.1) +
                        0.3 * inputData_[1].interpol(t - 0.3) +
                        0.2 * inputData_[3].interpol(t - 0.2) +
                        0.1 * inputData_[4].interpol(t - 0.5) +
                        0.05 * (static_cast<double>(rand()) / RAND_MAX - 0.5);
        targetData_.addPoint(t, target);
    }
    targetData_.setName("target");

    // Save files
    std::string path_prefix = output_path;
    if (!output_path.empty() && output_path.back() != '/' && output_path.back() != '\\') {
        path_prefix += "/";
    }

    inputData_.write(path_prefix + "ga_test_input.csv");

    std::ofstream target_file(path_prefix + "ga_test_target.txt");
    if (target_file.is_open()) {
        for (size_t i = 0; i < targetData_.size(); ++i) {
            target_file << std::fixed << std::setprecision(6)
            << targetData_.getTime(i) << ","
            << targetData_.getValue(i) << std::endl;
        }
        target_file.close();
    }

    dataGenerated_ = true;
}
void SyntheticDataDialog::onGenerate()
{
    double t_start = tStartSpin_->value();
    double t_end = tEndSpin_->value();
    double dt = dtSpin_->value();
    std::string output_path = outputPathEdit_->text().toStdString();

    // Validate inputs
    if (t_end <= t_start) {
        QMessageBox::warning(this, "Invalid Range", "End time must be greater than start time.");
        return;
    }

    if (dt <= 0) {
        QMessageBox::warning(this, "Invalid Time Step", "Time step must be positive.");
        return;
    }

    // Generate data
    try {
        generateSyntheticData(t_start, t_end, dt, output_path);

        statusLabel_->setText(QString("✓ Data generated successfully! Files saved to: %1")
                                  .arg(QString::fromStdString(output_path)));
        statusLabel_->setStyleSheet("QLabel { color: green; font-weight: bold; }");

        // Enable plot and OK buttons
        plotInputsButton_->setEnabled(true);
        plotTargetButton_->setEnabled(true);
        okButton_->setEnabled(true);

        QMessageBox::information(this, "Success",
                                 QString("Synthetic data generated!\n\n"
                                         "Input file: ga_test_input.csv\n"
                                         "Target file: ga_test_target.txt\n\n"
                                         "Target formula:\n"
                                         "  0.4×temperature(t-0.1) +\n"
                                         "  0.3×pressure(t-0.3) +\n"
                                         "  0.2×concentration(t-0.2) +\n"
                                         "  0.1×velocity(t-0.5) +\n"
                                         "  noise"));

    } catch (const std::exception& e) {
        QMessageBox::critical(this, "Error",
                              QString("Failed to generate data: %1").arg(e.what()));
        statusLabel_->setText("✗ Generation failed");
        statusLabel_->setStyleSheet("QLabel { color: red; }");
    }
}

void SyntheticDataDialog::onPlotInputs()
{
    if (!dataGenerated_) {
        QMessageBox::warning(this, "No Data", "Please generate data first.");
        return;
    }

    ChartWindow *chartWindow = new ChartWindow(inputData_, this);
    chartWindow->setWindowTitle("Synthetic Input Time Series");
    chartWindow->setChartTitle("Synthetic Input Time Series");
    chartWindow->setAxisLabels("Time", "Value");
    chartWindow->setWindowModality(Qt::NonModal);

    chartWindow->setAttribute(Qt::WA_DeleteOnClose, true);
    chartWindow->show();
}

void SyntheticDataDialog::onPlotTarget()
{
    if (!dataGenerated_) {
        QMessageBox::warning(this, "No Data", "Please generate data first.");
        return;
    }

    // Create a TimeSeriesSet from the single TimeSeries
    TimeSeriesSet<double> targetSet;
    targetSet.append(targetData_);

    ChartWindow *chartWindow = new ChartWindow(targetSet, this);
    chartWindow->setWindowTitle("Synthetic Target Time Series");
    chartWindow->setChartTitle("Synthetic Target Time Series");
    chartWindow->setAxisLabels("Time", "Target Value");
    chartWindow->setWindowModality(Qt::NonModal);

    chartWindow->setAttribute(Qt::WA_DeleteOnClose, true);
    chartWindow->show();
}
