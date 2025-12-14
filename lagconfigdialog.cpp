#include "lagconfigdialog.h"
#include <QMessageBox>
#include <QGroupBox>
#include <QHBoxLayout>

LagConfigDialog::LagConfigDialog(NetworkArchitecture& architecture,
                                 const TimeSeriesSet<double>& inputData,
                                 QWidget *parent)
    : QDialog(parent)
    , architecture_(architecture)
    , inputData_(inputData)
{
    setWindowTitle("Configure Time Lags");
    setModal(true);
    setupUI();
    loadCurrentLags();
    resize(600, 500);
}

void LagConfigDialog::setupUI()
{
    QVBoxLayout* mainLayout = new QVBoxLayout(this);

    // Info label
    QLabel* infoLabel = new QLabel(
        "<b>Time Lag Configuration</b><br>"
        "Define which past time steps to use as input features for each series.<br>"
        "Example: Min=1, Max=5, Step=1 → lags [1, 2, 3, 4, 5]<br>"
        "Lag 0 means current value, Lag 1 means previous time step, etc.");
    infoLabel->setWordWrap(true);
    infoLabel->setStyleSheet("QLabel { padding: 10px; background-color: #e8f4f8; border-radius: 5px; }");
    mainLayout->addWidget(infoLabel);

    // Buttons for batch operations
    QHBoxLayout* batchButtonLayout = new QHBoxLayout();

    QPushButton* defaultsButton = new QPushButton("Reset to Defaults");
    defaultsButton->setToolTip("Set all series to lag [0] (current value only)");
    connect(defaultsButton, &QPushButton::clicked, this, &LagConfigDialog::onUseDefaults);
    batchButtonLayout->addWidget(defaultsButton);

    QPushButton* applyToAllButton = new QPushButton("Apply First to All");
    applyToAllButton->setToolTip("Copy first series configuration to all series");
    connect(applyToAllButton, &QPushButton::clicked, this, &LagConfigDialog::onApplyToAll);
    batchButtonLayout->addWidget(applyToAllButton);

    batchButtonLayout->addStretch();
    mainLayout->addLayout(batchButtonLayout);

    // Scroll area for series configurations
    QScrollArea* scrollArea = new QScrollArea();
    scrollArea->setWidgetResizable(true);
    scrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

    QWidget* scrollWidget = new QWidget();
    QVBoxLayout* scrollLayout = new QVBoxLayout(scrollWidget);

    // Create configuration for each series
    for (int i = 0; i < inputData_.size(); i++) {
        QGroupBox* groupBox = new QGroupBox(
            QString("Series %1: %2").arg(i).arg(QString::fromStdString(inputData_[i].name())));
        QFormLayout* formLayout = new QFormLayout(groupBox);

        SeriesLagConfig config;
        config.groupBox = groupBox;

        config.minLagSpin = new QSpinBox();
        config.minLagSpin->setRange(0, 1000);
        config.minLagSpin->setValue(0);
        config.minLagSpin->setToolTip("Minimum lag value (0 = current time)");
        formLayout->addRow("Min Lag:", config.minLagSpin);

        config.maxLagSpin = new QSpinBox();
        config.maxLagSpin->setRange(0, 1000);
        config.maxLagSpin->setValue(0);
        config.maxLagSpin->setToolTip("Maximum lag value");
        formLayout->addRow("Max Lag:", config.maxLagSpin);

        config.stepSpin = new QSpinBox();
        config.stepSpin->setRange(1, 100);
        config.stepSpin->setValue(1);
        config.stepSpin->setToolTip("Step between consecutive lags");
        formLayout->addRow("Step:", config.stepSpin);

        config.resultLabel = new QLabel("Resulting lags: [0]");
        config.resultLabel->setStyleSheet("QLabel { font-family: monospace; color: #0066cc; }");
        formLayout->addRow("Preview:", config.resultLabel);

        // Connect signals to update preview
        int index = i;  // Capture for lambda
        connect(config.minLagSpin, QOverload<int>::of(&QSpinBox::valueChanged),
                [this, index]() { updateResultLabel(index); });
        connect(config.maxLagSpin, QOverload<int>::of(&QSpinBox::valueChanged),
                [this, index]() { updateResultLabel(index); });
        connect(config.stepSpin, QOverload<int>::of(&QSpinBox::valueChanged),
                [this, index]() { updateResultLabel(index); });

        seriesConfigs_.push_back(config);
        scrollLayout->addWidget(groupBox);
    }

    scrollLayout->addStretch();
    scrollWidget->setLayout(scrollLayout);
    scrollArea->setWidget(scrollWidget);
    mainLayout->addWidget(scrollArea);

    // Dialog buttons
    QDialogButtonBox* dialogButtons = new QDialogButtonBox(
        QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    connect(dialogButtons, &QDialogButtonBox::accepted, this, &LagConfigDialog::onAccept);
    connect(dialogButtons, &QDialogButtonBox::rejected, this, &QDialog::reject);
    mainLayout->addWidget(dialogButtons);
}

void LagConfigDialog::loadCurrentLags()
{
    if (architecture_.lags.size() == inputData_.size()) {
        for (size_t i = 0; i < seriesConfigs_.size(); i++) {
            if (!architecture_.lags[i].empty()) {
                int minLag = *std::min_element(architecture_.lags[i].begin(),
                                               architecture_.lags[i].end());
                int maxLag = *std::max_element(architecture_.lags[i].begin(),
                                               architecture_.lags[i].end());

                seriesConfigs_[i].minLagSpin->setValue(minLag);
                seriesConfigs_[i].maxLagSpin->setValue(maxLag);

                // Estimate step
                if (architecture_.lags[i].size() > 1) {
                    int step = architecture_.lags[i][1] - architecture_.lags[i][0];
                    seriesConfigs_[i].stepSpin->setValue(step);
                }
            }
        }
    }

    // Update all previews
    for (size_t i = 0; i < seriesConfigs_.size(); i++) {
        updateResultLabel(i);
    }
}

void LagConfigDialog::updateResultLabel(int index)
{
    if (index < 0 || index >= seriesConfigs_.size()) return;

    int minLag = seriesConfigs_[index].minLagSpin->value();
    int maxLag = seriesConfigs_[index].maxLagSpin->value();
    int step = seriesConfigs_[index].stepSpin->value();

    QString result = "Resulting lags: [";
    bool first = true;
    int count = 0;

    for (int lag = minLag; lag <= maxLag && count < 20; lag += step) {
        if (!first) result += ", ";
        result += QString::number(lag);
        first = false;
        count++;
    }

    if (minLag <= maxLag) {
        int totalLags = ((maxLag - minLag) / step) + 1;
        if (totalLags > 20) {
            result += ", ...";
        }
        result += QString("] (%1 lags)").arg(totalLags);
    } else {
        result += "] (0 lags - invalid!)";
    }

    seriesConfigs_[index].resultLabel->setText(result);
}

void LagConfigDialog::onUseDefaults()
{
    for (auto& config : seriesConfigs_) {
        config.minLagSpin->setValue(0);
        config.maxLagSpin->setValue(0);
        config.stepSpin->setValue(1);
    }
}

void LagConfigDialog::onApplyToAll()
{
    if (seriesConfigs_.empty()) return;

    int minLag = seriesConfigs_[0].minLagSpin->value();
    int maxLag = seriesConfigs_[0].maxLagSpin->value();
    int step = seriesConfigs_[0].stepSpin->value();

    for (size_t i = 1; i < seriesConfigs_.size(); i++) {
        seriesConfigs_[i].minLagSpin->setValue(minLag);
        seriesConfigs_[i].maxLagSpin->setValue(maxLag);
        seriesConfigs_[i].stepSpin->setValue(step);
    }
}

void LagConfigDialog::onAccept()
{
    // Validate
    for (size_t i = 0; i < seriesConfigs_.size(); i++) {
        int minLag = seriesConfigs_[i].minLagSpin->value();
        int maxLag = seriesConfigs_[i].maxLagSpin->value();

        if (minLag > maxLag) {
            QMessageBox::warning(this, "Invalid Configuration",
                                 QString("Series %1: Min lag (%2) must be <= Max lag (%3)")
                                     .arg(i).arg(minLag).arg(maxLag));
            return;
        }
    }

    // Build lag configuration
    architecture_.lags.clear();

    for (size_t i = 0; i < seriesConfigs_.size(); i++) {
        int minLag = seriesConfigs_[i].minLagSpin->value();
        int maxLag = seriesConfigs_[i].maxLagSpin->value();
        int step = seriesConfigs_[i].stepSpin->value();

        std::vector<int> lags;
        for (int lag = minLag; lag <= maxLag; lag += step) {
            lags.push_back(lag);
        }

        architecture_.lags.push_back(lags);
    }

    accept();
}
