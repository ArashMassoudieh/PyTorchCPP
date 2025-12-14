#include "networkarchitecturedialog.h"
#include <QFormLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QMessageBox>

NetworkArchitectureDialog::NetworkArchitectureDialog(
    const TimeSeriesSet<double>& inputData,
    const TimeSeries<double>& targetData,
    const NetworkArchitecture& existingArchitecture,
    QWidget *parent)
    : QDialog(parent)
    , inputData_(inputData)
    , targetData_(targetData)
    , outputActivation_("linear")
    , model_(nullptr)
{
    setWindowTitle("Configure Network Architecture");
    setModal(true);
    setupUI();
    connectSignals();

    // Load existing architecture if configured
    if (existingArchitecture.isConfigured) {
        loadExistingArchitecture(existingArchitecture);
    }

    updateSummary();
    resize(600, 500);
}

void NetworkArchitectureDialog::loadExistingArchitecture(const NetworkArchitecture& arch)
{
    // Load hidden layers
    hiddenLayerSizes_ = arch.hiddenLayers;

    // Load activations (fill with "relu" if not enough stored)
    hiddenActivations_ = arch.activations;
    while (hiddenActivations_.size() < hiddenLayerSizes_.size()) {
        hiddenActivations_.push_back("relu");
    }

    // Load output activation
    outputActivation_ = arch.outputActivation;

    // Update UI
    updateLayersList();
    outputActivationCombo_->setCurrentText(QString::fromStdString(outputActivation_));
    updateSummary();

    // Change button text to indicate editing
    createButton_->setText("Update Network");
}

NetworkArchitecture NetworkArchitectureDialog::getArchitecture() const
{
    NetworkArchitecture arch;
    arch.isConfigured = true;
    arch.hiddenLayers = hiddenLayerSizes_;
    arch.activations = hiddenActivations_;
    arch.outputActivation = outputActivation_;

    // Create lags configuration (simple: all series, no lags)
    arch.lags.resize(inputData_.size());
    for (size_t i = 0; i < inputData_.size(); ++i) {
        arch.lags[i] = {0};  // Use current values only
    }

    return arch;
}

NetworkArchitectureDialog::~NetworkArchitectureDialog()
{
    // model_ is owned by the caller, don't delete here
}

void NetworkArchitectureDialog::setupUI()
{
    QVBoxLayout* mainLayout = new QVBoxLayout(this);

    // Input info
    QLabel* inputInfoLabel = new QLabel(
        QString("Input data: %1 time series, %2 samples")
            .arg(inputData_.size())
            .arg(targetData_.size()),
        this);
    inputInfoLabel->setStyleSheet("QLabel { font-weight: bold; color: #2c3e50; }");
    mainLayout->addWidget(inputInfoLabel);

    // Layer configuration group
    QGroupBox* layerConfigGroup = new QGroupBox("Add Hidden Layers", this);
    QFormLayout* layerConfigLayout = new QFormLayout(layerConfigGroup);

    layerSizeSpin_ = new QSpinBox(this);
    layerSizeSpin_->setRange(1, 1000);
    layerSizeSpin_->setValue(64);
    layerConfigLayout->addRow("Number of Nodes:", layerSizeSpin_);

    activationCombo_ = new QComboBox(this);
    activationCombo_->addItems({"relu", "tanh", "sigmoid", "linear"});
    activationCombo_->setCurrentText("relu");
    layerConfigLayout->addRow("Activation Function:", activationCombo_);

    QHBoxLayout* layerButtonLayout = new QHBoxLayout();
    addLayerButton_ = new QPushButton("Add Layer", this);
    addLayerButton_->setStyleSheet("QPushButton { background-color: #27ae60; color: white; }");
    layerButtonLayout->addWidget(addLayerButton_);

    removeLayerButton_ = new QPushButton("Remove Selected", this);
    removeLayerButton_->setEnabled(false);
    removeLayerButton_->setStyleSheet("QPushButton { background-color: #e74c3c; color: white; }");
    layerButtonLayout->addWidget(removeLayerButton_);

    layerConfigLayout->addRow(layerButtonLayout);
    mainLayout->addWidget(layerConfigGroup);

    // Layers list
    QLabel* layersLabel = new QLabel("Hidden Layers (in order):", this);
    mainLayout->addWidget(layersLabel);

    layersList_ = new QListWidget(this);
    layersList_->setMaximumHeight(150);
    mainLayout->addWidget(layersList_);

    // Output activation
    QGroupBox* outputGroup = new QGroupBox("Output Layer Configuration", this);
    QFormLayout* outputLayout = new QFormLayout(outputGroup);

    outputActivationCombo_ = new QComboBox(this);
    outputActivationCombo_->addItems({"linear", "relu", "tanh", "sigmoid"});
    outputActivationCombo_->setCurrentText("linear");
    outputLayout->addRow("Output Activation:", outputActivationCombo_);

    mainLayout->addWidget(outputGroup);

    // Network summary
    summaryLabel_ = new QLabel(this);
    summaryLabel_->setWordWrap(true);
    summaryLabel_->setStyleSheet("QLabel { background-color: #ecf0f1; padding: 10px; border-radius: 5px; }");
    mainLayout->addWidget(summaryLabel_);

    mainLayout->addStretch();

    // Dialog buttons
    QHBoxLayout* buttonLayout = new QHBoxLayout();
    buttonLayout->addStretch();

    createButton_ = new QPushButton("Create Network", this);
    createButton_->setStyleSheet("QPushButton { background-color: #3498db; color: white; padding: 8px; font-weight: bold; }");
    buttonLayout->addWidget(createButton_);

    cancelButton_ = new QPushButton("Cancel", this);
    buttonLayout->addWidget(cancelButton_);

    mainLayout->addLayout(buttonLayout);
    setLayout(mainLayout);
}

void NetworkArchitectureDialog::connectSignals()
{
    connect(addLayerButton_, &QPushButton::clicked, this, &NetworkArchitectureDialog::onAddLayer);
    connect(removeLayerButton_, &QPushButton::clicked, this, &NetworkArchitectureDialog::onRemoveLayer);
    connect(createButton_, &QPushButton::clicked, this, &NetworkArchitectureDialog::onCreateNetwork);
    connect(cancelButton_, &QPushButton::clicked, this, &QDialog::reject);
    connect(layersList_, &QListWidget::itemSelectionChanged, this, &NetworkArchitectureDialog::onLayerSelectionChanged);
    connect(outputActivationCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &NetworkArchitectureDialog::updateSummary);
}

void NetworkArchitectureDialog::onAddLayer()
{
    int layerSize = layerSizeSpin_->value();
    std::string activation = activationCombo_->currentText().toStdString();

    hiddenLayerSizes_.push_back(layerSize);
    hiddenActivations_.push_back(activation);

    updateLayersList();
    updateSummary();
}

void NetworkArchitectureDialog::onRemoveLayer()
{
    int currentRow = layersList_->currentRow();
    if (currentRow >= 0 && currentRow < static_cast<int>(hiddenLayerSizes_.size())) {
        hiddenLayerSizes_.erase(hiddenLayerSizes_.begin() + currentRow);
        hiddenActivations_.erase(hiddenActivations_.begin() + currentRow);

        updateLayersList();
        updateSummary();
    }
}

void NetworkArchitectureDialog::onLayerSelectionChanged()
{
    removeLayerButton_->setEnabled(layersList_->currentRow() >= 0);
}

void NetworkArchitectureDialog::updateLayersList()
{
    layersList_->clear();

    for (size_t i = 0; i < hiddenLayerSizes_.size(); ++i) {
        QString layerText = QString("Layer %1: %2 nodes (%3)")
        .arg(i + 1)
            .arg(hiddenLayerSizes_[i])
            .arg(QString::fromStdString(hiddenActivations_[i]));
        layersList_->addItem(layerText);
    }
}

int NetworkArchitectureDialog::calculateInputSize()
{
    // This is a simplified calculation - you might need to adjust based on your lag configuration
    // For now, assume we use all time series without lags
    return inputData_.size();
}

void NetworkArchitectureDialog::updateSummary()
{
    int inputSize = calculateInputSize();
    int outputSize = 1; // Single output for target

    QString summary = QString("<b>Network Architecture Summary:</b><br>");
    summary += QString("Input layer: %1 features<br>").arg(inputSize);

    if (hiddenLayerSizes_.empty()) {
        summary += QString("<i>No hidden layers (direct input → output)</i><br>");
    } else {
        for (size_t i = 0; i < hiddenLayerSizes_.size(); ++i) {
            summary += QString("Hidden layer %1: %2 nodes, %3 activation<br>")
            .arg(i + 1)
                .arg(hiddenLayerSizes_[i])
                .arg(QString::fromStdString(hiddenActivations_[i]));
        }
    }

    summary += QString("Output layer: %1 output(s), %2 activation<br>")
                   .arg(outputSize)
                   .arg(outputActivationCombo_->currentText());

    // Calculate total parameters
    int totalParams = 0;
    int prevSize = inputSize;
    for (int layerSize : hiddenLayerSizes_) {
        totalParams += (prevSize + 1) * layerSize; // weights + biases
        prevSize = layerSize;
    }
    totalParams += (prevSize + 1) * outputSize; // output layer

    summary += QString("<br><b>Total parameters: %1</b>").arg(totalParams);

    summaryLabel_->setText(summary);
}

void NetworkArchitectureDialog::onCreateNetwork()
{
    if (hiddenLayerSizes_.empty()) {
        QMessageBox::StandardButton reply = QMessageBox::question(
            this,
            "No Hidden Layers",
            "You haven't added any hidden layers. This will create a linear model.\n\nContinue?",
            QMessageBox::Yes | QMessageBox::No);

        if (reply == QMessageBox::No) {
            return;
        }
    }

    try {
        // Create new model
        model_ = new NeuralNetworkWrapper();

        // Set architecture
        model_->setHiddenLayers(hiddenLayerSizes_);

        // For now, use simple configuration - you can extend this to use lags
        std::vector<std::vector<int>> lags(inputData_.size());
        for (size_t i = 0; i < inputData_.size(); ++i) {
            lags[i] = {0}; // Use current values only (no lags)
        }
        model_->setLags(lags);

        // Get output activation
        outputActivation_ = outputActivationCombo_->currentText().toStdString();

        // Initialize network
        // Note: You'll need to add a method to set custom activations per layer
        // For now, using the default activation function
        model_->initializeNetwork(1, "relu"); // 1 output, default activation

        // Set the data
        double t_start = targetData_.mint();
        double t_end = targetData_.maxt();
        double dt = 0.1; // You might want to calculate this from data

        model_->setInputData(DataType::Train, inputData_, t_start, t_end, dt);
        model_->setTargetData(DataType::Train, targetData_, t_start, t_end, dt);

        QMessageBox::information(this, "Success",
                                 QString("Network created successfully!\n\n"
                                         "Hidden layers: %1\n"
                                         "Total parameters: %2\n\n"
                                         "You can now train this network.")
                                     .arg(hiddenLayerSizes_.size())
                                     .arg(model_->getTotalParameters()));

        accept(); // Close dialog with success

    } catch (const std::exception& e) {
        QMessageBox::critical(this, "Error",
                              QString("Failed to create network:\n%1").arg(e.what()));

        if (model_) {
            delete model_;
            model_ = nullptr;
        }
    }
}
