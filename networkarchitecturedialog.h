#ifndef NETWORKARCHITECTUREDIALOG_H
#define NETWORKARCHITECTUREDIALOG_H

#include <QDialog>
#include <QSpinBox>
#include <QComboBox>
#include <QPushButton>
#include <QListWidget>
#include <QVBoxLayout>
#include <QLabel>
#include <vector>
#include <string>
#include "neuralnetworkwrapper.h"
#include "TimeSeriesSet.h"
#include "TimeSeries.h"
#include "commontypes.h"  // Add this for NetworkArchitecture

/**
 * @brief Dialog for manually configuring neural network architecture
 */
class NetworkArchitectureDialog : public QDialog
{
    Q_OBJECT

public:
    explicit NetworkArchitectureDialog(
        const TimeSeriesSet<double>& inputData,
        const TimeSeries<double>& targetData,
        QWidget *parent = nullptr);

    // Constructor for editing existing architecture
    explicit NetworkArchitectureDialog(
        const TimeSeriesSet<double>& inputData,
        const TimeSeries<double>& targetData,
        const NetworkArchitecture& existingArchitecture,
        QWidget *parent = nullptr);

    ~NetworkArchitectureDialog();

    /**
     * @brief Get the configured neural network model
     * @return Pointer to configured NeuralNetworkWrapper (nullptr if not created)
     */
    NeuralNetworkWrapper* getModel() { return model_; }

    /**
     * @brief Check if model was successfully created
     * @return true if model is valid
     */
    bool isModelValid() const { return model_ != nullptr && model_->isInitialized(); }

    /**
     * @brief Get the configured architecture (without creating model)
     * @return NetworkArchitecture containing the configuration
     */
    NetworkArchitecture getArchitecture() const;

private Q_SLOTS:
    void onAddLayer();
    void onRemoveLayer();
    void onCreateNetwork();
    void onLayerSelectionChanged();

private:
    void setupUI();
    void connectSignals();
    void updateLayersList();
    void updateSummary();
    int calculateInputSize();
    void loadExistingArchitecture(const NetworkArchitecture& arch);  // Add this

    // UI Components
    QSpinBox* layerSizeSpin_;
    QComboBox* activationCombo_;
    QPushButton* addLayerButton_;
    QPushButton* removeLayerButton_;
    QListWidget* layersList_;
    QLabel* summaryLabel_;
    QComboBox* outputActivationCombo_;
    QPushButton* createButton_;
    QPushButton* cancelButton_;

    // Data
    TimeSeriesSet<double> inputData_;
    TimeSeries<double> targetData_;

    // Architecture specification
    std::vector<int> hiddenLayerSizes_;
    std::vector<std::string> hiddenActivations_;
    std::string outputActivation_;

    // Created model
    NeuralNetworkWrapper* model_;
};

#endif // NETWORKARCHITECTUREDIALOG_H
