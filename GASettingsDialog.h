#ifndef GASETTINGSDIALOG_H
#define GASETTINGSDIALOG_H

#include <QDialog>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QLineEdit>
#include <QPushButton>
#include <QGroupBox>
#include "ga.h"
#include "neuralnetworkwrapper.h"

class GASettingsDialog : public QDialog
{
    Q_OBJECT

public:
    explicit GASettingsDialog(GeneticAlgorithm<NeuralNetworkWrapper>* ga, QWidget* parent = nullptr);
    ~GASettingsDialog();

private Q_SLOTS:
    void onAccept();
    void onReject();
    void browseOutputPath();

private:
    void setupUI();
    void loadSettings();
    void saveSettings();
    bool validateInputs();

    QGroupBox* createPopulationGroup();
    QGroupBox* createGeneticOperatorsGroup();
    QGroupBox* createFilePathsGroup();

    GeneticAlgorithm<NeuralNetworkWrapper>* ga_;

    // UI controls
    QSpinBox* populationSpinBox_;
    QSpinBox* generationsSpinBox_;
    QDoubleSpinBox* mutationProbSpinBox_;
    QLineEdit* outputPathEdit_;
    QPushButton* browseOutputBtn_;
    QPushButton* okButton_;
    QPushButton* cancelButton_;
};

#endif // GASETTINGSDIALOG_H
