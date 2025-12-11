#include "GASettingsDialog.h"
#include <QVBoxLayout>
#include <QFormLayout>
#include <QGridLayout>
#include <QFileDialog>
#include <QMessageBox>
#include <QLabel>

GASettingsDialog::GASettingsDialog(GeneticAlgorithm<NeuralNetworkWrapper>* ga, QWidget* parent)
    : QDialog(parent)
    , ga_(ga)
{
    if (!ga_) {
        throw std::invalid_argument("GASettingsDialog: GA pointer cannot be null");
    }

    setWindowTitle("Genetic Algorithm Settings");
    setMinimumWidth(500);

    setupUI();
    loadSettings();

    connect(okButton_, &QPushButton::clicked, this, &GASettingsDialog::onAccept);
    connect(cancelButton_, &QPushButton::clicked, this, &GASettingsDialog::onReject);
    connect(browseOutputBtn_, &QPushButton::clicked, this, &GASettingsDialog::browseOutputPath);
}

GASettingsDialog::~GASettingsDialog()
{
}

void GASettingsDialog::setupUI()
{
    QVBoxLayout* mainLayout = new QVBoxLayout(this);

    mainLayout->addWidget(createPopulationGroup());
    mainLayout->addWidget(createGeneticOperatorsGroup());
    mainLayout->addWidget(createFilePathsGroup());

    // Buttons
    QHBoxLayout* buttonLayout = new QHBoxLayout();
    okButton_ = new QPushButton("OK", this);
    cancelButton_ = new QPushButton("Cancel", this);
    okButton_->setDefault(true);

    buttonLayout->addStretch();
    buttonLayout->addWidget(okButton_);
    buttonLayout->addWidget(cancelButton_);

    mainLayout->addLayout(buttonLayout);
}

QGroupBox* GASettingsDialog::createPopulationGroup()
{
    QGroupBox* group = new QGroupBox("Population Parameters", this);
    QFormLayout* layout = new QFormLayout(group);

    populationSpinBox_ = new QSpinBox(this);
    populationSpinBox_->setRange(10, 1000);
    populationSpinBox_->setSingleStep(10);
    layout->addRow("Population Size:", populationSpinBox_);

    generationsSpinBox_ = new QSpinBox(this);
    generationsSpinBox_->setRange(1, 1000);
    generationsSpinBox_->setSingleStep(10);
    layout->addRow("Generations:", generationsSpinBox_);

    return group;
}

QGroupBox* GASettingsDialog::createGeneticOperatorsGroup()
{
    QGroupBox* group = new QGroupBox("Genetic Operators", this);
    QFormLayout* layout = new QFormLayout(group);

    mutationProbSpinBox_ = new QDoubleSpinBox(this);
    mutationProbSpinBox_->setRange(0.0, 1.0);
    mutationProbSpinBox_->setSingleStep(0.01);
    mutationProbSpinBox_->setDecimals(4);
    layout->addRow("Mutation Probability:", mutationProbSpinBox_);

    return group;
}

QGroupBox* GASettingsDialog::createFilePathsGroup()
{
    QGroupBox* group = new QGroupBox("Output", this);
    QGridLayout* layout = new QGridLayout(group);

    QLabel* label = new QLabel("Output Path:", this);
    outputPathEdit_ = new QLineEdit(this);
    browseOutputBtn_ = new QPushButton("Browse...", this);

    layout->addWidget(label, 0, 0);
    layout->addWidget(outputPathEdit_, 0, 1);
    layout->addWidget(browseOutputBtn_, 0, 2);

    return group;
}

void GASettingsDialog::loadSettings()
{
    populationSpinBox_->setValue(ga_->Settings.totalpopulation);
    generationsSpinBox_->setValue(ga_->Settings.generations);
    mutationProbSpinBox_->setValue(ga_->Settings.mutation_probability);
    outputPathEdit_->setText(QString::fromStdString(ga_->Settings.outputpath));
}

void GASettingsDialog::saveSettings()
{
    ga_->Settings.totalpopulation = populationSpinBox_->value();
    ga_->Settings.generations = generationsSpinBox_->value();
    ga_->Settings.mutation_probability = mutationProbSpinBox_->value();
    ga_->Settings.outputpath = outputPathEdit_->text().toStdString();
}

bool GASettingsDialog::validateInputs()
{
    if (populationSpinBox_->value() < 10) {
        QMessageBox::warning(this, "Invalid Input",
                             "Population size must be at least 10.");
        return false;
    }
    return true;
}

void GASettingsDialog::onAccept()
{
    if (validateInputs()) {
        saveSettings();
        accept();
    }
}

void GASettingsDialog::onReject()
{
    reject();
}

void GASettingsDialog::browseOutputPath()
{
    QString dirname = QFileDialog::getExistingDirectory(
        this,
        "Select Output Directory",
        outputPathEdit_->text(),
        QFileDialog::ShowDirsOnly
        );

    if (!dirname.isEmpty()) {
        outputPathEdit_->setText(dirname);
    }
}
