#include "hydropinnwindow.h"

#include <QComboBox>
#include <QLabel>
#include <QMainWindow>
#include <QPushButton>
#include <QVBoxLayout>
#include <QWidget>

HydroPINNWindow::HydroPINNWindow(QWidget* parent)
    : QMainWindow(parent), statusLabel_(new QLabel(this)), modeCombo_(new QComboBox(this)),
      runButton_(new QPushButton("Run", this)) {
    setWindowTitle("HydroPINN - Experiment Runner");
    resize(640, 240);

    auto* central = new QWidget(this);
    auto* layout = new QVBoxLayout(central);

    auto* title = new QLabel("HydroPINN Modes", central);
    title->setStyleSheet("font-size: 18px; font-weight: bold;");

    modeCombo_->addItems({"ffn", "ffn_pinn", "lstm", "lstm_pinn"});

    layout->addWidget(title);
    layout->addWidget(modeCombo_);
    layout->addWidget(runButton_);
    layout->addWidget(statusLabel_);
    layout->addStretch();

    setCentralWidget(central);

    connect(runButton_, &QPushButton::clicked, this, &HydroPINNWindow::updateStatus);
    connect(modeCombo_, &QComboBox::currentTextChanged, this, [this](const QString&) { updateStatus(); });

    updateStatus();
}

void HydroPINNWindow::updateStatus() {
    statusLabel_->setText(QString("Ready to run mode: %1").arg(modeCombo_->currentText()));
}
