#pragma once

#include <QMainWindow>

class QLabel;
class QComboBox;
class QPushButton;

class HydroPINNWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit HydroPINNWindow(QWidget* parent = nullptr);

private:
    QLabel* statusLabel_;
    QComboBox* modeCombo_;
    QPushButton* runButton_;

    void updateStatus();
};
