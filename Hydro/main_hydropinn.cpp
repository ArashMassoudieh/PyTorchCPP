#include "hydropinnwindow.h"

#include <QApplication>
#include <QComboBox>
#include <QMessageBox>
#include <QPushButton>
#include <QSurfaceFormat>
#include <torch/torch.h>

#include <exception>
#include <iostream>

namespace {
QComboBox* findDataSourceCombo(HydroPINNWindow& window)
{
    const auto combos = window.findChildren<QComboBox*>();
    for (QComboBox* combo : combos) {
        if (combo->findText("Synthetic") >= 0 &&
            combo->findText("CSV File") >= 0 &&
            combo->findText("Hydro Package") >= 0) {
            return combo;
        }
    }
    return nullptr;
}

QPushButton* findButtonByText(HydroPINNWindow& window, const QString& text)
{
    const auto buttons = window.findChildren<QPushButton*>();
    for (QPushButton* button : buttons) {
        if (button->text() == text) return button;
    }
    return nullptr;
}

void configureContextSpecificPlotButtons(HydroPINNWindow& window)
{
    QComboBox* dataSource = findDataSourceCombo(window);
    QPushButton* inputsOutput = findButtonByText(window, "Show Inputs + Output");
    QPushButton* cumulativePhysics = findButtonByText(window, "Cumulative Physics Residual");

    if (inputsOutput) {
        inputsOutput->setText("Synthetic Inputs + Output");
        inputsOutput->setToolTip(
            "Plots the generated synthetic input channels and output. "
            "This action is intentionally unavailable for CSV and Hydro Package data; "
            "use target/predicted and hydrologic diagnostic plots for observed packages.");

        auto refreshInputsButton = [dataSource, inputsOutput]() {
            if (!dataSource) return;
            const bool synthetic = dataSource->currentText() == "Synthetic";
            inputsOutput->setEnabled(synthetic);
        };
        refreshInputsButton();
        if (dataSource) {
            QObject::connect(dataSource, &QComboBox::currentTextChanged,
                             inputsOutput, [refreshInputsButton](const QString&) {
                                 refreshInputsButton();
                             });
        }
    }

    if (cumulativePhysics) {
        cumulativePhysics->setText("Cumulative Physics Residual (PINN only)");
        cumulativePhysics->setToolTip(
            "Requires a successful physics-informed run with stored physics residuals. "
            "Plain FFN/LSTM runs do not produce this series. Current GIStoOHQ rainfall-runoff "
            "exports intentionally block PINN approaches until a separate rainfall-runoff "
            "physics profile is versioned.");
    }
}
}

int main(int argc, char *argv[])
{
#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
    // Must be set before QApplication is constructed.
    QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
    QCoreApplication::setAttribute(Qt::AA_UseHighDpiPixmaps);
#endif

    try {
        // Keep LibTorch conservative inside a GUI app.
        // This avoids CPU over-subscription when Qt, OpenMP, and Torch are all active.
        torch::set_num_threads(1);
        torch::set_num_interop_threads(1);

        QApplication app(argc, argv);

        QCoreApplication::setApplicationName("HydroPINN");
        QCoreApplication::setApplicationVersion("0.1");
        QCoreApplication::setOrganizationName("EnviroInformatics LLC");
        QApplication::setApplicationDisplayName("HydroPINN - Physics-Informed Hydrology");

        HydroPINNWindow window;
        configureContextSpecificPlotButtons(window);
        window.show();

        return app.exec();
    }
    catch (const c10::Error &e) {
        std::cerr << "LibTorch error:\n" << e.what() << std::endl;
        QMessageBox::critical(nullptr, "HydroPINN - LibTorch error", QString::fromStdString(e.what()));
        return EXIT_FAILURE;
    }
    catch (const std::exception &e) {
        std::cerr << "Application error:\n" << e.what() << std::endl;
        QMessageBox::critical(nullptr, "HydroPINN - Error", QString::fromUtf8(e.what()));
        return EXIT_FAILURE;
    }
    catch (...) {
        std::cerr << "Unknown application error." << std::endl;
        QMessageBox::critical(nullptr, "HydroPINN - Error", "Unknown application error.");
        return EXIT_FAILURE;
    }
}
