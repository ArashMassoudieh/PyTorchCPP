#include "hydropinnwindow.h"

#include <QApplication>
#include <QMessageBox>
#include <QSurfaceFormat>
#include <torch/torch.h>

#include <exception>
#include <iostream>

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
