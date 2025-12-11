#include "mainwindow.h"
#include <QApplication>
#include <ATen/ATen.h>

int main(int argc, char *argv[])
{
    // Configure PyTorch threading
    at::set_num_threads(1);         // intra-op threads
    at::set_num_interop_threads(1); // inter-op threads

    QApplication app(argc, argv);
    
    // Set application information
    app.setApplicationName("NeuroForge");
    app.setApplicationVersion("1.0");
    app.setOrganizationName("EnviroInformatics LLC");
    app.setApplicationDisplayName("NeuroForge - Neural Network GA Optimizer");
    
    MainWindow window;
    window.show();
    
    return app.exec();
}
