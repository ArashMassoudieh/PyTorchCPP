#include "hydropinnwindow.h"

#include <QApplication>
#include <ATen/ATen.h>

int main(int argc, char *argv[])
{
    at::set_num_threads(1);
    at::set_num_interop_threads(1);

    QApplication app(argc, argv);

    app.setApplicationName("HydroPINN");
    app.setApplicationVersion("0.1");
    app.setOrganizationName("EnviroInformatics LLC");
    app.setApplicationDisplayName("HydroPINN - Physics-Informed Hydrology");

    HydroPINNWindow window;
    window.show();

    return app.exec();
}
