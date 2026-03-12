#include "hydropinnwindow.h"

#include <QApplication>
#include <torch/torch.h>

int main(int argc, char *argv[])
{
    torch::set_num_threads(1);
    torch::set_num_interop_threads(1);

    QApplication app(argc, argv);

    app.setApplicationName("HydroPINN");
    app.setApplicationVersion("0.1");
    app.setOrganizationName("EnviroInformatics LLC");
    app.setApplicationDisplayName("HydroPINN - Physics-Informed Hydrology");

    HydroPINNWindow window;
    window.show();

    return app.exec();
}
