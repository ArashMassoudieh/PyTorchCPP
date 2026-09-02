#include "build_identity.h"

#include <QAction>
#include <QApplication>
#include <QMainWindow>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QTimer>

namespace {

void installBuildIdentity()
{
    QMainWindow* window = nullptr;
    for (QWidget* widget : QApplication::topLevelWidgets()) {
        if ((window = qobject_cast<QMainWindow*>(widget))) break;
    }
    if (!window) {
        QTimer::singleShot(100, [](){ installBuildIdentity(); });
        return;
    }
    if (window->property("hydro_build_identity_installed").toBool()) return;

    const QString identity = hydroBuildIdentity("HydroPINN");
    window->setWindowTitle("HydroPINN - Physics-Informed Hydrology | " + hydroBuildCommit());
    window->setProperty("hydro_build_identity", identity);

    QMenu* helpMenu = nullptr;
    if (window->menuBar()) {
        for (QAction* action : window->menuBar()->actions()) {
            if (action && action->menu() && QString(action->text()).remove('&').compare("Help", Qt::CaseInsensitive) == 0) {
                helpMenu = action->menu();
                break;
            }
        }
        if (!helpMenu) helpMenu = window->menuBar()->addMenu("Help");
    }

    if (helpMenu) {
        auto* action = helpMenu->addAction("Build Information...");
        action->setObjectName("HydroBuildInformationAction");
        QObject::connect(action, &QAction::triggered, window, [window, identity]() {
            QMessageBox::information(window, "HydroPINN Build Information", identity);
        });
    }

    window->setProperty("hydro_build_identity_installed", true);
}

void scheduleInstall()
{
    QTimer::singleShot(0, [](){ installBuildIdentity(); });
}

} // namespace

Q_COREAPP_STARTUP_FUNCTION(scheduleInstall)
