#include <QAction>
#include <QApplication>
#include <QMainWindow>
#include <QMenu>
#include <QMenuBar>
#include <QTimer>
#include <QToolBar>

namespace {

QMainWindow* mainWindow()
{
    for (QWidget* widget : QApplication::topLevelWidgets()) {
        if (auto* window = qobject_cast<QMainWindow*>(widget)) return window;
    }
    return nullptr;
}

QMenu* batchMenu(QMainWindow* window)
{
    if (!window || !window->menuBar()) return nullptr;
    for (QAction* action : window->menuBar()->actions()) {
        if (action && action->menu() && QString(action->text()).remove('&') == "Batch")
            return action->menu();
    }
    return nullptr;
}

void removeActionByText(QMenu* menu, const QString& text)
{
    if (!menu) return;
    const auto actions = menu->actions();
    for (QAction* action : actions) {
        if (!action) continue;
        if (QString(action->text()).remove('&') == text) {
            menu->removeAction(action);
            action->deleteLater();
        }
    }
}

void removeActionByText(QToolBar* toolbar, const QString& text)
{
    if (!toolbar) return;
    const auto actions = toolbar->actions();
    for (QAction* action : actions) {
        if (!action) continue;
        if (QString(action->text()).remove('&') == text) {
            toolbar->removeAction(action);
            action->deleteLater();
        }
    }
}

void normalizeSeparators(QMenu* menu)
{
    if (!menu) return;
    bool previousSeparator = true;
    const auto actions = menu->actions();
    for (QAction* action : actions) {
        if (!action) continue;
        if (action->isSeparator()) {
            if (previousSeparator) {
                menu->removeAction(action);
                action->deleteLater();
            }
            previousSeparator = true;
        } else {
            previousSeparator = false;
        }
    }
    const auto remaining = menu->actions();
    if (!remaining.isEmpty() && remaining.last()->isSeparator()) {
        QAction* last = remaining.last();
        menu->removeAction(last);
        last->deleteLater();
    }
}

void cleanUi()
{
    QMainWindow* window = mainWindow();
    if (!window) {
        QTimer::singleShot(100, [](){ cleanUi(); });
        return;
    }

    QMenu* menu = batchMenu(window);
    if (!menu) {
        QTimer::singleShot(100, [](){ cleanUi(); });
        return;
    }

    // Legacy supervised-only entry is fully superseded by Quick Sweep and
    // the unified five-method Sweep Manager.
    removeActionByText(menu, "Tuning Sweep...");

    // Defensive cleanup for older builds that may still inject these entries.
    removeActionByText(menu, "Sweep Presets");
    removeActionByText(menu, "Stage 2 Configure...");
    removeActionByText(menu, "Stage 3 Multi-seed Robustness...");

    // Make the remaining expert entry explicit.
    for (QAction* action : menu->actions()) {
        if (action && QString(action->text()).remove('&') == "Sweep Manager...") {
            action->setText("Advanced Sweep Manager...");
            action->setToolTip("Configure custom grids for FFN, FFN + PINN, LSTM, LSTM + PINN, and PINN.");
        }
    }

    if (QToolBar* toolbar = window->findChild<QToolBar*>("HydroBatchToolBar")) {
        removeActionByText(toolbar, "Tuning Sweep...");
        removeActionByText(toolbar, "Sweep Presets");
    }

    normalizeSeparators(menu);
}

void scheduleCleanup()
{
    // Run after startup-injected actions have had time to install.
    QTimer::singleShot(300, [](){ cleanUi(); });
}

} // namespace

Q_COREAPP_STARTUP_FUNCTION(scheduleCleanup)
