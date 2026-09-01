#include "batch_results_summary.h"

#include <QApplication>
#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QMainWindow>
#include <QProcess>
#include <QSet>
#include <QTimer>

namespace {

QString findRepoRoot()
{
    const QStringList starts = {QDir::currentPath(), QCoreApplication::applicationDirPath()};
    for (const QString& start : starts) {
        QDir dir(start);
        for (int depth = 0; depth < 10; ++depth) {
            if (QFileInfo::exists(dir.filePath("HydroPINN.pro")) ||
                QFileInfo::exists(dir.filePath("HydroBatch.pro"))) return dir.absolutePath();
            if (!dir.cdUp()) break;
        }
    }
    return {};
}

void collectSummaries(const QString& root, QStringList* out)
{
    QDir dir(root);
    if (!dir.exists()) return;
    const QFileInfoList files = dir.entryInfoList({"batch_summary.csv"}, QDir::Files | QDir::NoSymLinks);
    for (const QFileInfo& info : files) out->append(info.absoluteFilePath());
    const QFileInfoList dirs = dir.entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot | QDir::NoSymLinks);
    for (const QFileInfo& info : dirs) collectSummaries(info.absoluteFilePath(), out);
}

QMainWindow* mainWindow()
{
    for (QWidget* widget : QApplication::topLevelWidgets()) {
        if (auto* window = qobject_cast<QMainWindow*>(widget)) return window;
    }
    return nullptr;
}

bool hydroBatchIsRunning()
{
    for (QWidget* widget : QApplication::topLevelWidgets()) {
        const auto processes = widget->findChildren<QProcess*>();
        for (QProcess* process : processes) {
            if (!process || process->state() == QProcess::NotRunning) continue;
            const QString program = QFileInfo(process->program()).fileName();
            if (program.compare("HydroBatch", Qt::CaseInsensitive) == 0) return true;
        }
    }
    return false;
}

void installWatcher()
{
    QMainWindow* window = mainWindow();
    if (!window) {
        QTimer::singleShot(150, [](){ installWatcher(); });
        return;
    }
    if (window->property("hydro_batch_results_watcher").toBool()) return;

    const QString repo = findRepoRoot();
    if (repo.isEmpty()) return;
    const QString experimentRoot = repo + "/Hydro/experiments";

    auto* seen = new QSet<QString>();
    QStringList existing;
    collectSummaries(experimentRoot, &existing);
    for (const QString& path : existing) seen->insert(path);

    auto* timer = new QTimer(window);
    timer->setInterval(1500);
    QObject::connect(timer, &QTimer::timeout, window, [window, experimentRoot, seen]() {
        // HydroBatch appends batch_summary.csv incrementally. Do not surface a
        // partial table while any GUI-launched HydroBatch process is active.
        if (hydroBatchIsRunning()) return;

        QStringList summaries;
        collectSummaries(experimentRoot, &summaries);
        for (const QString& path : summaries) {
            if (seen->contains(path)) continue;
            const QFileInfo info(path);
            if (!info.exists() || info.size() <= 0) continue;
            seen->insert(path);
            const QString stage = info.dir().dirName();
            showHydroBatchResultsSummary(window, path,
                                         stage.isEmpty() ? "HydroBatch Results Summary"
                                                         : "HydroBatch Results - " + stage);
        }
    });
    window->setProperty("hydro_batch_results_watcher", true);
    timer->start();
}

void scheduleInstall()
{
    QTimer::singleShot(0, [](){ installWatcher(); });
}

} // namespace

Q_COREAPP_STARTUP_FUNCTION(scheduleInstall)
