#include "batch_results_summary.h"

#include <QDialog>
#include <QFile>
#include <QFileInfo>
#include <QHeaderView>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QTableWidget>
#include <QTextStream>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

QStringList parseCsvLine(const QString& line)
{
    QStringList fields;
    QString current;
    bool quoted = false;
    for (int i = 0; i < line.size(); ++i) {
        const QChar c = line.at(i);
        if (c == '"') {
            if (quoted && i + 1 < line.size() && line.at(i + 1) == '"') {
                current += '"';
                ++i;
            } else {
                quoted = !quoted;
            }
        } else if (c == ',' && !quoted) {
            fields << current;
            current.clear();
        } else {
            current += c;
        }
    }
    fields << current;
    return fields;
}

double numberOrNan(const QString& text)
{
    bool ok = false;
    const double value = text.trimmed().toDouble(&ok);
    return ok ? value : std::numeric_limits<double>::quiet_NaN();
}

QString fmt(double value, int precision = 3)
{
    if (!std::isfinite(value)) return "—";
    return QString::number(value, 'f', precision);
}

QString methodLabel(const QString& mode)
{
    if (mode == "ffn") return "FFN";
    if (mode == "ffn_pinn") return "FFN + PINN";
    if (mode == "lstm") return "LSTM";
    if (mode == "lstm_pinn") return "LSTM + PINN";
    if (mode == "pinn") return "PINN";
    return mode;
}

struct ResultRow {
    QString experimentId;
    QString mode;
    QString hidden;
    QString sequence;
    QString activation;
    double rmse = std::numeric_limits<double>::quiet_NaN();
    double mae = std::numeric_limits<double>::quiet_NaN();
    double r2 = std::numeric_limits<double>::quiet_NaN();
    double kge = std::numeric_limits<double>::quiet_NaN();
    double pbias = std::numeric_limits<double>::quiet_NaN();
    double physicsRmse = std::numeric_limits<double>::quiet_NaN();
    bool success = false;
};

QString statusFor(const ResultRow& r, bool best)
{
    if (!r.success) return "Failed";
    if (best) return "Best";
    if (!std::isfinite(r.r2)) return "Check metrics";
    if (r.r2 >= 0.50 && (!std::isfinite(r.kge) || r.kge >= 0.50)) return "Good";
    if (r.r2 >= 0.0) return "Fair";
    return "Needs tuning";
}

bool better(const ResultRow& a, const ResultRow& b)
{
    if (a.success != b.success) return a.success;
    const bool ar2 = std::isfinite(a.r2);
    const bool br2 = std::isfinite(b.r2);
    if (ar2 != br2) return ar2;
    if (ar2 && std::abs(a.r2 - b.r2) > 1e-12) return a.r2 > b.r2;
    const bool ak = std::isfinite(a.kge);
    const bool bk = std::isfinite(b.kge);
    if (ak != bk) return ak;
    if (ak && std::abs(a.kge - b.kge) > 1e-12) return a.kge > b.kge;
    const bool ap = std::isfinite(a.pbias);
    const bool bp = std::isfinite(b.pbias);
    if (ap != bp) return ap;
    if (ap && std::abs(std::abs(a.pbias) - std::abs(b.pbias)) > 1e-12)
        return std::abs(a.pbias) < std::abs(b.pbias);
    const bool armse = std::isfinite(a.rmse);
    const bool brmse = std::isfinite(b.rmse);
    if (armse != brmse) return armse;
    if (armse && std::abs(a.rmse - b.rmse) > 1e-12) return a.rmse < b.rmse;
    return a.experimentId < b.experimentId;
}

} // namespace

void showHydroBatchResultsSummary(QWidget* parent,
                                  const QString& summaryCsvPath,
                                  const QString& title)
{
    QFile file(summaryCsvPath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) return;
    QTextStream stream(&file);
    if (stream.atEnd()) return;

    const QStringList header = parseCsvLine(stream.readLine());
    auto column = [&](const QString& name) { return header.indexOf(name); };
    const int cExperiment = column("experiment_id");
    const int cMode = column("mode");
    const int cHidden = column("hidden_layers");
    const int cSequence = column("lstm_sequence_length");
    const int cActivation = column("activation");
    const int cSuccess = column("success");
    const int cRmse = column("rmse");
    const int cMae = column("mae");
    const int cR2 = column("r2");
    const int cNse = column("nse");
    const int cKge = column("kge");
    const int cPbias = column("pbias");
    const int cPhysicsRmse = column("physics_residual_rmse");

    QList<ResultRow> rows;
    while (!stream.atEnd()) {
        const QString line = stream.readLine();
        if (line.trimmed().isEmpty()) continue;
        const QStringList f = parseCsvLine(line);
        auto value = [&](int idx) -> QString { return idx >= 0 && idx < f.size() ? f.at(idx).trimmed() : QString(); };
        ResultRow r;
        r.experimentId = value(cExperiment);
        r.mode = value(cMode);
        r.hidden = value(cHidden);
        r.sequence = value(cSequence);
        r.activation = value(cActivation);
        const QString success = value(cSuccess).toLower();
        r.success = success == "yes" || success == "true" || success == "1";
        r.rmse = numberOrNan(value(cRmse));
        r.mae = numberOrNan(value(cMae));
        r.r2 = numberOrNan(value(cR2));
        if (!std::isfinite(r.r2)) r.r2 = numberOrNan(value(cNse));
        r.kge = numberOrNan(value(cKge));
        r.pbias = numberOrNan(value(cPbias));
        r.physicsRmse = numberOrNan(value(cPhysicsRmse));
        rows << r;
    }
    if (rows.isEmpty()) return;

    std::sort(rows.begin(), rows.end(), better);

    auto* dialog = new QDialog(parent);
    dialog->setAttribute(Qt::WA_DeleteOnClose);
    dialog->setWindowTitle(title.isEmpty() ? "HydroBatch Results Summary" : title);
    dialog->resize(1040, 560);
    auto* layout = new QVBoxLayout(dialog);

    auto* intro = new QLabel(
        "Ranked primarily by R²/NSE, then KGE, absolute PBIAS, and RMSE. "
        "Very low RMSE does not override negative R²/NSE or invalid KGE.", dialog);
    intro->setWordWrap(true);
    layout->addWidget(intro);

    auto* table = new QTableWidget(rows.size(), 10, dialog);
    table->setHorizontalHeaderLabels({"Rank","Method","Architecture","Memory","RMSE","MAE","R²/NSE","KGE","PBIAS %","Status"});
    table->setEditTriggers(QAbstractItemView::NoEditTriggers);
    table->setSelectionBehavior(QAbstractItemView::SelectRows);
    table->setAlternatingRowColors(true);
    for (int i = 0; i < rows.size(); ++i) {
        const ResultRow& r = rows.at(i);
        const QString architecture = r.hidden.isEmpty() ? "—" : r.hidden;
        const QString memory = r.sequence.isEmpty() || r.sequence == "0" ? "—" : r.sequence + " h";
        const QStringList values = {
            QString::number(i + 1), methodLabel(r.mode), architecture, memory,
            fmt(r.rmse, 5), fmt(r.mae, 5), fmt(r.r2, 3), fmt(r.kge, 3), fmt(r.pbias, 2), statusFor(r, i == 0)
        };
        for (int c = 0; c < values.size(); ++c) table->setItem(i, c, new QTableWidgetItem(values.at(c)));
    }
    table->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
    table->horizontalHeader()->setStretchLastSection(true);
    layout->addWidget(table, 1);

    const ResultRow& best = rows.first();
    auto* bestLabel = new QLabel(
        QString("Top ranked: %1  |  R²/NSE %2  |  KGE %3  |  PBIAS %4%  |  RMSE %5")
            .arg(methodLabel(best.mode), fmt(best.r2, 3), fmt(best.kge, 3), fmt(best.pbias, 2), fmt(best.rmse, 5)),
        dialog);
    bestLabel->setWordWrap(true);
    layout->addWidget(bestLabel);

    auto* row = new QHBoxLayout();
    auto* path = new QLabel("Summary: " + QFileInfo(summaryCsvPath).absoluteFilePath(), dialog);
    path->setTextInteractionFlags(Qt::TextSelectableByMouse);
    auto* close = new QPushButton("Close", dialog);
    row->addWidget(path, 1);
    row->addWidget(close);
    layout->addLayout(row);
    QObject::connect(close, &QPushButton::clicked, dialog, &QDialog::close);
    dialog->show();
}
