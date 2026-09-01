#pragma once

#include <QString>

class QWidget;

// Show a modeless ranked summary for a HydroBatch batch_summary.csv file.
// The dialog owns itself and can safely be called while a full pipeline
// continues with later stages.
void showHydroBatchResultsSummary(QWidget* parent,
                                  const QString& summaryCsvPath,
                                  const QString& title = QString());
