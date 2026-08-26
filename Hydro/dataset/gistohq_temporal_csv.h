#pragma once

#include "gistohq_hourly_harmonizer.h"

#include <string>
#include <vector>

/**
 * Loads GIStoOHQ temporal CSV assets in either wide form
 * (`timestamp,<variable>...`) or long form (`timestamp,variable,value`).
 * Empty values remain absent; they are never imputed.
 */
GisToOhqHourlyInputs loadGisToOhqTemporalCsvFiles(const std::vector<std::string>& paths);
