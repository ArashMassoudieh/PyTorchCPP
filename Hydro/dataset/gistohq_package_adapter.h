#pragma once

#include "gistohq_model_rows.h"

#include <string>
#include <vector>

struct GisToOhqPackageConfig {
    std::int64_t start_epoch_seconds = 0;
    std::int64_t end_epoch_seconds = 0;
    double catchment_area_m2 = 0.0;
    bool require_observed_discharge = true;
    double discharge_sample_support_seconds = 300.0;
    double minimum_discharge_coverage = 0.75;
    double latent_heat_mj_per_kg = 2.45;
};

struct GisToOhqPreparedPackage {
    std::vector<GisToOhqHourlyRow> hourly_rows;
    std::vector<GisToOhqModelRow> model_rows;
    bool has_observed_discharge = false;
};

/** Validates and prepares a native GIStoOHQ temporal export for HydroPINN. */
GisToOhqPreparedPackage prepareGisToOhqPackage(
    const std::string& package_directory,
    const GisToOhqPackageConfig& config);

/** Returns true when manifest.json declares schema_name=HydroPINNExport. */
bool isGisToOhqHydroPinnExport(const std::string& package_directory);

/** Reads study bounds and catchment area from the producer manifest and prepares the package. */
GisToOhqPreparedPackage prepareGisToOhqPackage(
    const std::string& package_directory,
    bool require_observed_discharge);
