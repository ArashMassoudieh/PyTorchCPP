#pragma once

#include "hydro_dataset_contract.h"

#include <map>
#include <optional>
#include <string>
#include <vector>

struct HydroObservation {
    std::string timestamp;
    std::string catchment_id;
    double elapsed_hours = 0.0;
    double precipitation_mm_per_hour = 0.0;
    double potential_et_mm_per_hour = 0.0;
    double observed_discharge_m3_per_second = 0.0;
    double observed_runoff_mm_per_hour = 0.0;
    std::optional<double> storage_mm;
};

struct HydroObservationDataset {
    std::string dataset_id;
    std::string schema_version;
    std::string profile;
    std::map<std::string, double> catchment_area_m2;
    std::map<std::string, std::string> variable_units;
    std::map<std::string, std::vector<HydroObservation>> observations_by_catchment;
};

struct HydroForecast {
    std::string issue_time;
    std::string valid_time;
    double lead_hours = 0.0;
    std::string catchment_id;
    std::string variable;
    double value = 0.0;
    std::string unit;
    std::string forecast_model;
    std::string model_cycle;
    std::string ensemble_member;
};

struct HydroPackageManifest {
    std::string schema_name;
    std::string schema_version;
    std::string profile;
    std::string dataset_id;
    std::string observations_file;
    std::string catchment_attributes_file;
    std::string quality_control_file;
    std::string variables_file;
    std::string forecast_file;
    std::string observations_sha256;
    std::string catchment_attributes_sha256;
    std::string variables_sha256;
    std::string forecast_sha256;
};

/** Loads validated generic hydro-observation CSV exports. */
class DDRRLoader {
public:
    bool load(const std::string& path);
    HydroObservationDataset loadObservations(
        const std::string& path,
        const std::map<std::string, double>& catchmentAreasM2,
        const HydroDatasetContract& contract = HydroDatasetContract::rainfallRunoffV1()) const;
    HydroObservationDataset loadPackageDirectory(
        const std::string& packageDirectory,
        const HydroDatasetContract& contract = HydroDatasetContract::rainfallRunoffV1()) const;
    std::vector<HydroForecast> loadForecasts(
        const std::string& path,
        const std::string& predictionTime) const;
    HydroPackageManifest loadManifest(const std::string& manifestPath) const;
};
