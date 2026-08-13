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
    std::map<std::string, double> catchment_area_m2;
    std::map<std::string, std::vector<HydroObservation>> observations_by_catchment;
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
};
