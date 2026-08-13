#include "../dataset/hydro_dataset_contract.h"

#include <cassert>
#include <cstdio>
#include <fstream>

int main() {
    const std::string validPath = "/tmp/hydropinn_contract_valid.csv";
    {
        std::ofstream out(validPath);
        out << "timestamp,catchment_id,precipitation,potential_et,observed_discharge,storage\n"
            << "2024-01-01T00:00:00Z,hickey_run,0,0.1,1.2,10\n"
            << "2024-01-01T01:00:00Z,hickey_run,2,0.1,1.4,10.5\n"
            << "2024-01-01T02:00:00Z,hickey_run,0,0.1,1.3,10.4\n";
    }
    HydroDatasetValidator validator;
    const auto valid = validator.validateCsv(validPath, HydroDatasetContract::observationsV1());
    assert(valid.valid);
    assert(valid.row_count == 3);
    const auto rainfallRunoff = HydroDatasetContract::rainfallRunoffV1();
    const auto waterBalance = HydroDatasetContract::waterBalanceV1();
    assert(rainfallRunoff.schema_name == "hydro-observations");
    assert(rainfallRunoff.profile == "rainfall-runoff");
    assert(waterBalance.profile == "water-balance");

    const std::string genericPath = "/tmp/hydro_contract_generic.csv";
    {
        std::ofstream out(genericPath);
        out << "timestamp,catchment_id,precipitation,potential_et,observed_discharge\n"
            << "2024-01-01T00:00:00Z,watershed_a,0,0.1,1.2\n"
            << "2024-01-01T01:00:00Z,watershed_a,1,0.1,1.3\n"
            << "2024-01-01T02:00:00Z,watershed_a,0,0.1,1.1\n";
    }
    assert(validator.validateCsv(genericPath, rainfallRunoff).valid);
    assert(!validator.validateCsv(genericPath, waterBalance).valid);

    const std::string multiPath = "/tmp/hydropinn_contract_multi.csv";
    {
        std::ofstream out(multiPath);
        out << "timestamp,catchment_id,precipitation,potential_et,observed_discharge,storage\n";
        for (int hour = 0; hour < 3; ++hour) {
            const std::string time = "2024-01-01T0" + std::to_string(hour) + ":00:00Z";
            out << time << ",sub_1,0,0.1,1.2,10\n"
                << time << ",sub_2,0,0.1,1.1,9\n";
        }
    }
    const auto multi = validator.validateCsv(multiPath, HydroDatasetContract::observationsV1());
    assert(multi.valid);
    assert(multi.row_count == 6);

    const std::string invalidPath = "/tmp/hydropinn_contract_invalid.csv";
    {
        std::ofstream out(invalidPath);
        out << "timestamp,catchment_id,precipitation,potential_et,observed_discharge\n"
            << "2024-01-01T01:00:00Z,hickey_run,-1,0.1,1.2\n"
            << "2024-01-01T00:00:00Z,hickey_run,0,0.1,1.3\n"
            << "2024-01-01T02:00:00Z,hickey_run,0,0.1,1.4\n";
    }
    const auto invalid = validator.validateCsv(invalidPath, HydroDatasetContract::observationsV1());
    assert(!invalid.valid);
    assert(!invalid.errors.empty());

    const std::string duplicatePath = "/tmp/hydropinn_contract_duplicate.csv";
    {
        std::ofstream out(duplicatePath);
        out << "timestamp,catchment_id,precipitation,potential_et,observed_discharge,storage\n"
            << "not-a-time,hickey_run,0,0.1,1.2,10\n"
            << "2024-01-01T01:00:00Z,hickey_run,0,0.1,1.3,10\n"
            << "2024-01-01T01:00:00Z,hickey_run,0,0.1,1.4,10\n";
    }
    const auto duplicate = validator.validateCsv(duplicatePath, HydroDatasetContract::observationsV1());
    assert(!duplicate.valid);
    assert(duplicate.errors.size() >= 2);
    std::remove(validPath.c_str());
    std::remove(multiPath.c_str());
    std::remove(genericPath.c_str());
    std::remove(invalidPath.c_str());
    std::remove(duplicatePath.c_str());
    return 0;
}
