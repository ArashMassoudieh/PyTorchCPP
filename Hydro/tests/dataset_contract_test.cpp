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
    std::remove(validPath.c_str());
    std::remove(invalidPath.c_str());
    return 0;
}
