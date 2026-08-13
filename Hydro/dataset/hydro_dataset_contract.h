#pragma once

#include <map>
#include <string>
#include <vector>

struct HydroVariableMetadata {
    std::string name;
    std::string unit;
    std::string role; // timestamp | forcing | target | state | identifier | forecast
    bool required = false;
    bool nonnegative = false;
};

struct HydroDatasetContract {
    std::string schema_name = "hydropinn-dataset";
    std::string schema_version = "1.0.0";
    std::vector<HydroVariableMetadata> variables;

    static HydroDatasetContract observationsV1(bool requireStorage = true);
};

struct HydroDatasetValidation {
    bool valid = false;
    std::size_t row_count = 0;
    std::vector<std::string> errors;
    std::vector<std::string> warnings;
};

class HydroDatasetValidator {
public:
    HydroDatasetValidation validateCsv(const std::string& path,
                                       const HydroDatasetContract& contract,
                                       bool hasHeader = true) const;
};
