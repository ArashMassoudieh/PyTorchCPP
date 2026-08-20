#include "../dataset/ddrr_loader.h"
#include "../dataset/hydro_checksum.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <filesystem>
#include <stdexcept>

int main() {
    const std::string path = "/tmp/hydro_loader_observations.csv";
    {
        std::ofstream out(path);
        out << "timestamp,catchment_id,precipitation,potential_et,observed_discharge,storage\n";
        for (int hour = 0; hour < 3; ++hour) {
            const std::string time = "2024-01-01T0" + std::to_string(hour) + ":00:00Z";
            out << time << ",a,1,0.1,1,10\n"
                << time << ",b,2,0.2,1,20\n";
        }
    }
    DDRRLoader loader;
    assert(loader.load(path));
    const auto dataset = loader.loadObservations(path, {{"a", 1.0e6}, {"b", 2.0e6}},
                                                  HydroDatasetContract::waterBalanceV1());
    assert(dataset.observations_by_catchment.size() == 2);
    const auto& a = dataset.observations_by_catchment.at("a");
    const auto& b = dataset.observations_by_catchment.at("b");
    assert(a.size() == 3 && b.size() == 3);
    assert(a[0].elapsed_hours == 0.0 && a[2].elapsed_hours == 2.0);
    assert(b[0].elapsed_hours == 0.0 && b[2].elapsed_hours == 2.0);
    assert(std::abs(a[0].observed_runoff_mm_per_hour - 3.6) < 1.0e-12);
    assert(std::abs(b[0].observed_runoff_mm_per_hour - 1.8) < 1.0e-12);
    assert(a[0].storage_mm.has_value() && *a[0].storage_mm == 10.0);

    bool rejected = false;
    try {
        (void)loader.loadObservations(path, {{"a", 1.0e6}});
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    assert(rejected);

    const std::string fractionalPath = "/tmp/hydro_loader_fractional.csv";
    {
        std::ofstream out(fractionalPath);
        out << "timestamp,catchment_id,precipitation,potential_et,observed_discharge\n"
            << "2024-01-01T00:00:00.000Z,a,1,0.1,1\n"
            << "2024-01-01T00:00:00.500Z,a,1,0.1,1\n"
            << "2024-01-01T00:00:01.000Z,a,1,0.1,1\n";
    }
    const auto fractional = loader.loadObservations(fractionalPath, {{"a", 1.0e6}});
    assert(std::abs(fractional.observations_by_catchment.at("a")[1].elapsed_hours - 0.5 / 3600.0) < 1.0e-12);

    const std::filesystem::path package = "/tmp/hydro_loader_package";
    std::filesystem::remove_all(package);
    std::filesystem::create_directories(package);
    std::filesystem::copy_file(path, package / "observations.csv");
    {
        std::ofstream out(package / "manifest.json");
        out << R"({
  "schema_name": "hydro-observations",
  "schema_version": "1.0.0",
  "profile": "water-balance",
  "dataset_id": "two-catchment-test",
  "observations_file": "observations.csv",
  "catchment_attributes_file": "catchment_attributes.csv",
  "variables_file": "variables.json",
  "quality_control_file": "quality_control.csv"
})";
    }
    {
        std::ofstream out(package / "quality_control.csv");
        out << "rule_id,severity,message\n"
            << "coverage,warning,Short test fixture\n";
    }
    {
        std::ofstream out(package / "catchment_attributes.csv");
        out << "catchment_id,area_m2,mean_slope\n"
            << "a,1000000,0.02\n"
            << "b,2000000,0.03\n";
    }
    {
        std::ofstream out(package / "variables.json");
        out << R"([
  {"name":"timestamp","unit":"UTC ISO-8601"},
  {"name":"catchment_id","unit":"1"},
  {"name":"precipitation","unit":"mm/h"},
  {"name":"potential_et","unit":"mm/h"},
  {"name":"observed_discharge","unit":"m3/s"},
  {"name":"storage","unit":"mm"}
])";
    }
    const auto packaged = loader.loadPackageDirectory(package.string(), HydroDatasetContract::waterBalanceV1());
    assert(packaged.dataset_id == "two-catchment-test");
    assert(packaged.schema_version == "1.0.0");
    assert(packaged.profile == "water-balance");
    assert(packaged.catchment_area_m2.at("a") == 1.0e6);
    assert(packaged.variable_units.at("observed_discharge") == "m3/s");
    assert(packaged.observations_by_catchment.at("b").size() == 3);

    {
        std::ofstream out(package / "variables.json");
        out << R"([{"name":"timestamp","unit":"UTC ISO-8601"},{"name":"catchment_id","unit":"1"},{"name":"precipitation","unit":"mm/h"},{"name":"potential_et","unit":"mm/h"},{"name":"observed_discharge","unit":"ft3/s"},{"name":"storage","unit":"mm"}])";
    }
    rejected = false;
    try { (void)loader.loadPackageDirectory(package.string(), HydroDatasetContract::waterBalanceV1()); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
    {
        std::ofstream out(package / "variables.json");
        out << R"([{"name":"timestamp","unit":"UTC ISO-8601"},{"name":"catchment_id","unit":"1"},{"name":"precipitation","unit":"mm/h"},{"name":"potential_et","unit":"mm/h"},{"name":"observed_discharge","unit":"m3/s"},{"name":"storage","unit":"mm"}])";
    }

    const std::string observationsDigest = sha256File((package / "observations.csv").string());
    const std::string attributesDigest = sha256File((package / "catchment_attributes.csv").string());
    {
        std::ofstream out(package / "manifest.json");
        out << "{\"schema_name\":\"hydro-observations\",\"schema_version\":\"1.0.0\","
            << "\"profile\":\"water-balance\",\"dataset_id\":\"checksummed\","
            << "\"observations_file\":\"observations.csv\",\"catchment_attributes_file\":\"catchment_attributes.csv\","
            << "\"quality_control_file\":\"quality_control.csv\","
            << "\"observations_sha256\":\"" << observationsDigest << "\","
            << "\"catchment_attributes_sha256\":\"" << attributesDigest << "\"}";
    }
    assert(loader.loadPackageDirectory(package.string(), HydroDatasetContract::waterBalanceV1()).dataset_id == "checksummed");
    {
        std::ofstream out(package / "observations.csv", std::ios::app);
        out << '\n';
    }
    rejected = false;
    try { (void)loader.loadPackageDirectory(package.string(), HydroDatasetContract::waterBalanceV1()); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
    std::filesystem::copy_file(path, package / "observations.csv", std::filesystem::copy_options::overwrite_existing);

    // Two-digit incompatible major versions must not pass by sharing a first digit.
    {
        std::ofstream out(package / "manifest.json");
        out << R"({"schema_name":"hydro-observations","schema_version":"10.0.0","profile":"water-balance","dataset_id":"bad-version","observations_file":"observations.csv","catchment_attributes_file":"catchment_attributes.csv"})";
    }
    rejected = false;
    try {
        (void)loader.loadPackageDirectory(package.string(), HydroDatasetContract::waterBalanceV1());
    } catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);

    // A filename containing two dots is safe; only a literal parent component is traversal.
    std::filesystem::copy_file(path, package / "observations..csv");
    {
        std::ofstream out(package / "manifest.json");
        out << R"({"schema_name":"hydro-observations","schema_version":"1.0.0","profile":"water-balance","dataset_id":"safe-dots","observations_file":"observations..csv","catchment_attributes_file":"catchment_attributes.csv"})";
    }
    const auto safeDots = loader.loadPackageDirectory(package.string(), HydroDatasetContract::waterBalanceV1());
    assert(safeDots.dataset_id == "safe-dots");

    {
        std::ofstream out(package / "manifest.json");
        out << R"({"schema_name":"hydro-observations","schema_version":"1.0.0","profile":"water-balance","dataset_id":"traversal","observations_file":"../observations.csv","catchment_attributes_file":"catchment_attributes.csv"})";
    }
    rejected = false;
    try { (void)loader.loadManifest((package / "manifest.json").string()); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);


    {
        std::ofstream out(package / "catchment_attributes.csv");
        out << "catchment_id,area_m2\n"
            << "a,-1\n"
            << "b,2000000\n";
    }
    rejected = false;
    try {
        (void)loader.loadPackageDirectory(package.string());
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    assert(rejected);

    {
        std::ofstream out(package / "catchment_attributes.csv");
        out << "catchment_id,area_m2\n"
            << "a,1000000\n"
            << "b,2000000\n";
    }
    {
        std::ofstream out(package / "quality_control.csv");
        out << "rule_id,severity,message\n"
            << "missing_discharge,error,Target is incomplete\n";
    }
    rejected = false;
    try {
        (void)loader.loadPackageDirectory(package.string(), HydroDatasetContract::waterBalanceV1());
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    assert(rejected);
    std::remove(path.c_str());
    std::remove(fractionalPath.c_str());
    std::filesystem::remove_all(package);
    return 0;
}
