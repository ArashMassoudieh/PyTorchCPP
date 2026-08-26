#include "../dataset/gistohq_package_adapter.h"

#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>

int main() {
    const std::filesystem::path root = "/tmp/gistohq_package_adapter";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root / "observations");
    { std::ofstream(root / "manifest.json") << R"({
"schema_name":"HydroPINNExport","schema_version":"1.1","profile":"water-balance-v1",
"site_id":"sligocreekdemo","start_date":"2024-01-01","end_date":"2024-01-01",
"catchment_area_m2":1000000})"; }
    {
        std::ofstream variables(root / "variables.json");
        variables << R"([
{"name":"PRECTOTCORR","unit":"mm/day"},{"name":"T2M","unit":"degC"},
{"name":"RH2M","unit":"%"},{"name":"WS2M","unit":"m/s"},
{"name":"ALLSKY_SFC_SW_DWN","unit":"MJ/m2/h"},
{"name":"EVPTRNS","unit":"MJ/m2/day"},{"name":"00060","unit":"ft3/s"}])";
    }
    {
        std::ofstream weather(root / "observations/temporal_1.csv");
        weather << "timestamp,PRECTOTCORR,T2M,RH2M,WS2M,ALLSKY_SFC_SW_DWN\n"
                << "2024-01-01T00:00:00Z,24,10,50,2,1\n"
                << "2024-01-01T01:00:00Z,48,11,51,3,2\n";
        std::ofstream(root / "observations/temporal_2.csv")
            << "timestamp,EVPTRNS\n2024-01-01T00:00:00Z,58.8\n";
        std::ofstream discharge(root / "observations/temporal_3.csv");
        discharge << "timestamp,00060\n";
        for (int minute = 0; minute < 120; minute += 5) {
            const int hour = minute / 60, within = minute % 60;
            discharge << "2024-01-01T0" << hour << ':'
                      << (within < 10 ? "0" : "") << within << ":00Z,10\n";
        }
    }
    GisToOhqPackageConfig config;
    config.start_epoch_seconds = 1704067200;
    config.end_epoch_seconds = config.start_epoch_seconds + 7200;
    config.catchment_area_m2 = 1.0e6;
    const auto prepared = prepareGisToOhqPackage(root.string(), config);
    assert(prepared.has_observed_discharge);
    assert(prepared.hourly_rows.size() == 2 && prepared.model_rows.size() == 2);
    assert(prepared.hourly_rows[0].precipitation_mm_per_hour == 1.0);
    assert(std::abs(prepared.hourly_rows[0].pet_mm_per_hour - 1.0) < 1.0e-12);
    assert(isGisToOhqHydroPinnExport(root.string()));
    const auto preparedFromManifest = prepareGisToOhqPackage(root.string(), true);
    assert(preparedFromManifest.model_rows.size() == 2);

    std::filesystem::remove(root / "observations/temporal_3.csv");
    auto variables = std::ifstream(root / "variables.json");
    std::string text((std::istreambuf_iterator<char>(variables)), {});
    const auto dischargeDeclaration = text.find(",{\"name\":\"00060\"");
    assert(dischargeDeclaration != std::string::npos);
    text.erase(dischargeDeclaration, text.find('}', dischargeDeclaration) - dischargeDeclaration + 1);
    std::ofstream(root / "variables.json", std::ios::trunc) << text;
    config.require_observed_discharge = false;
    const auto weatherOnly = prepareGisToOhqPackage(root.string(), config);
    assert(!weatherOnly.has_observed_discharge && weatherOnly.model_rows.size() == 2);
    std::filesystem::remove_all(root);
}
