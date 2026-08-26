#include "../dataset/gistohq_temporal_csv.h"

#include <cassert>
#include <filesystem>
#include <fstream>

int main() {
    const auto root = std::filesystem::path("/tmp/gistohq_temporal_csv_fixture");
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    const auto weather = root / "temporal_1.csv";
    const auto pet = root / "temporal_2.csv";
    const auto discharge = root / "temporal_3.csv";
    {
        std::ofstream out(weather, std::ios::binary);
        out << "timestamp,PRECTOTCORR,T2M,RH2M,WS2M,ALLSKY_SFC_SW_DWN\r\n"
            << "2024-01-01T00:00:00Z,24,10,50,2,0.5\r\n"
            << "2024-01-01T01:00:00.000Z,48,11,,3,0.6\r\n";
    }
    {
        std::ofstream out(pet);
        out << "timestamp_utc,variable,value,native_unit,provider_qualifiers\n"
            << "2024-01-01T00:00:00Z,EVPTRNS,58.8,MJ/m^2/day,\n";
    }
    {
        std::ofstream out(discharge);
        out << "timestamp_utc,variable,value,native_unit,provider_qualifiers\n"
            << "2024-01-01T01:00:00Z,00060,100,ft3/s,A\n"
            << "2024-01-01T01:05:00Z,00060,101,ft3/s,A\n";
    }
    const auto inputs = loadGisToOhqTemporalCsvFiles(
        {discharge.string(), weather.string(), pet.string()});
    assert(inputs.precipitation_mm_per_day.size() == 2);
    assert(inputs.temperature_c.size() == 2);
    assert(inputs.relative_humidity_percent.size() == 1); // empty value remains missing
    assert(inputs.pet_energy_mj_per_m2_per_day.size() == 1);
    assert(inputs.discharge_ft3_per_second.size() == 2);
    assert(inputs.discharge_ft3_per_second.front().value == 100.0);

    {
        std::ofstream out(root / "duplicate.csv");
        out << "timestamp,00060\n2024-01-01T01:00:00Z,99\n";
    }
    bool rejectedDuplicate = false;
    try {
        (void)loadGisToOhqTemporalCsvFiles({discharge.string(), (root / "duplicate.csv").string()});
    } catch (const std::runtime_error&) { rejectedDuplicate = true; }
    assert(rejectedDuplicate);

    {
        std::ofstream out(root / "bad_time.csv");
        out << "timestamp,T2M\n2024-02-30T00:00:00Z,10\n";
    }
    bool rejectedCalendar = false;
    try { (void)loadGisToOhqTemporalCsvFiles({(root / "bad_time.csv").string()}); }
    catch (const std::runtime_error&) { rejectedCalendar = true; }
    assert(rejectedCalendar);
    std::filesystem::remove_all(root);
    return 0;
}
