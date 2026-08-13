#include "../dataset/ddrr_loader.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
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
    std::remove(path.c_str());
    return 0;
}
