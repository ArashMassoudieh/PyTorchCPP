#include "hydrofusion_native_support_adapter.h"
#include <iostream>
#include <functional>
using namespace hydrofusion;
void require(bool ok) { if (!ok) throw std::runtime_error("test failure"); }
void rejects(const std::function<void()>& f) { bool failed=false; try { f(); } catch (const std::invalid_argument&) { failed=true; } require(failed); }
int main() {
    NativeAdapterContext c; c.end_x_exclusive=300; c.catchment_area_m2=2000000; c.expected_source_id="fabricated-native-v1";
    std::vector<GisToOhqHourlyRow> rows;
    std::vector<NativeSupportEvidence> evidence;
    for (int h=0; h<300; ++h) {
        GisToOhqHourlyRow r; r.epoch_seconds=c.epoch_zero_seconds+h*3600;
        r.observed_discharge_m3_per_second=2; r.observed_runoff_mm_per_hour=999; // deliberately wrong cached conversion
        r.discharge_valid=true; r.all_forcings_valid=true;
        r.precipitation_valid=r.temperature_valid=r.relative_humidity_valid=r.wind_valid=r.solar_valid=r.pet_valid=true;
        r.temperature_c=15; r.relative_humidity_percent=50;
        rows.push_back(r); evidence.push_back({r.epoch_seconds,true,false,1,c.expected_source_id});
    }
    auto adapt=[&]{return adaptNativeHourlyRows(rows,evidence,c);};
    auto out=adapt(); require(out.size()==300 && std::abs(out[0].observed_runoff_mm_per_hour-3.6)<1e-12);
    HourlyIndex idx(out); auto sample=buildSample(idx,out,100); require(sample && sample->predictor_availability_mask);
    evidence[90].native_observation=false; evidence[90].interpolated=true;
    out=adapt(); HourlyIndex ii(out); sample=buildSample(ii,out,100); require(sample && !sample->predictor_availability_mask);
    evidence[90].native_observation=true; rejects([&]{adapt();}); evidence[90].native_observation=false;
    evidence[90].interpolated=false; evidence[90].native_coverage=0.5; out=adapt(); require(!out[90].discharge_valid);
    evidence[90]={rows[90].epoch_seconds,true,false,1,c.expected_source_id};
    auto saved=rows[90]; rows.erase(rows.begin()+90); evidence.erase(evidence.begin()+90);
    out=adapt(); HourlyIndex gap(out); sample=buildSample(gap,out,100); require(sample && !sample->predictor_availability_mask);
    rows.insert(rows.begin()+90,saved); evidence.insert(evidence.begin()+90,{saved.epoch_seconds,true,false,1,c.expected_source_id});
    auto bad=c; bad.discharge_unit="cfs"; rejects([&]{adaptNativeHourlyRows(rows,evidence,bad);});
    bad=c; bad.precipitation_unit="mm/day"; rejects([&]{adaptNativeHourlyRows(rows,evidence,bad);});
    bad=c; bad.catchment_area_m2=0; rejects([&]{adaptNativeHourlyRows(rows,evidence,bad);});
    evidence[0].source_id="unknown"; rejects([&]{adapt();}); evidence[0].source_id=c.expected_source_id;
    evidence.push_back(evidence[0]); rejects([&]{adapt();}); evidence.pop_back();
    evidence.pop_back(); rejects([&]{adapt();}); evidence.push_back({rows.back().epoch_seconds,true,false,1,c.expected_source_id});
    rows[0].epoch_seconds++; rejects([&]{adapt();}); rows[0].epoch_seconds--;
    rows.back().epoch_seconds=c.epoch_zero_seconds+kSealStartX*3600; rejects([&]{adapt();}); rows.back().epoch_seconds=evidence.back().epoch_seconds;
    rows[0].observed_discharge_m3_per_second=std::numeric_limits<double>::quiet_NaN(); out=adapt(); require(!out[0].discharge_valid && std::isfinite(out[0].observed_runoff_mm_per_hour));
    rows[0].relative_humidity_percent=101; out=adapt(); require(!out[0].all_forcings_valid);
    rows[0].relative_humidity_percent=50;
    rows[0].observed_discharge_m3_per_second=std::numeric_limits<double>::max();
    out=adapt(); require(!out[0].discharge_valid);
    auto duplicate=rows[1]; rows[1].epoch_seconds=rows[0].epoch_seconds;
    rejects([&]{adapt();}); rows[1]=duplicate;
    bad=c; bad.epoch_zero_seconds=0; rejects([&]{adaptNativeHourlyRows(rows,evidence,bad);});
    bad=c; bad.end_x_exclusive=kSealStartX+1; rejects([&]{adaptNativeHourlyRows(rows,evidence,bad);});
    std::cout << "Native support adapter: all fabricated-package boundary checks passed\n";
}
