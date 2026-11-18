#include "hydrofusion_causal_native_adapter.h"
#include <iostream>
#include <functional>
using namespace hydrofusion;
void checked(bool value,int line){if(!value)throw std::runtime_error("check failed at line "+std::to_string(line));}
#define check(value) checked((value),__LINE__)
void rejects(const std::function<void()>& f){bool rejected=false;try{f();}catch(const std::invalid_argument&){rejected=true;}check(rejected);}
int main(){
    constexpr int64_t epoch=1420070400;
    std::vector<GisToOhqHourlyRow> rows;std::vector<NativeSupportEvidence> evidence;
    for(int h=0;h<240;++h){GisToOhqHourlyRow r;r.epoch_seconds=epoch+h*3600;
        r.observed_discharge_m3_per_second=10+h;r.discharge_valid=true;
        r.precipitation_mm_per_hour=h;r.temperature_c=15;r.relative_humidity_percent=50;r.wind_speed_m_per_second=1;r.solar_energy_mj_per_m2_per_hour=1;
        r.precipitation_valid=r.temperature_valid=r.relative_humidity_valid=r.wind_valid=r.solar_valid=true;
        // Deliberately poisoned current-day distributed PET: adapter must ignore it.
        r.pet_mm_per_hour=9999;r.pet_valid=r.all_forcings_valid=true;
        rows.push_back(r);evidence.push_back({r.epoch_seconds,true,false,1,"fabricated"});}
    std::vector<NativeDailyPetEnergy> days;for(int day=0;day<10;++day)days.push_back({epoch+day*86400,2.45*24*(day+1),true});
    NativeAdapterContext context;context.first_x=1;context.end_x_exclusive=241;context.catchment_area_m2=3600000;context.expected_source_id="fabricated";
    CausalNativePolicy policy{0,0,0,"fabricated release policy, not operational provider evidence"};
    auto out=adaptCausalNativeIntervals(rows,evidence,days,context,policy);
    check(out.front().x==1 && out.front().observed_runoff_mm_per_hour==10);
    check(!out[22].all_forcings_valid && out[23].all_forcings_valid);
    check(out[23].pet_mm_per_hour==1 && out[24].pet_mm_per_hour==1 && out[46].pet_mm_per_hour==1 && out[47].pet_mm_per_hour==2);
    auto changed=days;changed[1].energy_mj_m2_day=1e6;
    auto altered=adaptCausalNativeIntervals(rows,evidence,changed,context,policy);
    for(int i=0;i<47;++i)check(out[i].pet_mm_per_hour==altered[i].pet_mm_per_hour);
    check(out[47].pet_mm_per_hour!=altered[47].pet_mm_per_hour);
    // A one-day declared publication delay postpones first day until hour48.
    auto delayed=policy;delayed.daily_pet_delay_hours=24;
    auto lagged=adaptCausalNativeIntervals(rows,evidence,days,context,delayed);
    check(!lagged[46].all_forcings_valid && lagged[47].pet_mm_per_hour==1);
    delayed=policy;delayed.hourly_forcing_delay_hours=2;
    lagged=adaptCausalNativeIntervals(rows,evidence,days,context,delayed);
    check(!lagged[0].all_forcings_valid && !lagged[1].all_forcings_valid && lagged[23].precipitation_mm_per_hour==21);
    // Future forcing cannot affect an earlier predictor.
    auto future=rows;future[120].precipitation_mm_per_hour=1e6;
    altered=adaptCausalNativeIntervals(future,evidence,days,context,policy);
    for(int i=0;i<120;++i)check(out[i].precipitation_mm_per_hour==altered[i].precipitation_mm_per_hour);
    HourlyIndex index(out);auto sample=buildSample(index,out,120);
    check(sample && sample->predictor_availability_mask && sample->target_native_mask);
    check(sample->features.q_lag_0h==129 && sample->observed_target_q==130);
    check(std::abs(sample->features.met_pet_mm_per_h-5)<1e-12);
    // Physically completed hourly Q predicts the next completed hourly Q.
    check(sample->origin_x==120 && sample->target_x==121);
    auto bad=policy;bad.discharge_delay_hours=1;rejects([&]{adaptCausalNativeIntervals(rows,evidence,days,context,bad);});
    bad=policy;bad.daily_pet_delay_hours=-1;rejects([&]{adaptCausalNativeIntervals(rows,evidence,days,context,bad);});
    bad=policy;bad.availability_evidence.clear();rejects([&]{adaptCausalNativeIntervals(rows,evidence,days,context,bad);});
    auto duplicate=days;duplicate.push_back(days[0]);rejects([&]{adaptCausalNativeIntervals(rows,evidence,duplicate,context,policy);});
    auto sealed=rows;sealed.back().epoch_seconds=epoch+77015*3600;rejects([&]{adaptCausalNativeIntervals(sealed,evidence,days,context,policy);});
    auto missing=days;missing.erase(missing.begin()+4);
    altered=adaptCausalNativeIntervals(rows,evidence,missing,context,policy);check(!altered[119].all_forcings_valid);
    auto invalid=days;invalid[4].energy_mj_m2_day=std::numeric_limits<double>::infinity();
    altered=adaptCausalNativeIntervals(rows,evidence,invalid,context,policy);check(!altered[119].all_forcings_valid);
    std::cout<<"CAUSAL NATIVE INTERVAL/PET CHECKS PASSED\n";
}
