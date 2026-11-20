#include "hydrofusion_causal_physics.h"
#include "hydrofusion_synthetic_fixture.h"
#include <functional>
#include <iostream>
using namespace hydrofusion;
static int checks=0;
static void check(bool x){++checks;if(!x)throw std::runtime_error("v3 check "+std::to_string(checks));}
static void rejects(const std::function<void()>& f){bool bad=false;try{f();}catch(const std::invalid_argument&){bad=true;}check(bad);}
int main(){
 const CausalPhysicsParams p{};
 std::vector<HourlyRow> rows;
 for(int x=24;x<90;++x){auto r=makeSyntheticFixtureRow(x,1);r.rainfall_valid_end_x=x-24;r.rainfall_release_x=x;r.precipitation_mm_per_hour=x==30?10.:0.;r.pet_mm_per_hour=0;rows.push_back(r);}
 const auto result=causalPhysicalValidTimeDryForecasts(rows,p);
 // Direct reference at origin30: storm actually ended at6; route then dry25h.
 ReservoirLedger expected;stepCausalReservoirHourMean(expected,10,p);
 for(int i=0;i<24;++i)stepCausalReservoirHourMean(expected,0,p);
 check(std::abs(result.at(30)-stepCausalReservoirHourMean(expected,0,p))<1e-15);
 check(result.at(29)==0 && result.at(30)>0);
 check(result.at(30)<causalPhysicalHourlyMeans(rows,p).at(30));
 auto changed=rows;changed.back().precipitation_mm_per_hour=999;
 auto future=causalPhysicalValidTimeDryForecasts(changed,p);
 for(int x=24;x<89;++x)check(result.at(x)==future.at(x));
 changed=rows;for(auto&r:changed){r.observed_runoff_mm_per_hour=std::nan("");r.discharge_valid=false;}
 check(result==causalPhysicalValidTimeDryForecasts(changed,p));
 // Prefix inference exactly equals all-row replay; no target row needed.
 auto prefix=rows;prefix.resize(7);check(causalPhysicalValidTimeDryForecasts(prefix,p).at(30)==result.at(30));
 changed=rows;changed[7].rainfall_valid_end_x=-1;rejects([&]{causalPhysicalValidTimeDryForecasts(changed,p);});
 changed=rows;changed[7].rainfall_release_x=99;rejects([&]{causalPhysicalValidTimeDryForecasts(changed,p);});
 changed=rows;changed[7].rainfall_valid_end_x=38;rejects([&]{causalPhysicalValidTimeDryForecasts(changed,p);});
 changed=rows;changed[7].rainfall_valid_end_x-=1;rejects([&]{causalPhysicalValidTimeDryForecasts(changed,p);});
 changed=rows;changed[7].precipitation_mm_per_hour=std::nan("");rejects([&]{causalPhysicalValidTimeDryForecasts(changed,p);});
 changed=rows;changed.back().x=kSealStartX-1;rejects([&]{causalPhysicalValidTimeDryForecasts(changed,p);});
 changed=rows;changed.erase(changed.begin()+7);auto gap=causalPhysicalValidTimeDryForecasts(changed,p);check(gap.at(32)==0);
 changed=rows;changed[7].all_forcings_valid=false;auto missing=causalPhysicalValidTimeDryForecasts(changed,p);check(!missing.count(31)&&missing.at(32)==0);
 // Zero-delay is still a forecast of NEXT hour, not replay of completed-hour rain.
 auto zero=makeSyntheticFixtureRow(30,0);zero.precipitation_mm_per_hour=10;zero.pet_mm_per_hour=0;
 ReservoirLedger z;stepCausalReservoirHourMean(z,10,p);check(causalPhysicalValidTimeDryForecasts({zero},p).at(30)==stepCausalReservoirHourMean(z,0,p));
 check(std::abs(z.inputVolume-z.outflowVolume-z.fastStorage-z.slowStorage)<1e-12);
 // An already released interval remains causal when used a later hour.
 // Numeric routing depends on valid time, not an invented fresh release stamp.
 changed=rows;for(auto& r:changed)--r.rainfall_release_x;
 check(result==causalPhysicalValidTimeDryForecasts(changed,p));
 changed=rows;changed[7].rainfall_release_x=changed[7].rainfall_valid_end_x-1;
 rejects([&]{causalPhysicalValidTimeDryForecasts(changed,p);});
 // Bound total age, not just delivery delay, before dry-projection iteration.
 auto stale=makeSyntheticFixtureRow(300,0);stale.rainfall_valid_end_x=0;stale.rainfall_release_x=1;
 rejects([&]{causalPhysicalValidTimeDryForecasts({stale},p);});
 std::cout<<"PASS valid-time physics checks="<<checks<<'\n';
}
