#include "hydrofusion_discharge_state_forecast.h"
#include <iostream>
using namespace hydrofusion;
int main(){
 HourlyRow row;row.x=100;row.discharge_valid=row.all_forcings_valid=true;row.observed_runoff_mm_per_hour=2.;row.pet_mm_per_hour=.1;
 const auto issue=kRainEpochSeconds+100*3600;PreissueRain r{100,issue-600,issue-500,3.,1.};CausalPhysicsParams p{.4,.025,.7,0.};int checks=0;
 auto check=[&](bool b){if(!b)throw std::runtime_error("Assertion failed");++checks;};
 auto rejects=[&](auto f){bool rejected=false;try{f();}catch(const std::exception&){rejected=true;}check(rejected);};
 auto dry=dischargeStateForecast(row,r,p);check(std::abs(dry.mean-2*(.7*std::exp(-.4)+.3*std::exp(-.025)))<1e-12);
 p.gain=1.;auto wet=dischargeStateForecast(row,r,p);check(wet.mean>dry.mean);
 p.gain=.3;auto partial=dischargeStateForecast(row,r,p);check(std::abs(partial.mean-dry.mean-.3*(wet.mean-dry.mean))<1e-12);
 check(std::abs(partial.ledger.inputVolume-partial.ledger.outflowVolume-partial.ledger.fastStorage-partial.ledger.slowStorage)<1e-12);
 auto late=r;late.release=issue+1;rejects([&]{dischargeStateForecast(row,late,p);});
 auto invalid=row;invalid.x=kSealStartX-1;rejects([&]{dischargeStateForecast(invalid,r,p);});
 invalid=row;invalid.observed_runoff_mm_per_hour=std::numeric_limits<double>::quiet_NaN();rejects([&]{dischargeStateForecast(invalid,r,p);});
 invalid=row;invalid.discharge_valid=false;rejects([&]{dischargeStateForecast(invalid,r,p);});
 auto wrong=r;wrong.origin=99;rejects([&]{dischargeStateForecast(row,wrong,p);});
 p.fast=0;rejects([&]{dischargeStateForecast(row,r,p);});
 p={.4,.025,.7,1};r.rate=0;row.pet_mm_per_hour=0;check(std::abs(dischargeStateForecast(row,r,p).mean-dry.mean)<1e-12);
 std::cout<<"PASS "<<checks<<" discharge-state checks\n";
}
