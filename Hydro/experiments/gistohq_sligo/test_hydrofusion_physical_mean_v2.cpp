#include "hydrofusion_causal_physics.h"
#include "hydrofusion_synthetic_fixture.h"
#include <iostream>
using namespace hydrofusion;
static void check(bool ok){if(!ok)throw std::runtime_error("Hourly mean regression failed");}
int main(){
    CausalPhysicsParams p;ReservoirLedger old,mean;
    const double end=stepCausalReservoirs(old,1,p),average=stepCausalReservoirHourMean(mean,1,p);
    check(std::abs(end-.47103316576920246)<1e-14);
    check(std::abs(average-.27202786292788839)<1e-14);
    check(std::abs(mean.fastStorage-old.fastStorage)<1e-14&&std::abs(mean.slowStorage-old.slowStorage)<1e-14);
    for(int i=0;i<1000;++i)stepCausalReservoirHourMean(mean,0,p);
    check(std::abs(mean.inputVolume-mean.outflowVolume-mean.fastStorage-mean.slowStorage)<1e-12);
    check(std::abs(reservoirHourOutflow(0,1,1e-12)-5e-13)<1e-24);
    check(std::abs(reservoirHourOutflow(2,0,1e-12)-2e-12)<1e-23);
    auto rows=buildSyntheticFixture();auto base=causalPhysicalHourlyMeans(rows,p);
    rows[200].precipitation_mm_per_hour=100;auto changed=causalPhysicalHourlyMeans(rows,p);
    for(int i=0;i<200;++i)check(base.at(i)==changed.at(i));
    for(auto&r:rows)r.observed_runoff_mm_per_hour=999;
    check(changed==causalPhysicalHourlyMeans(rows,p));
    rows.erase(rows.begin()+100);auto gap=causalPhysicalHourlyMeans(rows,p);ReservoirLedger zero;
    check(gap.at(101)==stepCausalReservoirHourMean(zero,std::max(0.,rows[100].precipitation_mm_per_hour-rows[100].pet_mm_per_hour),p));
    rows[120].all_forcings_valid=false;check(!causalPhysicalHourlyMeans(rows,p).count(121));
    bool rejected=false;rows.back().x=kSealStartX;
    try{causalPhysicalHourlyMeans(rows,p);}catch(const std::invalid_argument&){rejected=true;}check(rejected);
    std::cout<<"PASS hourly analytic pulse, v1 unchanged, storage, mass, tiny rates, future/Q invariance, gaps, missing forcing, seal\n";
}
