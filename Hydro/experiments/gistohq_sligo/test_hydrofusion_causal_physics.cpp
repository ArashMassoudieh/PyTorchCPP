#include "hydrofusion_causal_physics.h"
#include "hydrofusion_synthetic_fixture.h"
#include "hydrofusion_log_residual.h"
#include <iostream>
using namespace hydrofusion;
void check(bool ok){if(!ok)throw std::runtime_error("physical check failed");}
int main(){
    CausalPhysicsParams p;ReservoirLedger s;
    for(int i=0;i<500;++i)stepCausalReservoirs(s,i<10?1:0,p);
    check(std::abs(s.inputVolume-s.outflowVolume-s.fastStorage-s.slowStorage)<1e-10);
    auto rows=buildSyntheticFixture();auto base=causalPhysicalForecasts(rows,p);
    rows[200].precipitation_mm_per_hour=100;
    auto changed=causalPhysicalForecasts(rows,p);
    for(int i=0;i<200;++i)check(base.at(i)==changed.at(i));
    // No discharge assimilation, even of a segment's first observed Q.
    for(auto& r:rows)r.observed_runoff_mm_per_hour=999;
    check(changed==causalPhysicalForecasts(rows,p));
    rows.erase(rows.begin()+100);auto gap=causalPhysicalForecasts(rows,p);
    ReservoirLedger zero;check(gap.at(101)==stepCausalReservoirs(zero,std::max(0.0,rows[100].precipitation_mm_per_hour-rows[100].pet_mm_per_hour),p));
    for(double q:{0.0,0.1,4.0})check(std::abs(reconstructQ(TargetMode::LogResidual,q,0,1)-q)<1e-12);
    bool failed=false;try{auto bad=p;bad.fast=0;causalPhysicalForecasts(rows,bad);}catch(const std::invalid_argument&){failed=true;}check(failed);
    auto prefix=buildSyntheticFixture();prefix.resize(101);HourlyIndex index(prefix);
    auto noTarget=buildSample(index,prefix,100,kSealStartX,false);
    check(noTarget && noTarget->predictor_availability_mask && !noTarget->target_native_mask);
    std::cout<<"Physical ledger/causality/gap/reset/zero-correction checks passed\n";
}
