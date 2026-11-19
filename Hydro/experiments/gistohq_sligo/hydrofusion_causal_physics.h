#pragma once
#include "hydrofusion_feature_builder.h"

namespace hydrofusion {
struct CausalPhysicsParams { double fast=1, slow=0.1, fraction=0.7, gain=1; };
struct ReservoirLedger { double fastStorage=0,slowStorage=0,inputVolume=0,outflowVolume=0; };
inline void validateCausalPhysics(const CausalPhysicsParams& p) {
    if(!std::isfinite(p.fast)||p.fast<=0||!std::isfinite(p.slow)||p.slow<=0||
       !std::isfinite(p.fraction)||p.fraction<0||p.fraction>1||!std::isfinite(p.gain)||p.gain<0)
        throw std::invalid_argument("Invalid causal physical parameters.");
}
inline double stepCausalReservoirs(ReservoirLedger& state,double effectiveRain,const CausalPhysicsParams& p) {
    validateCausalPhysics(p);
    if(!std::isfinite(effectiveRain)||effectiveRain<0||!std::isfinite(state.fastStorage)||state.fastStorage<0||
       !std::isfinite(state.slowStorage)||state.slowStorage<0)throw std::invalid_argument("Invalid causal physical state/input.");
    const double inflow=effectiveRain*p.gain;
    const auto update=[](double s,double r,double k){return std::exp(-k)*s+(-std::expm1(-k))*r/k;};
    const double fast=update(state.fastStorage,p.fraction*inflow,p.fast);
    const double slow=update(state.slowStorage,(1-p.fraction)*inflow,p.slow);
    const double out=state.fastStorage+state.slowStorage+inflow-fast-slow;
    const double q=p.fast*fast+p.slow*slow;
    if(!std::isfinite(fast)||!std::isfinite(slow)||!std::isfinite(out)||!std::isfinite(q)||q<0||out < -1e-10)
        throw std::overflow_error("Causal physical update overflow/domain failure.");
    state.fastStorage=fast;state.slowStorage=slow;
    state.inputVolume+=inflow;state.outflowVolume+=out;
    if(!std::isfinite(state.inputVolume)||!std::isfinite(state.outflowVolume))throw std::overflow_error("Physical ledger overflow.");
    return q;
}
inline std::map<int64_t,double> causalPhysicalForecasts(const std::vector<HourlyRow>& rows,const CausalPhysicsParams& params) {
    validateCausalPhysics(params);HourlyIndex index(rows);(void)index;
    // Guard every timestamp before touching forcing; full reserved datasets are
    // not admissible and must never be supplied for after-the-fact filtering.
    for(const auto& r:rows)if(r.x>=kSealStartX)throw std::invalid_argument("Causal physics preseal input required.");
    std::map<int64_t,double> output;ReservoirLedger state;int64_t previous=-2;double previousRain=0;
    for(const auto& r:rows) {
        const auto x=static_cast<int64_t>(r.x);
        if(!r.all_forcings_valid||!std::isfinite(r.precipitation_mm_per_hour)||r.precipitation_mm_per_hour<0||
           !std::isfinite(r.pet_mm_per_hour)||r.pet_mm_per_hour<0) {state={};previous=-2;continue;}
        if(x!=previous+1)state={};else stepCausalReservoirs(state,previousRain,params);
        const double rain=std::max(0.0,r.precipitation_mm_per_hour-r.pet_mm_per_hour);
        auto forecastState=state;output.emplace(x,stepCausalReservoirs(forecastState,rain,params));
        previous=x;previousRain=rain;
    }
    return output;
}
// Explicit v2 output semantics: exact mean outflow over one hour. The v1
// function above is intentionally unchanged for old checkpoint reconstruction.
inline double reservoirHourOutflow(double storage,double inflow,double rate){
    if(!std::isfinite(storage)||storage<0||!std::isfinite(inflow)||inflow<0||!std::isfinite(rate)||rate<=0)throw std::invalid_argument("Invalid hourly reservoir arguments.");
    const double released=-std::expm1(-rate);
    // 1-(1-exp(-k))/k loses precision as k approaches zero.
    const double fraction=rate<1e-4?rate*(.5+rate*(-1./6+rate*(1./24+rate*(-1./120+rate/720)))):1-released/rate;
    const double out=storage*released+inflow*fraction;
    if(!std::isfinite(out)||out<0)throw std::overflow_error("Hourly outflow overflow.");
    return out;
}
inline double stepCausalReservoirHourMean(ReservoirLedger& state,double effectiveRain,const CausalPhysicsParams& p){
    validateCausalPhysics(p);
    if(!std::isfinite(effectiveRain)||effectiveRain<0)throw std::invalid_argument("Invalid effective rain.");
    const double inflow=effectiveRain*p.gain;
    const double out=reservoirHourOutflow(state.fastStorage,p.fraction*inflow,p.fast)+reservoirHourOutflow(state.slowStorage,(1-p.fraction)*inflow,p.slow);
    const auto update=[](double s,double r,double k){return std::exp(-k)*s+(-std::expm1(-k)/k)*r;};
    const double fast=update(state.fastStorage,p.fraction*inflow,p.fast),slow=update(state.slowStorage,(1-p.fraction)*inflow,p.slow);
    if(!std::isfinite(fast)||!std::isfinite(slow)||!std::isfinite(out)||!std::isfinite(state.inputVolume+inflow)||!std::isfinite(state.outflowVolume+out))throw std::overflow_error("Hourly physical state overflow.");
    state.fastStorage=fast;state.slowStorage=slow;state.inputVolume+=inflow;state.outflowVolume+=out;
    return out; // integrated depth divided by exactly one hour
}
inline std::map<int64_t,double> causalPhysicalHourlyMeans(const std::vector<HourlyRow>& rows,const CausalPhysicsParams& params){
    validateCausalPhysics(params);HourlyIndex checked(rows);(void)checked;
    for(const auto&r:rows)if(r.x>=kSealStartX)throw std::invalid_argument("Causal physics preseal input required.");
    std::map<int64_t,double> result;ReservoirLedger state;int64_t previous=-2;double prior=0;
    for(const auto&r:rows){
        const auto x=static_cast<int64_t>(r.x);
        if(!r.all_forcings_valid||!std::isfinite(r.precipitation_mm_per_hour)||r.precipitation_mm_per_hour<0||!std::isfinite(r.pet_mm_per_hour)||r.pet_mm_per_hour<0){state={};previous=-2;continue;}
        if(x!=previous+1)state={};else stepCausalReservoirHourMean(state,prior,params);
        const double rain=std::max(0.,r.precipitation_mm_per_hour-r.pet_mm_per_hour);
        auto next=state;result.emplace(x,stepCausalReservoirHourMean(next,rain,params));previous=x;prior=rain;
    }
    return result;
}
// v3: rows are indexed by forecast origin/release hour, but rainfall belongs
// to [rainfall_valid_end_x-1, rainfall_valid_end_x). Commit ONLY newly released
// intervals on that historical timeline. Never commit the dry nowcast copy:
// a later release replaces the unknown interval with its actual available rain.
// Missing rainfall between the latest known interval and target is explicitly
// assumed zero, NOT asserted observed dry. PET is the estimate available at
// release, not future/current-day realized ET. No Q assimilation or calibration.
inline std::map<int64_t,double> causalPhysicalValidTimeDryForecasts(
    const std::vector<HourlyRow>& rows,const CausalPhysicsParams& params){
    validateCausalPhysics(params);HourlyIndex checked(rows);(void)checked;
    // Validate the complete identity domain before any forcing is inspected.
    for(const auto& row:rows){
        if(row.x>=kSealStartX-1)throw std::invalid_argument("v3 requires presealed origin AND target.");
        if(row.all_forcings_valid && (row.rainfall_valid_end_x<0 ||
           row.rainfall_valid_end_x>row.rainfall_release_x || row.rainfall_release_x!=row.x ||
           row.rainfall_release_x-row.rainfall_valid_end_x>168))
            throw std::invalid_argument("v3 requires explicit bounded valid-time/release provenance.");
    }
    std::map<int64_t,double> output;ReservoirLedger state;
    int64_t previousOrigin=-2,previousValidEnd=-2;
    for(const auto& row:rows){
        const auto origin=static_cast<int64_t>(row.x);
        if(!row.all_forcings_valid){state={};previousOrigin=previousValidEnd=-2;continue;}
        if(!std::isfinite(row.precipitation_mm_per_hour)||row.precipitation_mm_per_hour<0||
           !std::isfinite(row.pet_mm_per_hour)||row.pet_mm_per_hour<0)
            throw std::invalid_argument("v3 requires finite nonnegative available rain/PET.");
        const auto validEnd=row.rainfall_valid_end_x;
        // Changed delays would imply duplicate, missing or reordered historical
        // forcing. Reject within a segment rather than silently revising state.
        if(origin==previousOrigin+1 && validEnd!=previousValidEnd+1)
            throw std::invalid_argument("v3 release stream must advance valid time hourly.");
        if(origin!=previousOrigin+1)state={};
        stepCausalReservoirHourMean(state,std::max(0.,row.precipitation_mm_per_hour-row.pet_mm_per_hour),params);
        auto forecast=state;
        // state is at validEnd. Advance to origin, then forecast [origin,origin+1).
        for(int64_t hour=validEnd;hour<origin;++hour)stepCausalReservoirHourMean(forecast,0.,params);
        output.emplace(origin,stepCausalReservoirHourMean(forecast,0.,params));
        previousOrigin=origin;previousValidEnd=validEnd;
    }
    return output;
}
} // namespace hydrofusion
