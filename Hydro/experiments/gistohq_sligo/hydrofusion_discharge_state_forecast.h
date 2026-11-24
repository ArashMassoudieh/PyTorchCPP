#pragma once
#include "hydrofusion_preissue_rain.h"
namespace hydrofusion {
// EXPERIMENTAL observed-discharge state initialization, not the free-running v3
// reservoir. Qorigin is the observed mean of the immediately preceding hour.
// Under a dry preceding-hour reconstruction, allocate that outflow by fraction
// and infer end-of-hour stores. Each origin is a new analysis; its initial water
// is explicitly counted as analysis storage, not rainfall or a continuous ledger.
struct DischargeStateForecast { double mean=0,analysisStorage=0; ReservoirLedger ledger; };
inline DischargeStateForecast dischargeStateForecast(const HourlyRow& row,const PreissueRain& rain,const CausalPhysicsParams& p){
    validateCausalPhysics(p);
    if(!std::isfinite(row.x)||row.x<0||row.x!=std::floor(row.x)||row.x>=kSealStartX-1||
       !row.discharge_valid||!std::isfinite(row.observed_runoff_mm_per_hour)||row.observed_runoff_mm_per_hour<0||
       !row.all_forcings_valid||!std::isfinite(row.pet_mm_per_hour)||row.pet_mm_per_hour<0)
        throw std::invalid_argument("Observed-discharge analysis requires valid preseal origin Q/PET.");
    const auto origin=static_cast<int64_t>(row.x),issue=kRainEpochSeconds+origin*3600;
    if(rain.origin!=origin||rain.valid!=issue-600||rain.release<rain.valid||rain.release>issue||
       !std::isfinite(rain.rate)||rain.rate<0||!std::isfinite(rain.coverage)||rain.coverage<.75||rain.coverage>1.000000000001)
        throw std::invalid_argument("Unavailable observed-discharge analysis rain.");
    const auto store=[](double q,double rate){return q*std::exp(-rate)/(-std::expm1(-rate));};
    DischargeStateForecast out;
    out.ledger.fastStorage=store(row.observed_runoff_mm_per_hour*p.fraction,p.fast);
    out.ledger.slowStorage=store(row.observed_runoff_mm_per_hour*(1-p.fraction),p.slow);
    out.analysisStorage=out.ledger.fastStorage+out.ledger.slowStorage;
    if(!std::isfinite(out.analysisStorage)||out.analysisStorage<0)throw std::overflow_error("Analysis storage overflow.");
    out.ledger.inputVolume=out.analysisStorage; // Explicit analysis initial condition.
    out.mean=stepCausalReservoirHourMean(out.ledger,std::max(0.,rain.rate-row.pet_mm_per_hour),p);
    return out;
}
}
