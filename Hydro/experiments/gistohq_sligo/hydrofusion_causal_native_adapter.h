#pragma once
#include "hydrofusion_native_support_adapter.h"
#include <algorithm>

namespace hydrofusion {
// Interval records are native summaries over [epoch_seconds, epoch_seconds+3600).
// They are NEVER labeled at the interval start in a forecast feature stream.
struct NativeDailyPetEnergy {
    int64_t day_start_epoch_seconds = 0;
    double energy_mj_m2_day = 0;
    bool valid = false;
};
struct CausalNativePolicy {
    // Explicit availability assumptions, not claims of real-time provider delivery.
    // Negative defaults prevent accidental acceptance without a frozen policy.
    int64_t discharge_delay_hours = -1;
    int64_t hourly_forcing_delay_hours = -1;
    int64_t daily_pet_delay_hours = -1;
    std::string availability_evidence;
};

// Shared, torch-free row adapter for all causal forecast families. Native Q is
// the JUST COMPLETED hour at origin; target at origin+1 is the FOLLOWING hour.
// Nonzero Q-release latency needs a distinct horizon/target contract and is not
// silently approximated here. Upstream must explicitly establish Q availability.
inline std::vector<HourlyRow> adaptCausalNativeIntervals(
    const std::vector<GisToOhqHourlyRow>& intervals,
    const std::vector<NativeSupportEvidence>& intervalEvidence,
    const std::vector<NativeDailyPetEnergy>& dailyPet,
    const NativeAdapterContext& endpointContext,
    const CausalNativePolicy& policy) {
    if(policy.discharge_delay_hours!=0 || policy.hourly_forcing_delay_hours<0 || policy.hourly_forcing_delay_hours>168 ||
       policy.daily_pet_delay_hours<0 || policy.daily_pet_delay_hours>168 ||
       policy.availability_evidence.empty())
        throw std::invalid_argument("Explicit bounded causal availability policy required.");
    if(endpointContext.epoch_zero_seconds!=1420070400 || endpointContext.first_x<1 ||
       endpointContext.end_x_exclusive<=endpointContext.first_x || endpointContext.end_x_exclusive>kSealStartX)
        throw std::invalid_argument("Invalid causal endpoint development slice.");
    const auto lower=endpointContext.epoch_zero_seconds+(endpointContext.first_x-1)*3600;
    const auto upper=endpointContext.epoch_zero_seconds+(endpointContext.end_x_exclusive-1)*3600;
    std::map<int64_t,const GisToOhqHourlyRow*> byStart;
    int64_t previous=-1;
    // All identities, including daily intervals, checked before reading values.
    for(const auto& row:intervals){
        if(row.epoch_seconds<lower || row.epoch_seconds>=upper || row.epoch_seconds%3600 || row.epoch_seconds<=previous)
            throw std::invalid_argument("Causal native intervals must be sorted unique presealed hourly intervals.");
        byStart.emplace(row.epoch_seconds,&row);previous=row.epoch_seconds;
    }
    std::map<int64_t,const NativeDailyPetEnergy*> petByEnd;
    for(const auto& day:dailyPet){
        const auto start=day.day_start_epoch_seconds;
        // No days extending into/past the sealed observation interval; add only
        // after safe domain checks. Extra preseal days are permitted for lagging.
        if(start<endpointContext.epoch_zero_seconds || start>endpointContext.epoch_zero_seconds+kSealStartX*3600-86400 || start%86400)
            throw std::invalid_argument("Invalid or sealed daily PET interval.");
        if(!petByEnd.emplace(start+86400,&day).second)
            throw std::invalid_argument("Duplicate daily PET interval.");
    }
    std::vector<GisToOhqHourlyRow> shifted=intervals;
    std::vector<NativeSupportEvidence> shiftedEvidence=intervalEvidence;
    for(auto& entry:shiftedEvidence){
        if(!byStart.count(entry.epoch_seconds))throw std::invalid_argument("Unknown interval evidence identity.");
        entry.epoch_seconds+=3600;
    }
    for(auto& row:shifted){
        const auto origin=row.epoch_seconds+3600;
        const auto sourceStart=origin-policy.hourly_forcing_delay_hours*3600-3600;
        const auto forcing=byStart.find(sourceStart);
        if(forcing==byStart.end()){
            row.precipitation_valid=row.temperature_valid=row.relative_humidity_valid=row.wind_valid=row.solar_valid=false;
        }else{
            const auto& f=*forcing->second;
            row.precipitation_mm_per_hour=f.precipitation_mm_per_hour;
            row.temperature_c=f.temperature_c;row.relative_humidity_percent=f.relative_humidity_percent;
            row.wind_speed_m_per_second=f.wind_speed_m_per_second;row.solar_energy_mj_per_m2_per_hour=f.solar_energy_mj_per_m2_per_hour;
            row.precipitation_valid=f.precipitation_valid;row.temperature_valid=f.temperature_valid;
            row.relative_humidity_valid=f.relative_humidity_valid;row.wind_valid=f.wind_valid;row.solar_valid=f.solar_valid;
        }
        // Use ONLY the most recent complete UTC day whose release time is no
        // later than origin. Never distribute the current day's eventual total.
        const auto availableBefore=origin-policy.daily_pet_delay_hours*3600;
        const auto expectedDayEnd=availableBefore/86400*86400;
        const auto day=petByEnd.find(expectedDayEnd);
        row.pet_valid=false;row.pet_mm_per_hour=0;
        if(day!=petByEnd.end()){
            const auto& value=*day->second;
            if(value.valid && std::isfinite(value.energy_mj_m2_day) && value.energy_mj_m2_day>=0){
                row.pet_mm_per_hour=value.energy_mj_m2_day/2.45/24;
                row.pet_valid=std::isfinite(row.pet_mm_per_hour);
            }
        }
        row.all_forcings_valid=row.precipitation_valid && row.temperature_valid && row.relative_humidity_valid && row.wind_valid && row.solar_valid && row.pet_valid;
        row.epoch_seconds=origin;
    }
    return adaptNativeHourlyRows(shifted,shiftedEvidence,endpointContext);
}
} // namespace hydrofusion
