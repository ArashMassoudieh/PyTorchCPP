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
    int64_t daily_pet_utc_start_hour = 0; // Explicit provider-day offset; legacy UTC days unchanged.
    // Both -1 preserve the exact v1/v2 shared-weather policy. Independent
    // precipitation timing is opt-in only; age and release delay are distinct.
    int64_t rainfall_delay_hours = -1;
    int64_t rainfall_release_delay_hours = -1;
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
       policy.availability_evidence.empty() || policy.daily_pet_utc_start_hour<0 || policy.daily_pet_utc_start_hour>23)
        throw std::invalid_argument("Explicit bounded causal availability policy required.");
    const bool independentRain = policy.rainfall_delay_hours!=-1 || policy.rainfall_release_delay_hours!=-1;
    if(independentRain && (policy.rainfall_delay_hours<0 || policy.rainfall_delay_hours>168 ||
       policy.rainfall_release_delay_hours<0 || policy.rainfall_release_delay_hours>policy.rainfall_delay_hours))
        throw std::invalid_argument("Invalid independent rainfall age/release policy.");
    const auto rainAge=independentRain?policy.rainfall_delay_hours:policy.hourly_forcing_delay_hours;
    const auto rainRelease=independentRain?policy.rainfall_release_delay_hours:policy.hourly_forcing_delay_hours;
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
        if(start<endpointContext.epoch_zero_seconds || start>endpointContext.epoch_zero_seconds+kSealStartX*3600-86400 || start%86400!=policy.daily_pet_utc_start_hour*3600)
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
        if(independentRain){
            // Do not make temperature/humidity/wind/solar fresher when changing
            // only precipitation. Missing radar remains missing, never dry zero.
            const auto rain=byStart.find(origin-rainAge*3600-3600);
            row.precipitation_valid=false;
            if(rain!=byStart.end()){
                row.precipitation_mm_per_hour=rain->second->precipitation_mm_per_hour;
                row.precipitation_valid=rain->second->precipitation_valid;
            }
        }
        // Use ONLY the most recent complete provider day whose release time is no
        // later than origin. Never distribute the current day's eventual total.
        const auto availableBefore=origin-policy.daily_pet_delay_hours*3600;
        const auto dayOffset=policy.daily_pet_utc_start_hour*3600;
        const auto expectedDayEnd=(availableBefore-dayOffset)/86400*86400+dayOffset;
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
    auto adapted=adaptNativeHourlyRows(shifted,shiftedEvidence,endpointContext);
    for(auto& row:adapted){
        row.rainfall_valid_end_x=static_cast<int64_t>(row.x)-rainAge;
        row.rainfall_release_x=row.rainfall_valid_end_x+rainRelease;
    }
    return adapted;
}
} // namespace hydrofusion
