#pragma once
#include "hydrofusion_forecast_modes.h"
#include "../experiments/gistohq_sligo/hydrofusion_preissue_rain.h"
#include "../dataset/hydro_checksum.h"
#include <filesystem>
namespace hydrofusion {
inline bool rainHold(const HydroRunConfig& c){return c.hydrofusion_preissue_rain_recipe=="rate_hold_v1";}
inline void validateRainConfig(const HydroRunConfig& c){
    if(!forecastRecipeV5(c)){
        if(!c.hydrofusion_preissue_rain_path.empty()||!c.hydrofusion_preissue_rain_sha256.empty()||!c.hydrofusion_preissue_rain_provenance_sha256.empty())throw std::invalid_argument("Disabled rain recipe has unexpected data bindings.");return;
    }
    const auto hash=[](const std::string& h){return h.size()==64&&h.find_first_not_of("0123456789abcdef")==std::string::npos;};
    if((c.hydrofusion_preissue_rain_recipe!="zero_channel_v1"&&!rainHold(c))||!c.hydrofusion_development_only||!forecastRecipeV3(c)||c.hydrofusion_preissue_rain_path.empty()||!hash(c.hydrofusion_preissue_rain_sha256)||!hash(c.hydrofusion_preissue_rain_provenance_sha256))throw std::invalid_argument("Preissue rain requires explicit development-only v3 physics and bound weather/provenance.");
}
inline std::map<int64_t,PreissueRain> loadBoundPreissueRain(const HydroRunConfig& c,const std::filesystem::path& overridePath={}){
    validateRainConfig(c);if(!forecastRecipeV5(c))return {};
    const auto path=overridePath.empty()?std::filesystem::path(c.hydrofusion_preissue_rain_path):overridePath;
    if(sha256File(path.string())!=c.hydrofusion_preissue_rain_sha256||sha256File((path.parent_path()/"preissue_rain_provenance.json").string())!=c.hydrofusion_preissue_rain_provenance_sha256)throw std::invalid_argument("Preissue rain/provenance hash mismatch.");
    std::ifstream input(path);if(!input)throw std::invalid_argument("Missing preissue rain CSV.");return parsePreissueRain(input,c.validation_end_x);
}
inline std::set<int64_t> rainEligibleTargets(const HydroRunConfig& c,const std::map<int64_t,PreissueRain>& rain){
    std::set<int64_t> targets;for(const auto& row:rain)if(preissueRainHistoryAvailable(rain,row.first,c.lstm_sequence_length))targets.insert(row.first+1);return targets;
}
inline std::string rainSchema(const HydroRunConfig& c){
    validateRainConfig(c);std::ostringstream out;out<<"schema=hydrofusion_preissue_rain_v5\nrecipe="<<c.hydrofusion_preissue_rain_recipe<<"\nweather_sha256="<<c.hydrofusion_preissue_rain_sha256<<"\nprovenance_sha256="<<c.hydrofusion_preissue_rain_provenance_sha256<<"\nvalid_before_origin_seconds=600\nrelease_proxy=archive_http_last_modified\nminimum_basin_coverage=0.75\nfeature_units=mm/h\nfeature_order=last_channel_per_historical_step\nfuture_rain_policy="<<(rainHold(c)?"hold_available_rate_over_next_hour":"dry")<<"\nhistorical_delivery_proven=false\nexecution_scope=train_validation_only\n";return out.str();
}
}
