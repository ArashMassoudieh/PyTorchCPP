#pragma once
#include <string>
#include "hydro_run_types.h"
namespace hydrofusion {
inline bool isExtendedForecast(const std::string& name){return name=="ffn_causal_forecast"||name=="lstm_hydrofusion"||name=="pinn_causal_forecast"||name=="ffn_physical_forecast"||name=="lstm_physical_forecast";}
inline bool isPhysicalForecast(const std::string& name){return isExtendedForecast(name)&&name!="lstm_hydrofusion"&&name!="ffn_causal_forecast";}
inline bool isRecurrentForecast(const std::string& name){return name=="lstm_hydrofusion"||name=="lstm_physical_forecast";}
inline bool forecastRecipeV5(const HydroRunConfig& c){return c.hydrofusion_preissue_rain_recipe!="disabled";}
inline int forecastChannels(const HydroRunConfig& c){return (c.hydrofusion_context_recipe!="legacy20"?25:20)+(forecastRecipeV5(c)?1:0);}
inline bool forecastRecipeV4(const HydroRunConfig& c){return c.hydrofusion_context_recipe!="legacy20";}
inline bool forecastRecipeV2(const HydroRunConfig& c){return c.pinn_physics_profile=="causal_two_reservoir_mean_v2"||c.hydrofusion_objective!="log_mse_v1"||c.hydrofusion_ffn_activation!="tanh";}
inline bool forecastRecipeV3(const HydroRunConfig& c){return c.pinn_physics_profile=="causal_valid_time_dry_v3";}
inline std::string forecastCheckpointFormat(const HydroRunConfig& c){return forecastRecipeV5(c)?"hydrofusion-forecast-v5":forecastRecipeV4(c)?"hydrofusion-forecast-v4":forecastRecipeV3(c)?"hydrofusion-forecast-v3":forecastRecipeV2(c)?"hydrofusion-forecast-v2":"hydrofusion-forecast-v1";}
}
