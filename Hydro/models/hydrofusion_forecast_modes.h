#pragma once
#include <string>
namespace hydrofusion {
inline bool isExtendedForecast(const std::string& name){return name=="ffn_causal_forecast"||name=="lstm_hydrofusion"||name=="pinn_causal_forecast"||name=="ffn_physical_forecast"||name=="lstm_physical_forecast";}
inline bool isPhysicalForecast(const std::string& name){return isExtendedForecast(name)&&name!="lstm_hydrofusion"&&name!="ffn_causal_forecast";}
inline bool isRecurrentForecast(const std::string& name){return name=="lstm_hydrofusion"||name=="lstm_physical_forecast";}
}
