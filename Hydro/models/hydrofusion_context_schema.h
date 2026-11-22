#pragma once
#include "hydrofusion_forecast_modes.h"
#include "../experiments/gistohq_sligo/hydrofusion_context_recipe.h"
#include "../experiments/gistohq_sligo/hydrofusion_metadata_sidecar.h"
namespace hydrofusion {
inline std::string forecastContextSchema(const HydroRunConfig& c){
    if(!forecastRecipeV4(c)||!validContextRecipe(c.hydrofusion_context_recipe))throw std::invalid_argument("Context schema requires a v4 recipe.");
    parseForecastHistory(c.hydrofusion_history_hours,c.lstm_sequence_length);
    std::ostringstream out;out<<"schema=hydrofusion_context_v4\nrecipe="<<c.hydrofusion_context_recipe
        <<"\nexecution_scope="<<(c.hydrofusion_development_only?"train_validation_only":"exploratory_train_validation_test")<<"\nsequence_hours="<<c.lstm_sequence_length<<"\nffn_history_hours="<<c.hydrofusion_history_hours<<"\nfeature_order=";
    for(const auto& name:featureOrderNames())out<<name<<',';
    out<<"physical_current_mean,physical_next_mean,physical_fast_storage,physical_slow_storage,current_log_error\nfeature_units=";
    for(const auto& unit:featureUnits())out<<unit<<',';
    out<<"mm/h,mm/h,mm,mm,dimensionless\nreference="
        <<(c.hydrofusion_context_recipe=="zero25"?"persistence":c.hydrofusion_context_recipe=="absolute25"?"physical_next_mean":"max(0,Q_origin+physical_next_mean-physical_current_mean)")
        <<"\nbase_metadata_scope=first_20_channels\n";
    return out.str();
}
}
