#include "../../evaluation/hydrofusion_forecast_inference.h"
#include <iostream>
// Archive/config/scaler/recipe compatibility only; no fitting, raw observations
// or repeated all-row diagnostic. Full causal row parity is tested separately.
int main(int argc,char**argv){try{
    if(argc<2)throw std::invalid_argument("Pass existing managed v1 bundles.");
    torch::set_num_threads(1);
    for(int i=1;i<argc;++i){
        auto artifacts=HydroArtifactLoader().loadForInference(argv[i]);
        if(artifacts.models.size()!=1)throw std::runtime_error("Expected one model.");
        const auto approach=artifacts.models.begin()->first;
        if(artifacts.models.begin()->second.format!="hydrofusion-forecast-v1"||hydrofusion::forecastRecipeV2(artifacts.experiment.config))throw std::runtime_error("Expected unchanged v1 recipe.");
        HydroFusionForecastInference predictor(artifacts,argv[i],approach);
        std::cout<<"PASS managed v1 archive/config/scaler/recipe reload: "<<argv[i]<<'\n';
    }
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
