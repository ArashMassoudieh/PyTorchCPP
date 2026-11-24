// Feature assembly only: no neural model construction, fitting or test reads.
#include "hydrofusion_forecast_core.h"
#include "hydrofusion_prepared_native_package.h"
#include <iostream>
#include <iomanip>
int main(int argc,char** argv){try{
    if(argc!=6)throw std::invalid_argument("Pass development manifest, SHA, weather CSV provenance SHA and new support CSV.");
    using namespace hydrofusion;
    if(std::filesystem::exists(argv[5]))throw std::invalid_argument("New support output required.");
    HydroRunConfig c;c.train_end_x=17543;c.validation_end_x=26303;c.test_end_x=35063;
    c.hydrofusion_development_only=true;c.lstm_sequence_length=24;
    c.pinn_physics_profile="causal_valid_time_dry_v3";
    c.hydrofusion_preissue_rain_recipe="rate_hold_v1";
    c.hydrofusion_preissue_rain_path=argv[3];c.hydrofusion_preissue_rain_sha256=sha256File(argv[3]);
    c.hydrofusion_preissue_rain_provenance_sha256=argv[4];
    const auto rows=loadPreparedNativePackage(argv[1],argv[2],c.validation_end_x);
    const auto rain=loadBoundPreissueRain(c);const auto mask=rainEligibleTargets(c,rain);
    const auto data=prepareHydroFusionSequenceData(rows,TargetMode::LogResidual,{c.train_end_x,c.validation_end_x,c.test_end_x},24,true,&mask);
    if(data.train.x.size(0)!=13072||data.validation.x.size(0)!=7112||data.test.x.defined())throw std::runtime_error("Unexpected matched support.");
    const auto contexts=preissueRainContexts(rows,forecastPhysics(c),rain,true);
    auto train=data.train,validation=data.validation;
    addPreissueRainChannel(train,c,rain);addPreissueRainChannel(validation,c,rain);
    if(train.x.size(2)!=21||!torch::isfinite(train.x).all().item<bool>()||!torch::isfinite(validation.x).all().item<bool>())throw std::runtime_error("Native weather features invalid.");
    std::ofstream support(argv[5],std::ios::out|std::ios::trunc);support<<"split,target_x,observed,persistence\n"<<std::setprecision(17);
    for(const auto& item:std::vector<std::pair<std::string,const DatasetSplit*>>{{"train",&data.train},{"validation",&data.validation}}){
        const auto& split=*item.second;auto origins=split.originQ.contiguous();
        for(size_t i=0;i<split.targetX.size();++i)support<<item.first<<','<<split.targetX[i]<<','<<split.observedTargetQ[i]<<','<<origins.data_ptr<double>()[i]<<'\n';
    }
    support.close();if(!support)throw std::runtime_error("Support write failed.");
    std::cout<<std::setprecision(17)<<"PASS native preissue feature-only train="<<train.x.size(0)<<" validation="<<validation.x.size(0)<<" test=0 channels="<<train.x.size(2)<<" q_ref="<<data.qRef<<" weather_rows="<<rain.size()<<" contexts="<<contexts.size()<<'\n';
}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
