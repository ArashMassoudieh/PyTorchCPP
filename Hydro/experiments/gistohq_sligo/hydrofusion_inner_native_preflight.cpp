// Chronological internal-selection feature preflight; no models/fitting/test data.
#include "hydrofusion_forecast_core.h"
#include "hydrofusion_prepared_native_package.h"
#include <iostream>
#include <iomanip>
int main(int argc,char** argv){try{
 if(argc!=6)throw std::invalid_argument("manifest SHA weather provenanceSHA NEWsupport");
 using namespace hydrofusion;if(std::filesystem::exists(argv[5]))throw std::invalid_argument("Existing support");
 HydroRunConfig c;c.train_end_x=8759;c.validation_end_x=17543;c.test_end_x=26303;c.hydrofusion_development_only=true;c.lstm_sequence_length=24;
 c.pinn_physics_profile="causal_valid_time_dry_v3";c.hydrofusion_preissue_rain_recipe="rate_hold_v1";c.hydrofusion_preissue_rain_path=argv[3];c.hydrofusion_preissue_rain_sha256=sha256File(argv[3]);c.hydrofusion_preissue_rain_provenance_sha256=argv[4];
 const auto rows=loadPreparedNativePackage(argv[1],argv[2],17543);const auto rain=loadBoundPreissueRain(c);const auto mask=rainEligibleTargets(c,rain);
 auto data=prepareHydroFusionSequenceData(rows,TargetMode::LogResidual,{8759,17543,26303},24,true,&mask);
 if(data.train.targetX.empty()||data.validation.targetX.empty()||data.test.x.defined())throw std::runtime_error("Invalid internal support");
 addPreissueRainChannel(data.train,c,rain);addPreissueRainChannel(data.validation,c,rain);
 if(data.train.x.size(2)!=21||data.validation.x.size(2)!=21||!torch::isfinite(data.train.x).all().item<bool>()||!torch::isfinite(data.validation.x).all().item<bool>())throw std::runtime_error("Invalid feature assembly");
 std::ofstream out(argv[5]);out<<"split,target_x,observed,persistence\n"<<std::setprecision(17);
 for(const auto& item:std::vector<std::pair<std::string,const DatasetSplit*>>{{"train",&data.train},{"validation",&data.validation}}){const auto& s=*item.second;auto q=s.originQ.contiguous();for(size_t i=0;i<s.targetX.size();++i)out<<item.first<<','<<s.targetX[i]<<','<<s.observedTargetQ[i]<<','<<q.data_ptr<double>()[i]<<'\n';}
 out.close();if(!out)throw std::runtime_error("Support output failed");
 std::cout<<"PASS inner_train="<<data.train.targetX.size()<<" inner_validation="<<data.validation.targetX.size()<<" test=0 q_ref="<<std::setprecision(17)<<data.qRef<<'\n';
}catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
