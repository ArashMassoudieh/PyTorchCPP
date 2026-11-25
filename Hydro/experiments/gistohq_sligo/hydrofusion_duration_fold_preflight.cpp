// Chronological internal-selection feature preflight; no models/fitting/test data.
#include "hydrofusion_forecast_core.h"
#include "hydrofusion_prepared_native_package.h"
#include "../../evaluation/experiment_loader.h"
#include <iostream>
#include <iomanip>
int main(int argc,char** argv){try{
 if(argc!=3)throw std::invalid_argument("config NEWsupport");
 using namespace hydrofusion;if(std::filesystem::exists(argv[2]))throw std::invalid_argument("Existing support");
 torch::set_num_threads(2);auto c=HydroExperimentLoader().loadConfig(argv[1]).config;
 if(!c.hydrofusion_development_only||c.train_end_x<1||c.train_end_x>=c.validation_end_x||c.validation_end_x>17543||c.test_end_x!=17544||c.lstm_sequence_length!=24)throw std::invalid_argument("TRAIN-only chronological fold required");
 const auto rows=loadPreparedNativePackage(c.hydrofusion_native_manifest_path,c.hydrofusion_native_manifest_sha256,c.validation_end_x);const auto rain=loadBoundPreissueRain(c);const auto mask=rainEligibleTargets(c,rain);
 auto data=prepareHydroFusionSequenceData(rows,TargetMode::LogResidual,{c.train_end_x,c.validation_end_x,c.test_end_x},24,true,&mask);
 if(data.train.targetX.empty()||data.validation.targetX.empty()||data.test.x.defined())throw std::runtime_error("Invalid internal support");
 const auto contexts=preissueRainContexts(rows,forecastPhysics(c),rain,rainHold(c));
 addForecastContext(data.train,contexts,c,data.qRef);addForecastContext(data.validation,contexts,c,data.qRef);
 addPreissueRainChannel(data.train,c,rain);addPreissueRainChannel(data.validation,c,rain);
 if(data.train.x.size(2)!=forecastChannels(c)||data.validation.x.size(2)!=forecastChannels(c)||!torch::isfinite(data.train.x).all().item<bool>()||!torch::isfinite(data.validation.x).all().item<bool>())throw std::runtime_error("Invalid feature assembly");
 std::ofstream out(argv[2]);out<<"split,target_x,observed,persistence\n"<<std::setprecision(17);
 for(const auto& item:std::vector<std::pair<std::string,const DatasetSplit*>>{{"train",&data.train},{"validation",&data.validation}}){const auto& s=*item.second;auto q=s.originQ.contiguous();for(size_t i=0;i<s.targetX.size();++i)out<<item.first<<','<<s.targetX[i]<<','<<s.observedTargetQ[i]<<','<<q.data_ptr<double>()[i]<<'\n';}
 out.close();if(!out)throw std::runtime_error("Support output failed");
 std::cout<<"PASS inner_train="<<data.train.targetX.size()<<" inner_validation="<<data.validation.targetX.size()<<" test=0 q_ref="<<std::setprecision(17)<<data.qRef<<'\n';
}catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
