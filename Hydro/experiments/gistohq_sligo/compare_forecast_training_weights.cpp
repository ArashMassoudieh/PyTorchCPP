// Compare two managed bundles after a validation-only counterfactual. No fitting.
#include "../../evaluation/artifact_loader.h"
#include "hydrofusion_forecast_core.h"
#include <iostream>
using namespace hydrofusion;
int main(int argc,char**argv){try{
 if(argc!=3)throw std::invalid_argument("Two bundles required");
 torch::set_num_threads(2);
 auto a=HydroArtifactLoader().loadForInference(argv[1]),b=HydroArtifactLoader().loadForInference(argv[2]);
 if(a.models.size()!=1||b.models.size()!=1||a.models.begin()->first!=b.models.begin()->first)throw std::runtime_error("Model inventory mismatch");
 const auto name=a.models.begin()->first;const auto& c=a.experiment.config;
 if(c.hydrofusion_checkpoint_selection!="fixed_last_epoch_v1"||b.experiment.config.hydrofusion_checkpoint_selection!="fixed_last_epoch_v1")throw std::runtime_error("Fixed epoch policy required");
 auto load=[&](const std::vector<unsigned char>& bytes){std::string raw(bytes.begin(),bytes.end());std::istringstream stream(raw);torch::serialize::InputArchive archive,weights;archive.load_from(stream);archive.read("weights",weights);ForecastNet model(isRecurrentForecast(name),forecastHidden(c),false,forecastChannels(c),parseForecastHistory(c.hydrofusion_history_hours,c.lstm_sequence_length));model->load(weights);return model;};
 auto left=load(a.models.at(name).bytes),right=load(b.models.at(name).bytes);const auto p=left->named_parameters(),q=right->named_parameters();
 if(p.size()!=q.size())throw std::runtime_error("Parameter inventory mismatch");
 for(const auto& item:p)if(!q.contains(item.key())||!torch::equal(item.value(),q[item.key()]))throw std::runtime_error("Validation changed weights");
 const auto same=[](const HydroScalerState& x,const HydroScalerState& y){return x.method==y.method&&x.shape==y.shape&&x.offset==y.offset&&x.scale==y.scale;};
 if(!same(a.scalers.at(name).input,b.scalers.at(name).input)||!same(a.scalers.at(name).target,b.scalers.at(name).target))throw std::runtime_error("Validation changed scaler");
 std::cout<<"PASS validation-Q counterfactual: "<<p.size()<<" parameter tensors and both scalers bit-identical\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
