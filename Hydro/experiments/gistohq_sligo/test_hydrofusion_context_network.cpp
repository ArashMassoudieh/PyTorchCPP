#include "../../models/hydrofusion_forecast_core.h"
#include "hydrofusion_synthetic_fixture.h"
#include <functional>
#include <iostream>
using namespace hydrofusion;
int checks=0;
void check(bool ok){++checks;if(!ok)throw std::runtime_error("Context network check "+std::to_string(checks));}
void rejects(std::function<void()> f){bool bad=false;try{f();}catch(const std::exception&){bad=true;}check(bad);}
int main(){torch::set_num_threads(1);
 HydroRunConfig c;c.hydrofusion_synthetic_verification=true;c.train_end_x=150;c.validation_end_x=220;c.test_end_x=280;
 c.lstm_sequence_length=24;c.hidden_layers_csv="8";c.activation="tanh";c.normalization="standardize";c.optimizer="adam";c.shuffle_training=false;c.physics_weight=0;c.physics_dt=1;c.pinn_initial_flow=0;
 c.pinn_physics_profile="causal_valid_time_dry_v3";c.hydrofusion_context_recipe="anchored25";c.hydrofusion_history_hours="0,1,3,6,12,23";
 validateForecastConfig(c,"ffn_physical_forecast");check(forecastCheckpointFormat(c)=="hydrofusion-forecast-v4");
 auto recipe=recipeTensor(c,"ffn_physical_forecast");check(recipe[0].item<double>()==4);
 auto wrong=c;wrong.use_time_lagged_ffn=true;rejects([&]{validateForecastConfig(wrong,"ffn_physical_forecast");});
 wrong=c;wrong.hydrofusion_history_hours="0,24";rejects([&]{validateForecastConfig(wrong,"ffn_physical_forecast");});
 auto rows=buildSyntheticFixture();auto data=prepareHydroFusionSequenceData(rows,TargetMode::LogResidual,{150,220,280},24);
 const auto trace=causalPhysicalOriginContexts(rows,forecastPhysics(c));auto old=data.train.x.clone();
 addForecastContext(data.train,trace,c,data.qRef);check(data.train.x.size(2)==25);check(torch::equal(data.train.x.slice(2,0,20),old));
 auto zero=data.validation;c.hydrofusion_context_recipe="zero25";addForecastContext(zero,trace,c,data.qRef);check((zero.x.slice(2,20,25)==0).all().item<bool>());
 check(torch::equal(contextReferences(zero,trace,c),zero.originQ));c.hydrofusion_context_recipe="anchored25";
 ForecastNet net(false,8,false,25,parseForecastHistory(c.hydrofusion_history_hours,24));
 auto x=data.train.x.slice(0,0,2).clone().set_requires_grad(true);
 auto y=net->forward(x);std::vector<torch::Tensor> selected;
 for(auto h:parseForecastHistory(c.hydrofusion_history_hours,24))selected.push_back(x.select(1,23-h));
 check(torch::equal(y,net->ffn->forward(torch::cat(selected,1))));y.square().mean().backward();check(torch::isfinite(x.grad()).all().item<bool>());
 auto mask=torch::ones({24},torch::kBool);for(auto h:parseForecastHistory(c.hydrofusion_history_hours,24))mask[23-h]=false;
 check(x.grad().index({torch::indexing::Slice(),mask}).abs().sum().item<double>()==0);
 ForecastNet recurrent(true,8,false,25);check(recurrent->forward(x).sizes()==torch::IntArrayRef({2,1}));
 c.hydrofusion_history_hours="0";c.hydrofusion_development_only=true;
 auto dev=prepareHydroFusionSequenceData(buildSyntheticFixture(220),TargetMode::LogResidual,{150,220,280},24,true);
 check(!dev.test.x.defined()&&dev.test.targetX.empty());check(dev.qRef==data.qRef);
 rejects([&]{prepareHydroFusionSequenceData(rows,TargetMode::LogResidual,{150,220,280},24,true);});
 auto poisoned=rows;poisoned.back().observed_runoff_mm_per_hour=std::numeric_limits<double>::quiet_NaN();
 bool identityFirst=false;try{prepareHydroFusionSequenceData(poisoned,TargetMode::LogResidual,{150,220,280},24,true);}
 catch(const std::invalid_argument& e){identityFirst=std::string(e.what()).find("Development assembly refuses")!=std::string::npos;}
 check(identityFirst);
 auto malformed=trace;malformed.erase(data.train.originX.front()-23);auto split=dev.train;rejects([&]{addForecastContext(split,malformed,c,dev.qRef);});
 std::cout<<"PASS context network checks="<<checks<<'\n';
}
