#include "../../models/hydrofusion_forecast_core.h"
#include "../../evaluation/experiment_loader.h"
#include <iostream>
#include <functional>
using namespace hydrofusion;
static void check(bool ok){if(!ok)throw std::runtime_error("Forecast v2 unit check failed");}
static void rejects(const std::function<void()>&f){bool rejected=false;try{f();}catch(const std::exception&){rejected=true;}check(rejected);}
int main(int argc,char**argv){try{
    if(argc!=2)throw std::runtime_error("Pass existing synthetic FFN config");torch::set_num_threads(1);
    auto c=HydroExperimentLoader().loadConfig(argv[1]).config;
    validateForecastConfig(c,"ffn_causal_forecast");
    check(forecastCheckpointFormat(c)=="hydrofusion-forecast-v1");
    auto old=recipeTensor(c,"ffn_causal_forecast");check(old.numel()==13&&old[0].item<double>()==1);
    TensorScaler ys;HydroScalerState identity;identity.method="standardize";identity.shape={1,1};ys.importState(identity);
    auto pred=torch::tensor({{.2},{.4}},torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
    auto target=torch::tensor({{0.},{std::log(2.)}},torch::kFloat64);
    auto obs=torch::tensor({{0.},{1.}},torch::kFloat64),ref=torch::zeros_like(obs);
    check(torch::equal(forecastObjective(c,pred,target,ys,ref,obs,1,1),torch::mse_loss(pred,target)));
    c.hydrofusion_target_mode="direct_log_q";c.hydrofusion_objective="log_flow_mse_v2";
    auto loss=forecastObjective(c,pred,target,ys,ref,obs,1,2);
    auto physical=torch::expm1(pred);auto expected=.5*torch::mse_loss(pred,target)+.5*((physical-obs)/2).square().mean();
    check(std::abs(loss.item<double>()-expected.item<double>())<1e-12);loss.backward();
    check(torch::isfinite(pred.grad()).all().item<bool>()&&pred.grad().abs().sum().item<double>()>0);
    c.hydrofusion_target_mode="log_residual";
    auto residual=forecastObjective(c,pred,target,ys,torch::ones_like(ref),obs,1,2);
    auto residualFlow=torch::expm1(pred+std::log(2.));
    check(std::abs(residual.item<double>()-(.5*torch::mse_loss(pred,target)+.5*((residualFlow-obs)/2).square().mean()).item<double>())<1e-12);
    c.hydrofusion_target_mode="direct_log_q";
    check(recipeTensor(c,"ffn_causal_forecast").numel()==16&&forecastCheckpointFormat(c)=="hydrofusion-forecast-v2");
    rejects([&]{forecastObjective(c,pred,target,ys,ref,obs,1,0);});
    rejects([&]{forecastObjective(c,pred,target,ys,ref,obs,-1,1);});
    rejects([&]{forecastObjective(c,torch::full_like(pred,1000),target,ys,ref,obs,1,1);});
    rejects([&]{forecastObjective(c,pred,target,ys,ref,torch::full_like(obs,NAN),1,1);});
    rejects([&]{forecastObjective(c,pred,target,ys,ref.slice(0,0,1),obs,1,1);});
    DatasetSplit train;train.targetX={100,101};train.originX={99,100};train.observedTargetQ={1,3};
    check(forecastFlowScale(train,{150,220,280})==1);
    auto bad=train;bad.targetX={221,222};bad.originX={220,221};
    rejects([&]{forecastFlowScale(bad,{150,220,280});});
    bad=train;bad.targetX={77015,77016};bad.originX={77014,77015};
    rejects([&]{forecastFlowScale(bad,{150,220,280});});
    bad=train;bad.originX={std::numeric_limits<int64_t>::max(),100};rejects([&]{forecastFlowScale(bad,{150,220,280});});
    bad=train;bad.targetX={100,100};bad.originX={99,99};rejects([&]{forecastFlowScale(bad,{150,220,280});});
    auto unknown=c;unknown.hydrofusion_objective="unknown";rejects([&]{validateForecastConfig(unknown,"ffn_causal_forecast");});
    auto head=c;head.hydrofusion_ffn_activation="silu";validateForecastConfig(head,"ffn_causal_forecast");
    rejects([&]{validateForecastConfig(head,"lstm_hydrofusion");});
    torch::manual_seed(42);ForecastNet first(false,8);
    torch::manual_seed(42);ForecastNet second(false,8,false);
    auto x=torch::ones({2,24,20});check(torch::equal(first->forward(x),second->forward(x)));
    ForecastNet bounded(false,8),unbounded(false,8,true);
    {torch::NoGradGuard noGrad;for(auto&t:bounded->parameters())t.fill_(1);for(auto&t:unbounded->parameters())t.fill_(1);}
    auto medium=torch::ones({1,24,20})*10,large=medium*100;
    check(bounded->forward(large).item<double>()==bounded->forward(medium).item<double>());
    check(unbounded->forward(large).item<double>()>50*unbounded->forward(medium).item<double>());
    auto physicalConfig=HydroExperimentLoader().loadConfig(argv[1]).config;physicalConfig.hydrofusion_target_mode="log_residual";
    physicalConfig.pinn_physics_profile="causal_two_reservoir_mean_v2";validateForecastConfig(physicalConfig,"pinn_causal_forecast");
    check(recipeTensor(physicalConfig,"pinn_causal_forecast")[13].item<double>()==1);
    std::cout<<"PASS v1 exact objective/net defaults; v2 formula/gradient/overflow/domain/Train-only-scale; recipe binding; unbounded FFN; physical v2 dispatch\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
