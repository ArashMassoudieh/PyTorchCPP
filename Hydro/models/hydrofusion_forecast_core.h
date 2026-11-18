#pragma once
#include "hydro_run_types.h"
#include "hydrofusion_forecast_modes.h"
#include "hydro_lstm_module.h"
#include "../experiments/gistohq_sligo/hydrofusion_sequence_assembly.h"
#include "../experiments/gistohq_sligo/hydrofusion_causal_physics.h"
#include "../dataset/tensor_scaler.h"
#include <sstream>

namespace hydrofusion {
inline int forecastHidden(const HydroRunConfig& c){
    size_t used=0;int hidden=std::stoi(c.hidden_layers_csv,&used);
    if(used!=c.hidden_layers_csv.size()||hidden<1||hidden>256)throw std::invalid_argument("Forecast prototype requires one hidden width in [1,256].");
    return hidden;
}
inline CausalPhysicsParams forecastPhysics(const HydroRunConfig& c){return {c.storage_coeff,c.lambda_decay,c.runoff_coeff,c.forcing_gain};}
inline TargetMode forecastTargetMode(const HydroRunConfig& c){
    if(c.hydrofusion_target_mode=="log_residual")return TargetMode::LogResidual;
    if(c.hydrofusion_target_mode=="direct_log_q")return TargetMode::DirectLogQ;
    throw std::invalid_argument("Unknown forecast target mode.");
}
inline void validateForecastConfig(const HydroRunConfig& c,const std::string& name){
    if(!isExtendedForecast(name)||c.use_hydro_package||c.use_csv_data||!c.hydrofusion_synthetic_verification)
        throw std::invalid_argument("Extended forecasts require explicit synthetic verification and reject package/CSV training.");
    validateSplitCutoffs({c.train_end_x,c.validation_end_x,c.test_end_x});forecastTargetMode(c);
    if(c.lstm_sequence_length!=24||c.activation!="tanh"||c.normalization!="standardize"||c.optimizer!="adam"||
       c.shuffle_training||c.weight_decay!=0||c.physics_weight!=0||c.epochs<1||c.batch_size<1||!std::isfinite(c.learning_rate)||c.learning_rate<=0)
        throw std::invalid_argument("Forecast prototype requires 24 steps, tanh, standardize, Adam and positive training settings.");
    forecastHidden(c);
    if(c.data_weight!=1||c.lstm_bias_weight!=0||c.shared_scale_volume_weight!=0||c.lstm_bias_weight_adaptive||
       c.physical_error_weight!=0||c.high_flow_loss_weight!=0||c.lstm_dropout!=0||
       c.use_mrms_spatial_cv_feature||c.multiscale_daily_history_days!=0||c.multiscale_matched_control||c.pinn_flow_exponent!=0)
        throw std::invalid_argument("Forecast recipe does not support extra objectives, nonlinear routing, dropout, MRMS CV or multiscale branches.");
    if(isPhysicalForecast(name)){
        validateCausalPhysics(forecastPhysics(c));
        if(c.pinn_physics_profile!="causal_two_reservoir_v1"||c.pinn_initial_flow!=0||c.physics_dt!=1||c.hydrofusion_target_mode!="log_residual")
            throw std::invalid_argument("Physical forecast requires explicit causal_two_reservoir_v1, zero initial flow, hourly dt and log_residual correction.");
    }
}
struct ForecastNetImpl:torch::nn::Module {
    bool recurrent;HydroLSTM lstm{nullptr};torch::nn::Sequential ffn;
    ForecastNetImpl(bool sequence,int hidden):recurrent(sequence){
        if(sequence){lstm=HydroLSTM(20,hidden,1,1);register_module("lstm",lstm);}
        else{ffn=torch::nn::Sequential(torch::nn::Linear(20,hidden),torch::nn::Tanh(),torch::nn::Linear(hidden,1));register_module("ffn",ffn);}
    }
    torch::Tensor forward(const torch::Tensor& x){return recurrent?lstm->forward(x):ffn->forward(x.select(1,x.size(1)-1));}
};
TORCH_MODULE(ForecastNet);
inline torch::Tensor forecastReference(const DatasetSplit& split,const std::map<int64_t,double>& physics,bool physical){
    if(!physical)return split.originQ;
    std::vector<double> q;for(auto x:split.originX){auto found=physics.find(x);if(found==physics.end())throw std::invalid_argument("Physical context missing.");q.push_back(found->second);}
    return torch::tensor(q,torch::kFloat64).reshape({-1,1});
}
inline torch::Tensor forecastTargets(const DatasetSplit& split,TargetMode mode,double qref,const torch::Tensor& reference){
    std::vector<double> values;auto refs=reference.contiguous();
    for(size_t i=0;i<split.observedTargetQ.size();++i)values.push_back(computeTrainingTarget(mode,refs.data_ptr<double>()[i],split.observedTargetQ[i],qref));
    auto tensor=torch::tensor(values,torch::kFloat64).to(torch::kFloat32).reshape({-1,1});
    if(!torch::isfinite(tensor).all().item<bool>())throw std::overflow_error("Forecast targets overflow float32.");
    return tensor;
}
inline std::vector<double> forecastReconstruct(TargetMode mode,const torch::Tensor& transformed,const torch::Tensor& reference,double qref){return reconstructHydroFusionColumn(mode,transformed,reference,qref);}
inline torch::Tensor recipeTensor(const HydroRunConfig& c,const std::string& name){
    return torch::tensor({1.0,static_cast<double>(isRecurrentForecast(name)),static_cast<double>(isPhysicalForecast(name)),
        static_cast<double>(name=="pinn_causal_forecast"),static_cast<double>(forecastHidden(c)),24.0,
        static_cast<double>(forecastTargetMode(c)==TargetMode::LogResidual),c.storage_coeff,c.lambda_decay,c.runoff_coeff,c.forcing_gain,c.pinn_initial_flow,c.physics_dt},torch::kFloat64);
}
} // namespace hydrofusion
