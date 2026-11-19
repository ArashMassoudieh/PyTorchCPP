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
inline std::map<int64_t,double> forecastPhysicalSeries(const std::vector<HourlyRow>& rows,const HydroRunConfig& c){
    if(c.pinn_physics_profile=="causal_two_reservoir_v1")return causalPhysicalForecasts(rows,forecastPhysics(c));
    if(c.pinn_physics_profile=="causal_two_reservoir_mean_v2")return causalPhysicalHourlyMeans(rows,forecastPhysics(c));
    if(c.pinn_physics_profile=="causal_valid_time_dry_v3")return causalPhysicalValidTimeDryForecasts(rows,forecastPhysics(c));
    throw std::invalid_argument("Unknown causal physical profile.");
}
inline TargetMode forecastTargetMode(const HydroRunConfig& c){
    if(c.hydrofusion_target_mode=="log_residual")return TargetMode::LogResidual;
    if(c.hydrofusion_target_mode=="direct_log_q")return TargetMode::DirectLogQ;
    throw std::invalid_argument("Unknown forecast target mode.");
}
inline void validateForecastConfig(const HydroRunConfig& c,const std::string& name){
    const bool native=!c.hydrofusion_native_manifest_path.empty() && c.hydrofusion_native_manifest_sha256.size()==64;
    if(!isExtendedForecast(name)||c.use_hydro_package||c.use_csv_data||
       (c.hydrofusion_synthetic_verification==native) ||
       (!native && (!c.hydrofusion_native_manifest_path.empty()||!c.hydrofusion_native_manifest_sha256.empty())))
        throw std::invalid_argument("Extended forecasts require explicit synthetic verification OR a pinned prepared native manifest, and reject legacy package/CSV training.");
    validateSplitCutoffs({c.train_end_x,c.validation_end_x,c.test_end_x});forecastTargetMode(c);
    if(c.lstm_sequence_length!=24||c.activation!="tanh"||c.normalization!="standardize"||c.optimizer!="adam"||
       c.shuffle_training||c.weight_decay!=0||c.physics_weight!=0||c.epochs<1||c.batch_size<1||!std::isfinite(c.learning_rate)||c.learning_rate<=0)
        throw std::invalid_argument("Forecast prototype requires 24 steps, tanh, standardize, Adam and positive training settings.");
    forecastHidden(c);
    if(c.hydrofusion_objective!="log_mse_v1"&&c.hydrofusion_objective!="log_flow_mse_v2")throw std::invalid_argument("Unknown forecast objective.");
    if(c.hydrofusion_ffn_activation!="tanh"&&c.hydrofusion_ffn_activation!="silu")throw std::invalid_argument("Unknown forecast FFN activation.");
    if((isRecurrentForecast(name)||name=="pinn_causal_forecast")&&c.hydrofusion_ffn_activation!="tanh")throw std::invalid_argument("SiLU option requires a forecast FFN.");
    if(name=="pinn_causal_forecast"&&c.hydrofusion_objective!="log_mse_v1")throw std::invalid_argument("Physical-only reference has no learned objective.");
    if(!isPhysicalForecast(name)&&(c.pinn_physics_profile=="causal_two_reservoir_mean_v2"||forecastRecipeV3(c)))throw std::invalid_argument("Hourly-mean profile requires a physical forecast.");
    if(c.data_weight!=1||c.lstm_bias_weight!=0||c.shared_scale_volume_weight!=0||c.lstm_bias_weight_adaptive||
       c.physical_error_weight!=0||c.high_flow_loss_weight!=0||c.lstm_dropout!=0||
       c.use_mrms_spatial_cv_feature||c.multiscale_daily_history_days!=0||c.multiscale_matched_control||c.pinn_flow_exponent!=0)
        throw std::invalid_argument("Forecast recipe does not support extra objectives, nonlinear routing, dropout, MRMS CV or multiscale branches.");
    if(isPhysicalForecast(name)){
        validateCausalPhysics(forecastPhysics(c));
        if((c.pinn_physics_profile!="causal_two_reservoir_v1"&&c.pinn_physics_profile!="causal_two_reservoir_mean_v2"&&!forecastRecipeV3(c))||c.pinn_initial_flow!=0||c.physics_dt!=1||c.hydrofusion_target_mode!="log_residual")
            throw std::invalid_argument("Physical forecast requires an explicit versioned causal profile, zero initial flow, hourly dt and log_residual correction.");
    }
}
struct ForecastNetImpl:torch::nn::Module {
    bool recurrent;HydroLSTM lstm{nullptr};torch::nn::Sequential ffn;
    ForecastNetImpl(bool sequence,int hidden,bool silu=false):recurrent(sequence){
        if(sequence&&silu)throw std::invalid_argument("FFN activation cannot change LSTM semantics.");
        if(sequence){lstm=HydroLSTM(20,hidden,1,1);register_module("lstm",lstm);}
        else{
            if(silu)ffn=torch::nn::Sequential(torch::nn::Linear(20,hidden),torch::nn::SiLU(),torch::nn::Linear(hidden,1));
            else ffn=torch::nn::Sequential(torch::nn::Linear(20,hidden),torch::nn::Tanh(),torch::nn::Linear(hidden,1));
            register_module("ffn",ffn);
        }
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
// Train-only standard deviation (same fixed constant-series fallback as the
// target scaler). Never accept validation/test observations for normalization.
inline double forecastFlowScale(const DatasetSplit& train,const SplitCutoffs& cutoffs){
    validateSplitCutoffs(cutoffs);
    if(train.targetX.size()!=train.observedTargetQ.size()||train.originX.size()!=train.targetX.size())throw std::invalid_argument("Training flow scale identities missing.");
    int64_t previous=-1;
    for(size_t i=0;i<train.targetX.size();++i){
        const auto x=train.targetX[i],origin=train.originX[i];
        if(origin<23||origin>=kSealStartX-1||x<0||x>=kSealStartX||x!=origin+1||x<=previous)throw std::invalid_argument("Invalid training scale identity.");
        const auto partition=assignSampleSplit(origin-23,x,cutoffs);
        if(!partition||*partition!=SplitPartition::Train)throw std::invalid_argument("Flow normalization must use training only.");
        previous=x;
    }
    if(train.observedTargetQ.empty())throw std::invalid_argument("Training discharge required.");
    for(double q:train.observedTargetQ)if(!std::isfinite(q)||q<0)throw std::invalid_argument("Invalid training discharge.");
    const double scale=torch::tensor(train.observedTargetQ,torch::kFloat64).std(false).item<double>();
    if(!std::isfinite(scale))throw std::runtime_error("Nonfinite training flow scale.");
    return scale<1e-12?1.:scale;
}
inline torch::Tensor forecastObjective(const HydroRunConfig& c,const torch::Tensor& predictedStandardized,
    const torch::Tensor& targetStandardized,const TensorScaler& ys,const torch::Tensor& reference,
    const torch::Tensor& observed,double qref,double trainingScale){
    // Exact old computation for every default/v1 run.
    auto logLoss=torch::mse_loss(predictedStandardized,targetStandardized);
    if(c.hydrofusion_objective=="log_mse_v1")return logLoss;
    if(c.hydrofusion_objective!="log_flow_mse_v2"||!std::isfinite(trainingScale)||trainingScale<=0)throw std::invalid_argument("Invalid flow-aware objective/scale.");
    validateQRef(qref);
    if(predictedStandardized.dim()!=2||predictedStandardized.size(1)!=1||predictedStandardized.sizes()!=targetStandardized.sizes()||
       predictedStandardized.sizes()!=reference.sizes()||reference.sizes()!=observed.sizes()||observed.numel()==0||
       !torch::isfinite(observed).all().item<bool>()||!torch::isfinite(reference).all().item<bool>()||
       !torch::isfinite(predictedStandardized).all().item<bool>()||!torch::isfinite(targetStandardized).all().item<bool>()||
       (observed<0).any().item<bool>()||(reference<0).any().item<bool>())throw std::invalid_argument("Invalid flow-aware loss tensors.");
    auto z=ys.inverseTransform(predictedStandardized).to(torch::kFloat64);
    if(forecastTargetMode(c)==TargetMode::LogResidual)z=z+torch::log1p(reference.to(torch::kFloat64)/qref);
    auto flow=torch::clamp_min(torch::expm1(z)*qref,0.);
    if(!torch::isfinite(flow).all().item<bool>())throw std::overflow_error("Flow-aware loss reconstruction overflow.");
    auto error=(flow-observed.to(torch::kFloat64))/trainingScale;
    // Fixed dimensionless 50:50 mix. Physical MSE includes mean-bias squared;
    // there is no fitted coefficient, test-based rescale or objective grid.
    return .5*logLoss.to(torch::kFloat64)+.5*error.square().mean();
}
inline torch::Tensor recipeTensor(const HydroRunConfig& c,const std::string& name){
    auto legacy=torch::tensor({1.0,static_cast<double>(isRecurrentForecast(name)),static_cast<double>(isPhysicalForecast(name)),
        static_cast<double>(name=="pinn_causal_forecast"),static_cast<double>(forecastHidden(c)),24.0,
        static_cast<double>(forecastTargetMode(c)==TargetMode::LogResidual),c.storage_coeff,c.lambda_decay,c.runoff_coeff,c.forcing_gain,c.pinn_initial_flow,c.physics_dt},torch::kFloat64);
    if(forecastRecipeV3(c)){
        legacy[0]=3.;
        // Profile fixes valid-time integration, dry-gap policy and hourly mean.
        return torch::cat({legacy,torch::tensor({3.,static_cast<double>(c.hydrofusion_objective=="log_flow_mse_v2"),static_cast<double>(c.hydrofusion_ffn_activation=="silu")},torch::kFloat64)});
    }
    if(!forecastRecipeV2(c))return legacy;
    legacy[0]=2.;
    return torch::cat({legacy,torch::tensor({static_cast<double>(c.pinn_physics_profile=="causal_two_reservoir_mean_v2"),
        static_cast<double>(c.hydrofusion_objective=="log_flow_mse_v2"),static_cast<double>(c.hydrofusion_ffn_activation=="silu")},torch::kFloat64)});
}
} // namespace hydrofusion
