#include "hydrofusion_forecast_wrapper.h"
#include "hydrofusion_forecast_core.h"
#include "../experiments/gistohq_sligo/hydrofusion_synthetic_fixture.h"
#include "../evaluation/hydro_metrics.h"
#include "../experiments/gistohq_sligo/hydrofusion_prepared_native_package.h"
#include <limits>

HydroRunResult HydroFusionForecastWrapper::train(const HydroRunConfig& c,const std::string& approach){
    using namespace hydrofusion;validateForecastConfig(c,approach);
    const auto rows=c.hydrofusion_synthetic_verification?buildSyntheticFixture():loadPreparedNativePackage(c.hydrofusion_native_manifest_path,c.hydrofusion_native_manifest_sha256);const auto mode=forecastTargetMode(c);
    auto data=prepareHydroFusionSequenceData(rows,mode,{c.train_end_x,c.validation_end_x,c.test_end_x},24);
    std::map<int64_t,double> physics;if(isPhysicalForecast(approach))physics=forecastPhysicalSeries(rows,c);
    const auto trainRef=forecastReference(data.train,physics,isPhysicalForecast(approach));
    const auto valRef=forecastReference(data.validation,physics,isPhysicalForecast(approach));
    const auto testRef=forecastReference(data.test,physics,isPhysicalForecast(approach));
    const bool physicalOnly=approach=="pinn_causal_forecast";
    const double qref=physicalOnly?1.0:data.qRef;
    auto trainY=forecastTargets(data.train,mode,qref,trainRef),valY=forecastTargets(data.validation,mode,qref,valRef);
    TensorScaler xs,ys;ForecastNet net{nullptr};HydroRunResult result;
    const double flowScale=c.hydrofusion_objective=="log_flow_mse_v2"?forecastFlowScale(data.train,{c.train_end_x,c.validation_end_x,c.test_end_x}):1.;
    torch::Tensor trainObserved,validationObserved;
    if(c.hydrofusion_objective=="log_flow_mse_v2"){
        trainObserved=torch::tensor(data.train.observedTargetQ,torch::kFloat64).reshape({-1,1});
        validationObserved=torch::tensor(data.validation.observedTargetQ,torch::kFloat64).reshape({-1,1});
    }
    if(!physicalOnly){
        xs.fit(data.train.x,"standardize");ys.fit(trainY,"standardize");
        torch::manual_seed(static_cast<uint64_t>(std::max(0,c.random_seed)));net=ForecastNet(isRecurrentForecast(approach),forecastHidden(c),c.hydrofusion_ffn_activation=="silu");
        torch::optim::Adam optimizer(net->parameters(),torch::optim::AdamOptions(c.learning_rate));
        double best=std::numeric_limits<double>::infinity();std::string bestBytes;
        for(int epoch=1;epoch<=c.epochs;++epoch){
            net->train();double sum=0;const auto n=data.train.x.size(0);
            for(int64_t start=0;start<n;start+=c.batch_size){const auto end=std::min(n,start+c.batch_size);
                optimizer.zero_grad();auto standardized=net->forward(xs.transform(data.train.x.slice(0,start,end)));
                auto loss=c.hydrofusion_objective=="log_mse_v1"?torch::mse_loss(standardized,ys.transform(trainY.slice(0,start,end))):
                    forecastObjective(c,standardized,ys.transform(trainY.slice(0,start,end)),ys,trainRef.slice(0,start,end),
                        trainObserved.slice(0,start,end),qref,flowScale);
                if(!std::isfinite(loss.item<double>()))throw std::runtime_error("Nonfinite forecast training loss.");
                loss.backward();optimizer.step();sum+=loss.item<double>()*(end-start);
            }
            torch::NoGradGuard guard;net->eval();const auto standardized=net->forward(xs.transform(data.validation.x));
            const double validation=(c.hydrofusion_objective=="log_mse_v1"?torch::mse_loss(standardized,ys.transform(valY)):
                forecastObjective(c,standardized,ys.transform(valY),ys,valRef,validationObserved,qref,flowScale)).item<double>();
            if(!std::isfinite(validation))throw std::runtime_error("Nonfinite forecast validation loss.");
            result.training_loss_history.push_back(sum/n);result.validation_loss_history.push_back(validation);
            if(validation<best){best=validation;result.best_epoch=epoch;torch::serialize::OutputArchive weights;net->save(weights);std::ostringstream bytes;weights.save_to(bytes);bestBytes=bytes.str();}
        }
        if(bestBytes.empty())throw std::runtime_error("No selected forecast checkpoint.");
        std::istringstream bytes(bestBytes);torch::serialize::InputArchive selected;selected.load_from(bytes);net->load(selected);net->eval();
        result.final_loss=result.training_loss_history.at(result.best_epoch-1);result.input_scaler=xs.exportState();result.target_scaler=ys.exportState();
    }else{result.input_scaler.method="none";result.target_scaler.method="none";result.best_epoch=0;result.final_loss=0;}
    torch::serialize::OutputArchive checkpoint;checkpoint.write("recipe",recipeTensor(c,approach));checkpoint.write("qref",torch::tensor(qref,torch::kFloat64));
    if(c.hydrofusion_objective=="log_flow_mse_v2")checkpoint.write("training_flow_scale",torch::tensor(flowScale,torch::kFloat64));
    if(!physicalOnly){torch::serialize::OutputArchive weights;net->save(weights);checkpoint.write("weights",weights);}
    std::ostringstream saved;checkpoint.save_to(saved);const auto raw=saved.str();result.model_checkpoint.assign(raw.begin(),raw.end());result.model_checkpoint_format=forecastCheckpointFormat(c);
    torch::NoGradGuard guard;
    if(physicalOnly){auto values=testRef.contiguous();result.y_pred.assign(values.data_ptr<double>(),values.data_ptr<double>()+values.numel());}
    else result.y_pred=forecastReconstruct(mode,ys.inverseTransform(net->forward(xs.transform(data.test.x))),testRef,qref);
    std::vector<double> valPred;
    if(physicalOnly){auto values=valRef.contiguous();valPred.assign(values.data_ptr<double>(),values.data_ptr<double>()+values.numel());}
    else valPred=forecastReconstruct(mode,ys.inverseTransform(net->forward(xs.transform(data.validation.x))),valRef,qref);
    result.validation_mse=0;for(size_t i=0;i<valPred.size();++i){const auto e=valPred[i]-data.validation.observedTargetQ[i];result.validation_mse+=e*e;}result.validation_mse/=valPred.size();
    result.y_true=data.test.observedTargetQ;result.split.assign(result.y_pred.size(),"test");
    for(auto x:data.test.targetX)result.x.push_back(static_cast<double>(x));
    result.success=true;populateHydroMetrics(result,result.y_true,result.y_pred);populateHydroPeakMetrics(result);
    result.hydrofusion_target_mode=c.hydrofusion_target_mode;result.hydrofusion_q_ref=qref;result.hydrofusion_train_end_x=c.train_end_x;result.hydrofusion_validation_end_x=c.validation_end_x;result.hydrofusion_test_end_x=c.test_end_x;result.hydrofusion_test_origin_x=data.test.originX;
    return result;
}
