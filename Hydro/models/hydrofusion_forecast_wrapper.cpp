#include "hydrofusion_forecast_wrapper.h"
#include "hydrofusion_forecast_core.h"
#include "../experiments/gistohq_sligo/hydrofusion_synthetic_fixture.h"
#include "../evaluation/hydro_metrics.h"
#include "../experiments/gistohq_sligo/hydrofusion_prepared_native_package.h"
#include <limits>

HydroRunResult HydroFusionForecastWrapper::train(const HydroRunConfig& c,const std::string& approach){
    using namespace hydrofusion;validateForecastConfig(c,approach);
    const auto rows=c.hydrofusion_synthetic_verification?buildSyntheticFixture(c.hydrofusion_development_only?std::min<int64_t>(299,c.validation_end_x):299):loadPreparedNativePackage(c.hydrofusion_native_manifest_path,c.hydrofusion_native_manifest_sha256,c.hydrofusion_development_only?c.validation_end_x:kSealStartX-1);const auto mode=forecastTargetMode(c);
    const auto rain=loadBoundPreissueRain(c);const auto targetMask=rainEligibleTargets(c,rain);
    auto data=prepareHydroFusionSequenceData(rows,mode,{c.train_end_x,c.validation_end_x,c.test_end_x},c.lstm_sequence_length,c.hydrofusion_development_only,forecastRecipeV5(c)?&targetMask:nullptr);
    std::map<int64_t,double> physics;if(isPhysicalForecast(approach))physics=forecastPhysicalSeries(rows,c);
    std::map<int64_t,CausalPhysicalOriginContext> contexts;
    if(forecastRecipeV4(c)||forecastRecipeV5(c)){
        contexts=forecastRecipeV5(c)?preissueRainContexts(rows,forecastPhysics(c),rain,rainHold(c)):causalPhysicalOriginContexts(rows,forecastPhysics(c));
        if(forecastRecipeV5(c)&&isPhysicalForecast(approach)){physics.clear();for(const auto& row:contexts)physics.emplace(row.first,row.second.next_hour_mean);}
        addForecastContext(data.train,contexts,c,data.qRef);addForecastContext(data.validation,contexts,c,data.qRef);
        if(!c.hydrofusion_development_only)addForecastContext(data.test,contexts,c,data.qRef);
    }
    if(forecastRecipeV5(c)){addPreissueRainChannel(data.train,c,rain);addPreissueRainChannel(data.validation,c,rain);}
    const auto trainRef=forecastRecipeV4(c)?contextReferences(data.train,contexts,c):forecastReference(data.train,physics,isPhysicalForecast(approach));
    const auto valRef=forecastRecipeV4(c)?contextReferences(data.validation,contexts,c):forecastReference(data.validation,physics,isPhysicalForecast(approach));
    const auto& evaluation=c.hydrofusion_development_only?data.validation:data.test;
    const auto testRef=forecastRecipeV4(c)?contextReferences(evaluation,contexts,c):forecastReference(evaluation,physics,isPhysicalForecast(approach));
    const bool physicalOnly=approach=="pinn_causal_forecast";
    const double qref=physicalOnly?1.0:data.qRef;
    auto trainY=forecastTargets(data.train,mode,qref,trainRef),valY=forecastTargets(data.validation,mode,qref,valRef);
    TensorScaler xs,ys;ForecastNet net{nullptr};HydroRunResult result;
    const double flowScale=c.hydrofusion_objective=="log_flow_mse_v2"?forecastFlowScale(data.train,{c.train_end_x,c.validation_end_x,c.test_end_x},c.lstm_sequence_length):1.;
    torch::Tensor trainObserved,validationObserved;
    if(c.hydrofusion_objective=="log_flow_mse_v2"){
        trainObserved=torch::tensor(data.train.observedTargetQ,torch::kFloat64).reshape({-1,1});
        validationObserved=torch::tensor(data.validation.observedTargetQ,torch::kFloat64).reshape({-1,1});
    }
    if(!physicalOnly){
        xs.fit(data.train.x,"standardize");ys.fit(trainY,"standardize");
        torch::manual_seed(static_cast<uint64_t>(std::max(0,c.random_seed)));net=ForecastNet(isRecurrentForecast(approach),forecastHidden(c),c.hydrofusion_ffn_activation=="silu",forecastChannels(c),parseForecastHistory(c.hydrofusion_history_hours,c.lstm_sequence_length));
        torch::optim::Adam optimizer(net->parameters(),torch::optim::AdamOptions(c.learning_rate));
        double best=std::numeric_limits<double>::infinity();std::string bestBytes;
        for(int epoch=1;epoch<=c.epochs;++epoch){
            net->train();double sum=0;const auto n=data.train.x.size(0);
            torch::Tensor order;
            if(c.shuffle_training)order=torch::tensor(seededForecastWindowOrder(n,c.random_seed,epoch),torch::kLong);
            for(int64_t start=0;start<n;start+=c.batch_size){const auto end=std::min(n,start+c.batch_size);
                const auto indices=c.shuffle_training?order.slice(0,start,end):torch::Tensor{};
                const auto batch=[&](const torch::Tensor& tensor){return c.shuffle_training?tensor.index_select(0,indices):tensor.slice(0,start,end);};
                optimizer.zero_grad();auto standardized=net->forward(xs.transform(batch(data.train.x)));
                auto loss=c.hydrofusion_objective=="log_mse_v1"?torch::mse_loss(standardized,ys.transform(batch(trainY))):
                    forecastObjective(c,standardized,ys.transform(batch(trainY)),ys,batch(trainRef),
                        batch(trainObserved),qref,flowScale);
                if(!std::isfinite(loss.item<double>()))throw std::runtime_error("Nonfinite forecast training loss.");
                loss.backward();optimizer.step();sum+=loss.item<double>()*(end-start);
            }
            torch::NoGradGuard guard;net->eval();const auto standardized=net->forward(xs.transform(data.validation.x));
            const double validation=(c.hydrofusion_objective=="log_mse_v1"?torch::mse_loss(standardized,ys.transform(valY)):
                forecastObjective(c,standardized,ys.transform(valY),ys,valRef,validationObserved,qref,flowScale)).item<double>();
            if(!std::isfinite(validation))throw std::runtime_error("Nonfinite forecast validation loss.");
            result.training_loss_history.push_back(sum/n);result.validation_loss_history.push_back(validation);
            if(selectForecastCheckpoint(c.hydrofusion_checkpoint_selection,epoch,c.epochs,validation,best)){best=validation;result.best_epoch=epoch;torch::serialize::OutputArchive weights;net->save(weights);std::ostringstream bytes;weights.save_to(bytes);bestBytes=bytes.str();}
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
    else result.y_pred=forecastReconstruct(mode,ys.inverseTransform(net->forward(xs.transform(evaluation.x))),testRef,qref);
    std::vector<double> valPred;
    if(physicalOnly){auto values=valRef.contiguous();valPred.assign(values.data_ptr<double>(),values.data_ptr<double>()+values.numel());}
    else valPred=forecastReconstruct(mode,ys.inverseTransform(net->forward(xs.transform(data.validation.x))),valRef,qref);
    result.validation_mse=0;for(size_t i=0;i<valPred.size();++i){const auto e=valPred[i]-data.validation.observedTargetQ[i];result.validation_mse+=e*e;}result.validation_mse/=valPred.size();
    result.y_true=evaluation.observedTargetQ;result.split.assign(result.y_pred.size(),c.hydrofusion_development_only?"validation":"test");
    for(auto x:evaluation.targetX)result.x.push_back(static_cast<double>(x));
    result.success=true;populateHydroMetrics(result,result.y_true,result.y_pred);populateHydroPeakMetrics(result,c.hydrofusion_development_only?"validation":"test");
    result.hydrofusion_target_mode=c.hydrofusion_target_mode;result.hydrofusion_q_ref=qref;result.hydrofusion_train_end_x=c.train_end_x;result.hydrofusion_validation_end_x=c.validation_end_x;result.hydrofusion_test_end_x=c.test_end_x;result.hydrofusion_test_origin_x=evaluation.originX;
    return result;
}
