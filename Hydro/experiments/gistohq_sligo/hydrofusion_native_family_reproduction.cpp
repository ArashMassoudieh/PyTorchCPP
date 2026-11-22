#include "../../evaluation/hydrofusion_forecast_inference.h"
#include "hydrofusion_prepared_native_package.h"
#include <iostream>
#include <iomanip>
// Independent fresh-process all-family archive/scaler reload on ALL native test rows,
// plus a bounded sample through the real predictAt API. Also persists validation
// predictions for predeclared advancement criteria, never refitting scalers.
int main(int argc,char** argv){try{
    using namespace hydrofusion;
    if(argc!=3)throw std::invalid_argument("Pass managed bundle and NEW validation CSV.");
    const std::string directory=argv[1];
    if(std::filesystem::exists(argv[2]))throw std::invalid_argument("Refusing existing validation output.");
    torch::set_num_threads(2);
    auto artifacts=HydroArtifactLoader().loadForInference(directory);
    if(artifacts.models.size()!=1)throw std::runtime_error("One forecast model required.");
    const auto approach=artifacts.models.begin()->first;
    if(!isExtendedForecast(approach))throw std::runtime_error("Unsupported forecast family.");
    HydroFusionForecastInference predictor(artifacts,directory,approach);
    const auto& c=artifacts.experiment.config;
    const auto rows=loadPreparedNativePackage((std::filesystem::path(directory)/"native_input_manifest.json").string(),c.hydrofusion_native_manifest_sha256,c.hydrofusion_development_only?c.validation_end_x:kSealStartX-1);
    auto data=prepareHydroFusionSequenceData(rows,forecastTargetMode(c),{c.train_end_x,c.validation_end_x,c.test_end_x},c.lstm_sequence_length,c.hydrofusion_development_only);
    std::map<int64_t,CausalPhysicalOriginContext> contexts;
    if(forecastRecipeV4(c)){
        contexts=causalPhysicalOriginContexts(rows,forecastPhysics(c));
        addForecastContext(data.train,contexts,c,data.qRef);addForecastContext(data.validation,contexts,c,data.qRef);
        if(!c.hydrofusion_development_only)addForecastContext(data.test,contexts,c,data.qRef);
    }
    auto metadata=reloadHydroFusionMetadata((std::filesystem::path(directory)/"hydrofusion_metadata.txt").string());
    const bool physicalOnly=approach=="pinn_causal_forecast";
    if((physicalOnly?1.0:data.qRef)!=metadata.qRef)throw std::runtime_error("Training-only q_ref identity mismatch.");
    const auto& model=artifacts.models.at(approach);std::string raw(model.bytes.begin(),model.bytes.end());std::istringstream bytes(raw);
    torch::serialize::InputArchive archive,weights;archive.load_from(bytes);
    ForecastNet net{nullptr};
    if(!physicalOnly){archive.read("weights",weights);net=ForecastNet(isRecurrentForecast(approach),forecastHidden(c),c.hydrofusion_ffn_activation=="silu",forecastRecipeV4(c)?25:20,parseForecastHistory(c.hydrofusion_history_hours,c.lstm_sequence_length));net->load(weights);net->eval();}
    TensorScaler xs,ys;
    if(!physicalOnly){xs.importState(artifacts.scalers.at(approach).input);ys.importState(artifacts.scalers.at(approach).target);}
    std::map<int64_t,double> physics;
    if(isPhysicalForecast(approach))physics=forecastPhysicalSeries(rows,c);
    // Recompute transform statistics from the eligible TRAIN tensors only;
    // archive integrity alone cannot prove how a scaler was fitted.
    if(!physicalOnly){
        TensorScaler expectedX,expectedY;
        expectedX.fit(data.train.x,"standardize");
        const auto reference=forecastRecipeV4(c)?contextReferences(data.train,contexts,c):forecastReference(data.train,physics,isPhysicalForecast(approach));
        expectedY.fit(forecastTargets(data.train,forecastTargetMode(c),metadata.qRef,reference),"standardize");
        auto same=[](const HydroScalerState& a,const HydroScalerState& b){return a.method==b.method && a.shape==b.shape && a.offset==b.offset && a.scale==b.scale;};
        if(!same(expectedX.exportState(),artifacts.scalers.at(approach).input) || !same(expectedY.exportState(),artifacts.scalers.at(approach).target))
            throw std::runtime_error("Archived scalers differ from training-only eligible support.");
        if(c.hydrofusion_objective=="log_flow_mse_v2"){
            torch::Tensor scale;archive.read("training_flow_scale",scale);
            if(scale.scalar_type()!=torch::kFloat64 || scale.numel()!=1 || scale.item<double>()!=forecastFlowScale(data.train,{c.train_end_x,c.validation_end_x,c.test_end_x},c.lstm_sequence_length))
                throw std::runtime_error("Archived flow scale differs from training-only eligible support.");
        }
    }
    torch::NoGradGuard guard;
    auto predict=[&](const DatasetSplit& split){
        auto reference=(forecastRecipeV4(c)?contextReferences(split,contexts,c):forecastReference(split,physics,isPhysicalForecast(approach))).contiguous();
        if(physicalOnly)return std::vector<double>(reference.data_ptr<double>(),reference.data_ptr<double>()+reference.numel());
        return forecastReconstruct(forecastTargetMode(c),ys.inverseTransform(net->forward(xs.transform(split.x))),reference,metadata.qRef);
    };
    const auto& evaluation=c.hydrofusion_development_only?data.validation:data.test;
    auto predictions=predict(evaluation);
    const auto& result=artifacts.results.at(approach);
    auto map=readOriginTargetMapCsv(std::filesystem::path(directory)/"hydrofusion_origin_target_map.csv");validateOriginTargetMap(map,1);
    if(result.x.size()!=predictions.size()||map.size()!=predictions.size())throw std::runtime_error("Native test support size mismatch.");
    double maxAbs=0;size_t apiChecks=0;
    for(size_t i=0;i<predictions.size();++i){
        if(result.x[i]!=evaluation.targetX[i]||result.y_true[i]!=evaluation.observedTargetQ[i]||result.split[i]!=(c.hydrofusion_development_only?"validation":"test")||map[i]!=std::make_pair(evaluation.originX[i],evaluation.targetX[i]))throw std::runtime_error("Native test identity/value mismatch.");
        const double error=std::abs(predictions[i]-result.y_pred[i]);
        if(!std::isfinite(predictions[i])||!std::isfinite(result.y_pred[i])||error>1e-5+1e-5*std::abs(result.y_pred[i]))throw std::runtime_error("Native archive prediction mismatch.");
        maxAbs=std::max(maxAbs,error);
        if(i==0||i+1==predictions.size()||i%std::max<size_t>(1,predictions.size()/10)==0){
            std::vector<HourlyRow> prefix;for(const auto& row:rows)if(row.x<=evaluation.originX[i])prefix.push_back(row);
            const double actual=predictor.predictAt(prefix,evaluation.originX[i]);
            if(!std::isfinite(actual)||std::abs(actual-predictions[i])>1e-5+1e-5*std::abs(actual))throw std::runtime_error("Production target-free inference mismatch.");
            ++apiChecks;
        }
    }
    const auto validation=predict(data.validation);
    std::ofstream output(argv[2]);if(!output)throw std::runtime_error("Cannot create validation CSV.");
    output<<"approach,index,split,x,observed,predicted,residual\n"<<std::setprecision(17);
    for(size_t i=0;i<validation.size();++i){const auto observed=data.validation.observedTargetQ[i];
        if(!std::isfinite(validation[i]))throw std::runtime_error("Nonfinite validation output.");
        output<<approach<<','<<i<<",validation,"<<data.validation.targetX[i]<<','<<observed<<','<<validation[i]<<','<<validation[i]-observed<<'\n';}
    output.close();if(!output)throw std::runtime_error("Validation CSV write failed.");
    std::cout<<std::setprecision(17)<<(c.hydrofusion_development_only?"PASS all_validation_rows=":"PASS all_test_rows=")<<predictions.size()<<" production_target_free_checks="<<apiChecks<<" max_abs="<<maxAbs<<" validation_rows="<<validation.size()<<" q_ref="<<metadata.qRef<<'\n';
}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
