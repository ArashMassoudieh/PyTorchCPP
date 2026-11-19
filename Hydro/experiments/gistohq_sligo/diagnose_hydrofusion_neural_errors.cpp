#include "../../evaluation/hydrofusion_forecast_inference.h"
#include "hydrofusion_prepared_native_package.h"
#include <iostream>
#include <iomanip>
// Read-only neural diagnostic: reload existing checkpoint, reproduce test rows,
// then export raw target-space and physical errors on all three splits. No fit.
namespace {
std::vector<hydrofusion::HourlyRow> cachedRows;
std::optional<hydrofusion::PreparedData> cachedData;
std::string cachedManifest,cachedIntervals;
}
static int diagnoseOne(int argc,char** argv){try{
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
    const auto manifest=std::filesystem::path(directory)/"native_input_manifest.json";
    const auto intervals=std::filesystem::path(directory)/"intervals.csv";
    if(c.train_end_x!=17543||c.validation_end_x!=26303||c.test_end_x!=35063)throw std::runtime_error("Fixed diagnostic cutoffs required.");
    if(!cachedData){
        cachedRows=loadPreparedNativePackage(manifest.string(),c.hydrofusion_native_manifest_sha256);
        // x/observations/origins/q_ref are identical for both target modes;
        // data.y is never consumed below. True targets are recomputed per mode.
        cachedData=prepareHydroFusionSequenceData(cachedRows,TargetMode::DirectLogQ,{c.train_end_x,c.validation_end_x,c.test_end_x},24);
        cachedManifest=c.hydrofusion_native_manifest_sha256;cachedIntervals=sha256File(intervals.string());
    }else if(cachedManifest!=c.hydrofusion_native_manifest_sha256||cachedIntervals!=sha256File(intervals.string()))throw std::runtime_error("Cannot reuse prepared tensors across changed inputs.");
    const auto& rows=cachedRows;const auto& data=*cachedData;
    auto metadata=reloadHydroFusionMetadata((std::filesystem::path(directory)/"hydrofusion_metadata.txt").string());
    const bool physicalOnly=approach=="pinn_causal_forecast";
    if((physicalOnly?1.0:data.qRef)!=metadata.qRef)throw std::runtime_error("Training-only q_ref identity mismatch.");
    const auto& model=artifacts.models.at(approach);std::string raw(model.bytes.begin(),model.bytes.end());std::istringstream bytes(raw);
    torch::serialize::InputArchive archive,weights;archive.load_from(bytes);
    ForecastNet net{nullptr};
    if(!physicalOnly){archive.read("weights",weights);net=ForecastNet(isRecurrentForecast(approach),forecastHidden(c));net->load(weights);net->eval();}
    TensorScaler xs,ys;
    if(!physicalOnly){xs.importState(artifacts.scalers.at(approach).input);ys.importState(artifacts.scalers.at(approach).target);}
    std::map<int64_t,double> physics;
    if(isPhysicalForecast(approach))physics=causalPhysicalForecasts(rows,forecastPhysics(c));
    torch::NoGradGuard guard;
    auto predict=[&](const DatasetSplit& split){
        auto reference=forecastReference(split,physics,isPhysicalForecast(approach)).contiguous();
        if(physicalOnly)return std::vector<double>(reference.data_ptr<double>(),reference.data_ptr<double>()+reference.numel());
        return forecastReconstruct(forecastTargetMode(c),ys.inverseTransform(net->forward(xs.transform(split.x))),reference,metadata.qRef);
    };
    auto predictions=predict(data.test);
    const auto& result=artifacts.results.at(approach);
    auto map=readOriginTargetMapCsv(std::filesystem::path(directory)/"hydrofusion_origin_target_map.csv");validateOriginTargetMap(map,1);
    if(result.x.size()!=predictions.size()||map.size()!=predictions.size())throw std::runtime_error("Native test support size mismatch.");
    double maxAbs=0;size_t apiChecks=0;
    for(size_t i=0;i<predictions.size();++i){
        if(result.x[i]!=data.test.targetX[i]||result.y_true[i]!=data.test.observedTargetQ[i]||result.split[i]!="test"||map[i]!=std::make_pair(data.test.originX[i],data.test.targetX[i]))throw std::runtime_error("Native test identity/value mismatch.");
        const double error=std::abs(predictions[i]-result.y_pred[i]);
        if(!std::isfinite(predictions[i])||!std::isfinite(result.y_pred[i])||error>1e-5+1e-5*std::abs(result.y_pred[i]))throw std::runtime_error("Native archive prediction mismatch.");
        maxAbs=std::max(maxAbs,error);
        if(i==0||i+1==predictions.size()||i%std::max<size_t>(1,predictions.size()/10)==0){
            std::vector<HourlyRow> prefix;for(const auto& row:rows)if(row.x<=data.test.originX[i])prefix.push_back(row);
            const double actual=predictor.predictAt(prefix,data.test.originX[i]);
            if(!std::isfinite(actual)||std::abs(actual-predictions[i])>1e-5+1e-5*std::abs(actual))throw std::runtime_error("Production target-free inference mismatch.");
            ++apiChecks;
        }
    }
    if(physicalOnly)throw std::invalid_argument("Neural diagnostics require a learned checkpoint.");
    std::ofstream output(argv[2]);if(!output)throw std::runtime_error("Cannot create NEW diagnostic CSV.");
    output<<"approach,split,origin_x,target_x,observed,reference,q_ref,transformed_target,transformed_prediction,predicted,clipped\n"<<std::setprecision(17);
    auto exportSplit=[&](const DatasetSplit& split,const char* label){
        auto ref=forecastReference(split,physics,isPhysicalForecast(approach)).contiguous();
        auto raw=ys.inverseTransform(net->forward(xs.transform(split.x))).to(torch::kFloat64).contiguous();
        const auto physical=predict(split);
        for(size_t i=0;i<physical.size();++i){
            const double reference=ref.data_ptr<double>()[i],observed=split.observedTargetQ[i],estimate=raw.data_ptr<double>()[i];
            const double target=computeTrainingTarget(forecastTargetMode(c),reference,observed,metadata.qRef);
            const double reconstructedZ=estimate+(forecastTargetMode(c)==TargetMode::LogResidual?zTransform(reference,metadata.qRef):0);
            if(!std::isfinite(estimate)||!std::isfinite(target)||!std::isfinite(physical[i]))throw std::runtime_error("Nonfinite neural diagnostic.");
            output<<approach<<','<<label<<','<<split.originX[i]<<','<<split.targetX[i]<<','<<observed<<','<<reference<<','<<metadata.qRef<<','<<target<<','<<estimate<<','<<physical[i]<<','<<(reconstructedZ<0)<<'\n';
        }
    };
    exportSplit(data.train,"train");exportSplit(data.validation,"validation");exportSplit(data.test,"test");
    output.close();if(!output)throw std::runtime_error("Diagnostic CSV write failed.");
    std::cout<<std::setprecision(17)<<"PASS all_test_rows="<<predictions.size()<<" production_target_free_checks="<<apiChecks<<" max_abs="<<maxAbs<<" validation_rows="<<data.validation.targetX.size()<<" q_ref="<<metadata.qRef<<'\n';
}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}return 0;}

int main(int argc,char**argv){
    if(argc<3||argc%2!=1){std::cerr<<"Pass pairs of existing bundle and NEW output CSV\n";return 1;}
    for(int i=1;i<argc;i+=2){
        std::cout<<"DIAGNOSE "<<std::filesystem::path(argv[i]).filename().string()<<std::endl;
        char* one[]={argv[0],argv[i],argv[i+1]};if(diagnoseOne(3,one)!=0)return 1;
    }
    return 0;
}
