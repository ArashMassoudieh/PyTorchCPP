// Inspect weights/scalers only; no forward fitting, optimizer or target lookup.
#include "../../evaluation/hydrofusion_forecast_inference.h"
#include <iomanip>
#include <iostream>
int main(int argc,char**argv){try{
    using namespace hydrofusion;
    if(argc<3||std::filesystem::exists(argv[1]))throw std::invalid_argument("NEW output CSV followed by existing bundles required");
    torch::set_num_threads(2);torch::NoGradGuard guard;
    std::ofstream out(argv[1]);if(!out)throw std::runtime_error("Cannot create capacity CSV");
    out<<"bundle,approach,target_mode,q_ref,minimum_transformed_output,maximum_transformed_output,parameters\n"<<std::setprecision(17);
    for(int i=2;i<argc;++i){
        const std::string directory=argv[i];auto a=HydroArtifactLoader().loadForInference(directory);
        if(a.models.size()!=1)throw std::runtime_error("One neural forecast required");
        const auto name=a.models.begin()->first;const auto& c=a.experiment.config;
        if(!isExtendedForecast(name)||name=="pinn_causal_forecast")throw std::invalid_argument("Neural forecasts only");
        HydroFusionForecastInference managed(a,directory,name);
        const auto metadata=reloadHydroFusionMetadata((std::filesystem::path(directory)/"hydrofusion_metadata.txt").string());
        const auto& model=a.models.at(name);std::string raw(model.bytes.begin(),model.bytes.end());std::istringstream bytes(raw);
        torch::serialize::InputArchive archive,weights;archive.load_from(bytes);archive.read("weights",weights);
        ForecastNet net(isRecurrentForecast(name),forecastHidden(c));net->load(weights);net->eval();
        auto weight=isRecurrentForecast(name)?net->lstm->fc->weight:net->ffn->at<torch::nn::LinearImpl>(2).weight;
        auto bias=isRecurrentForecast(name)?net->lstm->fc->bias:net->ffn->at<torch::nn::LinearImpl>(2).bias;
        const double radius=weight.to(torch::kFloat64).abs().sum().item<double>(),center=bias.item<double>();
        const auto& state=a.scalers.at(name).target;
        if(state.method!="standardize"||state.offset.size()!=1||state.scale.size()!=1)throw std::runtime_error("One standardized target required");
        const double lo=(center-radius)*state.scale[0]+state.offset[0],hi=(center+radius)*state.scale[0]+state.offset[0];
        int64_t count=0;for(const auto&t:net->parameters())count+=t.numel();
        out<<std::filesystem::path(directory).filename().string()<<','<<name<<','<<c.hydrofusion_target_mode<<','<<metadata.qRef<<','<<lo<<','<<hi<<','<<count<<'\n';
    }
    out.close();if(!out)throw std::runtime_error("Capacity CSV write failed");
    std::cout<<"PASS inspected="<<argc-2<<" no_training\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
