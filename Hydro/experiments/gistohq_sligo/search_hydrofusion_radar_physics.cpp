// New radar-development physical-settings evaluation. No neural fitting or test Q.
#include "hydrofusion_prepared_native_package.h"
#include "hydrofusion_sequence_builder.h"
#include "hydrofusion_causal_physics.h"
#include <iomanip>
#include <iostream>
#include <array>
using namespace hydrofusion;
std::array<double,6> metrics(const std::vector<std::pair<double,double>>& values){
    if(values.empty())throw std::invalid_argument("Empty physical search support.");
    double a=0,b=0,sse=0,mae=0;for(auto v:values){a+=v.first;b+=v.second;sse+=(v.second-v.first)*(v.second-v.first);mae+=std::abs(v.second-v.first);}
    const double ma=a/values.size(),mb=b/values.size();double va=0,vb=0,cov=0;
    for(auto v:values){va+=(v.first-ma)*(v.first-ma);vb+=(v.second-mb)*(v.second-mb);cov+=(v.first-ma)*(v.second-mb);}
    if(va<=0||vb<=0||a<=0)throw std::invalid_argument("Undefined physical search scores.");
    std::array<double,6> out{1-sse/va,cov*cov/(va*vb),std::sqrt(sse/values.size()),mae/values.size(),100*(b-a)/a,static_cast<double>(values.size())};
    for(auto v:out)if(!std::isfinite(v))throw std::overflow_error("Physical search metrics overflow.");
    return out;
}
int main(int argc,char**argv){try{
    if(argc!=4||std::filesystem::exists(argv[3]))throw std::invalid_argument("Pass development manifest, its SHA256, NEW output CSV.");
    const SplitCutoffs cutoffs{17543,26303,35063};
    const auto rows=loadPreparedNativePackage(argv[1],argv[2],cutoffs.validation_end_x);
    const auto sequences=buildForecastSequences(rows,cutoffs,24);
    for(const auto&s:sequences)if(s.split!=SplitPartition::Train&&s.split!=SplitPartition::Validation)throw std::invalid_argument("Search test support forbidden.");
    std::ofstream out(argv[3]);out<<std::setprecision(17)<<"index,fast,slow,fraction,gain,split,nse,r2,rmse,mae,pbias_percent,rows\n";
    size_t index=0;
    for(double fast:{1.,2.})for(double slow:{.04,.06,.1})for(double fraction:{.4,.7})for(double gain:{.5,1.}){
        const auto pred=causalPhysicalValidTimeDryForecasts(rows,{fast,slow,fraction,gain});
        for(auto partition:{SplitPartition::Train,SplitPartition::Validation}){
            std::vector<std::pair<double,double>> values;
            for(const auto&s:sequences)if(s.split==partition)values.emplace_back(s.observed_target_q,pred.at(s.origin_x));
            auto m=metrics(values);out<<index<<','<<fast<<','<<slow<<','<<fraction<<','<<gain<<','<<(partition==SplitPartition::Train?"train":"validation");
            for(auto v:m)out<<','<<v;
            out<<'\n';
        }
        ++index;
    }
    out.close();if(!out)throw std::runtime_error("Physical search write failed.");
    std::cout<<"PASS physical_candidates="<<index<<" test_rows=0 neural_fits=0\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
