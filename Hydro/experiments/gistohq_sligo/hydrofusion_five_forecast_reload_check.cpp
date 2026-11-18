#include "../../evaluation/hydrofusion_forecast_inference.h"
#include "../../evaluation/inference_runner.h"
#include "hydrofusion_synthetic_fixture.h"
#include <iostream>
#include <iomanip>
int main(int argc,char** argv){
    try{
        if(argc!=3)throw std::invalid_argument("Pass managed bundle and extended approach.");
        using namespace hydrofusion;torch::set_num_threads(1);
        const std::string directory=argv[1],approach=argv[2];
        auto artifacts=HydroArtifactLoader().loadForInference(directory);
        HydroFusionForecastInference predictor(artifacts,directory,approach);
        bool genericRejected=false;try{HydroInferenceSession wrong(artifacts,approach);}catch(const std::exception&){genericRejected=true;}
        if(!genericRejected)throw std::runtime_error("Generic forecast output not rejected.");
        const auto rows=buildSyntheticFixture();const auto& c=artifacts.experiment.config;
        const auto eligible=buildForecastSequences(rows,{c.train_end_x,c.validation_end_x,c.test_end_x},24);
        std::vector<ForecastSequence> expectedSupport;for(const auto& s:eligible)if(s.split==SplitPartition::Test)expectedSupport.push_back(s);
        const auto map=readOriginTargetMapCsv(std::filesystem::path(directory)/"hydrofusion_origin_target_map.csv");
        validateOriginTargetMap(map,1);const auto& result=artifacts.results.at(approach);
        if(result.x.size()!=expectedSupport.size()||map.size()!=expectedSupport.size())throw std::runtime_error("Missing/extra eligible support.");
        double maxAbs=0,maxRel=0;
        for(size_t i=0;i<expectedSupport.size();++i){const auto& s=expectedSupport[i];
            if(result.x[i]!=s.target_x||map[i]!=std::make_pair(s.origin_x,s.target_x)||result.split[i]!="test"||result.y_true[i]!=s.observed_target_q)throw std::runtime_error("Timestamp/observed/split/map mismatch.");
            // Discard target and every future row entirely: production prediction
            // must not require or read them, including target native flags.
            std::vector<HourlyRow> prefix;for(const auto& row:rows)if(row.x<=s.origin_x)prefix.push_back(row);
            const auto actual=predictor.predictAt(prefix,s.origin_x),expected=result.y_pred[i];
            if(!std::isfinite(actual)||!std::isfinite(expected))throw std::runtime_error("Nonfinite parity input.");
            const auto absolute=std::abs(actual-expected),relative=absolute/std::max(1.0,std::abs(expected));
            maxAbs=std::max(maxAbs,absolute);maxRel=std::max(maxRel,relative);
            if(absolute>1e-5+1e-5*std::abs(expected))throw std::runtime_error("Fresh forecast parity failed.");
            auto perturbed=rows;for(auto& row:perturbed)if(row.x>s.origin_x){row.observed_runoff_mm_per_hour=999;row.precipitation_mm_per_hour=999;}
            if(predictor.predictAt(perturbed,s.origin_x)!=actual)throw std::runtime_error("Future values changed past forecast.");
        }
        bool gapRejected=false;auto gapped=rows;gapped.erase(gapped.begin()+200);
        try{predictor.predictAt(gapped,245);}catch(const std::exception&){gapRejected=true;}
        if(!gapRejected)throw std::runtime_error("Gapped history accepted.");
        bool missingRejected=false;auto missing=rows;missing[200].discharge_valid=false;
        try{predictor.predictAt(missing,245);}catch(const std::exception&){missingRejected=true;}
        if(!missingRejected)throw std::runtime_error("Non-native history accepted.");
        std::cout<<std::setprecision(17)<<"PASS "<<approach<<" rows="<<expectedSupport.size()<<" max_abs="<<maxAbs<<" max_rel="<<maxRel<<" (atol=1e-5,rtol=1e-5); target-free/future/gap/native/generic guards PASS\n";
    }catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
