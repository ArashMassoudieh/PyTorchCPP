#include "hydrofusion_prepared_native_package.h"
#include "hydrofusion_sequence_builder.h"
#include "hydrofusion_log_residual.h"
#include <iostream>
#include <set>
#include <iomanip>
int main(int argc,char** argv){try{
    if(argc!=3 && argc!=4)throw std::invalid_argument("Pass prepared manifest and sha256.");
    const auto rows=hydrofusion::loadPreparedNativePackage(argv[1],argv[2]);
    if(argc==4 && std::string(argv[3])=="--load-only"){std::cout<<"LOADED "<<rows.size()<<" endpoint rows\n";return 0;}
    const hydrofusion::SplitCutoffs cutoffs{17543,26303,35063};
    const auto sequences=hydrofusion::buildForecastSequences(rows,cutoffs,24);
    std::map<hydrofusion::SplitPartition,size_t> counts;
    std::set<int64_t> support;
    for(const auto& s:sequences){++counts[s.split];support.insert(s.target_x);}
    auto records=hydrofusion::runPersistenceBenchmark(hydrofusion::HourlyIndex(rows),rows,cutoffs);
    for(auto& record:records)record.included=record.included && support.count(record.target_x);
    const auto qref=hydrofusion::computeQRefFromTrainingSamples(records,cutoffs);
    std::cout<<"rows="<<rows.size()<<" train="<<counts[hydrofusion::SplitPartition::Train]<<" validation="<<counts[hydrofusion::SplitPartition::Validation]<<" test="<<counts[hydrofusion::SplitPartition::Test]<<" q_ref="<<qref<<'\n';
    if(argc==4){
        const std::filesystem::path output(argv[3]);
        if(std::filesystem::exists(output)||!std::filesystem::create_directory(output))throw std::runtime_error("New support directory required.");
        for(auto partition:{hydrofusion::SplitPartition::Train,hydrofusion::SplitPartition::Validation,hydrofusion::SplitPartition::Test}){
            const std::string name=partition==hydrofusion::SplitPartition::Train?"train":partition==hydrofusion::SplitPartition::Validation?"validation":"test";
            std::ofstream targets(output/(name+"_support.csv")),obs(output/(name+"_observations.csv")),persistence(output/(name+"_persistence.csv"));
            targets<<"x\n";obs<<"x,observed\n"<<std::setprecision(17);
            persistence<<"approach,index,split,x,observed,predicted,residual\n"<<std::setprecision(17);
            size_t i=0;
            for(const auto& s:sequences)if(s.split==partition){targets<<s.target_x<<'\n';obs<<s.target_x<<','<<s.observed_target_q<<'\n';
                const auto q=s.steps.back().q_lag_0h;
                persistence<<"persistence,"<<i++<<','<<name<<','<<s.target_x<<','<<s.observed_target_q<<','<<q<<','<<q-s.observed_target_q<<'\n';}
            if(!targets||!obs||!persistence)throw std::runtime_error("Support output failure.");
        }
    }
    for(auto partition:{hydrofusion::SplitPartition::Train,hydrofusion::SplitPartition::Validation,hydrofusion::SplitPartition::Test})
        if(!counts[partition])throw std::runtime_error("Empty matched support.");
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
