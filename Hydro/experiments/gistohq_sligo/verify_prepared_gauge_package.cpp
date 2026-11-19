// Read-only native package verification through the production shared loader.
#include "hydrofusion_prepared_native_package.h"
#include "hydrofusion_sequence_builder.h"
#include <iostream>
#include <iomanip>
int main(int argc,char** argv){
    if(argc!=3 && argc!=4 && argc!=7){std::cerr<<"usage: verify_prepared_gauge_package MANIFEST SHA256 [NEW_ROWS_CSV [TRAIN_END VALIDATION_END TEST_END]]\n";return 2;}
    try{
        const auto rows=hydrofusion::loadPreparedNativePackage(argv[1],argv[2]);
        std::ofstream exported;
        if(argc>=4){
            if(std::filesystem::exists(argv[3]))throw std::runtime_error("Refusing to overwrite row evidence");
            exported.open(argv[3]);
            if(!exported)throw std::runtime_error("Cannot write row evidence");
            exported<<std::setprecision(17)<<"x,q,p,t,rh,wind,solar,pet,forcing_valid,discharge_valid\n";
        }
        size_t validForcing=0,validQ=0;
        for(const auto& row:rows){
            if(row.x>=hydrofusion::kSealStartX)throw std::runtime_error("Reserved timestamp");
            validForcing+=row.all_forcings_valid;
            validQ+=row.discharge_valid;
            if(exported)exported<<row.x<<','<<row.observed_runoff_mm_per_hour<<','<<row.precipitation_mm_per_hour
                <<','<<row.temperature_c<<','<<row.relative_humidity_percent<<','<<row.wind_speed_m_per_second
                <<','<<row.solar_energy_mj_per_m2_per_hour<<','<<row.pet_mm_per_hour<<','<<row.all_forcings_valid
                <<','<<row.discharge_valid<<'\n';
        }
        size_t supportCounts[3]={0,0,0},sequenceCounts[3]={0,0,0};
        if(argc==7){
            auto cutoff=[](const char* text){
                const std::string token(text);
                if(token.empty() || token.find_first_not_of("0123456789")!=std::string::npos)
                    throw std::invalid_argument("Canonical cutoff integer required");
                return std::stoll(token);
            };
            hydrofusion::SplitCutoffs cutoffs{cutoff(argv[4]),cutoff(argv[5]),cutoff(argv[6])};
            hydrofusion::validateSplitCutoffs(cutoffs);
            hydrofusion::HourlyIndex index(rows);
            const auto support=hydrofusion::runPersistenceBenchmark(index,rows,cutoffs);
            const auto sequences=hydrofusion::buildForecastSequences(rows,cutoffs,24);
            const std::string sequencePath=std::string(argv[3])+".sequences.csv";
            if(std::filesystem::exists(sequencePath))throw std::runtime_error("Refusing to overwrite sequence evidence");
            std::ofstream seqReport(sequencePath);if(!seqReport)throw std::runtime_error("Cannot write sequence evidence");
            seqReport<<"origin_x,target_x,split,sequence_length\n";
            for(const auto& seq:sequences){
                ++sequenceCounts[static_cast<int>(seq.split)];
                seqReport<<seq.origin_x<<','<<seq.target_x<<','<<static_cast<int>(seq.split)<<','<<seq.steps.size()<<'\n';
            }
            const std::string path=std::string(argv[3])+".support.csv";
            if(std::filesystem::exists(path))throw std::runtime_error("Refusing to overwrite support evidence");
            std::ofstream report(path);if(!report)throw std::runtime_error("Cannot write support evidence");
            report<<"origin_x,target_x,predictor_available,target_native,split_assigned,split,included\n";
            for(const auto& record:support){
                report<<record.origin_x<<','<<record.target_x<<','<<record.predictor_availability_mask<<','
                      <<record.target_native_mask<<','<<record.split_assigned<<','<<static_cast<int>(record.split)<<','<<record.included<<'\n';
                if(record.included)++supportCounts[static_cast<int>(record.split)];
            }
        }
        std::cout<<std::setprecision(17)<<"{\"rows\":"<<rows.size()<<",\"first_endpoint_x\":"<<rows.front().x
                 <<",\"last_endpoint_x\":"<<rows.back().x<<",\"valid_forcing_rows\":"<<validForcing
                 <<",\"valid_discharge_rows\":"<<validQ<<",\"training_performed\":false,\"included_support\":["<<supportCounts[0]<<','<<supportCounts[1]<<','<<supportCounts[2]<<"],\"sequence_support\":["<<sequenceCounts[0]<<','<<sequenceCounts[1]<<','<<sequenceCounts[2]<<"]}\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
