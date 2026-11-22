#include "hydrofusion_prepared_native_package.h"
#include "hydrofusion_sequence_builder.h"
#include <iostream>
#include <iomanip>
using namespace hydrofusion;
int main(int argc,char**argv){try{
 if(argc!=4)throw std::invalid_argument("Pass control manifest, candidate manifest and NEW support CSV");
 if(std::filesystem::exists(argv[3]))throw std::invalid_argument("Support output already exists");
 auto a=loadPreparedNativePackage(argv[1],sha256File(argv[1]));auto b=loadPreparedNativePackage(argv[2],sha256File(argv[2]));
 if(a.size()!=b.size())throw std::runtime_error("Row count differs");
 for(size_t i=0;i<a.size();++i){const auto&x=a[i];const auto&y=b[i];
  if(x.x!=y.x || x.discharge_valid!=y.discharge_valid || x.observed_runoff_mm_per_hour!=y.observed_runoff_mm_per_hour || x.study_predictor_available!=y.study_predictor_available || x.temperature_c!=y.temperature_c || x.relative_humidity_percent!=y.relative_humidity_percent || x.wind_speed_m_per_second!=y.wind_speed_m_per_second || x.solar_energy_mj_per_m2_per_hour!=y.solar_energy_mj_per_m2_per_hour || x.pet_mm_per_hour!=y.pet_mm_per_hour)throw std::runtime_error("Nonrain input/Q/common mask changed");}
 const SplitCutoffs c{17543,26303,35063};auto sa=buildForecastSequences(a,c,24),sb=buildForecastSequences(b,c,24);
 if(sa.size()!=sb.size())throw std::runtime_error("Sequence support count differs");
 std::ofstream f(argv[3]);f<<"split,target_x,observed,persistence\n"<<std::setprecision(17);std::map<std::string,size_t>counts;
 for(size_t i=0;i<sa.size();++i){const auto&x=sa[i];const auto&y=sb[i];
  if(x.origin_x!=y.origin_x || x.target_x!=y.target_x || x.split!=y.split || x.observed_target_q!=y.observed_target_q || x.steps.back().q_lag_0h!=y.steps.back().q_lag_0h)throw std::runtime_error("Matched native support differs");
  const std::string split=x.split==SplitPartition::Train?"train":x.split==SplitPartition::Validation?"validation":"test";
  ++counts[split];f<<split<<','<<x.target_x<<','<<x.observed_target_q<<','<<x.steps.back().q_lag_0h<<'\n';}
 f.close();if(!f)throw std::runtime_error("Support write failed");
 for(const auto&v:counts){std::cout<<v.first<<'='<<v.second<<' ';}std::cout<<"PASS identical native support and nonrain/Q values\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
