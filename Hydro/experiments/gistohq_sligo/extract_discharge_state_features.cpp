#include "hydrofusion_prepared_native_package.h"
#include "hydrofusion_discharge_state_forecast.h"
#include <iomanip>
#include <iostream>
using namespace hydrofusion;
int main(int argc,char**argv){try{
 if(argc!=7&&argc!=11)throw std::invalid_argument("manifest SHA weather support split new-output [fast slow fraction gain]");
 const std::string split=argv[5];if(split!="train"&&split!="validation")throw std::invalid_argument("Bad split");
 const auto cutoff=split=="train"?17543:26303;
 const auto rows=loadPreparedNativePackage(argv[1],argv[2],cutoff);
 std::map<int64_t,const HourlyRow*> index;for(const auto&r:rows)index.emplace(static_cast<int64_t>(r.x),&r);
 std::ifstream weather(argv[3]);auto rain=parsePreissueRain(weather,26303);
 CausalPhysicsParams p;if(argc==11)p={rainNumber(argv[7]),rainNumber(argv[8]),rainNumber(argv[9]),rainNumber(argv[10])};validateCausalPhysics(p);
 if(std::filesystem::exists(argv[6]))throw std::invalid_argument("Output exists");
 std::ifstream support(argv[4]);std::string line;std::getline(support,line);if(line!="split,target_x,observed,persistence")throw std::invalid_argument("Support header");
 std::ofstream out(argv[6]);out<<"target_x,observed,origin_q,available_effective_rain,predicted\n"<<std::setprecision(17);
 std::set<int64_t>seen;size_t n=0;
 while(std::getline(support,line)){
  std::istringstream stream(line);std::string label,x,observed,persistence,extra;
  if(!std::getline(stream,label,',')||!std::getline(stream,x,','))throw std::invalid_argument("Missing identity");
  const auto target=rainInteger(x);if(target<1||target>=kSealStartX)throw std::invalid_argument("Reserved identity");
  if(label!=split)continue; // Excluded observed values never converted.
  if(target>cutoff||(split=="validation"&&target<=17543)||!seen.insert(target).second)throw std::invalid_argument("Split mismatch");
  if(!std::getline(stream,observed,',')||!std::getline(stream,persistence,',')||std::getline(stream,extra,','))throw std::invalid_argument("Malformed support row");
  const auto&origin=*index.at(target-1);const auto&truth=*index.at(target);
  const auto y=rainNumber(observed),q=rainNumber(persistence);
  if(!truth.discharge_valid||!origin.discharge_valid||std::abs(truth.observed_runoff_mm_per_hour-y)>1e-12||std::abs(origin.observed_runoff_mm_per_hour-q)>1e-12||!preissueRainHistoryAvailable(rain,target-1,24))throw std::invalid_argument("Native support/Q mismatch");
  const auto analysis=dischargeStateForecast(origin,rain.at(target-1),p);
  const auto& l=analysis.ledger;
  if(std::abs(l.inputVolume-l.outflowVolume-l.fastStorage-l.slowStorage)>1e-9*std::max(1.,l.inputVolume))throw std::runtime_error("Analysis water ledger mismatch");
  out<<target<<','<<y<<','<<q<<','<<std::max(0.,rain.at(target-1).rate-origin.pet_mm_per_hour)<<','<<(argc==11?analysis.mean:0.)<<'\n';++n;
 }
 if(n!=(split=="train"?13072:7112))throw std::runtime_error("Support count mismatch");
 out.close();if(!out)throw std::runtime_error("Output failed");std::cout<<"PASS "<<n<<" native discharge-analysis rows\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
