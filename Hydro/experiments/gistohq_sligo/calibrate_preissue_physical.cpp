// Training-only shape/gain calibration probe. Production routing equations, no neural fitting.
// Parameters are supplied explicitly; output is never an in-place package mutation.
#include "hydrofusion_prepared_native_package.h"
#include "hydrofusion_preissue_rain.h"
#include <iomanip>
#include <iostream>
using namespace hydrofusion;
int main(int argc,char**argv){try{
 if(argc!=10)throw std::invalid_argument("manifest SHA weather support split fast slow fraction new-output");
 const std::string split=argv[5];if(split!="train"&&split!="validation")throw std::invalid_argument("Invalid split");
 const int64_t cutoff=split=="train"?17543:26303;
 const auto rows=loadPreparedNativePackage(argv[1],argv[2],cutoff);
 std::ifstream weather(argv[3]);if(!weather)throw std::invalid_argument("Missing weather");auto rain=parsePreissueRain(weather,26303);
 CausalPhysicsParams p{rainNumber(argv[6]),rainNumber(argv[7]),rainNumber(argv[8]),1};validateCausalPhysics(p);
 auto wet=preissueRainContexts(rows,p,rain,true),dry=preissueRainContexts(rows,p,rain,false);
 std::map<int64_t,double> q;for(const auto&r:rows)if(r.discharge_valid&&std::isfinite(r.observed_runoff_mm_per_hour)&&r.observed_runoff_mm_per_hour>=0)q.emplace(static_cast<int64_t>(r.x),r.observed_runoff_mm_per_hour);
 if(std::filesystem::exists(argv[9]))throw std::invalid_argument("Output exists");
 std::ifstream support(argv[4]);std::string line;std::getline(support,line);if(line!="split,target_x,observed,persistence")throw std::invalid_argument("Bad support header");
 std::ofstream out(argv[9]);out<<"target_x,observed,wet_unit_gain,dry_unit_gain\n"<<std::setprecision(17);size_t n=0;std::set<int64_t> seen;
 while(std::getline(support,line)){
  std::istringstream s(line);std::string label,x,obs,persist,extra;
  if(!std::getline(s,label,',')||!std::getline(s,x,','))throw std::invalid_argument("Bad support identity");
  auto target=rainInteger(x);if(target<1||target>=kSealStartX)throw std::invalid_argument("Support outside seal");
  if(label!=split)continue; // Before reading excluded observations.
  if(target>cutoff||(split=="validation"&&target<=17543)||!seen.insert(target).second)throw std::invalid_argument("Support split mismatch");
  if(!std::getline(s,obs,',')||!std::getline(s,persist,',')||std::getline(s,extra,','))throw std::invalid_argument("Bad support row");
  double y=rainNumber(obs);if(q.find(target)==q.end()||std::abs(q.at(target)-y)>1e-12)throw std::invalid_argument("Native Q identity mismatch");
  if(!preissueRainHistoryAvailable(rain,target-1,24))throw std::invalid_argument("Weather support mismatch");
  auto a=wet.at(target-1),b=dry.at(target-1);
  if(a.ledger_at_origin.fastStorage!=b.ledger_at_origin.fastStorage||a.ledger_at_origin.slowStorage!=b.ledger_at_origin.slowStorage)throw std::runtime_error("Hold changed committed storage");
  double ledgerError=a.ledger_at_origin.inputVolume-a.ledger_at_origin.outflowVolume-a.ledger_at_origin.fastStorage-a.ledger_at_origin.slowStorage;
  if(std::abs(ledgerError)>1e-8*std::max(1.,a.ledger_at_origin.inputVolume))throw std::runtime_error("Water ledger failure");
  if(!std::isfinite(a.next_hour_mean)||!std::isfinite(b.next_hour_mean)||a.next_hour_mean<0||b.next_hour_mean<0)throw std::runtime_error("Nonfinite prediction");
  out<<target<<','<<y<<','<<a.next_hour_mean<<','<<b.next_hour_mean<<'\n';++n;
 }
 if(n!=(split=="train"?13072:7112))throw std::runtime_error("Unexpected support count");
 out.close();if(!out)throw std::runtime_error("Output failed");
 std::cout<<"PASS "<<split<<" rows="<<n<<" ledger/copy invariants verified\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
