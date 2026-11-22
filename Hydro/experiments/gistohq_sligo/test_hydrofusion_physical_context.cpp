#include "hydrofusion_causal_physics.h"
#include "hydrofusion_synthetic_fixture.h"
#include <functional>
#include <iostream>
#include <limits>
using namespace hydrofusion;
int checks=0;
void check(bool v){++checks;if(!v)throw std::runtime_error("Physical context check "+std::to_string(checks));}
void near(double a,double b){check(std::abs(a-b)<=1e-12*std::max({1.,std::abs(a),std::abs(b)}));}
void rejects(const std::function<void()>& f){bool bad=false;try{f();}catch(const std::invalid_argument&){bad=true;}catch(const std::overflow_error&){bad=true;}check(bad);}
int main(){
 const CausalPhysicsParams p{};
 for(int age:{0,1,24}){
  auto r=makeSyntheticFixtureRow(100,1);r.rainfall_valid_end_x=100-age;r.rainfall_release_x=100;r.precipitation_mm_per_hour=10;r.pet_mm_per_hour=0;
  auto trace=causalPhysicalOriginContexts({r},p).at(100);ReservoirLedger expected;
  double current=stepCausalReservoirHourMean(expected,10,p);
  for(int i=0;i<age;++i)current=stepCausalReservoirHourMean(expected,0,p);
  const auto atOrigin=expected;const double next=stepCausalReservoirHourMean(expected,0,p);
  check(trace.origin_x==100 && trace.segment_age_hours==0);near(trace.current_hour_mean,current);near(trace.next_hour_mean,next);
  near(trace.ledger_at_origin.fastStorage,atOrigin.fastStorage);near(trace.ledger_at_origin.slowStorage,atOrigin.slowStorage);
  near(trace.ledger_at_origin.inputVolume,trace.ledger_at_origin.outflowVolume+trace.ledger_at_origin.fastStorage+trace.ledger_at_origin.slowStorage);
  check(causalPhysicalValidTimeDryForecasts({r},p).at(100)==trace.next_hour_mean);
 }
 std::vector<HourlyRow> rows;
 for(int x=100;x<150;++x){auto r=makeSyntheticFixtureRow(x,1);r.rainfall_valid_end_x=x-1;r.rainfall_release_x=x;r.precipitation_mm_per_hour=x%5==0?3.:0.;r.pet_mm_per_hour=0;rows.push_back(r);}
 auto all=causalPhysicalOriginContexts(rows,p);auto altered=rows;
 for(auto& r:altered)if(r.x>120)r.precipitation_mm_per_hour=1000.;
 auto future=causalPhysicalOriginContexts(altered,p);std::vector<HourlyRow> prefix(rows.begin(),rows.begin()+21);auto pre=causalPhysicalOriginContexts(prefix,p);
 for(int x=100;x<=120;++x){check(all.at(x).current_hour_mean==pre.at(x).current_hour_mean);check(all.at(x).next_hour_mean==pre.at(x).next_hour_mean);check(all.at(x).next_hour_mean==future.at(x).next_hour_mean);check(all.at(x).ledger_at_origin.fastStorage==future.at(x).ledger_at_origin.fastStorage);}
 // Reading context must not change the committed solver's outputs or state.
 std::map<int64_t,CausalPhysicalOriginContext> captured;check(causalPhysicalValidTimeDryForecasts(rows,p)==causalPhysicalValidTimeDryForecastsWithContext(rows,p,&captured));
 rejects([&]{causalPhysicalValidTimeDryForecastsWithContext(rows,p,&captured);});
 altered=rows;altered[10].all_forcings_valid=false;auto gap=causalPhysicalOriginContexts(altered,p);check(!gap.count(110)&&gap.at(111).segment_age_hours==0&&gap.at(112).segment_age_hours==1);
 altered=rows;altered.erase(altered.begin()+10);gap=causalPhysicalOriginContexts(altered,p);check(!gap.count(110)&&gap.at(111).segment_age_hours==0);
 // Q is never assimilated by this trace producer; its output is forcing-only.
 altered=rows;for(auto&r:altered)r.observed_runoff_mm_per_hour=1e9;
 auto q=causalPhysicalOriginContexts(altered,p);check(q.at(120).next_hour_mean==all.at(120).next_hour_mean);
 auto c=all.at(120);c.current_hour_mean=c.next_hour_mean=3;check(dischargeAnchoredPhysicalReference(7,c)==7);
 c.current_hour_mean=2;c.next_hour_mean=5;check(dischargeAnchoredPhysicalReference(7,c)==10);
 c.current_hour_mean=10;c.next_hour_mean=1;check(dischargeAnchoredPhysicalReference(2,c)==0);
 rejects([&]{dischargeAnchoredPhysicalReference(-1,c);});rejects([&]{dischargeAnchoredPhysicalReference(std::numeric_limits<double>::quiet_NaN(),c);});
 auto bad=c;bad.next_hour_mean=-1;rejects([&]{dischargeAnchoredPhysicalReference(1,bad);});
 bad=c;bad.rainfall_release_x=bad.origin_x+1;rejects([&]{dischargeAnchoredPhysicalReference(1,bad);});
 bad=c;bad.origin_x=kSealStartX-1;rejects([&]{dischargeAnchoredPhysicalReference(1,bad);});
 bad=c;bad.current_hour_mean=0;bad.next_hour_mean=std::numeric_limits<double>::max();rejects([&]{dischargeAnchoredPhysicalReference(std::numeric_limits<double>::max(),bad);});
 std::cout<<"PASS physical origin context and discharge anchor checks="<<checks<<'\n';
}
