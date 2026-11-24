#include "hydrofusion_preissue_rain.h"
#include "hydrofusion_synthetic_fixture.h"
#include <iostream>
#include <stdexcept>
using namespace hydrofusion;
int main(){
 auto rows=buildSyntheticFixture();std::map<int64_t,PreissueRain> rain;
 for(int64_t i=0;i<300;++i)rain.emplace(i,PreissueRain{i,kRainEpochSeconds+i*3600-600,kRainEpochSeconds+i*3600-500,i%11==0?20.:0.,1.});
 const double gain=.26611573309982783;
 auto one=preissueRainContexts(rows,{1.,.4,.3,1.},rain,true);
 auto scaled=preissueRainContexts(rows,{1.,.4,.3,gain},rain,true);size_t n=0;
 for(const auto& item:one){const auto& b=scaled.at(item.first);
  if(std::abs(b.next_hour_mean-gain*item.second.next_hour_mean)>1e-12)throw std::runtime_error("Gain linearity failure");
  if(std::abs(b.ledger_at_origin.fastStorage-gain*item.second.ledger_at_origin.fastStorage)>1e-12)throw std::runtime_error("State scaling failure");
  ++n;
 }
 if(n<100)throw std::runtime_error("Vacuous fixture");
 std::cout<<"PASS "<<n<<" gain/output/state scaling checks\n";
}
