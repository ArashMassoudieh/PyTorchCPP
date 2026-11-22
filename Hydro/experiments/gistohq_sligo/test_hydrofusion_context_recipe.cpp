#include "hydrofusion_context_recipe.h"
#include <functional>
#include <iostream>
#include <limits>
using namespace hydrofusion;
int checks=0;void check(bool ok){++checks;if(!ok)throw std::runtime_error("Context recipe check "+std::to_string(checks));}
void rejects(std::function<void()> f){bool bad=false;try{f();}catch(const std::exception&){bad=true;}check(bad);}
int main(){
 check(parseForecastHistory("0,1,3,6,12,23",24)==std::vector<int64_t>({0,1,3,6,12,23}));
 for(const std::string text:{"","1","0,0","0,24","0,3,1","0,","0,nan","0,-1","0,1junk","0, 1","0,,1"})rejects([&]{parseForecastHistory(text,24);});
 CausalPhysicalOriginContext c;c.origin_x=100;c.rainfall_valid_end_x=99;c.rainfall_release_x=100;c.segment_age_hours=50;
 c.current_hour_mean=3;c.next_hour_mean=2;c.ledger_at_origin.fastStorage=5;c.ledger_at_origin.slowStorage=8;
 auto f=physicalContextChannels(4,1,c,"absolute25");check(f[0]==3&&f[1]==2&&f[2]==5&&f[3]==8);check(std::abs(f[4]-(std::log1p(4)-std::log1p(3)))<1e-14);
 check(physicalContextChannels(4,1,c,"zero25")==std::array<double,5>({0,0,0,0,0}));
 check(contextReference(4,c,"zero25")==4);check(contextReference(4,c,"absolute25")==2);check(contextReference(4,c,"anchored25")==3);
 c.next_hour_mean=3;check(contextReference(4,c,"anchored25")==4);
 rejects([&]{physicalContextChannels(4,0,c,"zero25");});
 c.ledger_at_origin.fastStorage=std::numeric_limits<double>::quiet_NaN();rejects([&]{physicalContextChannels(4,1,c,"zero25");});
 c.ledger_at_origin.fastStorage=1;c.segment_age_hours=101;rejects([&]{physicalContextChannels(4,1,c,"absolute25");});
 rejects([&]{contextReference(4,c,"unknown");});
 std::cout<<"PASS context recipe checks="<<checks<<'\n';
}
