#include "hydrofusion_preissue_rain.h"
#include "hydrofusion_synthetic_fixture.h"
#include <cassert>
#include <iostream>
using namespace hydrofusion;
int checks=0;template<class F>void rejects(F f){bool threw=false;try{f();}catch(const std::exception&){threw=true;}assert(threw);++checks;}
std::string header="target_x,origin_x,valid_seconds,issue_seconds,release_proxy_seconds,coverage,rate_mm_h,usable,status\n";
std::map<int64_t,PreissueRain> parse(const std::string& s,int64_t end=1){std::istringstream in(header+s);return parsePreissueRain(in,end);}
int main(){
 const std::string good="1,0,1420069800,1420070400,1420069860,1,5,1,WEATHER_VALUE_VERIFIED\n";
 auto r=parse(good);assert(r.at(0).rate==5);++checks;
 auto missing=parse("1,0,1420069800,1420070400,,,,0,WEATHER_REJECTED\n");assert(missing.empty());++checks;
 rejects([&]{parse("1,0,1420069800,1420070400,1420070401,1,5,1,WEATHER_VALUE_VERIFIED\n");});
 rejects([&]{parse("1,0,1420069800,1420070400,1420069860,1,nan,1,WEATHER_VALUE_VERIFIED\n");});
 rejects([&]{parse("1,0,1420069800,1420070400,1420069860,1,5junk,1,WEATHER_VALUE_VERIFIED\n");});
 rejects([&]{parse("77016,77015,0,0,0,0,nan,1,WEATHER_VALUE_VERIFIED\n");});
 rejects([&]{parse(good+good);});
 rejects([&]{parse("1,0,1420069800,1420070400,,,0,0,WEATHER_REJECTED\n");});
 rejects([&]{parse("1,0,1420069800,1420070400,1420069860,.74,5,1,WEATHER_VALUE_VERIFIED\n");});
 std::map<int64_t,PreissueRain> all;
 for(int64_t i=0;i<300;++i)all.emplace(i,PreissueRain{i,kRainEpochSeconds+i*3600-600,kRainEpochSeconds+i*3600-500,2.,1.});
 assert(preissueRainHistoryAvailable(all,100,24));++checks;auto gap=all;gap.erase(90);assert(!preissueRainHistoryAvailable(gap,100,24));++checks;
 auto rows=buildSyntheticFixture();auto dry=preissueRainContexts(rows,{},all,false),wet=preissueRainContexts(rows,{},all,true);
 for(const auto& c:dry){const auto& w=wet.at(c.first);assert(c.second.current_hour_mean==w.current_hour_mean&&c.second.ledger_at_origin.fastStorage==w.ledger_at_origin.fastStorage&&c.second.ledger_at_origin.slowStorage==w.ledger_at_origin.slowStorage);assert(w.next_hour_mean>c.second.next_hour_mean);++checks;}
 auto future=all;future.at(200).rate=1000;auto changed=preissueRainContexts(rows,{},future,true);assert(changed.at(100).next_hour_mean==wet.at(100).next_hour_mean);++checks;
 std::cout<<"PASS "<<checks<<" preissue rain checks\n";

}
