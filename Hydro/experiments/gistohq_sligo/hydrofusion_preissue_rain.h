#pragma once
#include "hydrofusion_causal_physics.h"
#include <fstream>
#include <sstream>
#include <limits>
#include <set>
namespace hydrofusion {
constexpr int64_t kRainEpochSeconds=1420070400;
struct PreissueRain {int64_t origin=0,valid=0,release=0;double rate=0,coverage=0;};
inline int64_t rainInteger(const std::string& s){
    if(s.empty()||s.find_first_not_of("0123456789")!=std::string::npos)throw std::invalid_argument("Invalid rain integer token.");
    size_t used=0;auto x=std::stoll(s,&used);if(used!=s.size())throw std::invalid_argument("Invalid rain integer token.");return x;
}
inline double rainNumber(const std::string& s){
    if(s.empty()||s.find_first_of(" \t\r\n")!=std::string::npos)throw std::invalid_argument("Invalid rain numeric token.");
    size_t used=0;double x=std::stod(s,&used);if(used!=s.size()||!std::isfinite(x)||x<0||x>std::numeric_limits<float>::max())throw std::invalid_argument("Invalid rain numeric value.");return x;
}
inline std::map<int64_t,PreissueRain> parsePreissueRain(std::istream& input,int64_t maxTarget){
    if(maxTarget<1||maxTarget>=kSealStartX)throw std::invalid_argument("Invalid rain development cutoff.");
    std::string line;std::getline(input,line);if(!line.empty()&&line.back()=='\r')line.pop_back();
    if(line!="target_x,origin_x,valid_seconds,issue_seconds,release_proxy_seconds,coverage,rate_mm_h,usable,status")throw std::invalid_argument("Rain CSV schema mismatch.");
    std::map<int64_t,PreissueRain> out;int64_t expected=1;
    while(std::getline(input,line)){
        if(!line.empty()&&line.back()=='\r')line.pop_back();
        std::vector<std::string> fields;std::istringstream row(line);std::string token;
        while(std::getline(row,token,','))fields.push_back(token);
        if(fields.size()!=9||line.empty()||line.back()==',')throw std::invalid_argument("Rain CSV field count mismatch.");
        const auto target=rainInteger(fields[0]);
        if(target!=expected||target>maxTarget||target>=kSealStartX)throw std::invalid_argument("Rain target identity outside declared development support.");
        const auto origin=rainInteger(fields[1]);
        if(origin!=target-1)throw std::invalid_argument("Rain origin/target mismatch.");
        const auto issue=kRainEpochSeconds+origin*3600,valid=issue-600;
        if(rainInteger(fields[2])!=valid||rainInteger(fields[3])!=issue)throw std::invalid_argument("Rain valid/issue identity mismatch.");
        if(fields[7]!="0"&&fields[7]!="1")throw std::invalid_argument("Invalid rain availability flag.");
        if(fields[7]=="1"){
            const auto release=rainInteger(fields[4]);const double coverage=rainNumber(fields[5]),rate=rainNumber(fields[6]);
            if(fields[8]!="WEATHER_VALUE_VERIFIED"||release<valid||release>issue||coverage<.75||coverage>1.000000000001)throw std::invalid_argument("Unavailable/late rain declared usable.");
            out.emplace(origin,PreissueRain{origin,valid,release,rate,coverage});
        }else if(!fields[6].empty()||fields[8]=="WEATHER_VALUE_VERIFIED")throw std::invalid_argument("Unavailable rain must remain missing, never zero filled.");
        ++expected;
    }
    if(expected!=maxTarget+1)throw std::invalid_argument("Rain CSV lacks explicit complete hourly availability inventory.");
    return out;
}
inline bool preissueRainHistoryAvailable(const std::map<int64_t,PreissueRain>& rain,int64_t origin,int64_t length){
    if(length<2||length>168||origin<length-1||origin>=kSealStartX-1)return false;
    for(auto h=origin-length+1;h<=origin;++h)if(rain.find(h)==rain.end())return false;
    return true;
}
inline std::map<int64_t,CausalPhysicalOriginContext> preissueRainContexts(
    const std::vector<HourlyRow>& rows,const CausalPhysicsParams& p,const std::map<int64_t,PreissueRain>& rain,bool hold){
    auto contexts=causalPhysicalOriginContexts(rows,p);
    std::map<int64_t,double> pet;for(const auto& row:rows)if(row.all_forcings_valid)pet.emplace(static_cast<int64_t>(row.x),row.pet_mm_per_hour);
    if(hold)for(auto& item:contexts){
        const auto found=rain.find(item.first);if(found==rain.end())continue;
        auto forecast=item.second.ledger_at_origin; // Never modify committed past state.
        item.second.next_hour_mean=stepCausalReservoirHourMean(forecast,std::max(0.,found->second.rate-pet.at(item.first)),p);
    }
    return contexts;
}
}
