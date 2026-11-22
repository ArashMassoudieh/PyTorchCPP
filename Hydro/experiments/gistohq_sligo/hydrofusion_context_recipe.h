#pragma once
#include "hydrofusion_causal_physics.h"
#include "hydrofusion_log_residual.h"
#include <array>
#include <sstream>
#include <string>

namespace hydrofusion {
// Explicit elapsed-hour offsets, newest first. Zero is compulsory; no row
// interpolation or gap bridging. A sequence supplies every intervening hour.
inline std::vector<int64_t> parseForecastHistory(const std::string& text,int64_t length){
    if(length<2||length>168||text.empty()||text.back()==',')throw std::invalid_argument("Invalid forecast history specification.");
    std::vector<int64_t> result;std::istringstream in(text);std::string token;
    while(std::getline(in,token,',')){
        if(token.empty()||token.find_first_not_of("0123456789")!=std::string::npos)throw std::invalid_argument("History offsets must be unsigned integer hours.");
        size_t used=0;const auto h=std::stoll(token,&used);
        if(used!=token.size()||h>=length||(!result.empty()&&h<=result.back()))throw std::invalid_argument("History offsets must be unique, ascending and within sequence.");
        result.push_back(h);
    }
    if(result.empty()||result.front()!=0)throw std::invalid_argument("Forecast history must include origin zero.");
    return result;
}
inline bool validContextRecipe(const std::string& recipe){return recipe=="zero25"||recipe=="absolute25"||recipe=="anchored25";}
inline std::array<double,5> physicalContextChannels(double originQ,double qref,
    const CausalPhysicalOriginContext& c,const std::string& recipe){
    if(!validContextRecipe(recipe))throw std::invalid_argument("Unknown physical-context recipe.");
    // Validate even the zero-channel control, so eligible support cannot differ.
    (void)dischargeAnchoredPhysicalReference(originQ,c);validateQRef(qref);
    const auto& s=c.ledger_at_origin;
    if(c.segment_age_hours>c.origin_x||!std::isfinite(s.fastStorage)||s.fastStorage<0||!std::isfinite(s.slowStorage)||s.slowStorage<0)
        throw std::invalid_argument("Invalid physical context storage/segment.");
    const double error=zTransform(originQ,qref)-zTransform(c.current_hour_mean,qref);
    if(!std::isfinite(error))throw std::overflow_error("Physical log-error overflow.");
    if(recipe=="zero25")return {0,0,0,0,0};
    return {c.current_hour_mean,c.next_hour_mean,s.fastStorage,s.slowStorage,error};
}
inline double contextReference(double originQ,const CausalPhysicalOriginContext& c,const std::string& recipe){
    if(!validContextRecipe(recipe))throw std::invalid_argument("Unknown physical-context reference.");
    const double anchored=dischargeAnchoredPhysicalReference(originQ,c);
    return recipe=="zero25"?originQ:recipe=="anchored25"?anchored:c.next_hour_mean;
}
} // namespace hydrofusion
