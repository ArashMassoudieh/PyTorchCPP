#pragma once
#include "hydrofusion_native_support_adapter.h"
#include <fstream>
#include <sstream>
#include <filesystem>

namespace hydrofusion {
// Deliberately new DEVELOPMENT interchange, NOT a GIStoOHQ manifest parser.
// Row schema fixes all units in the header; context supplies epoch/area/source.
// No real-source path is wired into model training by this reader.
struct NativeHourlyInputs { std::vector<GisToOhqHourlyRow> rows; std::vector<NativeSupportEvidence> evidence; std::vector<bool> common_origin_forcing_available; };
inline NativeHourlyInputs loadDevelopmentHourlyInputs(
    const std::filesystem::path& csv, const NativeAdapterContext& context, bool independentMasks=false) {
    std::ifstream stream(csv);
    if (!stream) throw std::invalid_argument("Cannot open development hourly package.");
    std::string line;
    const std::string header="epoch_seconds,q_m3_s,p_mm_h,t_c,rh_pct,wind_m_s,solar_mj_m2_h,pet_mm_h,forcing_valid,discharge_valid,native,interpolated,native_coverage,source_id";
    if (!std::getline(stream,line) || line != header+(independentMasks?",rainfall_valid,common_origin_forcing_available":""))
        throw std::invalid_argument("Unexpected development package schema/units.");
    std::vector<GisToOhqHourlyRow> rows;
    std::vector<NativeSupportEvidence> support;
    std::vector<bool> commonOrigin;
    auto fields=[&](const std::string& row) {
        std::vector<std::string> out; size_t start=0;
        for (;;) {const auto end=row.find(',',start);out.push_back(row.substr(start,end-start));if(end==std::string::npos)break;start=end+1;}
        if(out.size()!=(independentMasks?16u:14u))throw std::invalid_argument("Development package field count mismatch.");
        return out;
    };
    auto integer=[](const std::string& token) {
        if(token.empty() || token.find_first_not_of("0123456789")!=std::string::npos)
            throw std::invalid_argument("Malformed development package integer.");
        size_t used=0; int64_t n;
        try{n=std::stoll(token,&used);}catch(const std::exception&){throw std::invalid_argument("Integer overflow.");}
        if(used!=token.size())throw std::invalid_argument("Trailing integer content.");
        return n;
    };
    auto number=[](const std::string& token) {
        if(token.empty() || token.find_first_of(" \t\r\n")!=std::string::npos)
            throw std::invalid_argument("Empty/whitespace numeric field.");
        size_t used=0;double value;
        try{value=std::stod(token,&used);}catch(const std::exception&){throw std::invalid_argument("Malformed numeric field.");}
        if(used!=token.size() || !std::isfinite(value))throw std::invalid_argument("Non-finite/trailing numeric field.");
        return value;
    };
    auto flag=[](const std::string& token) {
        if(token!="0" && token!="1")throw std::invalid_argument("Boolean must be 0 or 1.");
        return token=="1";
    };
    // First pass reads timestamp tokens only. Reject sealed/out-of-slice rows
    // before converting or examining ANY observation/forcing token. No full-data
    // load followed by filtering; upstream must already have produced a slice.
    std::vector<std::vector<std::string>> raw;
    while(std::getline(stream,line)) {
        auto f=fields(line);const auto epoch=integer(f[0]);
        if(context.epoch_zero_seconds!=1420070400 || context.first_x<0 ||
           context.end_x_exclusive<=context.first_x || context.end_x_exclusive>kSealStartX)
            throw std::invalid_argument("Invalid preseal context.");
        const auto lower=context.epoch_zero_seconds+context.first_x*3600;
        const auto upper=context.epoch_zero_seconds+context.end_x_exclusive*3600;
        if(epoch<lower || epoch>=upper || (epoch-context.epoch_zero_seconds)%3600!=0)
            throw std::invalid_argument("Development package timestamp outside preseal hourly slice.");
        raw.push_back(std::move(f));
    }
    if(!stream.eof() || raw.empty())throw std::invalid_argument("Empty or unreadable development package.");
    for(const auto& f:raw) {
        GisToOhqHourlyRow r;r.epoch_seconds=integer(f[0]);r.observed_discharge_m3_per_second=number(f[1]);
        r.precipitation_mm_per_hour=number(f[2]);r.temperature_c=number(f[3]);r.relative_humidity_percent=number(f[4]);
        r.wind_speed_m_per_second=number(f[5]);r.solar_energy_mj_per_m2_per_hour=number(f[6]);r.pet_mm_per_hour=number(f[7]);
        r.all_forcings_valid=flag(f[8]);r.precipitation_valid=r.temperature_valid=r.relative_humidity_valid=
            r.wind_valid=r.solar_valid=r.pet_valid=r.all_forcings_valid;
        r.discharge_valid=flag(f[9]);
        if(independentMasks){r.precipitation_valid=flag(f[14]);commonOrigin.push_back(flag(f[15]));}
        support.push_back({r.epoch_seconds,flag(f[10]),flag(f[11]),number(f[12]),f[13]});rows.push_back(r);
    }
    return {std::move(rows),std::move(support),std::move(commonOrigin)};
}
inline std::vector<HourlyRow> loadDevelopmentHourlyPackage(
    const std::filesystem::path& csv, const NativeAdapterContext& context) {
    auto input=loadDevelopmentHourlyInputs(csv,context);
    return adaptNativeHourlyRows(input.rows,input.evidence,context);
}
} // namespace hydrofusion
