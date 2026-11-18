#include "hydrofusion_development_package.h"
#include <iostream>
using namespace hydrofusion;
int main(int argc,char** argv) {
    if(argc!=2)throw std::invalid_argument("Pass a fresh test-owned directory.");
    const std::filesystem::path dir=argv[1];
    if(!std::filesystem::create_directory(dir))throw std::invalid_argument("Refusing existing directory.");
    NativeAdapterContext c;c.end_x_exclusive=300;c.catchment_area_m2=2000000;c.expected_source_id="fabricated-v1";
    const std::string header="epoch_seconds,q_m3_s,p_mm_h,t_c,rh_pct,wind_m_s,solar_mj_m2_h,pet_mm_h,forcing_valid,discharge_valid,native,interpolated,native_coverage,source_id\n";
    const std::string row="1420070400,2,0.1,15,50,1,0.8,0.04,1,1,1,0,1,fabricated-v1\n";
    auto write=[&](const std::string& text){std::ofstream f(dir/"hourly.csv");f<<text;};
    write(header+row);auto data=loadDevelopmentHourlyPackage(dir/"hourly.csv",c);
    if(data.size()!=1 || !data[0].discharge_valid || std::abs(data[0].observed_runoff_mm_per_hour-3.6)>1e-12)throw std::runtime_error("conversion failure");
    int rejected=0;
    for(const auto& bad:std::vector<std::string>{
        "1420070400,2junk,0.1,15,50,1,0.8,0.04,1,1,1,0,1,fabricated-v1\n",
        "1420070400,nan,0.1,15,50,1,0.8,0.04,1,1,1,0,1,fabricated-v1\n",
        "1420070400,2,0.1,15,50,1,0.8,0.04,2,1,1,0,1,fabricated-v1\n",
        "1420070400,2,0.1,15,50,1,0.8,0.04,1,1,1,0,1,fabricated-v1,extra\n",
        "1420070400,2,0.1,15,50,1,0.8,0.04,1,1,1,0,1,unknown\n",
        "169733?000,DO_NOT_PARSE,0.1,15,50,1,0.8,0.04,1,1,1,0,1,fabricated-v1\n",
        row+row, std::string{}}) {
        write(header+bad);bool fail=false;try{loadDevelopmentHourlyPackage(dir/"hourly.csv",c);}catch(const std::invalid_argument&){fail=true;}
        if(!fail)throw std::runtime_error("corrupt package accepted");
        ++rejected;
    }
    // Exact synthetic seal boundary: ensure timestamp failure precedes NaN parser.
    write(header+std::to_string(c.epoch_zero_seconds+kSealStartX*3600)+",nan,0,0,0,0,0,0,1,1,1,0,1,fabricated-v1\n");
    bool sealed=false;try{loadDevelopmentHourlyPackage(dir/"hourly.csv",c);}catch(const std::invalid_argument& e){sealed=std::string(e.what()).find("timestamp")!=std::string::npos;}
    if(!sealed)throw std::runtime_error("seal must be checked first");
    write(header+"1420070400,2,0.1,15,50,1,0.8,0.04,1,1,0,1,1,fabricated-v1\n");
    if(loadDevelopmentHourlyPackage(dir/"hourly.csv",c)[0].discharge_valid)throw std::runtime_error("interpolated support accepted");
    std::cout<<"Development disk package: conversion, interpolation and "<<rejected+1<<" rejection checks passed\n";
}
