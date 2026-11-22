#pragma once
#include "hydrofusion_development_package.h"
#include "hydrofusion_causal_native_adapter.h"
#include "../../dataset/hydro_checksum.h"
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>

namespace hydrofusion {
// Prepared package is a NEW pinned interchange, not a permissive replacement for
// GIStoOHQ's legacy loader. Python preparation independently checks the producer.
inline std::vector<HourlyRow> loadPreparedNativePackage(const std::string& path,const std::string& digest,int64_t maximumTargetX=kSealStartX-1){
    if(digest.size()!=64 || sha256File(path)!=digest)throw std::invalid_argument("Native manifest hash mismatch.");
    std::ifstream f(path,std::ios::binary);std::string bytes((std::istreambuf_iterator<char>(f)),{});
    QJsonParseError error;auto document=QJsonDocument::fromJson(QByteArray::fromStdString(bytes),&error);
    if(error.error!=QJsonParseError::NoError || !document.isObject())throw std::invalid_argument("Malformed native manifest.");
    const auto m=document.object();
    auto text=[&](const char* key){if(!m[key].isString()||m[key].toString().isEmpty())throw std::invalid_argument("Missing native manifest string.");return m[key].toString().toStdString();};
    auto integer=[&](const char* key){if(!m[key].isDouble())throw std::invalid_argument("Missing native manifest integer.");const double n=m[key].toDouble();if(!std::isfinite(n)||n!=std::floor(n)||n<0||n>2000000000)throw std::invalid_argument("Invalid native manifest integer.");return static_cast<int64_t>(n);};
    const auto schema=text("schema");
    const bool independentRain=schema=="HydroFusionPreparedNative-v3";
    const bool referenceEt=schema=="HydroFusionPreparedNative-v2" || independentRain;
    if((schema!="HydroFusionPreparedNative-v1" && !referenceEt) || text("time_label")!="interval_end" ||
       text("study_type")!="idealized_retrospective_causal_hindcast")throw std::invalid_argument("Unsupported native package semantics.");
    const auto csvName=text("interval_csv");if(csvName!="intervals.csv")throw std::invalid_argument("Native CSV must be local intervals.csv.");
    auto csv=std::filesystem::path(path).parent_path()/csvName;
    if(std::filesystem::is_symlink(csv)||sha256File(csv.string())!=text("interval_csv_sha256"))throw std::invalid_argument("Native CSV hash/path mismatch.");
    NativeAdapterContext context;context.first_x=integer("first_interval_x");context.end_x_exclusive=integer("end_interval_x_exclusive");
    if(!m["catchment_area_m2"].isDouble()||!m["minimum_native_coverage"].isDouble())throw std::invalid_argument("Native area/coverage required.");
    context.catchment_area_m2=m["catchment_area_m2"].toDouble();context.minimum_native_coverage=m["minimum_native_coverage"].toDouble();context.expected_source_id=text("source_id");
    if(maximumTargetX<1||maximumTargetX>=kSealStartX||context.end_x_exclusive>maximumTargetX)throw std::invalid_argument("Native package exceeds authorized target cutoff before observation parsing.");
    // Existing adapter validates area/coverage; raw CSV first-pass validates ALL timestamps.
    const bool independentMasks=m.contains("interval_mask_schema");
    if(independentMasks && (!independentRain || text("interval_mask_schema")!="independent_rain_and_common_origin_v1"))
        throw std::invalid_argument("Versioned independent native interval masks required.");
    auto inputs=loadDevelopmentHourlyInputs(csv,context,independentMasks);
    CausalNativePolicy policy{integer("discharge_delay_hours"),integer("hourly_forcing_delay_hours"),integer("daily_pet_delay_hours"),text("availability_evidence")};
    const char* rainKeys[]={"rainfall_delay_hours","rainfall_release_delay_hours","rainfall_product","rainfall_availability_basis","rainfall_availability_evidence"};
    if(independentRain){
        policy.rainfall_delay_hours=integer("rainfall_delay_hours");
        policy.rainfall_release_delay_hours=integer("rainfall_release_delay_hours");
        const auto product=text("rainfall_product");
        if(product!="MRMS_RadarOnly_QPE_01H" && product!="AORC_hourly" && product!="fabricated_verification")
            throw std::invalid_argument("Unsupported independent rainfall product; mixed fallback forbidden.");
        if(text("rainfall_availability_basis")!="assumed_fixed_hourly_delay")
            throw std::invalid_argument("Native v3 is a retrospective assumed-delay scenario, not operational certification.");
        text("rainfall_availability_evidence");
        if(product=="MRMS_RadarOnly_QPE_01H" && policy.rainfall_release_delay_hours<1)
            throw std::invalid_argument("Hourly radar policy cannot assert publication at the valid interval end.");
    }else{
        for(const auto key:rainKeys)if(m.contains(key))
            throw std::invalid_argument("Versioned native rainfall semantics required.");
    }
    if(referenceEt){
        policy.daily_pet_utc_start_hour=integer("daily_pet_utc_start_hour");
        if(text("pet_input_semantics")!="short_grass_reference_et")
            throw std::invalid_argument("Unsupported native reference ET semantics.");
    }else if(m.contains("daily_pet_utc_start_hour") || m.contains("pet_input_semantics")){
        throw std::invalid_argument("Versioned native ET semantics required.");
    }
    if(!m["daily_pet"].isArray())throw std::invalid_argument("Daily PET records required.");
    std::vector<NativeDailyPetEnergy> pet;
    for(const auto& value:m["daily_pet"].toArray()){
        if(!value.isObject())throw std::invalid_argument("Malformed daily PET row.");
        const auto row=value.toObject();
        const auto t=row["day_start_epoch_seconds"],energy=row[referenceEt ? "reference_et_mm_day" : "energy_mj_m2_day"],valid=row["valid"];
        if(!t.isDouble()||!energy.isDouble()||!valid.isBool()||!std::isfinite(t.toDouble())||t.toDouble()!=std::floor(t.toDouble())||t.toDouble()<1420070400||t.toDouble()>2000000000)
            throw std::invalid_argument("Malformed daily PET types/domain.");
        const double nativeEnergy=energy.toDouble()*(referenceEt ? 2.45 : 1.0);
        if(referenceEt && (!std::isfinite(nativeEnergy) || nativeEnergy<0))
            throw std::invalid_argument("Invalid native reference ET depth.");
        pet.push_back({static_cast<int64_t>(t.toDouble()),nativeEnergy,valid.toBool()});
    }
    context.first_x+=1;context.end_x_exclusive+=1;
    auto result=adaptCausalNativeIntervals(inputs.rows,inputs.evidence,pet,context,policy);
    if(static_cast<int64_t>(result.size())!=integer("row_count"))throw std::invalid_argument("Native row count mismatch.");
    // Study mask describes forcing availability AT this origin, after the
    // independent rain/met delays. Never alter Q/native-support evidence.
    // This removes sample eligibility, not physical forcing/state evolution.
    if(independentMasks){
        if(inputs.common_origin_forcing_available.size()!=result.size())throw std::invalid_argument("Native common-origin mask size mismatch.");
        for(size_t i=0;i<result.size();++i)result[i].study_predictor_available=inputs.common_origin_forcing_available[i];
    }
    return result;
}
} // namespace hydrofusion
