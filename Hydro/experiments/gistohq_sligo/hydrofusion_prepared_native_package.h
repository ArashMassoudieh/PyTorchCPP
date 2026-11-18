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
inline std::vector<HourlyRow> loadPreparedNativePackage(const std::string& path,const std::string& digest){
    if(digest.size()!=64 || sha256File(path)!=digest)throw std::invalid_argument("Native manifest hash mismatch.");
    std::ifstream f(path,std::ios::binary);std::string bytes((std::istreambuf_iterator<char>(f)),{});
    QJsonParseError error;auto document=QJsonDocument::fromJson(QByteArray::fromStdString(bytes),&error);
    if(error.error!=QJsonParseError::NoError || !document.isObject())throw std::invalid_argument("Malformed native manifest.");
    const auto m=document.object();
    auto text=[&](const char* key){if(!m[key].isString()||m[key].toString().isEmpty())throw std::invalid_argument("Missing native manifest string.");return m[key].toString().toStdString();};
    auto integer=[&](const char* key){if(!m[key].isDouble())throw std::invalid_argument("Missing native manifest integer.");const double n=m[key].toDouble();if(!std::isfinite(n)||n!=std::floor(n)||n<0||n>2000000000)throw std::invalid_argument("Invalid native manifest integer.");return static_cast<int64_t>(n);};
    if(text("schema")!="HydroFusionPreparedNative-v1" || text("time_label")!="interval_end" ||
       text("study_type")!="idealized_retrospective_causal_hindcast")throw std::invalid_argument("Unsupported native package semantics.");
    const auto csvName=text("interval_csv");if(csvName!="intervals.csv")throw std::invalid_argument("Native CSV must be local intervals.csv.");
    auto csv=std::filesystem::path(path).parent_path()/csvName;
    if(std::filesystem::is_symlink(csv)||sha256File(csv.string())!=text("interval_csv_sha256"))throw std::invalid_argument("Native CSV hash/path mismatch.");
    NativeAdapterContext context;context.first_x=integer("first_interval_x");context.end_x_exclusive=integer("end_interval_x_exclusive");
    if(!m["catchment_area_m2"].isDouble()||!m["minimum_native_coverage"].isDouble())throw std::invalid_argument("Native area/coverage required.");
    context.catchment_area_m2=m["catchment_area_m2"].toDouble();context.minimum_native_coverage=m["minimum_native_coverage"].toDouble();context.expected_source_id=text("source_id");
    // Existing adapter validates area/coverage; raw CSV first-pass validates ALL timestamps.
    auto inputs=loadDevelopmentHourlyInputs(csv,context);
    CausalNativePolicy policy{integer("discharge_delay_hours"),integer("hourly_forcing_delay_hours"),integer("daily_pet_delay_hours"),text("availability_evidence")};
    if(!m["daily_pet"].isArray())throw std::invalid_argument("Daily PET records required.");
    std::vector<NativeDailyPetEnergy> pet;
    for(const auto& value:m["daily_pet"].toArray()){
        if(!value.isObject())throw std::invalid_argument("Malformed daily PET row.");
        const auto row=value.toObject();
        const auto t=row["day_start_epoch_seconds"],energy=row["energy_mj_m2_day"],valid=row["valid"];
        if(!t.isDouble()||!energy.isDouble()||!valid.isBool()||!std::isfinite(t.toDouble())||t.toDouble()!=std::floor(t.toDouble())||t.toDouble()<1420070400||t.toDouble()>2000000000)
            throw std::invalid_argument("Malformed daily PET types/domain.");
        pet.push_back({static_cast<int64_t>(t.toDouble()),energy.toDouble(),valid.toBool()});
    }
    context.first_x+=1;context.end_x_exclusive+=1;
    auto result=adaptCausalNativeIntervals(inputs.rows,inputs.evidence,pet,context,policy);
    if(static_cast<int64_t>(result.size())!=integer("row_count"))throw std::invalid_argument("Native row count mismatch.");
    return result;
}
} // namespace hydrofusion
