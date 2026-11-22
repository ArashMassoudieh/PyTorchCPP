#include "hydrofusion_prepared_native_package.h"
#include "hydrofusion_sequence_builder.h"
#include <iostream>
#include <functional>
#include <unistd.h>
using namespace hydrofusion;
void check(bool v){if(!v)throw std::runtime_error("independent mask assertion failed");}
void rejects(const std::function<void()>& f){bool bad=false;try{f();}catch(const std::invalid_argument&){bad=true;}check(bad);}
int main(){
 auto dir=std::filesystem::temp_directory_path()/("hydrofusion_masks_"+std::to_string(getpid()));std::filesystem::create_directory(dir);
 auto csv=dir/"intervals.csv",manifest=dir/"native_input_manifest.json";
 auto writeCsv=[&](bool extra,bool malformed=false){std::ofstream f(csv);f<<"epoch_seconds,q_m3_s,p_mm_h,t_c,rh_pct,wind_m_s,solar_mj_m2_h,pet_mm_h,forcing_valid,discharge_valid,native,interpolated,native_coverage,source_id"<<(extra?",rainfall_valid,common_origin_forcing_available":"")<<'\n';
 for(int h=0;h<400;++h){f<<1420070400LL+h*3600<<",1,"<<h<<",15,50,1,1,0,1,1,1,0,1,fabricated";
 if(extra){f<<','<<(h==100?"0":malformed&&h==90?"2":"1")<<','<<(h==149?0:1);}f<<'\n';}};
 QJsonObject m{{"schema","HydroFusionPreparedNative-v3"},{"time_label","interval_end"},{"study_type","idealized_retrospective_causal_hindcast"},{"interval_csv","intervals.csv"},{"row_count",400},{"first_interval_x",0},{"end_interval_x_exclusive",400},{"catchment_area_m2",3600000},{"minimum_native_coverage",.75},{"source_id","fabricated"},{"discharge_delay_hours",0},{"hourly_forcing_delay_hours",24},{"daily_pet_delay_hours",24},{"daily_pet_utc_start_hour",0},{"pet_input_semantics","short_grass_reference_et"},{"availability_evidence","synthetic"},{"rainfall_delay_hours",1},{"rainfall_release_delay_hours",1},{"rainfall_product","fabricated_verification"},{"rainfall_availability_basis","assumed_fixed_hourly_delay"},{"rainfall_availability_evidence","synthetic"},{"interval_mask_schema","independent_rain_and_common_origin_v1"}};
 QJsonArray pet;for(int h=0;h<384;h+=24)pet.append(QJsonObject{{"day_start_epoch_seconds",double(1420070400LL+h*3600)},{"reference_et_mm_day",1.},{"valid",true}});m["daily_pet"]=pet;
 auto load=[&](){m["interval_csv_sha256"]=QString::fromStdString(sha256File(csv.string()));{std::ofstream f(manifest);f<<QJsonDocument(m).toJson().toStdString();}return loadPreparedNativePackage(manifest.string(),sha256File(manifest.string()));};
 writeCsv(true);auto rows=load();check(rows.size()==400);
 check(!rows[101].all_forcings_valid); // interval100 ends101, age1 => origin102
 check(rows[102].all_forcings_valid && rows[102].temperature_c==15);
 check(rows[149].all_forcings_valid && !rows[149].study_predictor_available && rows[149].discharge_valid);
 HourlyIndex index(rows);auto sample=buildSample(index,rows,150);check(sample && !sample->predictor_availability_mask && sample->target_native_mask);
 auto changed=rows;changed[149].study_predictor_available=true;HourlyIndex ci(changed);check(buildSample(ci,changed,150)->predictor_availability_mask);
 check(rows[149].observed_runoff_mm_per_hour==changed[149].observed_runoff_mm_per_hour);
 writeCsv(true,true);rejects([&](){load();});
 writeCsv(false);rejects([&](){load();});
 writeCsv(true);m["interval_mask_schema"]="unknown";rejects([&](){load();});
 m["interval_mask_schema"]="independent_rain_and_common_origin_v1";m["schema"]="HydroFusionPreparedNative-v2";rejects([&](){load();});
 std::filesystem::remove_all(dir);std::cout<<"PASS independent rain masks, delayed mapping, common support without Q/state mutation, malformed/schema rejection\n";
}
