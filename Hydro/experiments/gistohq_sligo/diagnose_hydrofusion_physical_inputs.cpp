// Read-only diagnostic: exact production causal adapter/router plus hourly ledger.
// Does not alter recipes, fit parameters, or replace any saved prediction.
#include "hydrofusion_prepared_native_package.h"
#include "hydrofusion_causal_physics.h"
#include "hydrofusion_sequence_builder.h"
#include <iomanip>
#include <iostream>
#include <set>
int main(int argc,char**argv){try{
    using namespace hydrofusion;
    if(argc!=4)throw std::invalid_argument("Pass manifest,sha256,NEWcsv");
    if(std::filesystem::exists(argv[3]))throw std::invalid_argument("Existing output forbidden");
    const auto rows=loadPreparedNativePackage(argv[1],argv[2]);
    const CausalPhysicsParams p{1,.1,.7,1};
    const auto production=causalPhysicalForecasts(rows,p);
    const auto seq=buildForecastSequences(rows,{17543,26303,35063},24);
    std::set<int64_t> eligible;
    for(const auto&s:seq)eligible.insert(s.origin_x);
    std::ofstream out(argv[3]);if(!out)throw std::runtime_error("Cannot open diagnostic output");
    out<<"origin_x,target_x,eligible,forcing_valid,q_valid,origin_q,p_mm_h,pet_mm_h,effective_mm_h,endpoint_q,interval_mean_q,storage_before,storage_after,ledger_error\n"<<std::setprecision(17);
    ReservoirLedger state;int64_t previous=-2;double previousRain=0;double maxError=0;
    for(const auto&r:rows){
        const auto x=static_cast<int64_t>(r.x);
        if(x>=kSealStartX)throw std::invalid_argument("Preseal required");
        if(!r.all_forcings_valid){state={};previous=-2;continue;}
        if(x!=previous+1)state={};else stepCausalReservoirs(state,previousRain,p);
        const double rain=std::max(0.,r.precipitation_mm_per_hour-r.pet_mm_per_hour);
        const double before=state.fastStorage+state.slowStorage;
        auto forecast=state;const double oldOut=forecast.outflowVolume;
        const double q=stepCausalReservoirs(forecast,rain,p);
        const double integrated=forecast.outflowVolume-oldOut;
        const double after=forecast.fastStorage+forecast.slowStorage;
        const double error=before+rain*p.gain-integrated-after;
        if(q!=production.at(x)||!std::isfinite(integrated)||std::abs(error)>1e-9)throw std::runtime_error("Production/ledger parity failed");
        maxError=std::max(maxError,std::abs(error));
        out<<x<<','<<x+1<<','<<eligible.count(x)<<",1,"<<r.discharge_valid<<','<<r.observed_runoff_mm_per_hour<<','<<r.precipitation_mm_per_hour<<','<<r.pet_mm_per_hour<<','<<rain<<','<<q<<','<<integrated<<','<<before<<','<<after<<','<<error<<'\n';
        previous=x;previousRain=rain;
    }
    out.close();if(!out)throw std::runtime_error("Diagnostic write failed");
    // Exact rectangular one-hour rainfall probe: endpoint != hourly mean.
    ReservoirLedger zero;const double endpoint=stepCausalReservoirs(zero,1,p);
    std::cout<<std::setprecision(17)<<"PASS ledger_max_error="<<maxError<<" pulse_endpoint="<<endpoint<<" pulse_hour_mean="<<zero.outflowVolume<<" eligible="<<seq.size()<<'\n';
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
