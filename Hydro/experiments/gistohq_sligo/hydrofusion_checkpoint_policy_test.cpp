#include "../../models/hydrofusion_checkpoint_policy.h"
#include <iostream>
using namespace hydrofusion;
int main(){int n=0;auto check=[&](bool b){if(!b)throw std::runtime_error("Checkpoint assertion");++n;};
 for(int e=1;e<=5;++e)for(double v:{0.,1.,100.})check(selectForecastCheckpoint("fixed_last_epoch_v1",e,5,v,2.)==(e==5));
 check(selectForecastCheckpoint("best_validation_v1",1,5,3.,std::numeric_limits<double>::infinity()));
 check(!selectForecastCheckpoint("best_validation_v1",2,5,3.,3.));check(selectForecastCheckpoint("best_validation_v1",2,5,2.,3.));
 auto rejects=[&](auto f){bool b=false;try{f();}catch(const std::exception&){b=true;}check(b);};
 rejects([&]{selectForecastCheckpoint("unknown",1,5,1.,2.);});rejects([&]{selectForecastCheckpoint("fixed_last_epoch_v1",6,5,1.,2.);});rejects([&]{selectForecastCheckpoint("fixed_last_epoch_v1",1,5,std::numeric_limits<double>::quiet_NaN(),2.);});
 std::cout<<"PASS "<<n<<" checkpoint policy checks\n";}
