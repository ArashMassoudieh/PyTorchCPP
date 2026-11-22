#include "../../evaluation/hydro_metrics.h"
#include <iostream>
int main(){
 HydroRunResult r;r.x={1,2,3,4};r.y_true={1,4,10,20};r.y_pred={1,3,10,15};r.split={"validation","validation","test","test"};
 auto legacy=r;populateHydroPeakMetrics(legacy);if(legacy.high_flow_rmse!=5||legacy.low_flow_rmse!=0)throw std::runtime_error("Legacy peak support changed");
 auto validation=r;populateHydroPeakMetrics(validation,"validation");if(validation.high_flow_rmse!=1||validation.low_flow_rmse!=0)throw std::runtime_error("Validation peak support mismatch");
 r.split.assign(4,"validation");bool rejected=false;try{populateHydroPeakMetrics(r);}catch(const std::invalid_argument&){rejected=true;}
 if(!rejected)throw std::runtime_error("Test default silently used validation");
 populateHydroPeakMetrics(r,"validation");if(r.high_flow_rmse!=5)throw std::runtime_error("Declared validation-only support failed");
 std::cout<<"PASS explicit development peak metrics and legacy default\n";
}
