#include "hydrofusion_causal_physics.h"
#include <iostream>
using namespace hydrofusion;
static void require(bool condition){if(!condition)throw std::runtime_error("Physical diagnostic regression failed");}
int main(){
    const CausalPhysicsParams p{1,.1,.7,1};
    ReservoirLedger pulse;
    const double endpoint=stepCausalReservoirs(pulse,1,p);
    const double exactEnd=.7*(-std::expm1(-1.))+.3*(-std::expm1(-.1));
    const double exactMean=1-(.7*(-std::expm1(-1.))/1+.3*(-std::expm1(-.1))/.1);
    require(std::abs(endpoint-exactEnd)<1e-12);
    require(std::abs(pulse.outflowVolume-exactMean)<1e-12);
    require(std::abs(endpoint-pulse.outflowVolume)>.19);
    // The fast fraction is a partition, not a runoff-loss coefficient.
    for(double fraction:{0.,.7,1.}){
        ReservoirLedger s;CausalPhysicsParams recipe{1,.1,fraction,1};
        stepCausalReservoirs(s,1,recipe);
        for(int i=0;i<1000;++i)stepCausalReservoirs(s,0,recipe);
        require(std::abs(s.outflowVolume-1)<1e-10);
        require(std::abs(s.inputVolume-s.outflowVolume-s.fastStorage-s.slowStorage)<1e-10);
    }
    // Dry-hour PET does not enter state loss: max(0-PET,0) is always zero.
    ReservoirLedger low,high;stepCausalReservoirs(low,.9,p);stepCausalReservoirs(high,.9,p);
    for(int i=0;i<1000;++i){stepCausalReservoirs(low,std::max(0.-.1,0.),p);stepCausalReservoirs(high,std::max(0.-100.,0.),p);}
    require(low.outflowVolume==high.outflowVolume);
    require(std::abs(low.outflowVolume-.9)<1e-10);
    std::cout<<"PASS pulse_endpoint_vs_hour_mean; mass_closure; partition_is_not_loss; dry_PET_cannot_deplete_storage\n";
}
