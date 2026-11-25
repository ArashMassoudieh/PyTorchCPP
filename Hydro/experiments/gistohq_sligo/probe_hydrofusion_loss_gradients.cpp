// Synthetic diagnostic of the exact production objective; no model fitting/data.
#include "hydrofusion_forecast_core.h"
#include <iostream>
#include <iomanip>
using namespace hydrofusion;
int main(){try{
 torch::set_num_threads(2);HydroRunConfig c;c.hydrofusion_objective="log_flow_mse_v2";c.hydrofusion_target_mode="log_residual";
 TensorScaler ys;ys.fit(torch::tensor({-1.,1.},torch::kFloat64).reshape({2,1}),"standardize");
 const double qref=.1,scale=.2,ref=.5;int checks=0;
 std::cout<<std::setprecision(17)<<"observed,predicted_flow,latent_z,loss,gradient,finite_difference,physical_gradient,log_gradient\n";
 for(double observed:{.01,.5,2.,10.})for(double predicted:{-.05,0.01,.25,.5,2.,8.}){
  double z=std::log1p(predicted/qref),r=z-std::log1p(ref/qref),target=std::log1p(observed/qref)-std::log1p(ref/qref);
  auto x=torch::tensor({r},torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true)).reshape({1,1});x.retain_grad();
  auto y=torch::tensor({target},torch::kFloat64).reshape({1,1}),reference=torch::full({1,1},ref,torch::kFloat64),obs=torch::full({1,1},observed,torch::kFloat64);
  auto loss=forecastObjective(c,x,y,ys,reference,obs,qref,scale);loss.backward();double g=x.grad().item<double>();
  auto value=[&](double a){return forecastObjective(c,torch::full({1,1},a,torch::kFloat64),y,ys,reference,obs,qref,scale).item<double>();};
  const double eps=1e-6,fd=(value(r+eps)-value(r-eps))/(2*eps);
  const double physical=predicted<0?0:(predicted-observed)*(predicted+qref)/(scale*scale),logpart=r-target;
  if(!std::isfinite(g)||std::abs(g-fd)>1e-5*std::max(1.,std::abs(g))||std::abs(g-physical-logpart)>1e-9*std::max(1.,std::abs(g)))throw std::runtime_error("Gradient decomposition mismatch");
  if((predicted<observed&&g>=0)||(predicted>observed&&g<=0))throw std::runtime_error("Wrong correction direction");
  ++checks;std::cout<<observed<<','<<std::max(0.,predicted)<<','<<z<<','<<loss.item<double>()<<','<<g<<','<<fd<<','<<physical<<','<<logpart<<'\n';
 }
 std::cerr<<"PASS "<<checks<<" production autograd/finite-difference/direction checks\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
