// Synthetic design probe ONLY. Production training/recipes are unchanged.
#include "hydrofusion_forecast_core.h"
#include <algorithm>
#include <numeric>
#include <random>
#include <iostream>
using namespace hydrofusion;
static std::vector<int64_t> order(int64_t n,uint64_t seed,uint64_t epoch){
 if(n<1||epoch<1)throw std::invalid_argument("Positive TRAIN count/epoch required");
 std::vector<int64_t> v(static_cast<size_t>(n));std::iota(v.begin(),v.end(),0);
 // Local RNG cannot change the network initialization/global torch RNG.
 std::seed_seq key{static_cast<uint32_t>(seed),static_cast<uint32_t>(seed>>32),static_cast<uint32_t>(epoch),static_cast<uint32_t>(epoch>>32),0x48594653u};
 std::mt19937_64 rng(key);std::shuffle(v.begin(),v.end(),rng);return v;
}
int main(){try{
 torch::set_num_threads(2);torch::NoGradGuard guard;int checks=0;
 auto check=[&](bool b){if(!b)throw std::runtime_error("Window shuffle assertion");++checks;};
 constexpr int64_t n=17,t=24,c=26;auto a=order(n,42,1),b=order(n,42,1),d=order(n,42,2);
 check(a==b);check(a!=d);auto sorted=a;std::sort(sorted.begin(),sorted.end());for(int64_t i=0;i<n;++i)check(sorted[i]==i);
 auto idx=torch::tensor(a,torch::kLong);auto x=torch::arange(n*t*c,torch::kFloat32).reshape({n,t,c})/1000;
 auto labels=torch::arange(n,torch::kFloat32).reshape({n,1});auto y=2*labels+1,reference=3*labels+2,observed=5*labels+4;
 auto sx=x.index_select(0,idx),sy=y.index_select(0,idx),sr=reference.index_select(0,idx),so=observed.index_select(0,idx);
 for(int64_t i=0;i<n;++i){check(torch::equal(sx[i],x[a[i]]));check(sy[i].item<float>()==2*a[i]+1);check(sr[i].item<float>()==3*a[i]+2);check(so[i].item<float>()==5*a[i]+4);}
 // Each complete [24,26] window is unchanged: no temporal/channel shuffling.
 check(torch::equal(sx.index_select(0,torch::argsort(idx)),x));
 TensorScaler scale;scale.fit(x,"standardize");check(torch::equal(scale.transform(sx),scale.transform(x).index_select(0,idx)));
 torch::manual_seed(123);auto before=torch::rand({5});auto unused=order(n,42,1);(void)unused;auto after=torch::rand({5});
 torch::manual_seed(123);check(torch::equal(before,torch::rand({5})));check(torch::equal(after,torch::rand({5})));
 for(bool recurrent:{false,true})for(auto history:std::vector<std::vector<int64_t>>{{0},{0,1,3,6,12,23}}){
  if(recurrent&&history.size()!=1)continue;
  torch::manual_seed(42);auto net=ForecastNet(recurrent,16,false,c,history);net->eval();
  auto canonical=net->forward(x),shuffled=net->forward(sx);
  check(torch::allclose(shuffled,canonical.index_select(0,idx),1e-5,1e-6));
  check(torch::allclose(net->forward(x),canonical,1e-5,1e-6)); // no persistent sample state
  check(torch::allclose(torch::mse_loss(canonical,y),torch::mse_loss(shuffled,sy),1e-5,1e-6));
 }
 bool bad=false;try{order(0,42,1);}catch(const std::invalid_argument&){bad=true;}check(bad);
 std::cout<<"PASS "<<checks<<" synthetic shuffling checks: permutation, all aligned tensors, within-window order, scaler transform, independent RNG, FFN sparse history/LSTM batch equivariance\n";
 std::cout<<"NOT production integration, checkpoint binding, training convergence, or skill evidence; no fitted model\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
