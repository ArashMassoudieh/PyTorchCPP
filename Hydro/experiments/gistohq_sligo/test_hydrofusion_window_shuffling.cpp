// Synthetic invariants for the opt-in production window-order helper.
#include "hydrofusion_forecast_core.h"
#include <algorithm>
#include <numeric>
#include <random>
#include <iostream>
using namespace hydrofusion;
static std::vector<int64_t> order(int64_t n,int seed,int epoch){
 return seededForecastWindowOrder(n,seed,epoch);
}

int main(){try{
 torch::set_num_threads(2);torch::NoGradGuard guard;int checks=0;
 auto check=[&](bool b){if(!b)throw std::runtime_error("Window shuffle assertion");++checks;};
 constexpr int64_t n=17,t=24,c=26;auto a=order(n,42,1),b=order(n,42,1),d=order(n,42,2);
 check(a==b);check(a!=d);check(a!=order(n,43,1));auto sorted=a;std::sort(sorted.begin(),sorted.end());for(int64_t i=0;i<n;++i)check(sorted[i]==i);
 auto idx=torch::tensor(a,torch::kLong);auto x=torch::arange(n*t*c,torch::kFloat32).reshape({n,t,c})/1000;
 auto labels=torch::arange(n,torch::kFloat32).reshape({n,1});auto y=2*labels+1,reference=3*labels+2,observed=5*labels+4;
 auto sx=x.index_select(0,idx),sy=y.index_select(0,idx),sr=reference.index_select(0,idx),so=observed.index_select(0,idx);
 for(int64_t i=0;i<n;++i){check(torch::equal(sx[i],x[a[i]]));check(sy[i].item<float>()==2*a[i]+1);check(sr[i].item<float>()==3*a[i]+2);check(so[i].item<float>()==5*a[i]+4);}
 std::vector<int64_t> seen;
 for(int64_t start=0;start<n;start+=6){auto end=std::min(n,start+6);auto batch=idx.slice(0,start,end);
  check(batch.numel()==end-start);auto bx=x.index_select(0,batch),by=y.index_select(0,batch),br=reference.index_select(0,batch),bo=observed.index_select(0,batch);
  for(int64_t i=0;i<batch.numel();++i){auto j=batch[i].item<int64_t>();seen.push_back(j);check(torch::equal(bx[i],x[j]));check(by[i].item<float>()==2*j+1);check(br[i].item<float>()==3*j+2);check(bo[i].item<float>()==5*j+4);}
 }
 std::sort(seen.begin(),seen.end());for(int64_t i=0;i<n;++i)check(seen[i]==i);
 HydroRunConfig config;config.hidden_layers_csv="16";config.lstm_sequence_length=24;config.shuffle_training=false;config.random_seed=42;auto legacy=recipeTensor(config,"ffn_causal_forecast");
 config.shuffle_training=true;auto bound=recipeTensor(config,"ffn_causal_forecast");check(bound.numel()==legacy.numel()+3);check(torch::equal(bound.slice(0,0,legacy.numel()),legacy));
 config.random_seed=43;check(!torch::equal(recipeTensor(config,"ffn_causal_forecast"),bound));
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
 std::cout<<"Not CLI/reload acceptance, training convergence, or skill evidence; no fitted model\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
