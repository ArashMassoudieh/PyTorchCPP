// Bounded no-neural calibration evaluator. Only train/validation truth is read.
// The canonical rows are independently hash-pinned by the campaign launcher.
#include "hydrofusion_causal_physics.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <array>
using namespace hydrofusion;
namespace {
using Table=std::vector<std::vector<std::string>>;
Table read(const std::string& path,const std::string& header,size_t fields){
 std::ifstream f(path);if(!f)throw std::invalid_argument("Cannot read input");std::string line;
 if(!std::getline(f,line))throw std::invalid_argument("Wrong CSV schema");
 if(!line.empty()&&line.back()=='\r')line.pop_back();
 if(line!=header)throw std::invalid_argument("Wrong CSV schema");
 Table table;
 while(std::getline(f,line)){
  if(!line.empty()&&line.back()=='\r')line.pop_back();
  std::vector<std::string> row;std::stringstream s(line);std::string cell;
  while(std::getline(s,cell,','))row.push_back(cell);
  if(row.size()!=fields||line.empty()||line.back()==',')throw std::invalid_argument("Malformed CSV row");
  table.push_back(row);}
 if(table.empty())throw std::invalid_argument("Empty CSV");
 return table;
}
double number(const std::string& s){size_t used=0;double v=std::stod(s,&used);if(used!=s.size()||!std::isfinite(v))throw std::invalid_argument("Invalid number");return v;}
int64_t hour(const std::string& s){double x=number(s);if(x<0||x>=kSealStartX||x!=std::floor(x))throw std::invalid_argument("Invalid/sealed timestamp");return static_cast<int64_t>(x);}
std::map<int64_t,double> truth(const Table& t,bool training){
 // Entire identity pass BEFORE any observed value, including wrong split rows.
 int64_t last=-1;for(const auto&r:t){auto x=hour(r[0]);if(x<=last||(training?x>17543:(x<=17543||x>26303)))throw std::invalid_argument("Wrong truth split/order");last=x;}
 std::map<int64_t,double> out;for(const auto&r:t){double q=number(r[1]);if(q<0)throw std::invalid_argument("Negative truth");out.emplace(hour(r[0]),q);}return out;
}
std::array<double,3> score(const std::map<int64_t,double>& observed,const std::map<int64_t,double>& predictions){
 double sum=0,sse=0,mae=0,pred=0;for(const auto&v:observed)sum+=v.second;
 double mean=sum/observed.size(),sst=0;
 for(const auto&v:observed){auto q=predictions.at(v.first-1);auto d=q-v.second;sse+=d*d;mae+=std::abs(d);pred+=q;sst+=(v.second-mean)*(v.second-mean);}
 if(!std::isfinite(sse)||!std::isfinite(sst)||sst<=0||sum<=0||!std::isfinite(pred)||!std::isfinite(mae))throw std::invalid_argument("Undefined score");
 return {1-sse/sst,mae/observed.size(),100*(pred-sum)/sum};
}
}
int main(int argc,char**argv){try{
 if(argc!=6)throw std::invalid_argument("canonical.csv train.csv validation.csv candidates.csv NEW_output.csv");
 if(std::filesystem::exists(argv[5]))throw std::invalid_argument("Refusing existing output");
 const auto raw=read(argv[1],"x,q,p,t,rh,wind,solar,pet,forcing_valid,discharge_valid",10);
 auto trainTable=read(argv[2],"x,observed",2),validationTable=read(argv[3],"x,observed",2);
 // Inspect all row identities; never parse canonical Q (column1) at ANY time.
 int64_t last=-1;for(const auto&r:raw){auto x=hour(r[0]);if(x<=last)throw std::invalid_argument("Unsorted canonical rows");last=x;}
 const auto train=truth(trainTable,true),validation=truth(validationTable,false);
 const auto candidates=read(argv[4],"index,fast,slow,fraction,gain",5);
 if(candidates.size()!=24)throw std::invalid_argument("Exactly24 frozen candidates required");
 std::vector<CausalPhysicsParams> params;size_t index=0;
 for(double fast:{1.,2.})for(double slow:{.04,.06,.1})for(double fraction:{.4,.7})for(double gain:{.5,1.}){
  const auto&r=candidates.at(index);if(number(r[0])!=static_cast<double>(index)||number(r[1])!=fast||number(r[2])!=slow||number(r[3])!=fraction||number(r[4])!=gain)throw std::invalid_argument("Candidate grid changed");
  params.push_back({fast,slow,fraction,gain});++index;}
 std::vector<HourlyRow> rows;
 for(const auto&r:raw){auto x=hour(r[0]);if(x>26302)break;HourlyRow out;out.x=static_cast<double>(x);out.rainfall_release_x=x;out.rainfall_valid_end_x=x-24;
  const double flag=number(r[8]);if(flag!=0&&flag!=1)throw std::invalid_argument("Invalid forcing flag");out.all_forcings_valid=flag==1;
  // Invalid forcing remains explicitly unavailable; not filled with old values.
  if(out.all_forcings_valid){out.precipitation_mm_per_hour=number(r[2]);out.pet_mm_per_hour=number(r[7]);}
  rows.push_back(out);}
 // Compute all results before opening the new output. No test truth or scores.
 std::vector<std::array<double,6>> scores;
 for(const auto&p:params){auto pred=causalPhysicalValidTimeDryForecasts(rows,p);auto a=score(train,pred),b=score(validation,pred);scores.push_back({a[0],a[1],a[2],b[0],b[1],b[2]});}
 std::ofstream f(argv[5]);if(!f)throw std::runtime_error("Cannot write output");f<<std::setprecision(17)<<"index,train_nse,train_mae,train_pbias_percent,validation_nse,validation_mae,validation_pbias_percent\n";
 for(size_t i=0;i<scores.size();++i){f<<i;for(double v:scores[i])f<<','<<v;f<<'\n';}f.close();if(!f)throw std::runtime_error("Output write failed");
 std::cout<<"PASS candidates=24 train_rows="<<train.size()<<" validation_rows="<<validation.size()<<" test_scores=0 canonical_Q_parsed=0\n";
}catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
