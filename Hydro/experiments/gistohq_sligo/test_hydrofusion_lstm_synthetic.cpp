#include "hydrofusion_sequence_assembly.h"
#include "hydrofusion_synthetic_fixture.h"
#include "hydrofusion_development_package.h"
#include "../../models/hydro_lstm_module.h"
#include "../../dataset/tensor_scaler.h"
#include <filesystem>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <unistd.h>
#include <sys/wait.h>
using namespace hydrofusion;
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
// Self-contained verification bundle, NOT an operational HydroArtifactLoader
// bundle or a new HydroBatch dispatch. All inputs are fabricated.
void saveScaler(torch::serialize::OutputArchive& archive,const std::string& name,const TensorScaler& scaler){
    auto state=scaler.exportState();
    archive.write(name+"_offset",torch::tensor(state.offset,torch::kFloat64));
    archive.write(name+"_scale",torch::tensor(state.scale,torch::kFloat64));
    archive.write(name+"_shape",torch::tensor(state.shape,torch::kInt64));
}
TensorScaler loadScaler(torch::serialize::InputArchive& archive,const std::string& name){
    torch::Tensor offset,scale,shape;archive.read(name+"_offset",offset);archive.read(name+"_scale",scale);archive.read(name+"_shape",shape);
    require(offset.scalar_type()==torch::kFloat64 && scale.scalar_type()==torch::kFloat64 && shape.scalar_type()==torch::kInt64,"scaler types");
    require(offset.dim()==1 && scale.dim()==1 && shape.dim()==1,"scaler ranks");
    HydroScalerState state;state.method="standardize";
    state.offset.assign(offset.data_ptr<double>(),offset.data_ptr<double>()+offset.numel());state.scale.assign(scale.data_ptr<double>(),scale.data_ptr<double>()+scale.numel());
    state.shape.assign(shape.data_ptr<int64_t>(),shape.data_ptr<int64_t>()+shape.numel());
    TensorScaler scaler;scaler.importState(state);return scaler;
}
torch::Tensor physical(HydroLSTM& model,const TensorScaler& xs,const TensorScaler& ys,const DatasetSplit& split,TargetMode mode,double qref){
    torch::NoGradGuard guard;model->eval();
    auto target=ys.inverseTransform(model->forward(xs.transform(split.x))).to(torch::kFloat64).contiguous();
    require(target.dim()==2 && target.size(1)==1 && torch::isfinite(target).all().item<bool>(),"nonfinite model target");
    auto q=reconstructHydroFusionColumn(mode,target,split.originQ,qref);
    return torch::tensor(q,torch::kFloat64).reshape({-1,1});
}
void reload(const std::filesystem::path& path){
    torch::serialize::InputArchive archive;archive.load_from(path.string());
    torch::Tensor modeValue,qValue,expected,x,originQ,ids,shape;
    archive.read("mode",modeValue);archive.read("qref",qValue);archive.read("expected",expected);
    archive.read("inputs",x);archive.read("origin_q",originQ);archive.read("ids",ids);archive.read("model_shape",shape);
    const auto mode=static_cast<TargetMode>(modeValue.item<int64_t>());validateTargetMode(mode);const auto qref=qValue.item<double>();validateQRef(qref);
    require(shape.numel()==3 && shape[0].item<int64_t>()==20 && shape[1].item<int64_t>()==8 && shape[2].item<int64_t>()==1,"model shape mismatch");
    require(x.dim()==3 && x.size(1)==24 && x.size(2)==20 && x.size(0)>0 && torch::isfinite(x).all().item<bool>(),"input schema");
    require(ids.dim()==2 && ids.size(1)==2 && ids.size(0)==x.size(0),"identities shape");
    require(((ids.select(1,1)-ids.select(1,0))==1).all().item<bool>() && (ids>=0).all().item<bool>() && (ids<kSealStartX).all().item<bool>(),"identities horizon/seal");
    require(originQ.sizes()==expected.sizes() && expected.dim()==2 && expected.size(0)==x.size(0) && expected.size(1)==1 && torch::isfinite(expected).all().item<bool>(),"expected schema/nonfinite");
    require(torch::equal(originQ.squeeze(1).to(torch::kFloat32),x.select(1,23).select(1,0)),"origin identity mismatch");
    auto xs=loadScaler(archive,"x");auto ys=loadScaler(archive,"y");
    HydroLSTM model(20,8,1,1);torch::serialize::InputArchive weights;archive.read("weights",weights);model->load(weights);
    DatasetSplit data;data.x=x;data.originQ=originQ;
    auto actual=physical(model,xs,ys,data,mode,qref);
    require(torch::isfinite(actual).all().item<bool>() && torch::equal(actual,expected),"fresh physical-Q reload mismatch");
    std::cout<<"Fresh LSTM physical-Q reload: "<<actual.size(0)<<" rows exact\n";
}
void run(const std::filesystem::path& path,TargetMode mode,const char* executable){
    NativeAdapterContext context;context.end_x_exclusive=300;context.catchment_area_m2=2000000;context.expected_source_id="fabricated-lstm-v1";
    const auto package=path.parent_path()/(path.stem().string()+"-hourly.csv");
    { std::ofstream file(package);file<<std::setprecision(17);
      file<<"epoch_seconds,q_m3_s,p_mm_h,t_c,rh_pct,wind_m_s,solar_mj_m2_h,pet_mm_h,forcing_valid,discharge_valid,native,interpolated,native_coverage,source_id\n";
      for(const auto& row:buildSyntheticFixture())file<<context.epoch_zero_seconds+static_cast<int64_t>(row.x)*3600<<','
        <<row.observed_runoff_mm_per_hour*context.catchment_area_m2/3600000.0<<','<<row.precipitation_mm_per_hour<<','<<row.temperature_c<<','
        <<row.relative_humidity_percent<<','<<row.wind_speed_m_per_second<<','<<row.solar_energy_mj_per_m2_per_hour<<','<<row.pet_mm_per_hour
        <<",1,1,1,0,1,"<<context.expected_source_id<<'\n';
      require(static_cast<bool>(file),"fabricated package write failed"); }
    const auto rows=loadDevelopmentHourlyPackage(package,context);
    auto data=prepareHydroFusionSequenceData(rows,mode,{150,220,280},24);
    require(data.train.x.size(1)==24 && data.train.x.size(2)==20,"sequence shape");
    // Exact same support in both target modes, with training-only scale.
    const auto other=prepareHydroFusionSequenceData(rows,mode==TargetMode::LogResidual?TargetMode::DirectLogQ:TargetMode::LogResidual,{150,220,280},24);
    require(data.qRef==other.qRef && data.train.targetX==other.train.targetX && data.test.targetX==other.test.targetX && torch::equal(data.train.x,other.train.x),"mode support parity");
    auto alteredRows=rows;
    for(auto& row:alteredRows)if(row.x>150)row.observed_runoff_mm_per_hour+=10;
    auto altered=prepareHydroFusionSequenceData(alteredRows,mode,{150,220,280},24);
    require(altered.qRef==data.qRef && torch::equal(altered.train.x,data.train.x) && torch::equal(altered.train.y,data.train.y),"validation/test influenced Train data or q_ref");
    TensorScaler xs,ys;xs.fit(data.train.x,"standardize");ys.fit(data.train.y,"standardize");
    TensorScaler alteredXs,alteredYs;alteredXs.fit(altered.train.x,"standardize");alteredYs.fit(altered.train.y,"standardize");
    require(xs.exportState().offset==alteredXs.exportState().offset && ys.exportState().offset==alteredYs.exportState().offset,"nontraining values influenced scaler offsets");
    torch::manual_seed(42);HydroLSTM model(20,8,1,1);torch::optim::Adam optimizer(model->parameters(),torch::optim::AdamOptions(0.01));
    double best=std::numeric_limits<double>::infinity();int bestEpoch=0;std::string bestWeights;
    for(int epoch=1;epoch<=12;++epoch){
        model->train();optimizer.zero_grad();auto loss=torch::mse_loss(model->forward(xs.transform(data.train.x)),ys.transform(data.train.y));
        require(std::isfinite(loss.item<double>()),"training loss nonfinite");loss.backward();optimizer.step();
        torch::NoGradGuard guard;model->eval();auto validation=torch::mse_loss(model->forward(xs.transform(data.validation.x)),ys.transform(data.validation.y)).item<double>();
        require(std::isfinite(validation),"validation loss nonfinite");
        if(validation<best){best=validation;bestEpoch=epoch;torch::serialize::OutputArchive checkpoint;model->save(checkpoint);std::ostringstream bytes;checkpoint.save_to(bytes);bestWeights=bytes.str();}
    }
    require(bestEpoch>0,"missing selected checkpoint");std::istringstream bytes(bestWeights);torch::serialize::InputArchive checkpoint;checkpoint.load_from(bytes);model->load(checkpoint);
    { torch::NoGradGuard guard;model->eval();
      const auto restoredValidation=torch::mse_loss(model->forward(xs.transform(data.validation.x)),ys.transform(data.validation.y)).item<double>();
      require(std::isfinite(restoredValidation) && std::abs(restoredValidation-best)<1e-8,"best validation checkpoint not restored"); }
    auto q=physical(model,xs,ys,data.test,mode,data.qRef);
    torch::serialize::OutputArchive archive,weights;model->save(weights);archive.write("weights",weights);
    saveScaler(archive,"x",xs);saveScaler(archive,"y",ys);
    archive.write("mode",torch::tensor(static_cast<int64_t>(mode)));archive.write("qref",torch::tensor(data.qRef,torch::kFloat64));
    archive.write("model_shape",torch::tensor({20,8,1},torch::kInt64));archive.write("expected",q);archive.write("inputs",data.test.x);archive.write("origin_q",data.test.originQ);
    archive.write("ids",torch::stack({torch::tensor(data.test.originX,torch::kInt64),torch::tensor(data.test.targetX,torch::kInt64)},1));archive.save_to(path.string());
    reload(path);
    auto pid=fork();require(pid>=0,"fork failed");if(pid==0){execl(executable,executable,"--reload",path.c_str(),static_cast<char*>(nullptr));_exit(127);}int status=0;require(waitpid(pid,&status,0)==pid && WIFEXITED(status) && WEXITSTATUS(status)==0,"child reload failure");
    std::cout<<"Synthetic LSTM mode "<<static_cast<int>(mode)<<": selected epoch "<<bestEpoch<<"/12\n";
}
int main(int argc,char** argv){
    torch::set_num_threads(1);
    if(argc==3 && std::string(argv[1])=="--reload"){reload(argv[2]);return 0;}
    require(argc==2,"Pass a fresh owned output directory");std::filesystem::path dir=argv[1];require(std::filesystem::create_directory(dir),"refusing existing output");
    run(dir/"residual.pt",TargetMode::LogResidual,argv[0]);run(dir/"direct.pt",TargetMode::DirectLogQ,argv[0]);
    std::cout<<"All synthetic LSTM sequence/scaling/fit/selection/reload checks passed\n";
}
