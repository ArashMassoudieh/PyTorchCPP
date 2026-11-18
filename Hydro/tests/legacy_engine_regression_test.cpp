// Synthetic-only regression coverage for finalized legacy engine changes.
// Reuse the existing fabricated-package writer; no real observations are opened.
#define main existing_multiscale_smoke_main
#include "../experiments/gistohq_sligo/synthetic_multiscale_smoke_test.cpp"
#undef main
#include "../models/ffn_wrapper.h"
#include "../models/lstm_wrapper.h"
#include "../models/pinn_wrapper.h"
#include "../models/residual_pinn_wrapper.h"
#include "../dataset/hydro_tensor_builder.h"

void requireLegacy(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}
void checkLegacySplits(const HydroRunResult& r, const HydroRunConfig& c) {
    requireLegacy(r.success && !r.x.empty(), "empty/failed model result");
    requireLegacy(r.x.size() == r.split.size(), "unaligned split labels");
    bool sawTrain=false, sawValidation=false, sawTest=false, sawBeyond=false;
    for (std::size_t i=0;i<r.x.size();++i) {
        const double x=r.x[i];
        requireLegacy(x<77016 && std::isfinite(r.y_pred[i]), "nonfinite/sealed synthetic result");
        const std::string expected=x<=c.train_end_x?"train":x<=c.validation_end_x?"validation":x<=c.test_end_x?"test":"beyond_test_end";
        requireLegacy(r.split[i]==expected,"wrong exact-x split at "+std::to_string(x));
        sawTrain|=expected=="train";sawValidation|=expected=="validation";
        sawTest|=expected=="test";sawBeyond|=expected=="beyond_test_end";
    }
    requireLegacy(sawTrain&&sawValidation&&sawTest&&sawBeyond,"fixture must exercise all four partitions");
}
void checkLossChanged(const HydroRunResult& a,const HydroRunResult& b) {
    requireLegacy(!a.training_loss_history.empty()&&!b.training_loss_history.empty(),"loss histories missing");
    requireLegacy(std::isfinite(b.training_loss_history.front()),"weighted loss not finite");
    requireLegacy(a.training_loss_history.front()!=b.training_loss_history.front(),"nonzero weight had no effect on training objective");
}
int main(int argc,char**argv) {
    try {
        torch::set_num_threads(1);torch::set_num_interop_threads(1);
        const std::filesystem::path root=argc>1?argv[1]:"/tmp/hydropinn_legacy_engine_fixture";
        buildSyntheticPackage(root/"package",40);
        // Remove six weather hours: sequence position now differs from exported x.
        const auto weather=root/"package/observations/temporal_1.csv";
        std::ifstream input(weather);std::vector<std::string> lines;std::string line;
        for(int i=-1;std::getline(input,line);++i) if(i<250||i>255) lines.push_back(line);
        input.close();{std::ofstream output(weather);for(const auto& l:lines)output<<l<<'\n';}
        {std::ofstream cv(root/"package/observations/temporal_4.csv");
         cv<<"timestamp_utc,variable,value,native_unit,provider_qualifiers\n";
         for(int h=0;h<960;++h)cv<<isoUtc(1704067200LL+h*3600LL)<<",PRECIP_MRMS_SPATIAL_CV,0.25,1,SYNTHETIC\n";}
        HydroRunConfig c;c.use_hydro_package=true;c.hydro_package_path=(root/"package").string();
        c.epochs=1;c.batch_size=32;c.hidden_layers_csv="8";c.lstm_sequence_length=6;
        c.train_end_x=479;c.validation_end_x=719;c.test_end_x=879;
        c.normalization="standardize";c.random_seed=42;c.evaluate_metrics=true;
        c.storage_coeff=.2;c.lambda_decay=.03;c.runoff_coeff=.5;c.pinn_initial_flow=0.;
        torch::Tensor x,y,t,segments;loadHydroPackageTensors(c,x,y,t,segments);
        auto seven=c;seven.use_mrms_spatial_cv_feature=true;
        torch::Tensor x7,y7,t7,s7;loadHydroPackageTensors(seven,x7,y7,t7,s7);
        requireLegacy(x.size(1)==6&&x7.size(1)==7,"optional CV feature width");
        requireLegacy(torch::equal(x,x7.slice(1,0,6))&&torch::equal(y,y7)&&torch::equal(t,t7),"CV option changed existing features/support");
        requireLegacy(torch::allclose(x7.slice(1,6,7),torch::full_like(x7.slice(1,6,7),.25)),"CV values not transported");
        auto invalid=seven;invalid.use_latent_storage_physics=true;bool rejected=false;
        try{loadHydroPackageTensors(invalid,x,y,t,segments);}catch(const std::runtime_error&){rejected=true;}
        requireLegacy(rejected,"latent/CV incompatible mode must reject");
        auto ffn=FFNWrapper().train(c);checkLegacySplits(ffn,c);
        auto lstm=LSTMWrapper().train(c);checkLegacySplits(lstm,c);
        for(double endpoint:lstm.x) requireLegacy(!(endpoint>=250&&endpoint<=260),"LSTM window crossed missing-hour gap");
        auto repeat=LSTMWrapper().train(c);requireLegacy(allclose(lstm.y_pred,repeat.y_pred,0.),"zero-option reproducibility");
        auto high=c;high.high_flow_loss_weight=1.;high.high_flow_loss_quantile=.9;
        auto highResult=LSTMWrapper().train(high);checkLegacySplits(highResult,high);checkLossChanged(lstm,highResult);
        auto physical=c;physical.use_latent_storage_physics=true;physical.pinn_physics_profile="pinn_two_reservoir_hybrid";
        auto physics=PINNWrapper().train(physical);checkLegacySplits(physics,physical);
        // Fixed-state routing must not initialize itself from target discharge.
        const auto changedPackage=root/"changed_target_package";
        std::filesystem::copy(root/"package",changedPackage,std::filesystem::copy_options::recursive);
        const auto dischargeFile=changedPackage/"observations/temporal_3.csv";
        std::ifstream dischargeInput(dischargeFile);std::vector<std::string> dischargeLines;
        std::getline(dischargeInput,line);dischargeLines.push_back(line);
        while(std::getline(dischargeInput,line)) {
            const auto comma=line.find(',');
            dischargeLines.push_back(line.substr(0,comma+1)+std::to_string(2.*std::stod(line.substr(comma+1))));
        }
        dischargeInput.close();{std::ofstream changed(dischargeFile);for(const auto& l:dischargeLines)changed<<l<<'\n';}
        auto changedPhysics=physical;changedPhysics.hydro_package_path=changedPackage.string();
        requireLegacy(allclose(physics.y_pred,PINNWrapper().train(changedPhysics).y_pred,0.),"fixed physics baseline consulted target observations");
        auto residual=physical;residual.pinn_physics_profile="ffn_residual_pinn_hybrid";
        checkLegacySplits(FFNResidualPINNWrapper().train(residual),residual);
        auto shared=physical;shared.pinn_physics_profile="lstm_shared_scale_ablation_zero";
        auto sharedZero=LSTMPINNWrapper().train(shared);checkLegacySplits(sharedZero,shared);
        auto volume=shared;volume.shared_scale_volume_weight=1.;auto volumeResult=LSTMPINNWrapper().train(volume);checkLossChanged(sharedZero,volumeResult);
        auto daily=shared;daily.multiscale_daily_history_days=2;daily.multiscale_daily_hidden_size=4;
        auto dailyZero=LSTMPINNWrapper().train(daily);checkLegacySplits(dailyZero,daily);
        auto dailyVolume=daily;dailyVolume.shared_scale_volume_weight=1.;checkLossChanged(dailyZero,LSTMPINNWrapper().train(dailyVolume));
        auto control=daily;control.multiscale_matched_control=true;auto controlZero=LSTMPINNWrapper().train(control);checkLegacySplits(controlZero,control);
        requireLegacy(controlZero.x==dailyZero.x,"daily/control eligibility mismatch");
        auto controlVolume=control;controlVolume.shared_scale_volume_weight=1.;checkLossChanged(controlZero,LSTMPINNWrapper().train(controlVolume));
        const auto exported=root/"configs";HydroExperimentExporter exporter;
        exporter.exportRun(exported.string(),"multiscale",daily,{{"lstm_multiscale",dailyZero}});
        const auto reloaded=HydroInferenceRunner().runSharedScaleAblation(daily,(exported/"multiscale").string(),"lstm_multiscale",false);
        requireLegacy(allclose(dailyZero.y_pred,reloaded.predicted,1e-6)&&dailyZero.split==reloaded.split,"daily export/reload parity");
        auto urban=shared;urban.pinn_physics_profile="lstm_shared_scale_ablation_urban";urban.multiscale_daily_history_days=2;
        auto urbanResult=LSTMPINNWrapper().train(urban);checkLegacySplits(urbanResult,urban);
        exporter.exportRun(exported.string(),"urban",urban,{{"lstm_pinn",urbanResult}});
        exporter.exportRun(exported.string(),"highflow",high,{{"lstm",highResult}});
        exporter.exportRun(exported.string(),"ffn",c,{{"ffn",ffn}});
        exporter.exportRun(exported.string(),"physics",physical,{{"pinn",physics}});
        exporter.exportRun(exported.string(),"residual",residual,{{"ffn_pinn",FFNResidualPINNWrapper().train(residual)}});
        std::cout<<"PASS: exact cutoff/gap labels for FFN, LSTM, standalone physics, residual and shared-scale arms; CV off/on support; zero-option repeat; nonzero high-flow/volume objectives; daily checkpoint reload.\n";
        return 0;
    } catch(const std::exception& ex) {std::cerr<<"FAIL: "<<ex.what()<<'\n';return 1;}
}
