#pragma once
#include "artifact_loader.h"
#include "../models/hydrofusion_forecast_core.h"
#include "../experiments/gistohq_sligo/hydrofusion_metadata_sidecar.h"
#include "../dataset/hydro_checksum.h"
#include "../models/hydrofusion_context_schema.h"

// Managed-bundle, raw-hourly-context inference for the five additive modes.
// No labels/q_ref refitting and no target row are needed to predict at origin.
class HydroFusionForecastInference {
    HydroRunConfig config_;std::string approach_;double qref_=0;
    hydrofusion::TargetMode mode_=hydrofusion::TargetMode::LogResidual;
    TensorScaler xs_,ys_;hydrofusion::ForecastNet net_{nullptr};
public:
    HydroFusionForecastInference(const HydroInferenceArtifacts& artifacts,const std::string& directory,const std::string& approach)
        :config_(artifacts.experiment.config),approach_(approach){
        using namespace hydrofusion;validateForecastConfig(config_,approach_);
        HydroArtifactLoader().validateBundleManifest(directory);
        if(!config_.hydrofusion_synthetic_verification){
            const auto native=std::filesystem::path(directory)/"native_input_manifest.json";
            if(sha256File(native.string())!=config_.hydrofusion_native_manifest_sha256)
                throw std::invalid_argument("Native input provenance binding mismatch.");
        }
        const auto model=artifacts.models.find(approach_);const auto scaler=artifacts.scalers.find(approach_);
        if(model==artifacts.models.end()||scaler==artifacts.scalers.end()||model->second.format!=forecastCheckpointFormat(config_))throw std::invalid_argument("Missing/mismatched forecast artifact.");
        auto metadata=reloadHydroFusionMetadata((std::filesystem::path(directory)/"hydrofusion_metadata.txt").string());
        validateMetadataBoundToBundle(metadata,model->second.sha256,sha256File((std::filesystem::path(directory)/"scalers.csv").string()));
        mode_=forecastTargetMode(config_);qref_=metadata.qRef;
        if(forecastRecipeV4(config_)){
            std::ifstream schema(std::filesystem::path(directory)/"hydrofusion_context_schema.txt",std::ios::binary);
            const std::string text((std::istreambuf_iterator<char>(schema)),std::istreambuf_iterator<char>());
            if(text!=forecastContextSchema(config_))throw std::invalid_argument("Context feature schema/config mismatch.");
        }
        if(metadata.mode!=mode_||metadata.cutoffs.train_end_x!=config_.train_end_x||metadata.cutoffs.validation_end_x!=config_.validation_end_x||metadata.cutoffs.test_end_x!=config_.test_end_x)
            throw std::invalid_argument("Forecast metadata/config mismatch.");
        std::string bytes(model->second.bytes.begin(),model->second.bytes.end());std::istringstream input(bytes);
        torch::serialize::InputArchive checkpoint;checkpoint.load_from(input);torch::Tensor recipe,qref;
        checkpoint.read("recipe",recipe);checkpoint.read("qref",qref);
        if(!torch::equal(recipe,recipeTensor(config_,approach_))||qref.numel()!=1||qref.item<double>()!=qref_)
            throw std::invalid_argument("Forecast checkpoint recipe/scale mismatch.");
        if(config_.hydrofusion_objective=="log_flow_mse_v2"){
            torch::Tensor scale;checkpoint.read("training_flow_scale",scale);
            if(scale.scalar_type()!=torch::kFloat64||scale.numel()!=1||!std::isfinite(scale.item<double>())||scale.item<double>()<=0)throw std::invalid_argument("Missing/invalid training-only physical loss scale.");
        }
        if(approach_=="pinn_causal_forecast"){
            if(qref_!=1||scaler->second.input.method!="none"||scaler->second.target.method!="none")throw std::invalid_argument("Physical-only baseline must not have fitted transforms.");
        }else{
            xs_.importState(scaler->second.input);ys_.importState(scaler->second.target);
            if(scaler->second.input.shape!=std::vector<int64_t>{1,1,forecastRecipeV4(config_)?25:20}||scaler->second.target.shape!=std::vector<int64_t>{1,1})throw std::invalid_argument("Forecast scaler shape mismatch.");
            net_=ForecastNet(isRecurrentForecast(approach_),forecastHidden(config_),config_.hydrofusion_ffn_activation=="silu",forecastRecipeV4(config_)?25:20,parseForecastHistory(config_.hydrofusion_history_hours,config_.lstm_sequence_length));torch::serialize::InputArchive weights;checkpoint.read("weights",weights);net_->load(weights);net_->eval();
        }
    }
    double predictAt(const std::vector<hydrofusion::HourlyRow>& supplied,int64_t origin){
        using namespace hydrofusion;
        if(config_.hydrofusion_development_only&&origin>=config_.validation_end_x)throw std::invalid_argument("Development model refuses test-period inference.");
        const auto history=config_.lstm_sequence_length-1;
        if(origin<history||origin>=kSealStartX-1)throw std::invalid_argument("Invalid forecast origin.");
        const SplitCutoffs cutoffs{config_.train_end_x,config_.validation_end_x,config_.test_end_x};
        auto partition=assignSampleSplit(origin-history,origin+1,cutoffs);
        if(!partition||*partition==SplitPartition::BeyondTestEnd)throw std::invalid_argument("Forecast context crosses partition/end.");
        // Check all supplied identities before values; prefix creation touches no
        // row after origin. Callers need not supply target-time observations.
        HourlyIndex suppliedIndex(supplied);(void)suppliedIndex;
        for(const auto& row:supplied)if(row.x>=kSealStartX)throw std::invalid_argument("Presealed forecast context required.");
        std::vector<HourlyRow> prefix;for(const auto& row:supplied){if(row.x>origin)break;prefix.push_back(row);}
        HourlyIndex index(prefix);std::vector<torch::Tensor> steps;
        for(int64_t h=origin-history;h<=origin;++h){auto sample=buildSample(index,prefix,h,kSealStartX,false);
            if(!sample||!sample->predictor_availability_mask)throw std::invalid_argument("Incomplete native forecast history.");
            steps.push_back(featuresToTensorRow(sample->features));}
        double reference=steps.back()[0].item<double>();
        if(isPhysicalForecast(approach_))reference=forecastPhysicalSeries(prefix,config_).at(origin);
        if(approach_=="pinn_causal_forecast")return reference;
        if(forecastRecipeV4(config_)){
            const auto contexts=causalPhysicalOriginContexts(prefix,forecastPhysics(config_));
            DatasetSplit split;split.x=torch::stack(steps).unsqueeze(0);split.originX={origin};
            addForecastContext(split,contexts,config_,qref_);steps.clear();
            for(int64_t t=0;t<split.x.size(1);++t)steps.push_back(split.x[0][t]);
            reference=contextReference(prefix.at(*index.rowForHour(origin)).observed_runoff_mm_per_hour,contexts.at(origin),config_.hydrofusion_context_recipe);
        }
        torch::NoGradGuard guard;auto x=torch::stack(steps).unsqueeze(0);auto target=ys_.inverseTransform(net_->forward(xs_.transform(x)));
        if(!torch::isfinite(target).all().item<bool>())throw std::runtime_error("Nonfinite forecast output.");
        // Preserve original double-precision Q(origin), not its rounded feature.
        if(!forecastRecipeV4(config_)&&!isPhysicalForecast(approach_))reference=prefix.at(*index.rowForHour(origin)).observed_runoff_mm_per_hour;
        return reconstructQ(mode_,reference,target.item<double>(),qref_);
    }
};
