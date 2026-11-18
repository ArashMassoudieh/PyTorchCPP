#include "hydrofusion_sequence_builder.h"
#include "hydrofusion_synthetic_fixture.h"
#include <iostream>
using namespace hydrofusion;
void check(bool ok) {if(!ok)throw std::runtime_error("sequence test failure");}
int main() {
    auto rows=buildSyntheticFixture(); SplitCutoffs cutoffs{150,220,280};
    auto one=buildForecastSequences(rows,cutoffs,1);
    HourlyIndex index(rows); auto records=runPersistenceBenchmark(index,rows,cutoffs);
    size_t n=0;for(auto& r:records)if(r.included)++n;check(one.size()==n);
    auto sequences=buildForecastSequences(rows,cutoffs,24);check(!sequences.empty());
    for(const auto& s:sequences) {
        check(s.steps.size()==24 && s.target_x==s.origin_x+1);
        auto sample=buildSample(index,rows,s.origin_x);
        check(sample && sample->features.q_lag_0h==s.steps.back().q_lag_0h);
        auto p=assignSampleSplit(s.origin_x-23,s.target_x,cutoffs);check(p && *p==s.split);
    }
    auto baseline=sequences;
    // Future values cannot alter already-ended sequences.
    rows[250].observed_runoff_mm_per_hour=999;
    auto changed=buildForecastSequences(rows,cutoffs,24);
    for(size_t i=0;i<baseline.size();++i)if(baseline[i].origin_x<250)
        check(changed[i].steps.back().q_lag_0h==baseline[i].steps.back().q_lag_0h);
    rows.erase(rows.begin()+200);auto gapped=buildForecastSequences(rows,cutoffs,24);
    for(const auto& s:gapped) check(!(s.origin_x>=200 && s.origin_x<295));
    bool rejected=false;try{buildForecastSequences(rows,cutoffs,0);}catch(const std::invalid_argument&){rejected=true;}check(rejected);
    std::cout<<"Causal sequence builder: parity, partition, future-value and gap checks passed\n";
}
