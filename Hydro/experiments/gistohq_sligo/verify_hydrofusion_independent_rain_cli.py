#!/usr/bin/env python3
"""Fabricated native-v3 independent rainfall through all five forecast families.

No native research fits. Seven variants include both nonphysical target modes;
legacy native-v2 predictions must remain byte-identical to a pinned old binary.
"""
import argparse,hashlib,json,os,resource,shutil,subprocess
from pathlib import Path
from test_corrected_gauge_package import inputs
from prepare_corrected_gauge_package import prepare
from verify_hydrofusion_reference_et_cli import NAMES
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run(binary,verifier,baseline,output):
    binary,verifier,baseline,output=[Path(p).absolute() for p in (binary,verifier,baseline,output)]
    output.mkdir(parents=True,exist_ok=False)
    native,weather=inputs(output)
    prepare(native,weather,output/'prepared_v2',producer_catalog=weather/'station_catalog.json',original_export_manifest=weather/'station_source_manifest.json',station_metadata=weather/'station.rdb')
    legacy=output/'prepared_v2/native_input_manifest.json'
    shutil.copytree(legacy.parent,output/'prepared_v3')
    manifest=output/'prepared_v3/native_input_manifest.json';m=json.loads(manifest.read_text())
    m.update(schema='HydroFusionPreparedNative-v3',rainfall_delay_hours=1,rainfall_release_delay_hours=1,
             rainfall_product='fabricated_verification',rainfall_availability_basis='assumed_fixed_hourly_delay',
             rainfall_availability_evidence='Fabricated one-hour release delay, not provider/operational evidence.')
    manifest.write_text(json.dumps(m,indent=2)+'\n')
    shutil.copytree(manifest.parent,output/'prepared_earlier_release')
    earlier=output/'prepared_earlier_release/native_input_manifest.json';em=dict(m,rainfall_delay_hours=2);earlier.write_text(json.dumps(em,indent=2)+'\n')
    dispatch=HERE.parents[2]/'build-hydrobatch'/('independent_rain_'+hashlib.sha256(str(output).encode()).hexdigest()[:12]);dispatch.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MALLOC_ARENA_MAX='2',CUDA_VISIBLE_DEVICES='')
    def limits():resource.setrlimit(resource.RLIMIT_AS,(12*1024**3,12*1024**3))
    receipt=dict(synthetic_only=True,binary_sha256=sha(binary),verifier_sha256=sha(verifier),baseline_sha256=sha(baseline),v2_sha256=sha(legacy),v3_sha256=sha(manifest),commands=[],legacy_identical=[])
    def command(name,args,expected=0,reason=None):
        with (output/(name+'.log')).open('x') as log:p=subprocess.run(list(map(str,args)),stdout=log,stderr=subprocess.STDOUT,env=env,timeout=600,preexec_fn=limits)
        text=(output/(name+'.log')).read_text()
        if p.returncode!=expected or reason and reason not in text:raise RuntimeError(name+': '+text[-2500:])
        receipt['commands'].append(dict(name=name,arguments=list(map(str,args)),exit_code=p.returncode))
    def config(name,manifest,modern=False):
        c=json.loads((HERE/'hydrofusion_five_method_checks'/(name+'.json')).read_text())
        c.update(epochs=3,hydrofusion_synthetic_verification=False,hydrofusion_native_manifest_path=str(manifest),hydrofusion_native_manifest_sha256=sha(manifest))
        if modern:
            if not name.startswith('pinn_'):c['hydrofusion_objective']='log_flow_mse_v2'
            if 'physical' in name or name.startswith('pinn_'):c['physics_profile']='causal_valid_time_dry_v3'
        return c
    for tag,path,modern in [('legacy',legacy,False),('timing_v3_control',legacy,True),('new',manifest,True),('earlier_release',earlier,True)]:
        jobs=[]
        for name in NAMES:
            c=config(name,path,modern);file=dispatch/(tag+'_'+name+'.json');file.write_text(json.dumps(c,indent=2)+'\n');jobs.append(name.removesuffix('_direct_log_q').removesuffix('_log_residual')+' '+file.name)
        batch=dispatch/(tag+'.batch');batch.write_text('\n'.join(jobs)+'\n')
        command(tag+'_cli',[binary,batch,output/(tag+'_bundles')])
        if tag in ('legacy','timing_v3_control'):
            command(tag+'_baseline_cli',[baseline,batch,output/(tag+'_baseline_bundles')])
            for name in NAMES:
                a=output/(tag+'_baseline_bundles')/name/'predictions.csv';b=output/(tag+'_bundles')/name/'predictions.csv'
                if a.read_bytes()!=b.read_bytes():raise RuntimeError('Legacy prediction regression: '+name)
                receipt['legacy_identical'].append(dict(tag=tag,name=name,sha256=sha(a)))
        for name in NAMES:command(tag+'_'+name+'_reload',[verifier,output/(tag+'_bundles')/name,name.removesuffix('_direct_log_q').removesuffix('_log_residual')])
    bad=[('negative_age','rainfall_delay_hours',-1,'Invalid native manifest integer'),
         ('fractional_age','rainfall_delay_hours',1.5,'Invalid native manifest integer'),
         ('too_old','rainfall_delay_hours',169,'Invalid independent rainfall'),
         ('unreleased','rainfall_release_delay_hours',2,'Invalid independent rainfall'),
         ('missing_release','rainfall_release_delay_hours',None,'Missing native manifest integer'),
         ('mixed_product','rainfall_product','generic_blended_precipitation','Unsupported independent rainfall product'),
         ('false_operational_claim','rainfall_availability_basis','operational_certified','retrospective assumed-delay'),
         ('missing_evidence','rainfall_availability_evidence','', 'Missing native manifest string'),
         ('downgrade','schema','HydroFusionPreparedNative-v2','Versioned native rainfall semantics required')]
    for name,key,value,reason in bad:
        folder=output/name;shutil.copytree(manifest.parent,folder);path=folder/'native_input_manifest.json';data=dict(m)
        if value is None:data.pop(key)
        else:data[key]=value
        path.write_text(json.dumps(data));c=config(NAMES[0],path,True);c['experiment_id']=name;file=dispatch/(name+'.json');file.write_text(json.dumps(c));batch=dispatch/(name+'.batch');batch.write_text('ffn_causal_forecast '+file.name+'\n')
        command(name,[binary,batch,output/(name+'_bundles')],1,reason)
    # Hash binding rejects changed native timing even without using target values.
    tampered=output/'tampered';shutil.copytree(output/'new_bundles'/NAMES[0],tampered);p=tampered/'native_input_manifest.json';d=json.loads(p.read_text());d['rainfall_delay_hours']=24;p.write_text(json.dumps(d))
    command('archive_native_hash_rejection',[verifier,tampered,'ffn_causal_forecast'],1,'Artifact bundle integrity check failed: native_input_manifest.json')
    shutil.copytree(dispatch,output/'dispatch_configs')
    (output/'acceptance.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('PASS28CLI/fresh-reload variants across5families;14byte-identical legacy predictions;10negative checks. Synthetic only.')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','verifier','baseline','output'):p.add_argument(name,type=Path)
    a=p.parse_args();run(a.binary,a.verifier,a.baseline,a.output)
