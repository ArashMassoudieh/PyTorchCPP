#!/usr/bin/env python3
"""Bounded development-only HydroPINN search backend, shared by CLI and Qt.
No test ranking, retries, implicit candidates, or reserved-data access.
"""
import argparse,csv,fcntl,hashlib,json,math,os,re,signal,subprocess,sys,time
from pathlib import Path
EXPERIMENTS=Path(__file__).resolve().parents[1]/'experiments/gistohq_sligo'
sys.path.insert(0,str(EXPERIMENTS))
from run_hydrofusion_six_fit_screen import directory_bytes
import resource,ctypes
FAMILIES={'ffn_causal_forecast','lstm_hydrofusion','ffn_physical_forecast','lstm_physical_forecast','pinn_causal_forecast'}
SAFE=re.compile(r'^[A-Za-z0-9_-]{1,100}$')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
 def pairs(items):
  out={}
  for k,v in items:
   if k in out:raise ValueError('Duplicate JSON key: '+k)
   out[k]=v
  return out
 return json.loads(Path(p).read_text(),object_pairs_hook=pairs,parse_constant=lambda v:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))
def write(p,v,exclusive=False):
 data=json.dumps(v,indent=2,allow_nan=False)+'\n'
 if exclusive:
  with Path(p).open('x') as f:f.write(data);f.flush();os.fsync(f.fileno())
 else:
  temp=Path(str(p)+'.tmp');temp.write_text(data);temp.replace(p)
def number(v,low,high):
 if isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v) or not low<=v<=high:raise ValueError('Numeric value outside allowed range')
 return v
def integer(v,low,high):
 number(v,low,high)
 if not isinstance(v,int):raise ValueError('Integer required')
 return v
def identity(v):
 if not re.fullmatch(r'\d+',v):raise ValueError('Integer target identity required')
 x=int(v)
 if not 0<x<77016:raise ValueError('Reserved/invalid target identity')
 return x
def goals(g):
 if set(g)!={'nse','r_squared'}:raise ValueError('Separate NSE and squared Pearson R2 goals required')
 for v in g.values():number(v,0,1)
 return g

def support(path,c):
 rows={k:{} for k in ('train','validation')}
 with Path(path).open() as f:
  for r in csv.DictReader(f):
   x=identity(r['target_x']);s=r['split']
   if s not in rows or not (x<=c['train_end_x'] if s=='train' else c['train_end_x']<x<=c['validation_end_x']) or x in rows[s]:raise ValueError('Support scope/identity mismatch before values')
   y,p=float(r['observed']),float(r['persistence'])
   if any(not math.isfinite(v) or v<0 for v in (y,p)):raise ValueError('Invalid support value')
   rows[s][x]=(y,p)
 if min(map(len,rows.values()))<3:raise ValueError('Insufficient support')
 return rows

def scores(rows,q90):
 y=[v[0] for v in rows.values()];p=[v[1] for v in rows.values()];n=len(y);ym=math.fsum(y)/n;pm=math.fsum(p)/n
 yy=math.fsum((v-ym)**2 for v in y);pp=math.fsum((v-pm)**2 for v in p);s=math.fsum(y)
 if yy<=0 or pp<=0 or s<=0:raise ValueError('Undefined NSE/correlation/bias')
 high=[(a,b) for a,b in zip(y,p) if a>q90]
 if not high:raise ValueError('No validation high-flow support')
 return dict(nse=1-math.fsum((a-b)**2 for a,b in zip(y,p))/yy,r_squared=(math.fsum((a-ym)*(b-pm) for a,b in zip(y,p))**2)/(yy*pp),mae=math.fsum(abs(a-b) for a,b in zip(y,p))/n,pbias=100*math.fsum(b-a for a,b in zip(y,p))/s,high_rmse=math.sqrt(math.fsum((a-b)**2 for a,b in high)/len(high)))
def evaluate(path,family,c,s):
 rows={}
 with Path(path).open() as f:
  for r in csv.DictReader(f):
   # Gate identity/scope before reading observation or prediction values.
   x=identity(r['x'])
   if r['split']!='validation' or r['approach']!=family or x not in s['validation'] or x in rows:raise ValueError('Non-validation or unmatched predictions')
   y,p=float(r['observed']),float(r['predicted'])
   if y!=s['validation'][x][0] or not math.isfinite(p) or p<0:raise ValueError('Prediction/truth invalid')
   rows[x]=(y,p)
 if set(rows)!=set(s['validation']):raise ValueError('Incomplete matched support')
 q=sorted(v[0] for v in s['train'].values());pos=.9*(len(q)-1);i=int(pos);q90=q[i]+(q[min(i+1,len(q)-1)]-q[i])*(pos-i)
 return dict(model=scores(rows,q90),persistence=scores(s['validation'],q90),rows=len(rows),training_q90=q90)

def validate_request(r):
 goals(r['goals']);limits=r['limits']
 if set(limits)!={'epochs','wall_seconds_per_fit','rss_bytes','threads','gpu','artifact_bytes_per_fit','max_fits'}:raise ValueError('Explicit finite resource limits required')
 for k,a,b in [('epochs',1,1000),('wall_seconds_per_fit',1,86400),('rss_bytes',1024**2,64*1024**3),('threads',1,16),('artifact_bytes_per_fit',1000,10**10),('max_fits',1,500)]:integer(limits[k],a,b)
 if limits['gpu'] is not False:raise ValueError('CPU-only first controller version')
 jobs=r['jobs']
 if not jobs or len(jobs)>limits['max_fits']:raise ValueError('Finite candidate inventory exceeds budget')
 seen=set();groups={};supports={}
 for j in jobs:
  for k in ('id','candidate','fold'):
   if not isinstance(j[k],str) or not SAFE.fullmatch(j[k]):raise ValueError('Unsafe job/candidate/fold name')
  if j['id'] in seen or j['family'] not in FAMILIES:raise ValueError('Duplicate job or unsupported family')
  seen.add(j['id']);c=load(j['config'])
  if c.get('hydrofusion_development_only') is not True or c.get('hydrofusion_synthetic_verification') is not False:raise ValueError('Real cropped development-only config required')
  a,b,e=[integer(c[k],1,77015) for k in ('train_end_x','validation_end_x','test_end_x')]
  if not a<b<e:raise ValueError('Chronological cutoff ordering')
  if c['epochs']!=limits['epochs'] or c.get('random_seed')!=j['seed']:raise ValueError('Epoch/seed inventory mismatch')
  integer(j['seed'],0,2**31-1)
  manifest=load(c['hydrofusion_native_manifest_path'])
  if manifest.get('execution_scope')!='train_validation_only' or manifest.get('target_cutoff_x')!=b or manifest.get('end_interval_x_exclusive')!=b:raise ValueError('Physically cropped development package required')
  for key in ('hydrofusion_native_manifest','hydrofusion_preissue_rain'):
   if sha(c[key+'_path'])!=c[key+'_sha256']:raise ValueError('Input hash mismatch')
  interval=Path(c['hydrofusion_native_manifest_path']).parent/manifest['interval_csv']
  if Path(manifest['interval_csv']).is_absolute() or '..' in Path(manifest['interval_csv']).parts:raise ValueError('Unsafe interval path')
  if sha(interval)!=manifest['interval_csv_sha256']:raise ValueError('Native interval hash mismatch')
  s=support(j['support'],c);signature=(a,b,sha(j['support']))
  if j['fold'] in supports and supports[j['fold']]!=signature:raise ValueError('Unmatched candidate support within fold')
  supports[j['fold']]=signature;key=(j['family'],j['candidate']);pair=(j['fold'],j['seed'])
  if pair in groups.setdefault(key,set()):raise ValueError('Duplicate candidate fold/seed')
  groups[key].add(pair)
 if len({tuple(sorted(v)) for v in groups.values()})!=1:raise ValueError('All candidates require identical declared folds/seeds')
 rule=r['rule']
 if set(rule)!={'min_nse_gain','max_mae_ratio','max_high_rmse_ratio','max_abs_bias_increase','required_fraction'}:raise ValueError('Explicit consistency/tail/bias rule required')
 number(rule['min_nse_gain'],0,1);number(rule['max_mae_ratio'],1,2);number(rule['max_high_rmse_ratio'],1,2);number(rule['max_abs_bias_increase'],0,100);number(rule['required_fraction'],0.01,1)
 return groups

def prepare(request,root,override=None):
 r=load(request)
 if override is not None:r['goals']=goals(override)
 validate_request(r)
 root=Path(root).absolute();root.mkdir(exist_ok=False);(root/'jobs').mkdir();(root/'configs').mkdir();pins={}
 for key in ('binary','verifier'):
  src=Path(r[key]).resolve(strict=True);dst=root/key
  import shutil
  shutil.copy2(src,dst);r[key]=str(dst);pins[str(dst)]=sha(dst)
 dispatch=Path(__file__).resolve().parents[2]/'build-hydrobatch'/('goal_'+hashlib.sha256(str(root).encode()).hexdigest()[:16]);dispatch.mkdir(exist_ok=False)
 for j in r['jobs']:
  src=Path(j['config']).resolve();c=load(src);c['experiment_id']=j['id'];cfg=dispatch/(j['id']+'.json');write(cfg,c,True);batch=dispatch/(j['id']+'.batch');batch.write_text(j['family']+' '+cfg.name+'\n');j['config']=str(cfg);j['batch']=str(batch);j['support']=str(Path(j['support']).resolve());write(root/'configs'/cfg.name,c,True)
  m=load(c['hydrofusion_native_manifest_path']);paths=[src,cfg,batch,root/'configs'/cfg.name,Path(j['support']),Path(c['hydrofusion_native_manifest_path']),Path(c['hydrofusion_native_manifest_path']).parent/m['interval_csv'],Path(c['hydrofusion_preissue_rain_path']),Path(c['hydrofusion_preissue_rain_path']).parent/'preissue_rain_provenance.json']
  for p in paths:pins[str(p.resolve())]=sha(p)
 # Snapshot all imported controller dependencies to detect code drift before resume.
 for p in (Path(__file__).resolve(),EXPERIMENTS/'run_hydrofusion_six_fit_screen.py'):pins[str(p)]=sha(p)
 r.update(schema='HydroPINNGoalBatch-v1',pins=pins,no_retries=True,no_expansion=True,scope='reused_development_only_not_blind',goal_revision=0)
 write(root/'plan.json',r,True);write(root/'state.json',dict(plan_sha256=sha(root/'plan.json'),status='READY',charged_fits=0),True);print(json.dumps(dict(status='READY',root=str(root),jobs=len(r['jobs']))))

def verify_plan(root):
 p=load(root/'plan.json');state=load(root/'state.json')
 if sha(root/'plan.json')!=state['plan_sha256']:raise ValueError('Plan changed')
 for path,h in p['pins'].items():
  if sha(path)!=h:raise ValueError('Pinned input/binary/source changed: '+path)
 revisions=root/'goal_revisions'
 if revisions.exists():
  previous=state['plan_sha256']
  for i,q in enumerate(sorted(revisions.glob('*.json')),1):
   revision=load(q)
   if q.name!=f'{i:06d}.json' or revision['previous_sha256']!=previous:raise ValueError('Goal revision chain mismatch')
   p['goals']=goals(revision['goals']);previous=sha(q)
 validate_request(p)
 return p,state

def report(root,p):
 grouped={};done=[]
 for j in p['jobs']:
  d=root/'jobs'/j['id'];receipt=d/'verified.json'
  if not receipt.exists():continue
  rec=load(receipt)
  for path,h in rec['hashes'].items():
   if sha(d/path)!=h:raise ValueError('Verified result changed')
  c=load(j['config']);actual=evaluate(d/'validation_predictions.csv',j['family'],c,support(j['support'],c))
  if actual!=rec['scores']:raise ValueError('Result score mismatch')
  grouped.setdefault((j['family'],j['candidate']),[]).append(dict(actual,fold=j['fold'],seed=j['seed'],job=j['id']));done.append(j['id'])
 expected=validate_request(p);result=[]
 for key,rows in grouped.items():
  if len(rows)!=len(expected[key]):continue
  rule=p['rule'];g=p['goals'];passing=[]
  for z in rows:
   m,b=z['model'],z['persistence'];passing.append(m['nse']>=g['nse'] and m['r_squared']>=g['r_squared'] and m['nse']>=b['nse']+rule['min_nse_gain'] and m['mae']<=b['mae']*rule['max_mae_ratio'] and m['high_rmse']<=b['high_rmse']*rule['max_high_rmse_ratio'] and abs(m['pbias'])<=abs(b['pbias'])+rule['max_abs_bias_increase'])
  means={k:math.fsum(z['model'][k] for z in rows)/len(rows) for k in rows[0]['model']};qualified=sum(passing)/len(passing)>=rule['required_fraction'] and means['nse']>=g['nse'] and means['r_squared']>=g['r_squared']
  result.append(dict(family=key[0],candidate=key[1],mean=means,per_fold_seed=rows,qualified=qualified,passing=sum(passing),planned=len(rows)))
 out=dict(completed=len(done),planned=len(p['jobs']),candidates=result,goal_met=all(any(z['qualified'] and z['family']==family for z in result) for family in {j['family'] for j in p['jobs']}),goals=p['goals'],scope=p['scope'],test_rows=0)
 write(root/'report.json',out);return out

def execute(command,output,env,limits,cancel):
 parent=os.getpid()
 def hard_limits():
  if ctypes.CDLL(None).prctl(1,signal.SIGKILL)!=0:raise RuntimeError('Cannot set parent-death guard')
  if os.getppid()!=parent:os.kill(os.getpid(),signal.SIGKILL)
  resource.setrlimit(resource.RLIMIT_AS,(limits['rss_bytes'],limits['rss_bytes']));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
  allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,set(allowed[:limits['threads']]))
  cap=limits['wall_seconds_per_fit']*limits['threads'];resource.setrlimit(resource.RLIMIT_CPU,(cap,cap))
 with (output/'training.log').open('x') as log:
  child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True,preexec_fn=hard_limits);start=time.monotonic()
  try:
   while child.poll() is None:
    if cancel.exists():raise RuntimeError('CANCELLED_DURING_FIT: claimed fit cannot be retried')
    if time.monotonic()-start>limits['wall_seconds_per_fit'] or directory_bytes(output)>limits['artifact_bytes_per_fit']:raise RuntimeError('RESOURCE_BUDGET_EXHAUSTED')
    time.sleep(.2)
   if child.returncode:raise RuntimeError('ENGINE_FAILED: no automatic retry')
  except BaseException:
   if child.poll() is None:
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=5)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
   raise


def run(root):
 root=Path(root).resolve(strict=True)
 with (root/'controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);p,state=verify_plan(root)
  if state['status'] in ('FAILED','INTERRUPTED','CANCELLED_DURING_FIT','RESOURCE_BUDGET_EXHAUSTED'):raise ValueError('Failed/interrupted inventory cannot be retried')
  if state['status'] in ('GOAL_MET','INVENTORY_EXHAUSTED'):return report(root,p)
  # Claims survive interruption; never repeat uncertain work.
  for j in p['jobs']:
   d=root/'jobs'/j['id']
   if (d/'claim.json').exists() and not (d/'verified.json').exists():
    state['status']='INTERRUPTED';write(root/'state.json',state);raise ValueError('Unverified prior claim; no automatic retry')
  state['charged_fits']=sum((root/'jobs'/j['id']/'claim.json').exists() for j in p['jobs'])
  for j in p['jobs']:
   verify_plan(root)
   if (root/'CANCEL').exists():state['status']='CANCELLED';write(root/'state.json',state);return report(root,p)
   d=root/'jobs'/j['id']
   if (d/'verified.json').exists():continue
   if state['charged_fits']>=p['limits']['max_fits']:state['status']='FIT_BUDGET_EXHAUSTED';write(root/'state.json',state);return report(root,p)
   d.mkdir(exist_ok=False);write(d/'claim.json',dict(plan_sha256=state['plan_sha256'],started=time.time()),True);state['charged_fits']+=1;state['status']='RUNNING';write(root/'state.json',state);print('START '+j['id'],flush=True)
   env=dict(os.environ,OMP_NUM_THREADS=str(p['limits']['threads']),MKL_NUM_THREADS=str(p['limits']['threads']),OPENBLAS_NUM_THREADS=str(p['limits']['threads']),CUDA_VISIBLE_DEVICES='',MALLOC_ARENA_MAX='2')
   try:
    execute([p['binary'],j['batch'],str(d/'bundles')],d,env,p['limits'],root/'CANCEL');bundle=d/'bundles'/j['id']
    with (d/'reload.log').open('x') as log:subprocess.run([p['verifier'],str(bundle),str(d/'validation_predictions.csv')],stdout=log,stderr=subprocess.STDOUT,env=env,check=True,timeout=3600)
    if 'production_target_free_checks=12' not in (d/'reload.log').read_text():raise ValueError('Missing semantic target-free verification')
    c=load(j['config']);value=evaluate(d/'validation_predictions.csv',j['family'],c,support(j['support'],c));exported=evaluate(bundle/'predictions.csv',j['family'],c,support(j['support'],c))
    if value!=exported:raise ValueError('Fresh reload/exported metrics differ')
    if directory_bytes(d)>p['limits']['artifact_bytes_per_fit']:raise ValueError('Artifact budget exceeded')
    hashes={str(q.relative_to(d)):sha(q) for q in d.rglob('*') if q.is_file()};write(d/'verified.json',dict(scores=value,hashes=hashes),True)
   except BaseException as e:
    state['status']='CANCELLED_DURING_FIT' if 'CANCELLED_DURING_FIT' in str(e) else 'RESOURCE_BUDGET_EXHAUSTED' if 'RESOURCE_BUDGET_EXHAUSTED' in str(e) else 'FAILED';state['error']=str(e);write(root/'state.json',state);raise
   out=report(root,p);print(json.dumps(out),flush=True)
   if out['goal_met']:state['status']='GOAL_MET';write(root/'state.json',state);return out
  state['status']='INVENTORY_EXHAUSTED';write(root/'state.json',state);return report(root,p)

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=['prepare','run','status','cancel','revise-goals']);ap.add_argument('root',type=Path);ap.add_argument('--request',type=Path);ap.add_argument('--nse',type=float);ap.add_argument('--r-squared',type=float);a=ap.parse_args()
 if a.command=='prepare':
  if not a.request:ap.error('--request required')
  prepare(a.request,a.root,None if a.nse is None and a.r_squared is None else dict(nse=a.nse,r_squared=a.r_squared))
 elif a.command=='run':print(json.dumps(run(a.root),indent=2))
 elif a.command=='revise-goals':
  with (a.root/'controller.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);p,state=verify_plan(a.root);g=goals(dict(nse=a.nse,r_squared=a.r_squared));d=a.root/'goal_revisions';d.mkdir(exist_ok=True);old=sorted(d.glob('*.json'));previous=sha(old[-1]) if old else state['plan_sha256'];write(d/f'{len(old)+1:06d}.json',dict(goals=g,previous_sha256=previous),True)
   if state['status']=='GOAL_MET':state['status']='READY'
   write(a.root/'state.json',state);p,_=verify_plan(a.root);print(json.dumps(report(a.root,p)))
 elif a.command=='cancel':write(a.root/'CANCEL',dict(requested=time.time()),True);print('Cancellation requested; current fit will be terminated and cannot be retried.')
 else:
  p,s=verify_plan(a.root);print(json.dumps(dict(state=s,report=load(a.root/'report.json') if (a.root/'report.json').exists() else None),indent=2))
if __name__=='__main__':
 try:main()
 except Exception as e:print('ERROR: '+str(e),file=sys.stderr);sys.exit(1)
