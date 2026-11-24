#!/usr/bin/env python3
"""Continue ONLY an unvisited weather tail, preserving a stopped extraction.
Reuses the frozen verified extractor and all previous per-hour records. No fits.
"""
import argparse,hashlib,json,shutil,subprocess,sys
from pathlib import Path
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def prepare(root,previous):
 root,previous=map(lambda p:Path(p).absolute(),(root,previous));failure=json.loads((previous/'failure.json').read_text());old=json.loads((previous/'plan.json').read_text());progress=json.loads((previous/'progress.json').read_text())
 if failure['error']!='Extraction wall ceiling':raise ValueError('Only a wall-limited unvisited tail is supported')
 n=failure['completed_hours'];files=sorted((previous/'records').glob('*.json'))
 if not 0<len(old['jobs'])-n<=500 or [int(f.stem) for f in files]!=list(range(1,n+1)):raise ValueError('Previous record prefix/tail bound mismatch')
 if progress['completed_hours']>n:raise ValueError('Invalid previous progress identity')
 root.mkdir(parents=True,exist_ok=False);(root/'provenance').mkdir()
 for name in ['extract_mrms_preissue_verified.py','mrms_preissue_geometry.py']:shutil.copy2(previous/'provenance'/name,root/'provenance'/name)
 shutil.copy2(Path(__file__),root/'provenance'/Path(__file__).name)
 p=dict(old);p['prior_records']={str(int(f.stem)):str(f) for f in files};p['pins']=dict(old['pins'])
 for f in files:p['pins'][str(f)]=sha(f)
 for f in (root/'provenance').iterdir():p['pins'][str(f)]=sha(f)
 for f in [previous/'failure.json',previous/'plan.json',previous/'progress.json']:p['pins'][str(f)]=sha(f)
 # Conservative bound charges every hour since the last checkpoint as a full
 # max-size download plus prefix, rather than undercounting prior transfer.
 p['prior_bytes']=progress['compressed_bytes']+(n-progress['completed_hours'])*(old['limits']['per_response_bytes']+32768)
 if p['prior_bytes']>=p['limits']['compressed_bytes']:raise ValueError('No remaining byte allowance')
 p['continuation']=dict(previous=str(previous),completed_hours=n,unvisited_hours=len(old['jobs'])-n,parent_wall_seconds=3600,previous_bytes_basis='conservative last-checkpoint plus maximum full-file/prefix per completed hour')
 (root/'plan.json').write_text(json.dumps(p,indent=2)+'\n');print(json.dumps(p['continuation']),flush=True)
def launch(root):
 root=Path(root).resolve(strict=True);p=json.loads((root/'plan.json').read_text());source=root/'provenance/extract_mrms_preissue_verified.py'
 if sha(source)!=p['pins'][str(source)]:raise ValueError('Frozen extractor changed')
 with (root/'continuation_claim.json').open('x') as f:json.dump(dict(plan_sha256=sha(root/'plan.json'),parent_wall_seconds=3600),f)
 try:
  result=subprocess.run([sys.executable,'-u',str(source),str(root),'--launch'],timeout=3600,check=True)
 except BaseException as e:
  with (root/'continuation_failure.json').open('x') as f:json.dump(dict(error=str(e),no_retry=True),f)
  raise
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('root');g=p.add_mutually_exclusive_group(required=True);g.add_argument('--prepare');g.add_argument('--launch',action='store_true');a=p.parse_args()
 if a.prepare:prepare(a.root,a.prepare)
 else:launch(a.root)
