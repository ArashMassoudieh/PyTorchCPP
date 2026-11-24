#!/usr/bin/env python3
"""Independent weather record/raw-hash audit. Never reads observed Q values."""
import argparse,csv,hashlib,json,math,collections
from pathlib import Path
from extract_mrms_preissue_verified import identity,validated_prefix
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check_record(row):
 valid,origin=identity(row)
 if row['valid_seconds']!=int(valid.timestamp()) or row['issue_seconds']!=int(origin.timestamp()):raise ValueError('Weather identity mismatch')
 if type(row['usable']) is not bool or row['historical_delivery_proven'] is not False:raise ValueError('Invalid availability/provenance flag')
 if row['usable']:
  if row['status']!='WEATHER_VALUE_VERIFIED' or row['unit']!='mm/h':raise ValueError('Invalid weather status/units')
  rate,coverage=row['rate_mm_h'],row['coverage']
  if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in [rate,coverage]) or rate<0 or not .75<=coverage<=1.00000000001:raise ValueError('Invalid weather values/coverage')
  release=row['release_proxy_seconds']
  if type(release) is not int or not row['valid_seconds']<=release<=row['issue_seconds']:raise ValueError('Late or invalid release proxy')
  if row['release_basis']!='archive_http_last_modified':raise ValueError('Unknown release basis')
 elif row['rate_mm_h'] is not None:raise ValueError('Unavailable weather must not contain a rate')
def audit(root,prefix=False):
 root=Path(root).resolve(strict=True);plan=json.loads((root/'plan.json').read_text());claim=json.loads((root/'launch_claim.json').read_text());assert claim['plan_sha256']==sha(root/'plan.json')
 if prefix:
  failure=json.loads((root/'failure.json').read_text());assert failure['error']=='Extraction wall ceiling';n=failure['completed_hours']
 else:
  done=json.loads((root/'completion.json').read_text());assert done['plan_sha256']==claim['plan_sha256'] and not (root/'failure.json').exists();n=26303
 out=root/'independent_record_audit';out.mkdir(exist_ok=False);paths=sorted((root/'records').glob('*.json'));assert [int(f.stem) for f in paths]==list(range(1,n+1))
 sources={};counts=collections.Counter();available=set();rates=[];monthly={};raw_count=0;raw_bytes=0
 for f in paths:
  row=json.loads(f.read_text());check_record(row);sources[str(f)]=sha(f);leaf=row;leaf_file=f;seen=set()
  while 'prior_record_source' in leaf:
   target=Path(leaf['prior_record_source']);assert str(target) not in seen;seen.add(str(target));assert len(seen)<=4 and sha(target)==leaf['prior_record_sha256'];leaf=json.loads(target.read_text());leaf_file=target
  for k in ['target_x','rate_mm_h','usable','status','valid_seconds','issue_seconds']:assert row[k]==leaf[k]
  base=leaf_file.parents[1]
  if 'compressed_sha256' in leaf:
   raw=base/'raw'/Path(leaf['url']).name;h=hashlib.sha256()
   with raw.open('rb') as stream:
    first=stream.read(32768);h.update(first)
    while chunk:=stream.read(1048576):h.update(chunk)
   assert h.hexdigest()==leaf['compressed_sha256'];sources[str(raw)]=h.hexdigest();raw_count+=1;raw_bytes+=raw.stat().st_size
   if row['usable']:
    geometry=validated_prefix(first)
    for k,v in geometry.items():assert abs(v-leaf['grid'][k])<1e-9
  if 'compressed_prefix_sha256' in leaf:
   raw=base/'prefixes'/(Path(leaf['url']).name+'.prefix');assert sha(raw)==leaf['compressed_prefix_sha256'];sources[str(raw)]=leaf['compressed_prefix_sha256']
  counts[row.get('reason',row['status'])]+=1;month=row['valid_utc'][:7];m=monthly.setdefault(month,dict(hours=0,usable=0));m['hours']+=1;m['usable']+=int(row['usable'])
  if row['usable']:available.add(row['target_x'])
  rates.append(row)
  if len(rates)%2000==0:print('AUDITED',len(rates),'raw_bytes',raw_bytes,flush=True)
 if not prefix:
  assert len(available)==done['usable_hours']
  with (out/'weather_rate_sidecar.csv').open('x') as stream:
   writer=csv.writer(stream);writer.writerow(['target_x','origin_x','valid_seconds','issue_seconds','release_proxy_seconds','coverage','rate_mm_h','usable','status'])
   for row in rates:writer.writerow([row['target_x'],row['target_x']-1,row['valid_seconds'],row['issue_seconds'],row.get('release_proxy_seconds',''),row.get('coverage',''),row['rate_mm_h'] if row['usable'] else '',int(row['usable']),row['status']])
 # Read timestamp identities only from old frozen support, never Q columns.
 support=Path('/home/arash/Dropbox/HydroPINN paper/results/hydrofusion_context_history_pilot_20261122/support.csv');coverage={s:dict(total=0,latest_rate=0,all24_rates=0) for s in ['train','validation']}
 with support.open() as stream:
  for row in csv.DictReader(stream):
   x=int(row['target_x']);s=row['split'];assert 1<=x<=26303 and s in coverage
   coverage[s]['total']+=1;coverage[s]['latest_rate']+=int(x in available);coverage[s]['all24_rates']+=int(all(x-k in available for k in range(24)))
 record=dict(status='PREFIX_AUDITED_NOT_COMPLETE' if prefix else 'WEATHER_RECORDS_AND_RAW_HASHES_VERIFIED_NOT_NATIVE_INPUT_ACCEPTANCE',hours=n,usable=len(available),raw_files_hashed=raw_count,raw_bytes_hashed=raw_bytes,source_plan_sha256=claim['plan_sha256'],rejections=dict(counts),monthly=monthly,identity_only_matched_support=coverage,new_fits=0,observed_Q_values_read=0,historical_delivery_proven=False,limitation='All record identities/values/coverage/release and saved raw hashes checked; actual GRIB geometry independently read for every usable raw. Rate values came from tested producer decoding; this audit does not independently re-extract every basin mean. Not a trained-native-feature or operational-delivery certification.')
 (out/'source_hashes.json').write_text(json.dumps(sources,indent=2)+'\n');(out/'review.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('root');p.add_argument('--prefix',action='store_true');a=p.parse_args();audit(a.root,a.prefix)
