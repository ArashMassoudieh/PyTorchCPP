#!/usr/bin/env python3
"""Combine independently audited prefix and new tail without rehashing old raws.
Checks all new record hashes; prior raw hashes are inherited from the immediately
preceding audit of the stopped, immutable prefix. No Q values or model fitting.
"""
import argparse,csv,hashlib,json,collections
from pathlib import Path
from verify_mrms_weather_records import check_record,sha
from extract_mrms_preissue_verified import validated_prefix

def finalize(root,prefix):
 root,prefix=Path(root).resolve(strict=True),Path(prefix).resolve(strict=True);done=json.loads((root/'completion.json').read_text());plan=json.loads((root/'plan.json').read_text());claim=json.loads((root/'launch_claim.json').read_text())
 assert done['hours']==26303 and done['plan_sha256']==claim['plan_sha256']==sha(root/'plan.json') and not (root/'failure.json').exists()
 audit=prefix/'independent_record_audit';previous=json.loads((audit/'review.json').read_text());hashes=json.loads((audit/'source_hashes.json').read_text());assert previous['status']=='PREFIX_AUDITED_NOT_COMPLETE' and previous['hours']==25950
 out=root/'independent_final_audit';out.mkdir(exist_ok=False);rows=[];sources={str(audit/'review.json'):sha(audit/'review.json'),str(audit/'source_hashes.json'):sha(audit/'source_hashes.json')};available=set();counts=collections.Counter();new_raw_bytes=0
 for x in range(1,26304):
  f=root/'records'/f'{x:05d}.json';row=json.loads(f.read_text());check_record(row);assert row['target_x']==x;sources[str(f)]=sha(f)
  if x<=25950:
   original=prefix/'records'/f.name;assert row['prior_record_source']==str(original) and row['prior_record_sha256']==hashes[str(original)]==sha(original);old=json.loads(original.read_text())
   for k,v in old.items():
    if k not in ['prior_record_source','prior_record_sha256']:assert row[k]==v
  else:
   assert 'prior_record_source' not in row
   if 'compressed_sha256' in row:
    raw=root/'raw'/Path(row['url']).name;assert sha(raw)==row['compressed_sha256'];sources[str(raw)]=row['compressed_sha256'];new_raw_bytes+=raw.stat().st_size
    if row['usable']:
     with raw.open('rb') as stream:geometry=validated_prefix(stream.read(32768))
     for k,v in geometry.items():assert abs(v-row['grid'][k])<1e-9
   if 'compressed_prefix_sha256' in row:
    p=root/'prefixes'/(Path(row['url']).name+'.prefix');assert sha(p)==row['compressed_prefix_sha256'];sources[str(p)]=row['compressed_prefix_sha256']
  rows.append(row);counts[row.get('reason',row['status'])]+=1
  if row['usable']:available.add(x)
 assert len(available)==done['usable_hours']==24861
 with (out/'weather_rate_sidecar.csv').open('x') as stream:
  writer=csv.writer(stream);writer.writerow(['target_x','origin_x','valid_seconds','issue_seconds','release_proxy_seconds','coverage','rate_mm_h','usable','status'])
  for row in rows:writer.writerow([row['target_x'],row['target_x']-1,row['valid_seconds'],row['issue_seconds'],row.get('release_proxy_seconds',''),row.get('coverage',''),row['rate_mm_h'] if row['usable'] else '',int(row['usable']),row['status']])
 support=Path('/home/arash/Dropbox/HydroPINN paper/results/hydrofusion_context_history_pilot_20261122/support.csv');matched={s:dict(total=0,latest_rate=0,all24_rates=0) for s in ['train','validation']}
 with support.open() as stream:
  for row in csv.DictReader(stream):
   x=int(row['target_x']);s=row['split'];assert 1<=x<=26303 and s in matched;matched[s]['total']+=1;matched[s]['latest_rate']+=int(x in available);matched[s]['all24_rates']+=int(all(x-k in available for k in range(24)))
 sources[str(out/'weather_rate_sidecar.csv')]=sha(out/'weather_rate_sidecar.csv')
 report=dict(status='WEATHER_EXTRACTION_AND_SOURCE_AUDIT_COMPLETE_NATIVE_INTEGRATION_PENDING',hours=26303,usable=24861,unusable=1442,prior_audited_hours=25950,new_hours=353,new_raw_bytes_hashed=new_raw_bytes,all_saved_raw_hashes_verified_in_two_disjoint_audits=True,plan_sha256=claim['plan_sha256'],identity_only_matched_support=matched,rejections=dict(counts),sidecar_sha256=sources[str(out/'weather_rate_sidecar.csv')],observed_Q_values_read=0,new_fits=0,historical_delivery_proven=False,limitations=['Prior stopped-prefix raw checksums and geometry are inherited from its immediately preceding independent audit, not recomputed twice.','All records validated; not every basin mean independently re-extracted.','Archive modification timestamps are proxies, not historical end-user delivery certification.','Not yet integrated or reload-verified as a trained native feature; no performance claim.'])
 (out/'source_hashes.json').write_text(json.dumps(sources,indent=2)+'\n');(out/'review.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('root');p.add_argument('prefix');a=p.parse_args();finalize(a.root,a.prefix)
