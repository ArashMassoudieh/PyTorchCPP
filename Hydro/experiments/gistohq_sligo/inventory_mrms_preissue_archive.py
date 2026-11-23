#!/usr/bin/env python3
"""Bounded weather-only daily-listing inventory; presence is NOT input acceptance."""
import argparse,csv,datetime as dt,gzip,hashlib,json,re,time,urllib.request,urllib.error
from pathlib import Path
EPOCH=dt.datetime(2015,1,1,tzinfo=dt.timezone.utc)
PATTERN=re.compile(r'href="(PrecipRate_00\.00_(\d{8})-(\d{6})\.grib2\.gz)"')

def hourly_candidates(html,day):
 names={}
 for name,date,stamp in PATTERN.findall(html):
  if date!=day.strftime('%Y%m%d'):raise ValueError('Directory contains wrong-date data')
  try:valid=dt.datetime.strptime(date+stamp,'%Y%m%d%H%M%S').replace(tzinfo=dt.timezone.utc)
  except ValueError:raise ValueError('Invalid archive timestamp')
  if valid.minute!=50 or valid.second:continue
  origin=valid+dt.timedelta(minutes=10);target=origin+dt.timedelta(hours=1);x=(target-EPOCH).total_seconds()/3600
  if x.is_integer() and 1<=x<=26303:names[name]=dict(name=name,valid_utc=valid.isoformat(),origin_utc=origin.isoformat(),target_x=int(x))
 return list(sorted(names.values(),key=lambda v:v['target_x']))

def run(root):
 root=Path(root);plan=json.loads((root/'plan.json').read_text());claim=root/'launch_claim.json'
 with claim.open('x') as f:json.dump(dict(plan_sha256=hashlib.sha256((root/'plan.json').read_bytes()).hexdigest(),no_retries=True),f)
 (root/'listings').mkdir(exist_ok=False);records=[];bytes_used=0;start=time.monotonic();days=[dt.date.fromisoformat(d) for d in plan['dates']]
 try:
  for day in days:
   if time.monotonic()-start>plan['wall_seconds']:raise RuntimeError('Wall limit')
   url=f'https://mtarchive.geol.iastate.edu/{day:%Y/%m/%d}/mrms/ncep/PrecipRate/';record=dict(date=day.isoformat(),url=url)
   try:
    cached=root/'cached'/f'{day:%Y%m%d}.html'
    if cached.exists():data=cached.read_bytes();record['reused_saved_listing']=True
    else:
     with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'HydroPINN bounded weather archive inventory'}),timeout=20) as response:data=response.read(plan['per_response_bytes']+1)
    bytes_used+=len(data)
    if len(data)>plan['per_response_bytes'] or bytes_used>plan['total_bytes']:raise RuntimeError('Byte limit')
    (root/'listings'/f'{day:%Y%m%d}.html.gz').write_bytes(gzip.compress(data,mtime=0))
    candidates=hourly_candidates(data.decode('utf-8'),day);record.update(status='LISTING_PARSED',listing_sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),candidates=candidates)
   except urllib.error.HTTPError as e:record.update(status='HTTP_ERROR',code=e.code)
   except (TimeoutError,urllib.error.URLError) as e:record.update(status='REQUEST_FAILED',error=str(e))
   records.append(record)
   with (root/'progress.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
   if len(records)%30==0:print('days',len(records),'of',len(days),'bytes',bytes_used,flush=True)
  candidates=[dict(url=r['url']+c['name'],**c) for r in records for c in r.get('candidates',[])]
  assert len({c['target_x'] for c in candidates})==len(candidates)
  result=dict(status='INVENTORY_COMPLETE_NOT_INPUT_ACCEPTANCE',days=len(records),candidates=len(candidates),missing_or_failed_days=[r['date'] for r in records if r['status']!='LISTING_PARSED'],bytes=bytes_used,observations=0,new_fits=0,spatial_coverage_verified=False,historical_release_verified=False,limitations=['Directory presence only; every acquired grid still needs geometry/value/native-support checks.','HTTP modification dates are archive proxies, not historical end-user delivery proof.'])
  with (root/'candidate_hours.json').open('x') as f:json.dump(candidates,f,indent=2)
  with (root/'completion.json').open('x') as f:json.dump(result,f,indent=2)
  print(json.dumps(result),flush=True)
 except BaseException as e:
  with (root/'failure.json').open('x') as f:json.dump(dict(completed_days=len(records),error=str(e),no_retry=True),f)
  raise
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('root');run(p.parse_args().root)
