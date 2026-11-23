#!/usr/bin/env python3
"""Immutable weather-only basin extraction. No Q or model training dependencies."""
import zlib
import argparse,datetime as dt,email.utils,gzip,hashlib,io,json,math,os,shutil,sys,time,urllib.request
from pathlib import Path
from mrms_preissue_geometry import regular_centers,parse_grid_prefix
EPOCH=dt.datetime(2015,1,1,tzinfo=dt.timezone.utc)
GEOMETRY_KEYS=['Ni','Nj','latitudeOfFirstGridPointInDegrees','longitudeOfFirstGridPointInDegrees','latitudeOfLastGridPointInDegrees','longitudeOfLastGridPointInDegrees','iDirectionIncrementInDegrees','jDirectionIncrementInDegrees','scanningMode']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def validated_prefix(data):
 if len(data)!=32768:raise ValueError('Expected exactly32KB compressed prefix')
 prefix=zlib.decompressobj(16+zlib.MAX_WBITS).decompress(data,4096)
 geometry=parse_grid_prefix(prefix);regular_centers(geometry);return geometry

def identity(job):
 x=job['target_x']
 if type(x) is not int or not 1<=x<=26303:raise ValueError('Target outside development scope')
 origin=EPOCH+dt.timedelta(hours=x-1);valid=origin-dt.timedelta(minutes=10)
 expected=f'PrecipRate_00.00_{valid:%Y%m%d-%H%M%S}.grib2.gz'
 if Path(job['url']).name!=expected:raise ValueError('URL/valid/origin/target identity mismatch')
 return valid,origin

def timestamp_metadata(headers,valid,origin):
 text=next((v for k,v in headers.items() if k.lower()=='last-modified'),None)
 if text is None:raise ValueError('Missing release proxy')
 release=email.utils.parsedate_to_datetime(text)
 if release.tzinfo is None or not valid<=release<=origin:raise ValueError('Archive release proxy outside causal interval')
 return release

def verify(plan):
 if plan['schema'] not in ['WeatherOnlyPreissueBasinExtractionPreflight-v2','WeatherOnlyPreissueBasinExtraction-v2']:raise ValueError('Unknown extractor schema')
 preflight=plan['schema']=='WeatherOnlyPreissueBasinExtractionPreflight-v2'
 if plan['limits']!=dict(compressed_bytes=32000000 if preflight else 30000000000,per_response_bytes=5000000,decompressed_bytes=200000000,wall_seconds=600 if preflight else 43200,workers=1):raise ValueError('Extraction bounds changed')
 for p,h in plan['pins'].items():
  if sha(p)!=h:raise ValueError('Pinned source changed: '+p)
 targets=set()
 for j in plan['jobs']:
  identity(j)
  if j['target_x'] in targets:raise ValueError('Duplicate hour')
  targets.add(j['target_x'])
 if preflight:
  cache=json.loads(Path(plan['cache']).read_text())
  if not 1<=len(targets)<=16 or any(not j['available'] or Path(j['url']).name not in cache for j in plan['jobs']):raise ValueError('Preflight must use cached weather only')
 elif targets!=set(range(1,26304)):raise ValueError('Incomplete declared inventory')

def prepare(root,inventory,gaps,preflight=False,prior=None):
 root,inventory,gaps=map(lambda p:Path(p).absolute(),(root,inventory,gaps));root.mkdir(parents=True,exist_ok=False);(root/'provenance').mkdir();(root/'cached').mkdir()
 merged={j['target_x']:j for j in json.loads((inventory/'candidate_hours.json').read_text()) if 1<=j['target_x']<=26303}
 for j in json.loads((gaps/'receipt.json').read_text())['samples']:
  if j['status']==200:merged[j['target_x']]=j
 jobs=[]
 for x in range(1,26304):
  origin=EPOCH+dt.timedelta(hours=x-1);valid=origin-dt.timedelta(minutes=10);name=f'PrecipRate_00.00_{valid:%Y%m%d-%H%M%S}.grib2.gz';j=dict(target_x=x,url=f'https://mtarchive.geol.iastate.edu/{valid:%Y/%m/%d}/mrms/ncep/PrecipRate/{name}',available=x in merged)
  identity(j);jobs.append(j)
 here=Path(__file__).resolve().parent
 sources=[Path(__file__),here/'mrms_preissue_geometry.py',Path('/mnt/3rd900/Projects/GIStoOHQ/ohqbuilder/watershed_data/mrms.py'),Path('/home/arash/Dropbox/HydroPINN paper/data/hydrofusion_gauge_forcing_preflight_2015_2018/gistohq_forcing/basin.geojson'),inventory/'candidate_hours.json',inventory/'independent_review.json',gaps/'receipt.json']
 for p in sources:shutil.copy2(p,root/'provenance'/p.name)
 # Frozen weather-only samples may be reused by exact filename and hash. No new
 # request is needed for them; their saved HTTP headers remain the provenance.
 cache={};base=inventory.parent
 samples=[(base/'asof_radar_metadata/preciprate/raw_sample_receipt.json','samples','raw_samples'),(base/'asof_radar_calendar/geometry_checked_quarterly/receipt.json','samples','raw')]
 names={Path(j['url']).name for j in jobs}
 for receipt,key,folder in samples:
  for row in json.loads(receipt.read_text())[key]:
   name=Path(row['url']).name
   if name not in names:continue
   p=receipt.parent/folder/name
   if sha(p)!=row['compressed_sha256']:raise ValueError('Cached raw hash mismatch')
   shutil.copy2(p,root/'cached'/name);cache[name]=dict(path=str(root/'cached'/name),sha256=sha(p),headers=row['headers'])
 (root/'cache.json').write_text(json.dumps(cache,indent=2)+'\n')
 pins={str(p):sha(p) for p in sources};pins.update({str(p):sha(p) for p in (root/'provenance').iterdir()});pins[str(root/'cache.json')]=sha(root/'cache.json')
 for j in cache.values():pins[j['path']]=j['sha256']
 if preflight:jobs=[j for j in jobs if Path(j['url']).name in cache]
 plan=dict(schema='WeatherOnlyPreissueBasinExtractionPreflight-v2' if preflight else 'WeatherOnlyPreissueBasinExtraction-v2',jobs=jobs,pins=pins,limits=dict(compressed_bytes=32000000 if preflight else 30000000000,per_response_bytes=5000000,decompressed_bytes=200000000,wall_seconds=600 if preflight else 43200,workers=1),basin=str(root/'provenance/basin.geojson'),cache=str(root/'cache.json'),no_retries=True,observations=0,new_fits=0,release_basis='archive_http_last_modified',historical_delivery_proven=False)
 plan['prior_records']={};plan['prior_bytes']=0
 if prior:
  prior=Path(prior).resolve(strict=True);failure=json.loads((prior/'failure.json').read_text())
  if failure['error']!='Compressed byte ceiling':raise ValueError('Unexpected previous failure')
  for f in sorted((prior/'records').glob('*.json')):
   rec=json.loads(f.read_text());x=rec['target_x'];identity(rec)
   raw=prior/'raw'/Path(rec['url']).name
   if sha(raw)!=rec['compressed_sha256']:raise ValueError('Previous raw hash mismatch')
   plan['prior_records'][str(x)]=str(f);plan['pins'][str(f)]=sha(f);plan['pins'][str(raw)]=sha(raw);plan['prior_bytes']+=rec['compressed_bytes']
  if len(plan['prior_records'])!=failure['completed_hours'] or set(map(int,plan['prior_records']))!=set(range(1,failure['completed_hours']+1)):raise ValueError('Previous support is not a complete prefix')
  oldplan=json.loads((prior/'plan.json').read_text());plan['prior_bytes']+=oldplan['limits']['per_response_bytes']+1
  plan['pins'][str(prior/'plan.json')]=sha(prior/'plan.json');plan['pins'][str(prior/'failure.json')]=sha(prior/'failure.json')
 (root/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');verify(plan);print('PREPARED',len(jobs),'hours;',sum(j['available'] for j in jobs),'available;',len(cache),'cached; no launch')

def launch(root):
 root=Path(root).resolve(strict=True);plan=json.loads((root/'plan.json').read_text());verify(plan)
 with (root/'launch_claim.json').open('x') as f:json.dump(dict(plan_sha256=sha(root/'plan.json'),started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),no_retries=True),f)
 import resource
 for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='2'
 resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
 import eccodes
 from shapely.geometry import shape
 sys.path.insert(0,'/mnt/3rd900/Projects/GIStoOHQ')
 from ohqbuilder.watershed_data.mrms import basin_cell_intersection_weights_m2,area_weighted_basin_average
 geom=shape(json.loads(Path(plan['basin']).read_text())['features'][0]['geometry']);cache=json.loads(Path(plan['cache']).read_text());weights={};limits=plan['limits'];start=time.monotonic();total=plan.get('prior_bytes',0);done=0;usable=0
 (root/'raw').mkdir(exist_ok=False);(root/'prefixes').mkdir(exist_ok=False);(root/'records').mkdir(exist_ok=False)
 try:
  for job in plan['jobs']:
   if time.monotonic()-start>limits['wall_seconds']:raise RuntimeError('Extraction wall ceiling')
   if total>=limits['compressed_bytes']:raise RuntimeError('Aggregate byte ceiling')
   valid,origin=identity(job);name=Path(job['url']).name;record=dict(job,valid_seconds=int(valid.timestamp()),issue_seconds=int(origin.timestamp()),valid_utc=valid.isoformat(),issue_utc=origin.isoformat(),usable=False,rate_mm_h=None,historical_delivery_proven=False)
   if str(job['target_x']) in plan.get('prior_records',{}):
    reference=Path(plan['prior_records'][str(job['target_x'])]);record=json.loads(reference.read_text());record['prior_record_source']=str(reference);record['prior_record_sha256']=sha(reference);usable+=int(record['usable'])
   elif not job['available']:record['status']='EXACT_FILE_ABSENT'
   else:
    # A bounded request error is recorded once; no retry/fallback or zero fill.
    compressed=None;headers={}
    try:
     if name in cache:
      entry=cache[name];compressed=Path(entry['path']).read_bytes();headers=entry['headers'];record['reused_cache']=True
     else:
      with urllib.request.urlopen(urllib.request.Request(job['url'],headers={'User-Agent':'HydroPINN bounded development weather extraction','Range':'bytes=0-32767'}),timeout=30) as response:
       headers=dict(response.headers)
       timestamp_metadata(headers,valid,origin)
       if response.status!=206 or not response.headers.get('Content-Range','').startswith('bytes 0-32767/'):raise ValueError('Server did not honor bounded prefix range')
       prefix=response.read(32769)
      total+=len(prefix)
      if len(prefix)!=32768:raise ValueError('Invalid prefix byte extent')
      prefix_path=root/'prefixes'/(name+'.prefix');prefix_path.write_bytes(prefix);record.update(compressed_prefix_sha256=sha(prefix_path),headers=headers)
      geometry=validated_prefix(prefix);record['prefix_grid']=geometry
      with urllib.request.urlopen(urllib.request.Request(job['url'],headers={'User-Agent':'HydroPINN bounded development weather extraction'}),timeout=30) as response:compressed=response.read(limits['per_response_bytes']+1);headers=dict(response.headers)
      if compressed[:32768]!=prefix:raise ValueError('Source changed between prefix and full request')
    except ValueError as e:compressed=None;record.update(status='WEATHER_REJECTED',reason=str(e),headers=headers if 'headers' in locals() else {})
    except Exception as e:compressed=None;record.update(status='REQUEST_FAILED',reason=str(e))
    if compressed is not None:
     total+=len(compressed)
     if len(compressed)>limits['per_response_bytes'] or total>limits['compressed_bytes']:raise RuntimeError('Compressed byte ceiling')
     raw=root/'raw'/name
     with raw.open('xb') as f:f.write(compressed)
     record.update(compressed_sha256=sha(raw),compressed_bytes=len(compressed),headers=headers)
     g=None
     try:
      release=timestamp_metadata(headers,valid,origin);record.update(release_proxy_seconds=int(release.timestamp()),release_basis='archive_http_last_modified')
      with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as stream:message=stream.read(limits['decompressed_bytes']+1)
      if len(message)>limits['decompressed_bytes'] or message[:4]!=b'GRIB':raise ValueError('Invalid/capped GRIB')
      n=int.from_bytes(message[8:16],'big')
      if message[n-4:n]!=b'7777' or any(message[n:]):raise ValueError('Invalid GRIB extent')
      g=eccodes.codes_new_from_message(message[:n]);meta={k:eccodes.codes_get(g,k) for k in GEOMETRY_KEYS};record['grid']=meta
      if [eccodes.codes_get(g,k) for k in ['discipline','parameterCategory','parameterNumber']]!=[209,6,1] or eccodes.codes_get(g,'stepType')!='instant':raise ValueError('Wrong weather product/units semantics')
      if eccodes.codes_get(g,'validityDate')!=int(valid.strftime('%Y%m%d')) or eccodes.codes_get(g,'validityTime')!=int(valid.strftime('%H%M')):raise ValueError('GRIB valid time mismatch')
      lat,lon=regular_centers(meta);key=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()
      if key not in weights:
       a,b,w=basin_cell_intersection_weights_m2(geom,lat,lon,cell_deg=meta['iDirectionIncrementInDegrees']);ii={v:i for i,v in enumerate(lat)};jj={round(v-360,10):j for j,v in enumerate(lon)};positions=[ii[x]*meta['Ni']+jj[round(y,10)] for x,y in zip(a,b)];weights[key]=(a,b,w,positions)
       with (root/('weights_'+key+'.json')).open('x') as f:json.dump(dict(grid=meta,cells=[dict(latitude=x,longitude=y,weight_m2=z,flat_index=i) for x,y,z,i in zip(a,b,w,positions) if z>0]),f,indent=2)
      a,b,w,positions=weights[key];values=list(map(float,eccodes.codes_get_double_elements(g,'values',positions)));values=[v if math.isfinite(v) and v>=0 else float('nan') for v in values];stats=area_weighted_basin_average(a,b,values,w);coverage=stats['weighted_area_m2']/sum(w)
      record.update(coverage=coverage,geometry_sha256=key,unit='mm/h',unit_basis='NOAA local MRMS discipline209/category6/parameter1')
      if coverage<.75:raise ValueError('Insufficient valid basin coverage')
      rate=stats['mean']
      if rate is None or not math.isfinite(rate) or rate<0:raise ValueError('Invalid precipitation rate')
      record.update(status='WEATHER_VALUE_VERIFIED',usable=True,rate_mm_h=rate);usable+=1
     except Exception as e:record.update(status='WEATHER_REJECTED',reason=str(e))
     finally:
      if g is not None:eccodes.codes_release(g)
   with (root/'records'/f"{job['target_x']:05d}.json").open('x') as f:json.dump(record,f,allow_nan=False)
   done+=1
   if done%100==0:
    progress=dict(completed_hours=done,usable_hours=usable,compressed_bytes=total,elapsed_seconds=time.monotonic()-start);(root/'progress.json').write_text(json.dumps(progress,indent=2)+'\n');print(json.dumps(progress),flush=True)
  with (root/'completion.json').open('x') as f:json.dump(dict(status='EXTRACTION_COMPLETE_REQUIRES_INDEPENDENT_REVIEW',hours=done,usable_hours=usable,compressed_bytes=total,plan_sha256=sha(root/'plan.json'),observations=0,new_fits=0,historical_delivery_proven=False),f,indent=2)
 except BaseException as e:
  with (root/'failure.json').open('x') as f:json.dump(dict(completed_hours=done,error=str(e),no_retry=True),f)
  raise
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('root');g=p.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',nargs=2,metavar=('INVENTORY','GAPS'));g.add_argument('--launch',action='store_true');p.add_argument('--preflight',action='store_true');p.add_argument('--prior',type=Path);a=p.parse_args()
 if a.prepare:prepare(a.root,*a.prepare,preflight=a.preflight,prior=a.prior)
 else:launch(a.root)
