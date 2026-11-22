#!/usr/bin/env python3
"""Prepare new native-v3 AORC24/radar1 input packages; preserve Q/met/PET.
Input-package comparison (source AND age), not an isolated timing experiment.
Common origin forcing mask uses only availability already knowable at origin.
"""
import argparse,csv,hashlib,json,math,shutil
from pathlib import Path
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
BASE_HEADER='epoch_seconds,q_m3_s,p_mm_h,t_c,rh_pct,wind_m_s,solar_mj_m2_h,pet_mm_h,forcing_valid,discharge_valid,native,interpolated,native_coverage,source_id'.split(',')
def prepare(native,radar,output):
 m=json.loads((native/'native_input_manifest.json').read_text());receipt=json.loads((radar/'receipt.json').read_text())
 if m['schema']!='HydroFusionPreparedNative-v2' or m['interval_csv_sha256']!=sha(native/'intervals.csv'):raise ValueError('Unverified native-v2 source')
 if receipt['csv_sha256']!=sha(radar/'basin_rainfall.csv'):raise ValueError('Radar CSV hash mismatch')
 if m['catchment_area_m2']!=16737500 or m['first_interval_x']!=0 or m['end_interval_x_exclusive']!=35064:raise ValueError('Wrong study area/interval')
 with (native/'intervals.csv').open() as f:
  reader=csv.DictReader(f);assert reader.fieldnames==BASE_HEADER;base=list(reader)
 with (radar/'basin_rainfall.csv').open() as f:rain=list(csv.DictReader(f))
 assert len(base)==len(rain)==35064
 # ALL identities before converting any observation or forcing value.
 for i,(a,b) in enumerate(zip(base,rain)):
  expected=1420070400+i*3600
  if a['epoch_seconds']!=str(expected) or b['interval_start_epoch']!=str(expected) or b['valid_end_epoch']!=str(expected+3600):raise ValueError('Nonmatching hourly/preseal identity')
 for a,b in zip(base,rain):
  for key in ['forcing_valid','discharge_valid','native','interpolated']:
   if a[key] not in ('0','1'):raise ValueError('Nonboolean native flag')
  c=float(b['spatial_coverage']);v=b['rainfall_valid']
  if not math.isfinite(c) or not 0<=c<=1 or v!=str(int(c>=.75)):raise ValueError('Invalid radar coverage/mask')
  if v=='1' and (not math.isfinite(float(b['p_mm_h'])) or float(b['p_mm_h'])<0):raise ValueError('Invalid accepted radar rain')
  if v=='0' and b['p_mm_h']!='':raise ValueError('Missing radar must remain missing')
 common=[]
 for i in range(len(base)):
  # Origin=i+1; AORC valid-end=origin-24, radar valid-end=origin-1.
  old=i-24;new=i-1
  common.append(old>=0 and new>=0 and base[old]['forcing_valid']=='1' and rain[new]['rainfall_valid']=='1')
 output.mkdir(parents=True,exist_ok=False)
 packages={}
 for arm in ['aorc24_control','radar1_candidate']:
  folder=output/arm;folder.mkdir();p=folder/'intervals.csv'
  with p.open('w',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=BASE_HEADER+['rainfall_valid','common_origin_forcing_available'],lineterminator='\n');writer.writeheader()
   for i,a in enumerate(base):
    row=dict(a);row['common_origin_forcing_available']=str(int(common[i]))
    row['rainfall_valid']=a['forcing_valid']
    if arm=='radar1_candidate':row['p_mm_h']=rain[i]['p_mm_h'] if rain[i]['rainfall_valid']=='1' else '0';row['rainfall_valid']=rain[i]['rainfall_valid']
    writer.writerow(row)
  nm=dict(m,schema='HydroFusionPreparedNative-v3',interval_mask_schema='independent_rain_and_common_origin_v1',interval_csv_sha256=sha(p),rainfall_delay_hours=24 if arm=='aorc24_control' else 1,rainfall_release_delay_hours=24 if arm=='aorc24_control' else 1,rainfall_product='AORC_hourly' if arm=='aorc24_control' else 'MRMS_RadarOnly_QPE_01H',rainfall_availability_basis='assumed_fixed_hourly_delay',rainfall_availability_evidence='Retrospective assumed release scenario. One-hour radar age is NOT historical operational delivery certification.',radar_receipt_sha256=sha(radar/'receipt.json'),common_support_rule='At origin: AORC met valid at origin-24h AND radar rain valid at origin-1h. Applied only to sample eligibility, not Q support or physical forcing/state.',comparison_scope='Rainfall input-package source-and-age comparison, no isolated timing attribution.',research_training_authorization='Latest explicit human all-method instruction; campaign records govern limits, not this descriptive field.')
  (folder/'native_input_manifest.json').write_text(json.dumps(nm,indent=2)+'\n');packages[arm]=dict(manifest_sha256=sha(folder/'native_input_manifest.json'),interval_csv_sha256=sha(p))
 report=dict(schema='HydroFusionMatchedRadarPackages-v1',source_manifest_sha256=sha(native/'native_input_manifest.json'),radar_receipt_sha256=sha(radar/'receipt.json'),packages=packages,rows=len(base),common_origin_forcing_hours=sum(common),observations_and_native_flags='Copied unchanged; missing rainfall numeric placeholder is excluded by its separate false mask.',coverage=receipt['annual_coverage'],status='PREPARED_PENDING_CPP_SUPPORT_AND_RELOAD_VERIFICATION')
 (output/'preparation_receipt.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for n in ['native','radar','output']:p.add_argument(n,type=Path)
 a=p.parse_args();prepare(a.native,a.radar,a.output)
