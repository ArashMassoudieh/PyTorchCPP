"""Prototype strict pre-issue weather contract; NOT wired into model training.
UTC epoch seconds are explicit. Missing or unavailable weather returns None,
never a fabricated dry hour. Fixed sample is issue minus600seconds.
"""
import math
EPOCH_SECONDS=1420070400
SEAL_X=77016

def integer(v):
 if isinstance(v,bool) or not isinstance(v,int):raise ValueError('Exact integer timestamp required')
 return v

def select_rate(records,issue_seconds,development_target_cutoff=26303):
 issue=integer(issue_seconds);cutoff=integer(development_target_cutoff)
 if not 0<cutoff<SEAL_X:raise ValueError('Invalid development cutoff')
 delta=issue-EPOCH_SECONDS
 if delta<0 or delta%3600:raise ValueError('Issue must identify an hourly native origin')
 target_x=delta//3600+1
 # Guard forecast identity before any weather value inspection.
 if target_x>cutoff or target_x>=SEAL_X:raise ValueError('Forecast target outside development scope')
 desired=issue-600;seen=set();chosen=None
 for record in records:
  valid=integer(record['valid_seconds']);release=integer(record['release_proxy_seconds'])
  if valid in seen:raise ValueError('Duplicate weather valid time')
  seen.add(valid)
  if valid!=desired:continue
  if release<valid:raise ValueError('Release proxy precedes valid time')
  if release>issue:continue
  if record.get('release_basis')!='archive_http_last_modified':raise ValueError('Unknown release provenance')
  coverage=record['coverage'];rate=record['rate_mm_h']
  if isinstance(rate,bool) or isinstance(coverage,bool) or not all(isinstance(v,(float,int)) and math.isfinite(v) for v in [coverage,rate]) or not 0<=coverage<=1 or rate<0:raise ValueError('Invalid rain value/coverage')
  if coverage<.75:continue
  chosen=dict(rate_mm_h=float(rate),valid_seconds=valid,release_proxy_seconds=release,issue_seconds=issue,target_x=target_x,coverage=float(coverage),historical_delivery_proven=False)
 return chosen
