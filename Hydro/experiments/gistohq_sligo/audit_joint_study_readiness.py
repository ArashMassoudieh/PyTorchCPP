#!/usr/bin/env python3
"""Metadata-only ten-year readiness audit. NEVER opens observation/forcing CSVs."""
import argparse,datetime,hashlib,json
from pathlib import Path
EPOCH=datetime.datetime(2015,1,1,tzinfo=datetime.timezone.utc)
SEAL=77016

def inspect_metadata(legacy,native,weather_provenance):
    records=[];values=[]
    for path in [legacy,native,weather_provenance]:
        path=Path(path)
        if path.suffix!='.json':raise ValueError('Only explicit JSON manifests/provenance permitted')
        raw=path.read_bytes();values.append(json.loads(raw));records.append(dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest()))
    old,current,weather=values
    area_old=float(old['catchment_area_m2']);area_current=float(current['catchment_area_m2'])
    end=datetime.datetime(2025,1,1,tzinfo=datetime.timezone.utc)
    hours=int((end-EPOCH).total_seconds()/3600)
    blockers=[]
    if area_old!=area_current:blockers.append('Legacy catchment area differs from corrected gauge catchment; no automatic rescale/replacement.')
    if old.get('source_failed_qc_rule_ids'):blockers.append('Legacy manifest records failed QC rules; declared dates are not verified complete coverage.')
    if current['end_interval_x_exclusive']<hours:blockers.append('Corrected accepted native development package does not span ten years.')
    if hours>=SEAL:blockers.append('Full ten-year evaluation reaches reserved targets; this audit does not authorize or access them.')
    blockers.append('Preissue rainfall acceptance currently covers only the bounded development sidecar, not full ten-year hourly availability/geometry/release evidence.')
    return dict(status='TEN_YEAR_NOT_READY_METADATA_ONLY',declared_legacy_start=old.get('study_start'),declared_legacy_end=old.get('study_end'),
        legacy_area_m2=area_old,corrected_gauge_area_m2=area_current,legacy_qc_failures=old.get('source_failed_qc_rule_ids',[]),
        corrected_native_declared_rows=current['row_count'],corrected_native_target_cutoff=current.get('target_cutoff_x'),
        corrected_native_execution_scope=current.get('execution_scope'),
        ten_year_expected_hours=hours,reserved_start_x=SEAL,reserved_start_utc=(EPOCH+datetime.timedelta(hours=SEAL)).isoformat(),
        blockers=blockers,metadata=records,observation_csv_reads=0,forcing_csv_reads=0,training_launched=False,
        limitations=['Metadata declarations only; no claim of raw data completeness or verified ten-year skill.','Preserve old catchment/outlet/input packages and reserved targets.','Continue joint development only within the existing accepted chronological support.'])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['legacy','native','weather_provenance','new_output']:p.add_argument(name,type=Path)
    a=p.parse_args();report=inspect_metadata(a.legacy,a.native,a.weather_provenance)
    with a.new_output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
