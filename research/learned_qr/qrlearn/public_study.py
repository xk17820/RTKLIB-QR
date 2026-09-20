"""Auditable public-data adapters; never reads an estimated solution to alter truth.

UrbanNav HK attitude convention reproduces the publisher's
`tools/gt_vis/convert_novatel_to_pose.py`, blob23be5f2f66aa744df1b25acd3b9b0b59eddb03d2.
Rear splitter lever arm GNSS->reference is [0,-.560,-.070] metres in
right/forward/up, reported by the dataset maintainer in issue37, comment1477896168.
Manual calibration accuracy is approximately 5 cm. No fitted alignment is used.
"""
from __future__ import annotations
from collections import Counter
import csv
import datetime as dt
import hashlib
import math
from pathlib import Path
import numpy as np
from .reference import Reference, enu_rotation

GPS_EPOCH = dt.datetime(1980, 1, 6)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def dms_to_degrees(degrees, minutes, seconds):
    d, m, s = map(float, (degrees, minutes, seconds))
    if not np.isfinite([d,m,s]).all() or not 0 <= m < 60 or not 0 <= s < 60:
        raise ValueError('invalid DMS coordinate')
    return math.copysign(abs(d) + m/60 + s/3600, d)


def llh_to_ecef(lat_deg, lon_deg, height):
    if not np.isfinite([lat_deg,lon_deg,height]).all() or not -90 <= lat_deg <= 90 or not -180 <= lon_deg <= 180:
        raise ValueError('invalid ellipsoidal latitude/longitude/height')
    lat,lon = math.radians(lat_deg),math.radians(lon_deg)
    e2=6.6943799901413165e-3
    n=6378137.0/math.sqrt(1-e2*math.sin(lat)**2)
    return np.array([(n+height)*math.cos(lat)*math.cos(lon),
                     (n+height)*math.cos(lat)*math.sin(lon),
                     (n*(1-e2)+height)*math.sin(lat)])


def span_body_to_enu(roll_deg, pitch_deg, heading_deg):
    if not np.isfinite([roll_deg,pitch_deg,heading_deg]).all():
        raise ValueError('non-finite reference attitude')
    phi, theta, psi = map(math.radians, (roll_deg,pitch_deg,-heading_deg))
    cp,sp,ct,st,cy,sy = math.cos(phi),math.sin(phi),math.cos(theta),math.sin(theta),math.cos(psi),math.sin(psi)
    return np.array([[cy*cp-sy*st*sp,-sy*ct,cy*sp+sy*st*cp],
                     [sy*cp+cy*st*sp,cy*ct,sy*sp-cy*st*cp],
                     [-ct*sp,st,ct*cp]])


def reference_to_splitter_antenna(reference_xyz, roll_deg, pitch_deg, heading_deg):
    x=np.asarray(reference_xyz,dtype=np.float64)
    gnss_to_reference=np.array([0.,-.560,-.070])
    return x-enu_rotation(x).T@span_body_to_enu(roll_deg,pitch_deg,heading_deg)@gnss_to_reference


def _write_reference(path, rows):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['gps_week','tow_s','x_m','y_m','z_m','valid'])
        w.writerows(rows)
    Reference.load(path)


def convert_hk_reference(source, destination):
    """Convert the publisher's raw SPAN reference to the rear splitter antenna.

    Quality codes are inventoried, not silently interpreted as integer truth.
    Every finite published reference row is retained; no solution-error filter.
    """
    a=np.loadtxt(source,skiprows=2,ndmin=2)
    if a.shape[1] != 20:raise ValueError('expected 20 columns in UrbanNav raw reference')
    rows=[];qualities=Counter()
    for record in a:
        _,week,tow=record[:3]
        if not np.isfinite([week,tow]).all() or week<0 or week != int(week) or not 0 <= tow < 604800:
            raise ValueError('invalid original reference GPST')
        qualities[str(int(record[19])) if np.isfinite(record[19]) else 'nonfinite']+=1
        valid=bool(np.isfinite(record[3:10]).all() and np.isfinite(record[16:19]).all())
        if valid:
            xyz=llh_to_ecef(dms_to_degrees(*record[3:6]),dms_to_degrees(*record[6:9]),record[9])
            xyz=reference_to_splitter_antenna(xyz,*record[16:19])
        else:xyz=np.full(3,np.nan)
        rows.append([int(week),format(tow,'.9f'),*xyz,int(valid)])
    _write_reference(destination,rows)
    return {'source_sha256':sha256(source),'converted_sha256':sha256(destination),'rows':len(rows),
            'source_quality_counts':dict(qualities),'quality_policy':'all finite published reference rows; no post-hoc error or quality-code selection',
            'time_policy':'original GPS week and TOW; no Unix conversion or fitted offset',
            'lever_applied_to_reference_only':True,'gnss_to_reference_body_rfu_m':[0.,-.560,-.070],
            'manual_lever_uncertainty_m':.05,'fitted_position_or_time_alignment':False}


def convert_tokyo_reference(source,destination):
    a=np.loadtxt(source,delimiter=',',skiprows=1,ndmin=2)
    if a.shape[1] < 8:raise ValueError('expected Tokyo reference GPS TOW, week, LLH and ECEF columns')
    rows=[]
    for row in a:
        tow,week=row[:2]
        if not np.isfinite([week,tow]).all() or week != int(week) or not 0 <= tow < 604800:
            raise ValueError('invalid Tokyo reference GPST')
        xyz=row[5:8];valid=int(np.isfinite(xyz).all())
        rows.append([int(week),format(tow,'.9f'),*xyz,valid])
    _write_reference(destination,rows)
    return {'source_sha256':sha256(source),'converted_sha256':sha256(destination),'rows':len(rows),
            'time_policy':'original GPS week and TOW','coordinate_policy':'published ECEF, no fitted translation or lever correction',
            'reference_point_alignment_verified':False,'fitted_position_or_time_alignment':False}


def validate_partitions(partitions):
    """A renamed copy or a second receiver from a campaign is not an independent split."""
    seen_ids=set();campaigns={};contents={};inventory=[]
    for split,routes in partitions.items():
        for r in routes:
            if not r.get('campaign'):raise ValueError('every route requires a campaign identifier')
            if r['id'] in seen_ids:raise ValueError('duplicate route id')
            seen_ids.add(r['id'])
            digest=sha256(r['rover'])
            if digest in contents and contents[digest] != split:raise ValueError('rover content overlaps partitions')
            if r['campaign'] in campaigns and campaigns[r['campaign']] != split:raise ValueError('campaign overlaps partitions')
            contents[digest]=split;campaigns[r['campaign']]=split
            inventory.append({'id':r['id'],'split':split,'campaign':r['campaign'],'rover_sha256':digest})
    return {'route_count':len(inventory),'campaign_count':len(campaigns),'routes':inventory}


def scheduled_slots(times,period):
    t=np.asarray(times,dtype=float)
    if not len(t) or t.ndim != 1 or not np.isfinite(t).all() or np.any(np.diff(t)<=0):
        raise ValueError('observation timestamps must increase strictly')
    if not np.isfinite(period) or period<=0:raise ValueError('period must be positive')
    origin=round(t[0]/period)*period
    slots=np.rint((t-origin)/period).astype(np.int64)
    # Jitter is assigned only for denominators; original measurements are never retimed.
    if np.max(np.abs(t-(origin+slots*period)))>min(.04,period*.4):
        raise ValueError('timestamps do not fit the declared sampling schedule')
    if len(np.unique(slots)) != len(slots):raise ValueError('duplicate scheduled observation slot')
    if slots[0] != 0:raise ValueError('invalid schedule origin')
    n=int(slots[-1])+1
    return {'scheduled_slots':n,'missing_observation_slots':n-len(t),'row_slots':slots.tolist(),
            'original_times':t.tolist(),'schedule_origin_gpst_s':float(origin),'period_s':float(period)}


def rinex_inventory(path,period):
    times=[]
    with Path(path).open(encoding='ascii',errors='strict') as f:
        for line in f:
            if not line.startswith('>'):continue
            # RINEX event flags 2..5 can omit the timestamp (e.g. new header).
            # The native parser consumes these records; they are not rover slots.
            if len(line)>31 and line[31:32].strip() and line[31] not in '01':continue
            fields=line[1:].split()
            if len(fields)<7:raise ValueError('malformed RINEX3 epoch')
            if int(fields[6]) not in (0,1):continue
            year,month,day,hour,minute=map(int,fields[:5]);seconds=float(fields[5])
            stamp=dt.datetime(year,month,day,hour,minute)+dt.timedelta(seconds=seconds)
            times.append((stamp-GPS_EPOCH).total_seconds())
    s=scheduled_slots(times,period)
    return {**s,'raw_epochs':len(times),'first_gpst_s':times[0],'last_gpst_s':times[-1],
            'sha256':sha256(path),'span_s':times[-1]-times[0]}


def merge_hourly_base(paths,destination):
    """Concatenate complete chronological RINEX3 bodies, retaining one header."""
    if Path(destination).exists():raise FileExistsError(destination)
    headers=[];bodies=[]
    for p in paths:
        text=Path(p).read_text(encoding='ascii').splitlines(keepends=True)
        end=next((i for i,s in enumerate(text) if 'END OF HEADER' in s),None)
        if end is None:raise ValueError('base file lacks RINEX header')
        headers.append(text[:end+1]);bodies.append(text[end+1:])
    def contract(h):
        return [s for s in h if 'SYS / # / OBS TYPES' in s or 'APPROX POSITION XYZ' in s or 'REC # / TYPE / VERS' in s]
    if any(contract(h)!=contract(headers[0]) for h in headers[1:]):raise ValueError('base observation schema/receiver/position changed')
    Path(destination).parent.mkdir(parents=True,exist_ok=True)
    with Path(destination).open('x',encoding='ascii') as f:
        f.writelines(headers[0])
        for b in bodies:f.writelines(b)
    rinex_inventory(destination,1.)
    return sha256(destination)
