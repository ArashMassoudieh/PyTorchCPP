"""Validate GRIB regular-grid point geometry before basin intersection.
No half-cell correction or endpoint interpolation is inferred for malformed grids.
"""
import math

def regular_centers(g):
 if g['scanningMode']!=0:raise ValueError('Unsupported GRIB scanning mode')
 ni,nj=g['Ni'],g['Nj']
 if type(ni) is not int or type(nj) is not int or min(ni,nj)<2:raise ValueError('Invalid grid dimensions')
 lat,lon=g['latitudeOfFirstGridPointInDegrees'],g['longitudeOfFirstGridPointInDegrees']
 di,dj=g['iDirectionIncrementInDegrees'],g['jDirectionIncrementInDegrees']
 if not all(math.isfinite(v) for v in [lat,lon,di,dj,g['latitudeOfLastGridPointInDegrees'],g['longitudeOfLastGridPointInDegrees']]) or min(di,dj)<=0:raise ValueError('Invalid grid coordinates/increments')
 if abs(di-dj)>1e-9:raise ValueError('Non-square angular grid unsupported')
 expected_lat=lat-(nj-1)*dj;expected_lon=lon+(ni-1)*di
 if abs(expected_lat-g['latitudeOfLastGridPointInDegrees'])>5e-6 or abs(expected_lon-g['longitudeOfLastGridPointInDegrees'])>5e-6:raise ValueError('Inconsistent first/last/count/increment grid geometry; no inferred correction')
 return [lat-i*dj for i in range(nj)],[lon+i*di for i in range(ni)]

def parse_grid_prefix(data):
 """Read template3.0 geometry from a bounded decompressed GRIB2 prefix."""
 if len(data)<16 or data[:4]!=b'GRIB' or data[7]!=2:raise ValueError('Invalid GRIB2 prefix')
 offset=16
 while offset+5<=len(data):
  size=int.from_bytes(data[offset:offset+4],'big');section=data[offset+4]
  if size<5 or offset+size>len(data):raise ValueError('Truncated GRIB section prefix')
  s=data[offset:offset+size]
  if section==3:
   if size<72 or int.from_bytes(s[12:14],'big')!=0:raise ValueError('Unsupported grid template')
   basic=int.from_bytes(s[38:42],'big');sub=int.from_bytes(s[42:46],'big')
   if basic==0:scale=1e-6
   elif basic!=0xffffffff and sub not in (0,0xffffffff):scale=basic/sub
   else:raise ValueError('Invalid angular scale')
   def unsigned(a,b):return int.from_bytes(s[a:b],'big')
   def angle(a,b):
    v=unsigned(a,b);return (-1 if v&0x80000000 else 1)*(v&0x7fffffff)*scale
   g=dict(Ni=unsigned(30,34),Nj=unsigned(34,38),latitudeOfFirstGridPointInDegrees=angle(46,50),longitudeOfFirstGridPointInDegrees=angle(50,54),latitudeOfLastGridPointInDegrees=angle(55,59),longitudeOfLastGridPointInDegrees=angle(59,63),iDirectionIncrementInDegrees=unsigned(63,67)*scale,jDirectionIncrementInDegrees=unsigned(67,71)*scale,scanningMode=s[71])
   if unsigned(6,10)!=g['Ni']*g['Nj']:raise ValueError('Grid point count mismatch')
   return g
  offset+=size
 raise ValueError('No grid definition in prefix')
