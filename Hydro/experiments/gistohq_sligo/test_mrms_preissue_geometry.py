import unittest
from mrms_preissue_geometry import regular_centers,parse_grid_prefix
class GeometryTests(unittest.TestCase):
 def setUp(self):self.g=dict(Ni=7000,Nj=3500,scanningMode=0,latitudeOfFirstGridPointInDegrees=54.995,longitudeOfFirstGridPointInDegrees=230.005,latitudeOfLastGridPointInDegrees=20.005001,longitudeOfLastGridPointInDegrees=299.994998,iDirectionIncrementInDegrees=.01,jDirectionIncrementInDegrees=.01)
 def test_current_grid(self):
  a,b=regular_centers(self.g);self.assertAlmostEqual(a[0],54.995);self.assertAlmostEqual(a[-1],20.005);self.assertAlmostEqual(b[-1],299.995)
 def test_real_january2015_malformed_header(self):
  self.g.update(latitudeOfFirstGridPointInDegrees=55.,longitudeOfFirstGridPointInDegrees=230.,latitudeOfLastGridPointInDegrees=20.000001,longitudeOfLastGridPointInDegrees=299.999998)
  with self.assertRaisesRegex(ValueError,'Inconsistent'):regular_centers(self.g)
 def test_valid_shifted_grid_is_not_snapped(self):
  self.g.update(latitudeOfFirstGridPointInDegrees=55.,longitudeOfFirstGridPointInDegrees=230.,latitudeOfLastGridPointInDegrees=20.01,longitudeOfLastGridPointInDegrees=299.99)
  a,b=regular_centers(self.g);self.assertEqual(a[0],55.);self.assertEqual(b[0],230.)
 def test_scanning_rejected(self):
  self.g['scanningMode']=64
  with self.assertRaises(ValueError):regular_centers(self.g)
 def test_nonfinite_endpoint_rejected(self):
  self.g['latitudeOfLastGridPointInDegrees']=float('nan')
  with self.assertRaises(ValueError):regular_centers(self.g)
 def test_prefix_rejects_truncated_message(self):
  with self.assertRaises(ValueError):parse_grid_prefix(b'GRIB')
 def test_prefix_roundtrip(self):
  s=bytearray(72);s[:4]=(72).to_bytes(4,'big');s[4]=3;s[6:10]=(4).to_bytes(4,'big');s[30:34]=(2).to_bytes(4,'big');s[34:38]=(2).to_bytes(4,'big');s[38:42]=(1).to_bytes(4,'big');s[42:46]=(1000000).to_bytes(4,'big')
  for offset,value in [(46,55000000),(50,230000000),(55,54990000),(59,230010000),(63,10000),(67,10000)]:s[offset:offset+4]=value.to_bytes(4,'big')
  header=bytearray(16);header[:4]=b'GRIB';header[7]=2;g=parse_grid_prefix(header+s);a,b=regular_centers(g);self.assertEqual(a,[55.,54.99]);self.assertEqual(b,[230.,230.01])
if __name__=='__main__':unittest.main()
