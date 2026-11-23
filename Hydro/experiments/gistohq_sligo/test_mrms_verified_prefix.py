import gzip,unittest
from extract_mrms_preissue_verified import validated_prefix
class PrefixTests(unittest.TestCase):
 def message(self,bad=False):
  s=bytearray(72);s[:4]=(72).to_bytes(4,'big');s[4]=3;s[6:10]=(4).to_bytes(4,'big');s[30:34]=(2).to_bytes(4,'big');s[34:38]=(2).to_bytes(4,'big')
  for offset,value in [(46,55000000),(50,230000000),(55,54980000 if bad else 54990000),(59,230010000),(63,10000),(67,10000)]:s[offset:offset+4]=value.to_bytes(4,'big')
  header=bytearray(16);header[:4]=b'GRIB';header[7]=2;compressed=gzip.compress(header+s);return compressed+b'\0'*(32768-len(compressed))
 def test_valid_prefix(self):self.assertEqual(validated_prefix(self.message())['Ni'],2)
 def test_malformed_grid_rejected_before_full_download(self):
  with self.assertRaisesRegex(ValueError,'Inconsistent'):validated_prefix(self.message(True))
 def test_unbounded_prefix_rejected(self):
  with self.assertRaises(ValueError):validated_prefix(self.message()+b'x')
if __name__=='__main__':unittest.main()
