import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from slice_hydrofusion_producer_prefix import copy_prefix, HEADER_BYTES
from audit_hydrofusion_producer_package import EPOCH


class PrefixTests(unittest.TestCase):
    def test_never_reads_value_of_first_excluded_record(self):
        prefix = HEADER_BYTES + b'2015-01-01T00:00:00Z,00060,1,cfs,A\n'
        cutoff = b'2015-01-01T01:00:00Z,'
        class ProtectedStream(io.BytesIO):
            def read(self, n=-1):
                if n < 0 or self.tell() + n > len(prefix) + len(cutoff):
                    raise AssertionError('attempt to read protected target value')
                return super().read(n)
            def readline(self, n=-1):
                if self.tell() >= len(prefix):
                    raise AssertionError('attempt to read protected record remainder')
                return super().readline(n)
        original = Path.open
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / 'out.csv'
            def open_path(path, *args, **kwargs):
                if str(path) == 'protected':
                    self.assertEqual(kwargs.get('buffering'), 0)
                    return ProtectedStream(prefix + cutoff + b'00060,SECRET,cfs,A\n')
                return original(path, *args, **kwargs)
            with patch.object(Path, 'open', open_path):
                self.assertEqual(copy_prefix(Path('protected'), destination, EPOCH + 3600), 1)
            self.assertEqual(destination.read_bytes(), prefix)

    def test_bad_cutoff_rejected_before_source_open(self):
        with self.assertRaises(ValueError):
            copy_prefix(Path('/missing'), Path('/missing-output'), EPOCH)

    def test_unsorted_and_partial_records_reject(self):
        for data in (b'2015-01-01T00:00:00Z,00060,1,cfs,A', b'2015-01-01T00:00:00Z,00060,1,cfs,A\n2015-01-01T00:00:0'):
            with self.subTest(data=data), tempfile.TemporaryDirectory() as temp:
                source = Path(temp) / 'in.csv'
                source.write_bytes(HEADER_BYTES + data)
                with self.assertRaises(ValueError):
                    copy_prefix(source, Path(temp) / 'out.csv', EPOCH + 3600)


if __name__ == '__main__':
    unittest.main()
