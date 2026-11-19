import hashlib
from build_hydrofusion_frozen import source_snapshot_changes


def test_sync_case_rename_is_a_build_integrity_failure(tmp_path):
    p=tmp_path/'settings.cpp';p.write_bytes(b'original')
    expected={'settings.cpp':hashlib.sha256(p.read_bytes()).hexdigest()}
    assert source_snapshot_changes(tmp_path,expected)==[]
    p.rename(tmp_path/'settings (Case Conflict).cpp')
    assert source_snapshot_changes(tmp_path,expected)==['settings.cpp']


def test_source_edit_is_not_hidden_by_unchanged_filename(tmp_path):
    p=tmp_path/'model.h';p.write_bytes(b'old')
    expected={'model.h':hashlib.sha256(p.read_bytes()).hexdigest()}
    p.write_bytes(b'new')
    assert source_snapshot_changes(tmp_path,expected)==['model.h']
