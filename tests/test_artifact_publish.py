import errno
from unittest.mock import patch
import pytest
from bridge_evidence.contest.artifact_publish import publish_no_replace

def test_oss_without_hardlinks_uses_locked_replace(tmp_path):
    source=tmp_path/'pending';source.write_bytes(b'complete')
    target=tmp_path/'result'
    with patch('os.link',side_effect=OSError(errno.EOPNOTSUPP,'OSS')):
        publish_no_replace(source,target,tmp_path/'locks')
    assert target.read_bytes()==b'complete'

def test_fallback_never_overwrites_existing_output(tmp_path):
    source=tmp_path/'pending';source.write_bytes(b'new');target=tmp_path/'result';target.write_bytes(b'user')
    with patch('os.link',side_effect=OSError(errno.EOPNOTSUPP,'OSS')):
        with pytest.raises(FileExistsError):publish_no_replace(source,target,tmp_path/'locks')
    assert target.read_bytes()==b'user'
