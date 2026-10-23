"""Pure preparation helpers and isolated-launch contracts; no MinerU/model run."""
from copy import deepcopy
import hashlib
import importlib
import io
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import zipfile

from lxml import etree
import pytest

from bridge_evidence.contest.mineru_preparation import PreparationError, prepare_document
from bridge_evidence.contest.mineru_preparation import native_repair
from bridge_evidence.contest.mineru_preparation._worker import require_native_configuration
from bridge_evidence.contest.mineru_preparation.mineru_materials import prepare
from bridge_evidence.contest.mineru_preparation.source_docx import NS, source_projection, xml_tables

launcher = importlib.import_module("bridge_evidence.contest.mineru_preparation.prepare_document")


def docx_fixture(tmp_path, body=None):
    root = tmp_path / "official-training"
    root.mkdir(exist_ok=True)
    source = root / "bridge.docx"
    body = body or '<w:p><w:r><w:t>合成训练桥</w:t></w:r></w:p>'
    xml = ('<w:document xmlns:w="' + NS['w'] + '" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
           '<w:body>' + body + '</w:body></w:document>').encode()
    with zipfile.ZipFile(source, 'w') as archive:
        archive.writestr('word/document.xml', xml)
        archive.writestr('word/media/image1.png', b'fake-image-payload')
    return source, root


def write_success(command, **kwargs):
    source, output = Path(command[3]), Path(command[4])
    assert kwargs['check'] is False and kwargs['capture_output'] is True
    assert kwargs['cwd'] == str(output)
    assert not any(output.iterdir())
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    (output / 'manifest.json').write_text(json.dumps({'status': 'prepared', 'docx_sha256': sha}))
    for name in ('document.json', 'materials.json'):
        (output / name).write_text('{}')
    (output / 'document.md').write_text('合成训练桥')
    (output / 'parsed-input.docx').write_bytes(source.read_bytes())
    return SimpleNamespace(returncode=0, stdout='prepared\n', stderr='')


def test_launcher_uses_separate_offline_runtime_without_parent_environment_changes(tmp_path, monkeypatch):
    source, root = docx_fixture(tmp_path)
    output = tmp_path / 'new-output'
    calls = []
    before_path = list(sys.path)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1')
    monkeypatch.setenv('PYTHONPATH', '/unrelated/sidecar')
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return write_success(command, **kwargs)
    monkeypatch.setattr(launcher.subprocess, 'run', run)
    result = prepare_document(source, output, training_root=root, python_executable=sys.executable,
        expected_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    assert result['status'] == 'prepared'
    command, options = calls[0]
    assert command[1:3] == ['-m', 'bridge_evidence.contest.mineru_preparation._worker']
    assert command[5:7] == ['--training-root', str(root)]
    env = options['env']
    assert env['CUDA_VISIBLE_DEVICES'] == '' and env['HIP_VISIBLE_DEVICES'] == ''
    assert env['HF_HUB_OFFLINE'] == '1' and env['MINERU_MODEL_SOURCE'] == 'local'
    assert env['ACCELERATE_USE_CPU'] == 'true' and env['OMP_NUM_THREADS'] == '1'
    assert env['PYTHONPATH'] == str(Path(launcher.__file__).resolve().parents[3])
    assert launcher.os.environ['CUDA_VISIBLE_DEVICES'] == '0,1'
    assert launcher.os.environ['PYTHONPATH'] == '/unrelated/sidecar'
    assert sys.path == before_path
    audit = json.loads((output / 'launch.json').read_text())
    assert audit['separate_process'] is True and audit['status'] == 'prepared'


@pytest.mark.parametrize('case', ['outside_root', 'symlink_escape', 'wrong_suffix', 'hash', 'nonempty', 'file_output'])
def test_invalid_scope_and_existing_outputs_fail_before_child(tmp_path, monkeypatch, case):
    source, root = docx_fixture(tmp_path)
    output, expected = tmp_path / 'output', None
    if case in {'outside_root', 'symlink_escape'}:
        other = tmp_path / 'outside.docx'
        other.write_bytes(source.read_bytes())
        if case == 'symlink_escape':
            source = root / 'escaped.docx'
            source.symlink_to(other)
        else:
            source = other
    elif case == 'wrong_suffix':
        target = source.with_suffix('.doc')
        source.rename(target)
        source = target
    elif case == 'hash':
        expected = '0' * 64
    elif case == 'nonempty':
        output.mkdir()
        (output / 'materials.json').write_text('preserve me')
    else:
        output.write_text('preserve me')
    monkeypatch.setattr(launcher.subprocess, 'run', lambda *_args, **_kwargs: pytest.fail('must not launch'))
    with pytest.raises((ValueError, FileExistsError)):
        prepare_document(source, output, training_root=root, expected_sha256=expected)
    if case == 'nonempty':
        assert (output / 'materials.json').read_text() == 'preserve me'
    if case == 'file_output':
        assert output.read_text() == 'preserve me'


@pytest.mark.parametrize('case', ['nonzero', 'timeout', 'missing_manifest', 'wrong_sha'])
def test_failure_is_visible_and_never_reported_as_prepared(tmp_path, monkeypatch, case):
    source, root = docx_fixture(tmp_path)
    output = tmp_path / 'output'
    def run(command, **kwargs):
        (output / 'partial-evidence.json').write_text('preserved')
        if case == 'timeout':
            raise subprocess.TimeoutExpired(command, 1, output=b'partial log', stderr=b'timed out')
        if case == 'wrong_sha':
            (output / 'manifest.json').write_text(json.dumps({'status': 'prepared', 'docx_sha256': '0' * 64}))
        return SimpleNamespace(returncode=2 if case == 'nonzero' else 0,
            stdout='partial log', stderr='MinerU runtime unavailable')
    monkeypatch.setattr(launcher.subprocess, 'run', run)
    with pytest.raises(PreparationError):
        prepare_document(source, output, training_root=root)
    assert (output / 'partial-evidence.json').read_text() == 'preserved'
    assert json.loads((output / 'failure.json').read_text())['status'] == 'failed'
    assert json.loads((output / 'launch.json').read_text())['status'] == 'failed'
    assert (output / 'parser.stdout.log').read_text() == 'partial log'


def test_projection_preserves_every_word_text_and_does_not_change_original(tmp_path):
    source, _root = docx_fixture(tmp_path, '<w:p><w:r><w:t>原文一</w:t><w:drawing><a:blip/></w:drawing></w:r></w:p>'
        '<w:p><w:r><w:pict><w:p><w:r><w:t>文本框原文</w:t></w:r></w:p></w:pict><w:t>原文二</w:t></w:r></w:p>')
    original = source.read_bytes()
    payload, scope = source_projection(source)
    assert source.read_bytes() == original
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        tree = etree.fromstring(archive.read('word/document.xml'))
        assert tree.xpath('//w:t/text()', namespaces=NS) == ['原文一', '文本框原文', '原文二']
        assert not tree.xpath('//w:drawing', namespaces=NS)
        assert tree.xpath('//w:pict', namespaces=NS)  # text-bearing shape retained
        assert archive.read('word/media/image1.png') == b'fake-image-payload'
    assert scope['w_t_exactly_preserved'] is True and scope['image_references_removed'] == 1


def table_fixture(tmp_path):
    body = '''<w:p><w:r><w:t>第一章 检测结果</w:t></w:r></w:p>
    <w:tbl><w:tblGrid><w:gridCol/><w:gridCol/><w:gridCol/></w:tblGrid>
    <w:tr><w:trPr><w:tblHeader/></w:trPr><w:tc><w:tcPr><w:gridSpan w:val="2"/></w:tcPr><w:p><w:r><w:t>构件</w:t></w:r></w:p></w:tc>
    <w:tc><w:p><w:r><w:t>评分</w:t></w:r></w:p></w:tc></w:tr>
    <w:tr><w:tc><w:tcPr><w:gridSpan w:val="2"/></w:tcPr><w:p><w:r><w:t>上部结构</w:t></w:r></w:p></w:tc>
    <w:tc><w:p><w:r><w:t>88.23</w:t></w:r></w:p></w:tc></w:tr></w:tbl>'''
    source, _root = docx_fixture(tmp_path, body)
    raw = {'pages': [{'page_idx': 0, 'blocks': [
        {'type': 'paragraph_title', 'content': '第一章 检测结果', 'level': 1},
        {'type': 'table_body', 'content': '<table><tr><td colspan="2">构件</td><td>评分</td></tr>'
         '<tr><td colspan="2">上部结构</td><td>88.23</td></tr></table>'}]}]}
    return source, raw


def test_materials_preserve_exact_table_spans_coordinates_headers_and_provenance(tmp_path):
    source, raw = table_fixture(tmp_path)
    materials = prepare(raw, source, budget=1000)
    assert materials['docx_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    ref = '#/tables/0/r1/c0'
    cell = materials['source_map'][ref]
    assert (cell['row'], cell['column'], cell['col_span']) == (1, 0, 2)
    assert materials['source_map']['#/tables/0/r1/c2']['column'] == 2
    assert cell['xml_cell_path'].endswith('/w:tr[2]/w:tc[1]')
    assert materials['diagnostics']['conservation_failures'] == []
    assert materials['diagnostics']['source_table_count'] == 1
    assert all(packet['docx_sha256'] == materials['docx_sha256'] for packet in materials['packets'])
    assert any('原DOCX重复表头' in packet['context'] and '88.23' in packet['context'] for packet in materials['packets'])


def test_mismatched_native_spans_fail_instead_of_inventing_cell_mapping(tmp_path):
    source, raw = table_fixture(tmp_path)
    raw['pages'][0]['blocks'][1]['content'] = raw['pages'][0]['blocks'][1]['content'].replace('colspan="2"', 'colspan="1"')
    with pytest.raises(ValueError, match='unverified span'):
        prepare(raw, source)


def test_original_table_not_returned_by_parser_is_explicit_failure(tmp_path):
    source, raw = table_fixture(tmp_path)
    raw['pages'][0]['blocks'].pop()
    with pytest.raises(ValueError, match='Source tables were not all consumed'):
        prepare(raw, source)


def test_long_text_materials_reconstruct_original_with_character_budget(tmp_path):
    text = '历史检测原文与位置说明。' * 500
    source, _root = docx_fixture(tmp_path, '<w:p><w:r><w:t>' + text + '</w:t></w:r></w:p>')
    materials = prepare({'pages': [{'page_idx': 0, 'blocks': [{'type': 'paragraph', 'content': text}]}]}, source, budget=350)
    packets = materials['packets']
    assert len(packets) > 1 and all(len(p['context']) <= 350 for p in packets)
    assert ''.join(s['text'] for p in packets for s in p['segments']) == text
    assert materials['diagnostics']['conservation_failures'] == []


def test_coordinate_oracle_preserves_vertical_merge_and_nested_table(tmp_path):
    body = '''<w:tbl><w:tblGrid><w:gridCol/><w:gridCol/></w:tblGrid>
    <w:tr><w:tc><w:tcPr><w:vMerge w:val="restart"/></w:tcPr><w:p><w:r><w:t>跨行名</w:t></w:r></w:p></w:tc>
    <w:tc><w:p><w:r><w:t>外层</w:t></w:r></w:p><w:tbl><w:tblGrid><w:gridCol/></w:tblGrid><w:tr><w:tc><w:p><w:r><w:t>内层</w:t></w:r></w:p></w:tc></w:tr></w:tbl></w:tc></w:tr>
    <w:tr><w:tc><w:tcPr><w:vMerge/></w:tcPr><w:p/></w:tc><w:tc><w:p><w:r><w:t>第二行</w:t></w:r></w:p></w:tc></w:tr></w:tbl>'''
    source, _root = docx_fixture(tmp_path, body)
    with zipfile.ZipFile(source) as archive:
        tables = list(xml_tables(etree.fromstring(archive.read('word/document.xml'))).values())
    assert len(tables) == 2
    assert tables[0]['cells'][0]['row_span'] == 2
    assert tables[0]['cells'][1]['text'] == '外层'
    assert tables[1]['cells'][0]['text'] == '内层'
    assert tables[0]['nested_count'] == 1


@pytest.mark.parametrize('title,merge', [(True, False), (False, True), (True, True)])
def test_llm_features_are_rejected_explicitly(title, merge):
    config = SimpleNamespace(llm_aided=SimpleNamespace(features=SimpleNamespace(
        title_leveling=title, cross_page_table_cell_merge=merge)))
    with pytest.raises(RuntimeError, match='must both be disabled'):
        require_native_configuration(config)


def mock_native_runtime(tmp_path, monkeypatch):
    class Tables:
        def _normalize_table_colspans(self, markup):
            return 'original-normalizer'
    module = ModuleType('docvortex.analyzers.native.office.docx.tables')
    module._DocxTables = Tables
    monkeypatch.setitem(sys.modules, module.__name__, module)
    reviewed = tmp_path / 'tables.py'
    reviewed.write_text('synthetic reviewed parser')
    monkeypatch.setattr('inspect.getfile', lambda obj: str(reviewed))
    monkeypatch.setattr(native_repair, 'SOURCE_HASH', hashlib.sha256(reviewed.read_bytes()).hexdigest())
    monkeypatch.setattr(native_repair, 'version', lambda name: {'mineru': '4.0.12', 'docvortex': '0.5.16'}[name])
    return Tables, reviewed


def test_native_repair_is_scoped_and_restored_on_exception(tmp_path, monkeypatch):
    tables, reviewed = mock_native_runtime(tmp_path, monkeypatch)
    original = tables._normalize_table_colspans
    with pytest.raises(ValueError, match='parse failed'):
        with native_repair.preserve_native_spans() as receipt:
            assert tables()._normalize_table_colspans('<td colspan="2">原文</td>') == '<td colspan="2">原文</td>'
            assert receipt['tables_seen'] == 1 and receipt['package_modified'] is False
            with pytest.raises(RuntimeError, match='Nested'):
                with native_repair.preserve_native_spans():
                    pass
            raise ValueError('parse failed')
    assert tables._normalize_table_colspans is original
    assert reviewed.read_text() == 'synthetic reviewed parser'


@pytest.mark.parametrize('fault', ['version', 'source_hash'])
def test_unreviewed_native_runtime_is_rejected_before_hook(tmp_path, monkeypatch, fault):
    tables, reviewed = mock_native_runtime(tmp_path, monkeypatch)
    original = tables._normalize_table_colspans
    if fault == 'version':
        monkeypatch.setattr(native_repair, 'version', lambda name: '999')
    else:
        reviewed.write_text('different implementation')
    with pytest.raises(RuntimeError):
        with native_repair.preserve_native_spans():
            pytest.fail('unreviewed runtime cannot parse')
    assert tables._normalize_table_colspans is original
