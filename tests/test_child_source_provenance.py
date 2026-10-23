"""Child input proof must belong to this document, task and settled directory."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from bridge_evidence.contest import document_multiagent as pipeline
from bridge_evidence.contest.document_workspace import DocumentWorkspace
from bridge_evidence.contest.evidence_manifest import manifest_from_text
from bridge_evidence.contest.run_state import atomic_json, json_digest
from test_document_multiagent import candidate


def with_source_proof(result, manifest, directory):
    result['source_delivery'] = {
        'identity': {'document_sha256': manifest['document_sha256'],
            'manifest_sha256': json_digest(manifest), 'agent_id': result['task_id'],
            'run_id': str(directory.resolve())},
        'confirmed_ranges': deepcopy(result['entries']['reads']),
        'confirmed_requests': [{'request_id': 'synthetic-child-request',
            'evidence': {'input_delivery_confirmed': True}}],
        'main_agent_observed': False, 'semantic_review': False}
    return result


def mutate(result, field):
    source = result['source_delivery']
    if field == 'task':
        source['identity']['agent_id'] = 'another-child'
    elif field == 'document':
        source['identity']['document_sha256'] = 'other-document'
    elif field == 'manifest':
        source['identity']['manifest_sha256'] = 'other-parser-manifest'
    elif field == 'run':
        source['identity']['run_id'] += '-other-run'
    elif field == 'missing_run':
        source['identity']['run_id'] = ''
    elif field == 'ranges':
        source['confirmed_ranges'] = {}
    elif field == 'main_observed':
        source['main_agent_observed'] = True


@pytest.mark.parametrize('field', ['task', 'document', 'manifest', 'run', 'missing_run', 'ranges', 'main_observed'])
def test_identity_mismatch_rejects_child_source_proof(tmp_path, field):
    manifest = manifest_from_text('同一个合成桥的原文。')
    result = with_source_proof(candidate(tmp_path, manifest), manifest, tmp_path / 'actual-child-run')
    pipeline.validate_child_source_provenance(manifest, result,
        expected_run_id=str((tmp_path / 'actual-child-run').resolve()))
    mutate(result, field)
    with pytest.raises(ValueError, match='source-delivery provenance'):
        pipeline.validate_child_source_provenance(manifest, result,
            expected_run_id=str((tmp_path / 'actual-child-run').resolve()))


@pytest.mark.parametrize('field', ['task', 'document', 'manifest', 'missing_run', 'ranges', 'main_observed'])
def test_bad_proof_does_not_import_reads_or_candidates_into_main_workspace(tmp_path, field):
    manifest = manifest_from_text('合成桥原文。')
    result = with_source_proof(candidate(tmp_path, manifest), manifest, tmp_path / 'child')
    mutate(result, field)
    workspace = DocumentWorkspace(manifest, tmp_path / 'main', {'model': 'synthetic'})
    before = workspace.snapshot()
    with pytest.raises(ValueError, match='source-delivery provenance'):
        pipeline.import_candidates(workspace, [result])
    assert workspace.snapshot() == before


def test_matching_proof_preserves_child_origin_without_claiming_main_read_or_review(tmp_path):
    manifest = manifest_from_text('合成桥原文。')
    result = with_source_proof(candidate(tmp_path, manifest), manifest, tmp_path / 'child')
    workspace = DocumentWorkspace(manifest, tmp_path / 'main', {'model': 'synthetic'})
    assert pipeline.import_candidates(workspace, [result])['facts_imported'] == 1
    assert workspace.snapshot()['reads'] == result['entries']['reads']
    origin = json.loads((workspace.directory / 'imported-source-origins.json').read_text())
    assert origin['scope'] == 'document_union_of_agent_reading'
    assert origin['main_agent_personal_reading'] is False and origin['semantic_review'] is False
    assert origin['children'][0]['identity'] == result['source_delivery']['identity']


@pytest.mark.parametrize('field', ['task', 'document', 'run', 'matching'])
def test_controller_checks_proof_against_actual_settled_result_directory(tmp_path, monkeypatch, field):
    manifest = manifest_from_text('外层控制器的合成桥原文。')
    accepted = []

    def dispatcher(tasks, **kwargs):
        task = tasks[0]['id']
        # Deliberately different from the controller root. The validator must
        # derive identity from the actual settled result file's parent.
        actual_directory = tmp_path / 'settled-worker-location'
        actual_directory.mkdir()
        result = with_source_proof(candidate(tmp_path, manifest, task), manifest, actual_directory)
        if field != 'matching':
            mutate(result, field)
        path = actual_directory / 'result.json'
        atomic_json(path, result)
        return {'status': 'completed', 'results': [{'task_id': task, 'result_path': str(path),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'settled': True}]}

    def no_model_client(*_args, **_kwargs):
        accepted.append(True)
        raise RuntimeError('synthetic-stop-after-provenance-accepted')

    monkeypatch.setattr(pipeline, 'make_client', no_model_client)
    result = pipeline.run_multiagent(manifest, tmp_path / 'controller-run', {'model': 'synthetic'},
        dispatcher=dispatcher)
    assert result['complete'] is False
    if field == 'matching':
        assert accepted == [True]
        assert len(result['child_results']) == 1
        assert result['error']['message'] == 'synthetic-stop-after-provenance-accepted'
    else:
        assert accepted == [] and result['child_results'] == []
        assert 'source-delivery provenance' in result['error']['message']
    assert Path(result['partial_docx']).is_file()
