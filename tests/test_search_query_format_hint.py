"""Literal search remains literal; only double-encoded array syntax is explained."""
import json
import socket
import subprocess

import pytest

from bridge_evidence.contest.document_tools import DocumentTools
from bridge_evidence.contest.evidence_manifest import manifest_from_text


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("No network or subprocess in search format tests")
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(subprocess, "Popen", blocked)


def tools(text='上部结构存在裂缝。下部结构存在剥落。', **kwargs):
    return DocumentTools(manifest_from_text(text), count_tokens=len, max_tool_tokens=4096, **kwargs)


def test_double_encoded_array_gets_hint_even_when_literal_has_no_matches():
    search = tools()
    query = '["上部结构", "下部结构"]'
    result = search.search_in_document(query)
    assert result['ok'] and result['total_matches'] == 0
    hint = result['query_format_hint']
    assert hint['received_type'] == 'string' and hint['search_interpretation'] == 'one_literal_phrase'
    assert hint['search_terms_changed'] is False
    assert hint['or_query_example'] == {'query': ['上部结构', '下部结构']}
    assert search.receipts[-1]['arguments']['query'] == query
    actual_array = search.search_in_document(hint['or_query_example']['query'])
    assert actual_array['total_matches'] == 2 and 'query_format_hint' not in actual_array
    assert result['grants_coverage'] is False and actual_array['grants_coverage'] is False
    assert search.receipts[-1]['ranges'] == []  # Search hits are not read receipts.


def test_json_looking_literal_matches_still_preserve_exact_text_and_offsets():
    query = '["上部结构", "下部结构"]'
    search = tools('前文 ' + query + ' 后文。上部结构。')
    result = search.search_in_document(query)
    assert result['total_matches'] == 1
    assert result['matches'][0]['matched_query'] == query
    assert result['matches'][0]['match_start'] == 3
    assert result['matches'][0]['match_end'] == 3 + len(query)
    assert result['query_format_hint']['search_terms_changed'] is False


def test_native_response_budget_and_compact_source_keep_hint_inside_budget():
    search = tools(native_response_budget=True, compact_model_source=True)
    result = search.search_in_document('["上部结构", "下部结构"]')
    assert result['ok'] and result['query_format_hint']['or_query_example']['query'] == ['上部结构', '下部结构']
    assert search.receipts[-1]['result_token_scope'] == 'native_response'
    assert search.receipts[-1]['result_tokens'] <= search.max_tool_tokens


@pytest.mark.parametrize('query', [
    ['上部结构', '下部结构'], '上部结构', '上部结构 下部结构', '上部结构,下部结构',
    "['上部结构', '下部结构']", '["上部结构",]', '[]', '[NaN]', '[Infinity]',
    '[{"a":"one","a":"two"}]', '["上部结构", 1]', '{"query":["上部结构"]}', '"上部结构"',
])
def test_regular_phrases_arrays_or_invalid_json_are_not_guessed(query):
    result = tools().search_in_document(query)
    assert result['ok'] and 'query_format_hint' not in result


def test_hint_never_turns_whitespace_or_overlong_arrays_into_invalid_executable_examples():
    for query in ('[""]', '[" "]', json.dumps(['上部结构'] * 17, ensure_ascii=False)):
        result = tools().search_in_document(query)
        assert result['ok'] and 'array_constraints' in result['query_format_hint']
        assert 'or_query_example' not in result['query_format_hint']
        assert result['total_matches'] == 0


def test_large_example_is_omitted_without_truncating_or_changing_search_phrase():
    query = json.dumps(['x' * 10000])
    result = tools().search_in_document(query)
    assert result['ok'] and result['total_matches'] == 0
    hint = result['query_format_hint']
    assert hint['example_omitted_for_budget'] is True and 'or_query_example' not in hint
    assert hint['search_terms_changed'] is False


def test_hint_preserves_scope_and_continuation_of_repeated_literal_matches():
    query = '["上部结构", "下部结构"]'
    search = tools(query + '\n' + query + '\n' + query)
    first = search.search_in_document(query, scope='document', max_matches=1)
    assert first['total_matches'] == 3 and len(first['matches']) == 1
    assert first['continuation']['arguments'] == {'query': query, 'cursor': 1, 'scope': 'document', 'max_matches': 1}
    second = search.search_in_document(**first['continuation']['arguments'])
    assert second['total_matches'] == 3 and second['cursor'] == 1
    assert second['query_format_hint'] == first['query_format_hint']
