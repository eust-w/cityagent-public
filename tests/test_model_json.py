"""Strict model transport syntax; never salvage JSON from prose or malformed data."""

import pytest


@pytest.mark.parametrize("text", [
    '{"value":1}', ' \n {"value":1} \t',
    '```json\n{"value":1}\n```', '```\n{"value":1}\n```',
    ' \n```JSON\n{"value":1}\n``` \n',
])
def test_object_or_single_complete_fence(text):
    from bridge_evidence.contest.model_json import parse_json_object

    assert parse_json_object(text) == {"value": 1}


@pytest.mark.parametrize("text", [
    None, 1, '', '[]', 'null', '1',
    'before {"value":1}', '{"value":1} after', '{"value":1} {}',
    '```json\n{"value":1}\n``` suffix', 'prefix ```json\n{"value":1}\n```',
    '```json\n{"value":1}\n', '```python\n{"value":1}\n```',
    '```json\n{"value":1}\n```\n```json\n{}\n```',
    '{"value":1,"value":2}', '{"nested":{"value":1,"value":2}}',
    '{"value":NaN}', '{"value":Infinity}', '{"value":-Infinity}',
    '{"value":1e999}', '{"value":1,}', '```json\n{"value":1,}\n```',
])
def test_junk_ambiguity_and_nonfinite_data_are_rejected(text):
    from bridge_evidence.contest.model_json import parse_json_object

    with pytest.raises(ValueError):
        parse_json_object(text)


@pytest.mark.parametrize("text, expected", [
    ('{"rows":[{"a":"x"}]', {"rows": [{"a": "x"}]}),
    ('{"rows":[{"a":"x"', {"rows": [{"a": "x"}]}),
    ('```json\n{"a":{"b":"}"}\n```', {"a": {"b": "}"}}),
])
def test_missing_final_closers_completed_only_when_enabled(text, expected):
    from bridge_evidence.contest.model_json import parse_json_object

    assert parse_json_object(text, close_unbalanced=True) == expected
    with pytest.raises(ValueError):
        parse_json_object(text)


@pytest.mark.parametrize("text", [
    '{"a":"unterminated', '{"a":[1}', '{"a":1,', '{"a":', '{"a":1}}',
    'prefix {"a":1', '{"a":1,"a":2',
])
def test_close_unbalanced_never_salvages_damaged_content(text):
    from bridge_evidence.contest.model_json import parse_json_object

    with pytest.raises(ValueError):
        parse_json_object(text, close_unbalanced=True)
