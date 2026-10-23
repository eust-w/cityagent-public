"""Synthetic fixtures only: candidate parsing must never confer verification."""
import hashlib
import importlib
import importlib.util
from dataclasses import replace

import pytest

M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def api():
    name = "bridge_evidence.contest.native_math"
    assert importlib.util.find_spec(name) is not None, "OMML candidate API is missing"
    return importlib.import_module(name)


def part(body):
    return (f'<w:document xmlns:w="{W}" xmlns:m="{M}"><w:body>'
            f'<w:p>{body}</w:p></w:body></w:document>').encode()


def run(text):
    return f"<m:r><m:t>{text}</m:t></m:r>"


def math(body):
    return f"<m:oMath>{body}</m:oMath>"


def test_run_preserves_exact_bytes_unicode_whitespace_and_occurrence_identity():
    fragment = '<m:oMath >' + run('α &amp; β') + '</m:oMath>'
    source = part(fragment + fragment)
    result = api().extract_omml_candidates(source)
    assert result["inventory_complete"] is True
    assert result["source_xml_sha256"] == hashlib.sha256(source).hexdigest()
    a, b = result["occurrences"]
    assert [a["occurrence_index"], b["occurrence_index"]] == [0, 1]
    assert a["source_xml_span"] != b["source_xml_span"]
    assert a["raw_sha256"] == b["raw_sha256"] == hashlib.sha256(fragment.encode()).hexdigest()
    for occurrence in (a, b):
        span = occurrence["source_xml_span"]
        assert source[span["start"]:span["end"]] == occurrence["raw_xml"].encode() == fragment.encode()
        assert occurrence["namespace_context"]["m"] == M
        assert occurrence["status"] == "candidate"
        assert occurrence["candidate_ast"]["children"][0]["text"] == "α & β"
        assert occurrence["semantics_verified"] is False
        assert occurrence["glyph_fidelity"] == "unverified"


def test_subscript_keeps_base_subscript_order_and_explicit_control_properties():
    sub = ('<m:sSub><m:sSubPr><m:ctrlPr><w:rPr><w:rFonts w:ascii="Cambria Math"/>'
           '</w:rPr></m:ctrlPr></m:sSubPr><m:e>' + run('x') + '</m:e><m:sub>'
           + run('i') + '</m:sub></m:sSub>')
    result = api().extract_omml_candidates(part(math(run('A=') + sub + run('+1'))))
    occurrence = result['occurrences'][0]
    assert occurrence['status'] == 'candidate'
    children = occurrence['candidate_ast']['children']
    assert [x['type'] for x in children] == ['run', 'subscript', 'run']
    assert children[1]['base']['children'][0]['text'] == 'x'
    assert children[1]['subscript']['children'][0]['text'] == 'i'
    props = children[1]['properties'][0]['children'][0]['children'][0]
    assert props['children'][0]['attributes'] == {f'{{{W}}}ascii': 'Cambria Math'}


def test_run_properties_are_explicit_and_space_is_not_stripped():
    body = ('<m:r><m:rPr><m:sty m:val="b"/></m:rPr><w:rPr>'
            '<w:rFonts w:ascii="Symbol"/><w:i/><w:color w:val="000000"/>'
            '<w:kern w:val="0"/><w:sz w:val="21"/><w:lang w:val="en-US"/>'
            '</w:rPr><m:t xml:space="preserve"> x </m:t></m:r>')
    occurrence = api().extract_omml_candidates(part(math(body)))['occurrences'][0]
    assert occurrence['status'] == 'candidate'
    node = occurrence['candidate_ast']['children'][0]
    assert node['text'] == ' x '
    assert node['text_attributes'] == {'{http://www.w3.org/XML/1998/namespace}space': 'preserve'}
    assert len(node['properties']) == 2
    assert node['properties'][0]['children'][0]['attributes'] == {f'{{{M}}}val': 'b'}
    assert occurrence['glyph_fidelity'] == 'unverified'


@pytest.mark.parametrize('body', [
    '<m:f><m:num>' + run('a') + '</m:num><m:den>' + run('b') + '</m:den></m:f>',
    '<m:r surprise="1"><m:t>x</m:t></m:r>',
    '<m:r><m:t m:unknown="1">x</m:t></m:r>',
    '<m:r><m:rPr><m:sty m:val="future"/></m:rPr><m:t>x</m:t></m:r>',
    '<m:r><w:rPr><w:rFonts w:asciiTheme="majorAscii"/></w:rPr><m:t>x</m:t></m:r>',
    '<m:r><w:rPr><w:vanish/></w:rPr><m:t>x</m:t></m:r>',
    '<m:r><w:rPr><w:sz w:val="NaN"/></w:rPr><m:t>x</m:t></m:r>',
    '<m:r><m:t>x</m:t><m:t>y</m:t></m:r>',
    '<m:sSub><m:e>' + run('x') + '</m:e></m:sSub>',
    '<m:sSub><m:sub>' + run('i') + '</m:sub><m:e>' + run('x') + '</m:e></m:sSub>',
    '<m:sSub><m:e/><m:sub>' + run('i') + '</m:sub></m:sSub>',
    run('x') + 'unmodeled text',
    '<!-- unknown annotation -->' + run('x'),
    '<?instruction ignore?>' + run('x'),
])
def test_unknown_or_malformed_math_is_unresolved_without_losing_other_occurrences(body):
    source = part(math(body) + math(run('ok')))
    result = api().extract_omml_candidates(source)
    assert result['inventory_complete'] is True
    bad, good = result['occurrences']
    assert bad['status'] == 'unresolved' and bad['candidate_ast'] is None and bad['diagnostics']
    assert bad['raw_xml'] == math(body)
    assert good['status'] == 'candidate'
    assert not any(x['semantics_verified'] for x in result['occurrences'])


def test_nested_math_is_not_deduplicated_or_silently_flattened():
    result = api().extract_omml_candidates(part(math(run('x') + math(run('y')))))
    outer, inner = result['occurrences']
    assert outer['status'] == 'unresolved'
    assert inner['status'] == 'candidate'
    assert inner['ancestor_math_occurrence_index'] == 0


@pytest.mark.parametrize('blob,code', [
    (b'<!DOCTYPE x [<!ENTITY leak SYSTEM "file:///etc/passwd">]><x>&leak;</x>', 'XML_DTD_FORBIDDEN'),
    (b'<x>&undefined;</x>', 'INVALID_XML'),
    (b'<x>', 'INVALID_XML'),
    (b'<?xml version="1.0" encoding="ISO-8859-1"?><x/>', 'XML_ENCODING_UNSUPPORTED'),
    (b'\xff', 'XML_ENCODING_UNSUPPORTED'),
])
def test_unsafe_or_invalid_xml_never_claims_an_inventory(blob, code):
    result = api().extract_omml_candidates(blob)
    assert result['inventory_complete'] is False
    assert result['occurrences'] == [] and result['diagnostics'] == [code]


@pytest.mark.parametrize('field,value,code', [
    ('max_xml_bytes', 8, 'XML_BYTE_LIMIT'),
    ('max_depth', 2, 'XML_DEPTH_LIMIT'),
    ('max_nodes', 3, 'XML_NODE_LIMIT'),
    ('max_occurrences', 1, 'MATH_OCCURRENCE_LIMIT'),
])
def test_global_limits_fail_closed_without_partial_inventory(field, value, code):
    module = api()
    limits = replace(module.NativeMathLimits(), **{field: value})
    result = module.extract_omml_candidates(part(math(run('x')) * 2), limits=limits)
    assert not result['inventory_complete']
    assert result['occurrences'] == [] and result['diagnostics'] == [code]


def test_candidate_byte_limit_retains_the_complete_original_occurrence():
    module = api()
    source = part(math(run('x' * 80)))
    result = module.extract_omml_candidates(source, limits=replace(module.NativeMathLimits(), max_math_bytes=32))
    occurrence = result['occurrences'][0]
    assert result['inventory_complete'] and occurrence['status'] == 'unresolved'
    assert occurrence['diagnostics'] == ['MATH_BYTE_LIMIT']
    assert occurrence['raw_xml'] == math(run('x' * 80))


def test_nested_math_cannot_amplify_retained_raw_bytes_beyond_aggregate_budget():
    module = api()
    fragment = run('x' * 80)
    for _ in range(8):
        fragment = math(fragment)
    source = part(fragment)
    limits = replace(module.NativeMathLimits(), max_total_math_bytes=512)
    result = module.extract_omml_candidates(source, limits=limits)
    assert result['inventory_complete'] is False
    assert result['occurrences'] == []
    assert result['diagnostics'] == ['MATH_TOTAL_BYTE_LIMIT']


def test_repeated_namespace_metadata_respects_output_budget_without_partial_success():
    module = api()
    source = part(math(run('x')) * 3).replace(b'<w:document ', b'<w:document xmlns:extra="' + b'x' * 1024 + b'" ')
    limits = replace(module.NativeMathLimits(), max_output_bytes=2500)
    result = module.extract_omml_candidates(source, limits=limits)
    assert result['inventory_complete'] is False
    assert result['occurrences'] == []
    assert result['diagnostics'] == ['MATH_OUTPUT_BYTE_LIMIT']


@pytest.mark.parametrize('extra', [b'<!--x-->', b'<?x y?>'])
def test_node_budget_counts_comments_and_processing_instructions(extra):
    module = api()
    result = module.extract_omml_candidates(b'<r>' + extra * 3 + b'</r>',
                                          limits=replace(module.NativeMathLimits(), max_nodes=2))
    assert not result['inventory_complete']
    assert result['diagnostics'] == ['XML_NODE_LIMIT']


def test_unknown_math_namespace_is_retained_unresolved():
    source = b'<x:oMath xmlns:x="http://purl.oclc.org/ooxml/officeDocument/math"><x:r><x:t>x</x:t></x:r></x:oMath>'
    result = api().extract_omml_candidates(source)
    assert result['inventory_complete'] and len(result['occurrences']) == 1
    assert result['occurrences'][0]['status'] == 'unresolved'
    assert result['occurrences'][0]['diagnostics'] == ['UNSUPPORTED_MATH_NAMESPACE']


def test_output_budget_applies_to_empty_success_envelope():
    module = api()
    result = module.extract_omml_candidates(b'<r/>', limits=replace(module.NativeMathLimits(), max_output_bytes=1))
    assert not result['inventory_complete'] and result['occurrences'] == []
    assert result['diagnostics'] == ['MATH_OUTPUT_BYTE_LIMIT']


def test_empty_part_is_an_inventory_and_invalid_call_arguments_are_rejected():
    module = api()
    result = module.extract_omml_candidates(part(''))
    assert result['inventory_complete'] and result['occurrences'] == []
    with pytest.raises(TypeError):
        module.extract_omml_candidates('<x/>')
    with pytest.raises(ValueError):
        module.extract_omml_candidates(b'<x/>', limits=replace(module.NativeMathLimits(), max_depth=True))
