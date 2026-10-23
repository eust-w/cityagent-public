"""MinerU -> bounded section materials with explicit DOCX provenance.

MinerU supplies text, native titles and rich table content. OOXML supplies only
coordinates/header flags and source lookup. No bridge facts or missing words
are synthesized. The neutral graph is an internal representation, not Docling.
"""
import collections
import hashlib
import json
from pathlib import Path
import re
import zipfile
from lxml import etree as E, html
from .source_docx import NS, norm, xml_tables
from .section_materials import make_materials, TOC, numbered_level, short_title


def flatten(item):
    if isinstance(item, list): return ''.join(flatten(x) for x in item)
    if isinstance(item, str): return item
    if not isinstance(item, dict): return ''
    return flatten(item.get('content', ''))


def prepare(document, docx, *, budget=6000):
    docx = Path(docx)
    digest = hashlib.sha256(docx.read_bytes()).hexdigest()
    with zipfile.ZipFile(docx) as archive:
        tree = E.fromstring(archive.read('word/document.xml'))
    source_tables = list(xml_tables(tree).values())
    toc_targets = set()
    for field in tree.xpath('//w:instrText[contains(., "PAGEREF")]', namespaces=NS):
        match = re.search(r'PAGEREF\s+([^\s\\]+)', field.text or '')
        if match: toc_targets.add(match[1])
    bookmark_paragraphs = {}
    source_paragraphs = collections.defaultdict(list)
    for p in tree.xpath('//w:p', namespaces=NS):
        # Textboxes may contain paragraphs nested inside other paragraphs.
        text = ''.join(t.text or '' for t in p.xpath('.//w:t', namespaces=NS)
                       if t.xpath('ancestor::w:p[1]', namespaces=NS)[0] is p)
        path = tree.getroottree().getpath(p)
        if norm(text): source_paragraphs[norm(text)].append(path)
        anchors = {b.get('{'+NS['w']+'}name') for b in p.xpath('.//w:bookmarkStart',namespaces=NS)
                   if b.xpath('ancestor::w:p[1]',namespaces=NS)[0] is p} & toc_targets
        if anchors: bookmark_paragraphs[path] = sorted(anchors)
    graph = {'body': {'self_ref': '#/body', 'children': []}, 'texts': [], 'tables': [], 'groups': [], 'pictures': []}
    packets, sources, table_pairs, warnings = [], {}, [], []
    text_id = 0
    table_index = 0

    def add_text(text, locator, *, heading=False, level=1, navigation=False, scope=None):
        nonlocal text_id
        ref = f'#/texts/{text_id}'; text_id += 1
        node = {'self_ref': ref, 'text': text, 'children': [], 'label': 'section_header' if heading else 'text', 'level': level or 1}
        if navigation:
            match = TOC.match(text.strip())
            node.update(navigation=True, toc_title=match[1] if match else text.strip())
        graph['texts'].append(node)
        paths = source_paragraphs.get(norm(text), []) if norm(text) else []
        if scope:
            paths = [p for p in paths if p.startswith(scope['xml_cell_path'] + '/')]
        else:
            paths = [p for p in paths if '/w:tbl' not in p]
        if not heading and not navigation and len(paths)==1 and paths[0] in bookmark_paragraphs and short_title(text) and numbered_level(text) is not None:
            # Follow an explicit TOC target even when the displayed TOC title
            # is stale/shortened. Keep the actual body title unchanged.
            node.update(label='section_header',level=numbered_level(text),heading_basis='toc_bookmark_target')
            node['toc_target_anchors']=bookmark_paragraphs[paths[0]]
        # Multiple exact occurrences remain explicitly ambiguous. No page guess.
        sources[ref] = {'parser_locator': locator, 'xml_paragraph_candidates': paths,
                        'paragraph_match': 'unique_exact' if len(paths) == 1 else 'ambiguous_exact' if paths else 'not_exact',
                        **(scope or {})}
        if node.get('toc_target_anchors'):
            sources[ref]['toc_target_anchors']=node['toc_target_anchors']
        return ref

    def make_table(node, locator, parent=None):
        nonlocal table_index
        ti = table_index; table_index += 1
        if ti >= len(source_tables): raise ValueError('More MinerU tables than source tables')
        raw = source_tables[ti]; ref = f'#/tables/{ti}'
        target = {'self_ref': ref, 'children': [], 'data': {'table_cells': []}}
        graph['tables'].append(target)
        rows = node.xpath('./tr|./thead/tr|./tbody/tr|./tfoot/tr')
        if len(rows) != raw['rows']: raise ValueError(f"{raw['source_table_id']}: row count mismatch")
        source_by_row = collections.defaultdict(list)
        for c in raw['cells']: source_by_row[c['row']].append(c)
        cells, exact_anchors, grid_offsets = [], 0, []
        occupied = set()
        for ri, row in enumerate(rows):
            htmlcells = row.xpath('./td|./th')
            original = source_by_row[ri]
            if len(htmlcells) != len(original):
                raise ValueError(f"{raw['source_table_id']} row {ri}: cell count mismatch {len(htmlcells)} != {len(original)}")
            ci = 0
            for hi, (cell, src) in enumerate(zip(htmlcells, original)):
                while (ri, ci) in occupied: ci += 1
                rs, cs = int(cell.get('rowspan', 1)), int(cell.get('colspan', 1))
                if (rs, cs) != (src['row_span'], src['col_span']):
                    raise ValueError(f"{raw['source_table_id']} r{ri}c{src['column']}: unverified span {rs,cs} != {src['row_span'],src['col_span']}")
                occupied.update((r, c) for r in range(ri, ri+rs) for c in range(ci, ci+cs))
                if ci != src['column']:
                    grid_offsets.append({'row': ri, 'parsed_column': ci, 'source_column': src['column']})
                cid = f"{ref}/r{ri}/c{src['column']}"
                scope = {'source_table_id': raw['source_table_id'], 'xml_table_path': raw['xml_path'],
                         'row': ri, 'column': src['column'], 'row_span': rs, 'col_span': cs,
                         'parsed_column': ci, 'xml_cell_path': src['xml_cell_path'],
                         'match': 'same_order_row_cell_count_and_exact_spans'}
                sources[cid] = {'parser_locator': locator+f'/row:{ri}/cell:{hi}', **scope}
                parts = []
                buf = []
                current_heading = [False, 1]
                block_count = [0]
                def flush():
                    if not buf: return
                    text = ''.join(buf); buf.clear()
                    p_ref = add_text(text, sources[cid]['parser_locator']+f'/part:{block_count[0]}',
                                     heading=current_heading[0], level=current_heading[1], scope=scope)
                    parts.append({'kind': 'text', 'text': text, 'ref': p_ref, 'block_ref': p_ref})
                    block_count[0] += 1
                def visit(element):
                    if element.tag == 'table':
                        flush()
                        nested_ref = make_table(element, locator+f'/row:{ri}/cell:{hi}/nested:{len(parts)}', {'table': ref, 'cell': cid})
                        parts.append({'kind': 'table_ref', 'ref': nested_ref}); return
                    block = element.tag in ('p','li','div','caption','h1','h2','h3','h4','h5','h6')
                    previous = current_heading[:]
                    if block:
                        flush(); current_heading[:] = [bool(re.fullmatch(r'h[1-6]', element.tag)), int(element.tag[1]) if re.fullmatch(r'h[1-6]', element.tag) else 1]
                    if element.text: buf.append(element.text)
                    for child in element:
                        visit(child)
                        if child.tail: buf.append(child.tail)
                    if block:
                        flush(); current_heading[:] = previous
                if cell.text: buf.append(cell.text)
                for child in cell:
                    visit(child)
                    if child.tail: buf.append(child.tail)
                flush()
                text = ''.join(p['text'] for p in parts if p['kind'] == 'text')
                if norm(text) == norm(src['text']) and norm(src['text']): exact_anchors += 1
                cells.append({'id': cid, 'row': ri, 'column': src['column'], 'row_span': rs, 'col_span': cs, 'parts': parts})
                target['data']['table_cells'].append({'start_row_offset_idx': ri, 'start_col_offset_idx': src['column']})
                ci += cs
        if exact_anchors < min(2, sum(bool(norm(c['text'])) for c in raw['cells'])):
            raise ValueError(f"{raw['source_table_id']}: insufficient exact text anchors; refuse to guess source mapping")
        header_rows = raw['explicit_header_rows']
        opening = [c for c in cells if c['row'] in header_rows] if header_rows else [c for c in cells if c['row'] == 0]
        meta = {'table': ref, 'parent': parent, 'header_rows': header_rows, 'opening_context': opening,
                'source_table': raw['source_table_id'], 'source_xml_path': raw['xml_path']}
        packets.append({'metadata': meta, 'units': cells})
        sources[ref] = {'parser_locator': locator, 'source_table_id': raw['source_table_id'], 'xml_table_path': raw['xml_path']}
        table_pairs.append({'ref': ref, 'source_table': raw['source_table_id'], 'exact_text_anchors': exact_anchors,
                            'cells': len(cells), 'grid_offsets': grid_offsets})
        return ref

    def add_body(item, locator, navigation=False):
        kind = item.get('type'); value = item.get('content')
        navigation = navigation or kind == 'index'
        if kind == 'table_body':
            tree_html = html.fragment_fromstring(value, create_parent='div')
            for i, table in enumerate(tree_html.xpath('.//table[not(ancestor::table)]')):
                ref = make_table(table, locator+f'/html-table:{i}')
                graph['body']['children'].append({'$ref': ref})
        elif kind in ('table','list','index') and isinstance(value, list):
            for i, child in enumerate(value): add_body(child, locator+f'/content:{i}', navigation)
        else:
            text = flatten(item)
            ref = add_text(text, locator, heading=kind == 'paragraph_title' and not navigation,
                           level=item.get('level', 1), navigation=navigation)
            if kind in ('header', 'footer'):
                sources[ref].update(xml_paragraph_candidates=[], paragraph_match='header_footer_part_not_mapped')
            graph['body']['children'].append({'$ref': ref})
            packets.append({'metadata': {}, 'units': [{'id': ref, 'text': text, 'char_range': [0,len(text)]}]})

    for page in document['pages']:
        for bi, item in enumerate(page.get('blocks', [])):
            add_body(item, f"page:{page['page_idx']}/block:{item.get('index',bi)}")
    if table_index != len(source_tables): raise ValueError('Source tables were not all consumed')
    result = make_materials(graph, packets, budget=budget)
    result.update(id='sha256:'+digest, docx_sha256=digest, parser='MinerU 4.0.12 with native-span repair',
                  source_map=sources, table_mapping=table_pairs)
    counts = collections.Counter(v.get('paragraph_match') for v in sources.values() if 'paragraph_match' in v)
    result['diagnostics'].update(source_table_count=len(source_tables), source_mapping_counts=dict(counts),
                                 tables_with_grid_offsets=sum(bool(x['grid_offsets']) for x in table_pairs))
    result['diagnostics']['known_limits'] += ['Character budgets are not target-model token budgets.',
        'Logical page indices are not Word printed page numbers.',
        'Ambiguous or non-exact paragraph source matches remain explicit; table cell geometry is checked.',
        'Materials preserve parser content; this is not a semantic factual-quality audit.']
    for packet in result['packets']:
        packet.update(report_id=result['id'], docx_sha256=digest)
    if hashlib.sha256(docx.read_bytes()).hexdigest() != digest: raise RuntimeError('Source changed during material preparation')
    return result
