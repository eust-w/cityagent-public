"""Section-aware packing over the MinerU-derived neutral graph and source-mapped cells.

No facts are extracted or inferred. Native headings and exact TOC matches define
boundaries; numbering alone only proposes a review candidate. Original payload
segments, coordinates and source refs remain reconstructable.
"""
import collections,copy,hashlib,json,re

CN='一二三四五六七八九十百零〇'
TOC=re.compile(r'^(.*?)\s*(?:\t+|\.{2,}|…{2,})\s*[-—－]?\s*(?:\d+|[ivxIVX]+)\s*[-—－]?\s*$')
def compact(s):return re.sub(r'\s+','',s)
def short_title(s):
    return bool(s.strip()) and len(s.strip())<=100 and '\n' not in s.strip() and not re.search(r'[。；;！？!?]',s.strip().rstrip('。'))
def numeric_path(s):
    m=re.match(r'^(\d{1,2}(?:\.\d{1,2}){0,4})\s*[\u4e00-\u9fffA-Za-z]',s.strip())
    return tuple(m[1].split('.')) if m else None
def numbered_level(text,chapter_mode=False):
    s=text.strip()
    if not short_title(s):return None
    if re.match(r'^第['+CN+r'0-9]+[章节篇]\s*\S',s):return 1
    if re.match(r'^附[录表件](?:['+CN+r'0-9A-Za-z]+)?[：:\s]*\D',s):return 1
    m=re.match(r'^(\d{1,2}(?:\.\d{1,2}){0,4})\s*([\u4e00-\u9fffA-Za-z].*)$',s)
    if m and not re.match(r'^(?:年|月|日|m(?:m)?\b|cm\b|MPa\b|kN\b)',m[2]):return m[1].count('.')+1
    if re.match(r'^['+CN+r']+[、\s]+\S',s):return 2 if chapter_mode else 1
    if re.match(r'^[（(]['+CN+r']+[）)]\s*\S|^[㈠㈡㈢㈣㈤㈥㈦㈧㈨㈩]\s*\S',s):return 3 if chapter_mode else 2
    return None

def payload_inventory(packets):
    """Recover the old exact primary parts, independently of their chunking."""
    bodies=collections.defaultdict(list);cells=collections.defaultdict(list);metas={}
    for packet in packets:
        if 'table' in packet['metadata']:metas.setdefault(packet['metadata']['table'],copy.deepcopy(packet['metadata']))
        for u in packet['units']:
            (cells if 'parts' in u else bodies)[u['id']].append(u)
    body={ref:''.join(u['text'] for u in sorted(us,key=lambda u:u['char_range'][0])) for ref,us in bodies.items()}
    result={}
    for cid,us in cells.items():
        sample=us[0];out={k:copy.deepcopy(sample[k]) for k in ['id','row','column','row_span','col_span']}
        if len(us)==1 and 'part_index' not in sample:out['parts']=copy.deepcopy(sample['parts'])
        else:
            bypart=collections.defaultdict(list)
            for u in us:bypart[u['part_index']].append(u)
            parts=[]
            for pi,portions in sorted(bypart.items()):
                part=copy.deepcopy(portions[0]['parts'][0])
                if part['kind']=='text':part['text']=''.join(u['parts'][0]['text'] for u in sorted(portions,key=lambda u:u.get('char_range',[0])[0]))
                parts.append(part)
            out['parts']=parts
        result[cid]=out
    return body,result,metas

def make_materials(document,old_packets,budget=6000):
    body,cells,table_metas=payload_inventory(old_packets)
    objects={x['self_ref']:x for key in ['texts','tables','groups','pictures'] for x in document.get(key,[])}
    objects[document['body']['self_ref']]=document['body']
    table_cells=collections.defaultdict(list)
    for c in cells.values():table_cells[c['id'].split('/r')[0]].append(c)
    for cs in table_cells.values():cs.sort(key=lambda c:(c['row'],c['column']))
    records=[];seen=set();visited_tables=set()
    def mark_rich(ref):
        if ref.startswith('#/tables/'):return
        if ref in seen:return
        seen.add(ref)
        for child in objects.get(ref,{}).get('children',[]):mark_rich(child['$ref'])
    def cell_records(c):
        if not c['parts']:
            yield {'key':c['id']+'/empty','kind':'empty_cell','text':'','offset':[0,0],'cell':c['id'],'block':c['id']}
        for pi,part in enumerate(c['parts']):
            common={'key':c['id']+f'/part{pi}','kind':part['kind'],'cell':c['id'],'part_index':pi,
                    'ref':part.get('ref'),'block':part.get('block_ref') or part.get('ref') or c['id']+f'/part{pi}'}
            if part['kind']=='table_ref':yield {**common,'text':'','offset':[0,0]};continue
            text=part['text']
            # Plain cells sometimes contain several unstyled paragraphs. Keep
            # exact newlines/offsets while giving each line a heading opportunity.
            lines=text.splitlines(keepends=True) if 'ref' not in part else [text]
            cursor=0
            for li,line in enumerate(lines or ['']):
                yield {**common,'text':line,'offset':[cursor,cursor+len(line)],
                       'block':common['block']+(f'/line{li}' if len(lines)>1 else '')}
                cursor+=len(line)
    def walk(ref):
        if ref in seen:return
        seen.add(ref);obj=objects.get(ref,{})
        if ref.startswith('#/tables/'):
            visited_tables.add(ref)
            for c in table_cells[ref]:
                native=next((x for x in obj['data']['table_cells'] if x['start_row_offset_idx']==c['row'] and x['start_col_offset_idx']==c['column']),None)
                if native and 'ref' in native:mark_rich(native['ref']['$ref'])
                for record in cell_records(c):
                    record['table']=ref;records.append(record)
                    if record['kind']=='table_ref':walk(record['ref'])
            return
        if ref in body:records.append({'key':ref,'kind':'text','text':body[ref],'offset':[0,len(body[ref])],'ref':ref,'block':ref})
        for child in obj.get('children',[]):walk(child['$ref'])
    walk(document['body']['self_ref'])
    if set(table_cells)!=visited_tables:raise ValueError('Unreachable tables: '+repr(set(table_cells)-visited_tables))
    # Group inline pieces of the same paragraph, but never group across a nested
    # table or a different cell. This avoids losing split/bold heading words.
    blocks=[]
    for r in records:
        key=(r.get('cell'),r['block'])
        if blocks and blocks[-1]['key']==key and r['kind']=='text' and all(x['kind']=='text' for x in blocks[-1]['records']):
            blocks[-1]['records'].append(r);blocks[-1]['text']+=r['text']
        else:blocks.append({'key':key,'text':r['text'],'records':[r]})
    toc={};toc_blocks=set()
    for i,b in enumerate(blocks):
        record=b['records'][0]
        source_obj=objects.get(record.get('block'),objects.get(record.get('ref'),{}))
        if source_obj.get('navigation'):
            toc_blocks.add(i)
            title=source_obj.get('toc_title',b['text']).strip()
            if short_title(title):toc.setdefault(compact(title),[]).append(i)
        m=TOC.match(b['text'].strip())
        if m and short_title(m[1]) and numbered_level(m[1]) is not None:
            toc.setdefault(compact(m[1]),[]).append(i);toc_blocks.add(i)
    chapter_mode=any(re.match(r'^第['+CN+r'0-9]+章',b['text'].strip()) for b in blocks)
    sections=[{'id':'S0000','parent':None,'title':'未分节前置材料','headings':[],'basis':'preamble','source':None,'region':'content','level':0}]
    stack=[];current=sections[0];navigation=False;candidates=[]
    def source(b):
        r=b['records'][0]
        return {k:r[k] for k in ['key','ref','cell','table','offset'] if k in r}
    for i,b in enumerate(blocks):
        title=b['text'].strip();r=b['records'][0];obj=objects.get(r.get('block'),objects.get(r.get('ref'),{}))
        is_toc=i in toc_blocks or compact(title)=='目录'
        if is_toc:
            if not navigation:
                current={'id':f'S{len(sections):04d}','parent':None,'title':'目录导航','headings':[],
                   'basis':'toc_navigation','source':source(b),'region':'navigation','level':0}
                sections.append(current);navigation=True
        else:
            if navigation:current=sections[0];stack=[];navigation=False
            native=obj.get('label')=='section_header' and short_title(title)
            confirmed=compact(title) in toc and short_title(title)
            level=numbered_level(title,chapter_mode)
            # Captions/years/standard IDs do not become headings by number alone.
            if native or confirmed:
                level=level or obj.get('level',1)
                if not isinstance(level,int) or level<1:level=1
                stack=[s for s in stack if s['level']<level]
                warning=None;number=numeric_path(title);prior_parent=stack[-1] if stack else None
                if number and len(number)>1:
                    # A missing/erroneous number must not silently attach 8.3.1
                    # to 8.2. Keep the nearest compatible ancestor, never invent
                    # an absent 8.3 heading or silently correct the source title.
                    stack=[s for s in stack if not numeric_path(s['title']) or
                           (len(numeric_path(s['title']))<len(number) and number[:len(numeric_path(s['title']))]==numeric_path(s['title']))]
                    actual=numeric_path(stack[-1]['title']) if stack else None
                    if actual!=number[:-1]:
                        warning={'expected_parent_number':'.'.join(number[:-1]),
                                 'preceding_parent_title':prior_parent['title'] if prior_parent else None,
                                 'reason':'immediate_numbered_parent_missing_or_inconsistent',
                                 'action':'retain_source_title_and_only_compatible_ancestor; inspect_source_if_needed'}
                parent=stack[-1] if stack else None
                current={'id':f'S{len(sections):04d}','parent':parent['id'] if parent else None,
                    'title':title,'headings':[s['title'] for s in stack]+[title],
                    'basis':obj.get('heading_basis','native_heading') if native else 'exact_toc_match','source':source(b),
                    'toc_sources':[source(blocks[k]) for k in toc.get(compact(title),[])],
                    'region':'content','level':level,'native_level':obj.get('level') if native else None,
                    'hierarchy_warning':warning}
                sections.append(current);stack.append(current)
            elif level is not None:
                candidates.append({'title':title,'source':source(b),'reason':'numbered_text_without_native_heading_or_exact_toc_match','inherited_section':current['id']})
        for rec in b['records']:rec['section_id']=current['id']
    bysection={s['id']:s for s in sections}
    def describe_record(rec):
        if rec['kind']=='table_ref':return '↳'+rec['ref']
        return rec['text']
    def render(rs):
        section=bysection[rs[0]['section_id']];lines=[]
        if section['region']=='navigation':lines.append('目录导航（不是正文证据）：')
        elif section['headings']:lines.append('章节：'+' / '.join(section['headings']))
        else:lines.append('区域：未分节前置材料')
        if section.get('hierarchy_warning'):
            lines.append('层级待核对：未确认编号 '+section['hierarchy_warning']['expected_parent_number']+' 的直接父章节；保留原文编号。')
        table=rs[0].get('table')
        if table:
            meta=table_metas[table];lines.append('表 '+table+'；坐标从0起，合并跨度为行数×列数。')
            if meta.get('parent'):lines.append('所在单元格：'+meta['parent']['cell'])
            # External context may belong to an earlier section: label it as
            # shared background instead of silently making it current evidence.
            background=[]
            for u in meta.get('opening_context',[]):
                if 'parts' in u:
                    t=''.join(p['text'] if p['kind']=='text' else '↳'+p['ref'] for p in u['parts'])
                    background.append(f"r{u['row']}c{u['column']}: "+json.dumps(t,ensure_ascii=False))
                else:background.append('回查 '+u['cell_ref'])
            header='原DOCX重复表头' if meta.get('header_rows') else '表格开头背景，未确认表头'
            if sum(map(len,background))<=650:lines.append(header+'：'+'；'.join(background))
            elif background:lines.append(header+'：按单元格引用回查 '+', '.join(u.get('id',u.get('cell_ref','')) for u in meta['opening_context']))
            first=min(cells[x['cell']]['row'] for x in rs)
            for c in table_cells[table]:
                if c['row']<first<c['row']+c['row_span']:
                    t=''.join(p['text'] for p in c['parts'] if p['kind']=='text')
                    lines.append('跨行延续 '+c['id']+(': '+json.dumps(t,ensure_ascii=False) if len(t)<=160 else '（按引用回查）'))
        lines.append('本块原文：')
        groups=[]
        for r in rs:
            identity=r.get('cell') or r['key']
            if groups and groups[-1][0]==identity:groups[-1][1].append(r)
            else:groups.append((identity,[r]))
        for identity,items in groups:
            fragments=[];last_block=None
            for item in items:
                if fragments and item['block']!=last_block:fragments.append('\n')
                fragments.append(describe_record(item));last_block=item['block']
            label=identity
            if items[0].get('cell'):
                c=cells[identity];label=f"r{c['row']}c{c['column']}[{c['row_span']}x{c['col_span']}]"
            lines.append(label+': '+json.dumps(''.join(fragments),ensure_ascii=False))
        return '\n'.join(lines)
    packets=[];pending=[]
    def emit():
        nonlocal pending
        if not pending:return
        sec=bysection[pending[0]['section_id']]
        packets.append({'packet_id':f'P{len(packets):05d}','section_id':sec['id'],'headings':sec['headings'],
          'region':sec['region'],'table':pending[0].get('table'),'context':render(pending),'segments':pending})
        pending=[]
    for r in records:
        if pending and (r['section_id'],r.get('table'))!=(pending[0]['section_id'],pending[0].get('table')):emit()
        if pending and len(render(pending+[r]))>budget:emit()
        if len(render([r]))<=budget:pending.append(r);continue
        if r['kind']!='text':raise ValueError('Metadata exceeds budget for '+r['key'])
        cursor=0
        while cursor<len(r['text']):
            take=min(3000,len(r['text'])-cursor)
            piece={**r,'text':r['text'][cursor:cursor+take],'offset':[r['offset'][0]+cursor,r['offset'][0]+cursor+take]}
            while len(render([piece]))>budget and take>1:
                take//=2;piece['text']=r['text'][cursor:cursor+take];piece['offset'][1]=piece['offset'][0]+take
            if len(render([piece]))>budget:raise ValueError('Metadata leaves no text budget: '+r['key'])
            pending=[piece];emit();cursor+=take
    emit()
    expected={ref:('text',text) for ref,text in body.items()}
    for c in cells.values():
        if not c['parts']:expected[c['id']+'/empty']=('empty_cell','')
        for pi,part in enumerate(c['parts']):expected[c['id']+f'/part{pi}']=(part['kind'],part.get('text',part.get('ref')))
    observed=collections.defaultdict(list)
    for p in packets:
        for s in p['segments']:observed[s['key']].append(s)
    failures=[]
    for key,(kind,value) in expected.items():
        got=observed[key]
        if kind=='text':
            ordered=sorted(got,key=lambda s:s['offset'][0]);cursor=0
            for item in ordered:
                if item['offset'][0]!=cursor:failures.append(key+'/offset-gap-or-overlap')
                cursor=item['offset'][1]
            if ''.join(s['text'] for s in ordered)!=value:failures.append(key+'/text')
        elif len(got)!=1 or (kind=='table_ref' and got[0]['ref']!=value):failures.append(key+'/reference')
    failures+=list(set(observed)-set(expected))
    for sec in sections:sec['packet_ids']=[p['packet_id'] for p in packets if p['section_id']==sec['id']]
    diagnostics={'packets':len(packets),'emitted_chars':sum(len(p['context']) for p in packets),
      'max_chars':max((len(p['context']) for p in packets),default=0),
      'sections':sum(s['region']=='content' and s['level']>0 for s in sections),
      'native_headings':sum(s['basis']=='native_heading' for s in sections),
      'toc_recovered_headings':sum(s['basis']=='exact_toc_match' for s in sections),
      'toc_bookmark_headings':sum(s['basis']=='toc_bookmark_target' for s in sections),
      'in_table_headings':sum(s['level']>0 and bool((s['source'] or {}).get('cell')) for s in sections),
      'hierarchy_warnings':sum(bool(s.get('hierarchy_warning')) for s in sections),
      'unconfirmed_candidates':len(candidates),'source_segments':len(expected),
      'conservation_failures':failures,'body_refs':len(body),'cell_ids':len(cells),
      'known_limits':['Heading assignment is structural, not semantic accuracy.',
       'Number-only candidates inherit the last confirmed heading until reviewed.',
       'Images/OCR and document-specific role assignment are excluded.']}
    if failures: raise ValueError('Material conservation failed: '+repr(failures[:10]))
    if diagnostics['max_chars']>budget: raise ValueError('Material exceeds character budget')
    return {'packets':packets,'sections':sections,'unconfirmed_candidates':candidates,'diagnostics':diagnostics}
