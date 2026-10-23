"""Local OOXML projection and coordinate oracle; no parser/model dependency."""
import io,re,zipfile
from lxml import etree as E
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
Q='{'+NS['w']+'}'
def norm(s):return re.sub(r'\s+','',s or '')
def value(node,path):
    n=node.find(path,NS)
    return n.get(Q+'val') if n is not None else None

def source_projection(path):
    """In-memory text/table view, same scope as round 1; originals immutable."""
    out = io.BytesIO()
    with zipfile.ZipFile(path) as src, zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as dst:
        doc = E.fromstring(src.read('word/document.xml'))
        before = doc.xpath('//w:t/text()', namespaces=NS)
        removed = 0
        for node in list(doc.xpath('//w:drawing|//w:pict', namespaces=NS)):
            if node.getparent() is not None and not node.xpath('.//w:t', namespaces=NS):
                node.getparent().remove(node); removed += 1
        for node in doc.xpath('//*[local-name()="imagedata" or local-name()="blip"]'):
            if node.getparent() is not None:
                node.getparent().remove(node); removed += 1
        if before != doc.xpath('//w:t/text()', namespaces=NS):
            raise ValueError('Source projection changed original OOXML text')
        for name in src.namelist():
            dst.writestr(name, E.tostring(doc) if name == 'word/document.xml' else src.read(name))
    return out.getvalue(), {'image_references_removed':removed,'w_t_exactly_preserved':True,
                           'body_w_t_chars':sum(map(len,before))}

def xml_tables(doc):
    """Independent coordinate oracle incl. nested tables, wrappers and gridBefore."""
    tables={}
    for ti,table in enumerate(doc.xpath('//w:tbl', namespaces=NS)):
        path=table.getroottree().getpath(table)
        rows=table.findall('w:tr',NS);cells=[];active={};headers=[]
        for ri,row in enumerate(rows):
            flag=row.find('w:trPr/w:tblHeader',NS)
            if flag is not None and flag.get(Q+'val','true') not in ('0','false','off'):headers.append(ri)
            col=int(value(row,'w:trPr/w:gridBefore') or 0)
            for tc in row.xpath('./w:tc|./w:sdt/w:sdtContent/w:tc',namespaces=NS):
                span=int(value(tc,'w:tcPr/w:gridSpan') or 1)
                merge=tc.find('w:tcPr/w:vMerge',NS)
                continuation=merge is not None and merge.get(Q+'val','continue')=='continue'
                texts=[]
                for t in tc.xpath('.//w:t',namespaces=NS):
                    nearest=t.xpath('ancestor::w:tbl[1]',namespaces=NS)
                    if nearest and nearest[0] is table:texts.append(t.text or '')
                text=''.join(texts)
                if continuation and col in active:
                    active[col]['row_span']=ri+1-active[col]['row']
                    if norm(text):active[col].setdefault('continuation_text',[]).append({'row':ri,'text':text})
                else:
                    cell={'row':ri,'column':col,'row_span':1,'col_span':span,'text':text,
                          'xml_cell_path':tc.getroottree().getpath(tc)}
                    cells.append(cell);active[col]=cell
                col+=span
        tables[path]={'source_table_id':f'T{ti:04d}','xml_path':path,'rows':len(rows),
                      'columns':len(table.findall('w:tblGrid/w:gridCol',NS)),
                      'explicit_header_rows':headers,'cells':cells,
                      'nested_count':len(table.xpath('.//w:tbl',namespaces=NS)),
                      'text_chars':sum(len(c['text']) for c in cells)}
    return tables
