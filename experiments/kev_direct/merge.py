"""Feed actual Kev decisions to the unchanged TongAgents main-review workflow."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

from bridge_evidence.contest.document_material_run import prepare_material_run
from bridge_evidence.contest.document_multiagent import import_candidates
from bridge_evidence.contest.document_merge_progress import build_handoff_evidence
from bridge_evidence.contest.document_multiagent_worker import make_client
from bridge_evidence.contest.document_speed_workflow import SpeedMergeWorkflow
from bridge_evidence.contest.document_workflow import DocumentSettings
from bridge_evidence.contest.run_state import atomic_json,json_digest
from compact_workflow import CompactReviewMixin, PROMPT as COMPACT_PROMPT
from dedup_workflow import DedupReviewMixin
from deployment import existing_path, port_number, service_settings


class KevMainWorkflow(DedupReviewMixin, CompactReviewMixin, SpeedMergeWorkflow):
    def _reader_system_prompt(self):
        if getattr(self, 'deduplicate_review', False):
            return super()._reader_system_prompt()
        if getattr(self, 'compact_review', False):
            return COMPACT_PROMPT
        return ('本轮没有子Agent。下面沿用的“子Agent/分工/交接”接口接收的是预训练Kev-4B的选择结果，'
            '不是另一个生成模型写出的事实。Kev判断原文片段分类并从原文候选中选择字段；程序逐字复制描述，'
            '因此一条候选可能含多个对象，也可能误分类、漏字段或混入历史。你负责阅读附带原文、审核事实关联，'
            '按需局部修改。未确认的字段保留未知，遗漏如实记录，不能把候选数量当作覆盖证明。'
            '本轮仅官方训练报告。保留Word导出、实际文件自查和finish_report。\n'
            + ('本轮采用聚焦候选：先一次select_basic_info为每个已有字段选择有依据的值，不能只选桥名和日期。'
               '接着排除标题、引导句和明确无病害的误报；核对同对象重复、构件与类型错配以及逐病害历史字段。'
               '候选附带历史表行时，按上一次/本次/发展状况的列关系核对；无法确认保留未知。'
               '审核时可同轮提出多个本类小批工具调用，分别处理病害、建议；不写长篇分析。'
               if getattr(self, 'focused_kev_review', False) else '')
            + super()._reader_system_prompt())


def snapshot(port, require_idle=True):
    port=port_number(port)
    with urllib.request.urlopen(f'http://127.0.0.1:{port}/metrics',timeout=12) as f:
        metrics=f.read().decode()
    gauges=[l for l in metrics.splitlines() if l.startswith(('vllm:num_requests_running','vllm:num_requests_waiting'))]
    assert gauges, 'Service did not expose request gauges'
    idle=all(float(l.rsplit(' ',1)[1])==0 for l in gauges)
    if require_idle:
        assert idle,'Existing requests still active; do not interrupt them'
    with urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models',timeout=12) as f:models=json.load(f)
    processes=[]
    for p in Path('/proc').glob('[0-9]*'):
        try:
            raw=(p/'cmdline').read_bytes();args=raw.decode().split('\0')
            if 'vllm.entrypoints.openai.api_server' in args and '--port' in args and args[args.index('--port')+1]==str(port):
                processes.append({'pid':int(p.name),'cmdline_sha256':hashlib.sha256(raw).hexdigest(),
                    'start_ticks':(p/'stat').read_text().rsplit(')',1)[1].split()[19]})
        except (OSError,UnicodeError,ValueError,IndexError):pass
    assert len(processes)==1
    return {'process':processes[0], 'models':models, 'idle_gauges':gauges, 'idle':idle}


def run(folder,candidate_path,directory,tokenizer_path,plan_path,preflight=False,focused_review=False,
        *,main_port=None,main_model=None,summary_port=None,summary_model=None,
        compact_review=False,deduplicate_review=False):
    selected=service_settings(main_port,main_model,summary_port,summary_model)
    main_port,summary_port=selected['main_port'],selected['summary_port']
    existing_path(str(tokenizer_path))
    if deduplicate_review and not compact_review:
        raise ValueError('Deduplication review requires the full compact candidate handoff')
    started=time.monotonic();directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    folder=Path(folder);manifest=json.loads((folder/'source-projection.json').read_bytes())
    materials=json.loads((folder/'materials.json').read_bytes());candidate=json.loads(Path(candidate_path).read_bytes())
    plan=json.loads(Path(plan_path).read_bytes())
    assert candidate['document_sha256']==manifest['document_sha256']
    assert candidate['manifest_sha256']==json_digest(manifest) and candidate['subagent_calls']==0
    assert plan['document_sha256']==manifest['document_sha256'] and plan['manifest_sha256']==json_digest(manifest)
    before={str(p):snapshot(p) for p in (main_port,summary_port)}
    atomic_json(directory/'service-before.json',before)
    os.environ['CITYAGENT_STOP_OBSERVER_SERVER_PID']=str(before[str(main_port)]['process']['pid'])
    os.environ['CITYAGENT_SUMMARY_STOP_OBSERVER_SERVER_PID']=str(before[str(summary_port)]['process']['pid'])
    model=selected['main_model']
    assert model in {m['id'] for m in before[str(main_port)]['models']['data']}, 'Explicit main model is not served; refuse a silent fallback'
    summary=selected['summary_model']
    assert summary in {m['id'] for m in before[str(summary_port)]['models']['data']}, 'Explicit summary model is not served; refuse a silent fallback'
    config={'base_url':f'http://127.0.0.1:{main_port}','model':model,'enable_thinking':False,
        'thinking_tokens':4096,'capture_raw_sse':True,'diagnostic_token_ids':True,
        'native_repetition_detection':True,'raw_guard_tokenizer_path':str(tokenizer_path),
        'raw_guard_tokenizer_sha256':hashlib.sha256(Path(tokenizer_path).read_bytes()).hexdigest(),
        'summary':{'base_url':f'http://127.0.0.1:{summary_port}','model':summary},'input_scope':'official-training'}
    deadline=time.monotonic()+900
    client=make_client(config,deadline,capture_directory=directory/'merge/raw-sse')
    summary_client=make_client(config,deadline,summary=True,capture_directory=directory/'merge/summary-raw-sse')
    material_index,_=prepare_material_run(manifest,materials,folder/'source.docx',directory)
    task={'id':'kev-prefill','headings':[{'title':'MinerU材料上的预训练Kev候选，零子Agent'}],
          'primary_evidence_ids':[u['id'] for u in manifest['units']]}
    entry_state={'identity':{'task_id':task['id'],'manifest_sha256':json_digest(manifest),
                            'material_identity':deepcopy(material_index.identity)},
        'reads':candidate['reads'],'entries':{e['id']:e for e in candidate['entries']}}
    result={'task_id':task['id'],'status':'completed','entries':entry_state,'model_calls':0,
        'generation_attempts_settled':True,'handoff':{'unresolved':candidate['unresolved'],
            'summary':'Program wrapper around actual pretrained pointer decisions; no child Agent ran.'}}
    evidence=build_handoff_evidence(manifest,[result],material_index=material_index)
    # Supply the same pinned MinerU context to the merger. This is original
    # source navigation, not an invented alias-to-fact mapping or model claim.
    refs_to_packet={}
    for group in plan['groups']:
        for item in group['candidates']:
            for ref in item['source_refs']:
                refs_to_packet[(ref['evidence_id'],ref['start'],ref['end'])]=group['packet']
    units={u['id']:u['text'] for u in manifest['units']}
    for item in evidence.values():
        packets=dict.fromkeys(refs_to_packet[(r['evidence_id'],r['start'],r['end'])]
                             for r in item['source_refs'])
        for packet_id in (() if compact_review else packets):
            context=material_index.packet(packet_id)['context']
            spans=[]
            for span in context:
                if span['start']==span['end']:continue
                assert span['text']==units[span['evidence_id']][span['start']:span['end']]
                spans.append(deepcopy(span))
            item.setdefault('material_evidence',[]).append({
                'identity':deepcopy(material_index.identity),
                'source_ref':{'packet_id':packet_id},'context':spans,
                'source_metadata':{'kind':'pinned_mineru_packet_context'},
                'context_is_supplemental_original_text':True,
                'context_is_not_an_additional_model_claim':True})
    child_summary=[{'task_id':task['id'],'status':'completed','primary_evidence_ids':task['primary_evidence_ids'],
        'headings':['Kev候选审核'],'entry_count':len(candidate['entries']),'handoff':result['handoff'],
        'source':'Pretrained Kev-4B decision outputs; ZERO child Agent sessions'}]
    settings=DocumentSettings(max_model_len=262144,output_tokens=262144,dynamic_output_budget=True,
        single_document_agent=True,require_semantic_review=False,allow_quality_gaps=True,
        enable_thinking=False,temperature=1.0,report_timeout=900,max_model_calls=32 if deduplicate_review else 16,
        native_context_compression=True,compression_trigger_tokens=196608,max_sessions=1,
        session_iterations=1048576,session_calls=1048576,workspace_bootstrap=False)
    workflow=KevMainWorkflow(manifest,client,{'url':client.base_url,'model':client.model,'api_key':client.api_key,
        'timeout':900,'enable_thinking':False,'temperature':1.0},directory/'merge',settings=settings,
        external_deadline=deadline,compression_client=summary_client,child_summary=child_summary,
        handoff_evidence=evidence,material_index=material_index,direct_delivery=True,
        run_identity={'candidate_provider':'pretrained-kev4b-no-child-agents','direct_delivery':True,
            'speed_priority':True,'input_scope':'official-training','model_config_sha256':json_digest(config)})
    workflow.focused_kev_review = focused_review
    workflow.compact_review = compact_review
    workflow.deduplicate_review = deduplicate_review
    atomic_json(directory/'review-policy.json',{'deduplicate_review':deduplicate_review,
        'max_model_calls':settings.max_model_calls,'single_request_output_cap':workflow.main_request_output_cap,
        'reference_answers_provided':False,'candidate_sha256':hashlib.sha256(Path(candidate_path).read_bytes()).hexdigest()})
    atomic_json(directory/'source-manifest.json',manifest)
    atomic_json(directory/'handoff-evidence.json',evidence)
    atomic_json(directory/'candidate-import.json',import_candidates(workflow.workspace,[result],tasks=[task]))
    if preflight:
        _,prompt=workflow._entry()
        receipt={'preflight_passed':True,'model_requests':len(workflow._model_calls()),
            'candidate_count':len(candidate['entries']),'prompt_characters':len(prompt),
            'material_identity':material_index.identity,
            'counts':{s:len(workflow.workspace.snapshot()['draft'][s]) for s in ('基本信息','病害明细','维修建议')}}
        assert receipt['model_requests']==0
        atomic_json(directory/'preflight.json',receipt)
        print(json.dumps(receipt,ensure_ascii=False));return receipt
    outcome,error=None,None
    try:outcome=workflow.run()
    except Exception as exc:
        error={'type':type(exc).__name__,'message':str(exc)}
        if hasattr(exc,'receipt'):atomic_json(directory/'failure-receipt.json',exc.receipt)
    finally:atomic_json(directory/'parent-client-calls.json',workflow._model_calls())
    delivery=deepcopy(workflow.artifact_delivery)
    complete=outcome is not None and error is None
    if complete:
        assert delivery.get('complete') is True and delivery.get('incomplete') is False
        assert workflow.artifact_manager.matches_snapshot(workflow.workspace.snapshot(),delivery)
        assert hashlib.sha256(Path(delivery['docx_path']).read_bytes()).hexdigest()==delivery['file_sha256']
    # An unsuccessful request may still be settling; preserve its real status
    # instead of losing the result or claiming disconnect proved cancellation.
    after={str(p):snapshot(p, require_idle=False) for p in (main_port,summary_port)}
    assert all(before[p]['process']==after[p]['process'] for p in before)
    atomic_json(directory/'service-after.json',after)
    calls=[c for c in workflow._model_calls() if c.get('endpoint')=='/v1/chat/completions']
    summary_result={'complete':complete,'error':error,'subagent_calls':0,'main_seconds':time.monotonic()-started,
        'kev_total_seconds':candidate['total_seconds'],'kev_load_seconds':candidate['load_seconds'],
        'kev_inference_seconds':candidate['total_seconds']-candidate['load_seconds'],
        'kev_decision_requests':candidate['decision_requests'],'kev_input_tokens':candidate['input_tokens'],
        'main_model':model,'main_calls':len(calls),
        'main_input_tokens':sum((c.get('usage') or {}).get('prompt_tokens',0) for c in calls),
        'main_output_tokens':sum((c.get('usage') or {}).get('completion_tokens',0) for c in calls),
        'draft_counts':{s:len(workflow.workspace.snapshot()['draft'][s]) for s in ('基本信息','病害明细','维修建议')},
        'delivery':delivery,'services_unchanged':True,'exhaustive_quality_verified':False}
    atomic_json(directory/'result.json',summary_result)
    print(json.dumps(summary_result,ensure_ascii=False))
    return summary_result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-folder',required=True);p.add_argument('--candidates',required=True);p.add_argument('--output',required=True);p.add_argument('--tokenizer',required=True);p.add_argument('--plan',required=True);p.add_argument('--preflight',action='store_true')
    p.add_argument('--focused-review',action='store_true')
    p.add_argument('--main-port',type=port_number,required=True);p.add_argument('--main-model',required=True)
    p.add_argument('--summary-port',type=port_number,required=True);p.add_argument('--summary-model',required=True)
    p.add_argument('--compact-review',action='store_true')
    p.add_argument('--deduplicate-review',action='store_true')
    a=p.parse_args();run(a.source_folder,a.candidates,a.output,a.tokenizer,a.plan,a.preflight,a.focused_review,
                        main_port=a.main_port,main_model=a.main_model,
                        summary_port=a.summary_port,summary_model=a.summary_model,compact_review=a.compact_review,
                        deduplicate_review=a.deduplicate_review)
