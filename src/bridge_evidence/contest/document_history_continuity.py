"""Exact append-only history, with explicitly evidenced native compression.

No request mutation, inference or implicit reset. Only a durable authorized
native memory replacement may replace a closed old prefix. All recent tool
cycles and every export inspection pair remain literal prior wire messages.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import threading

HISTORY_POLICY='exact_merged_messages_authorized_native_compression_v2'
def _canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')
def _digest(value):return hashlib.sha256(_canonical(value)).hexdigest()

class HistoryContinuityError(ValueError):
    def __init__(self,code,details):
        self.code=code;self.details=deepcopy(details)
        super().__init__('Continuous Agent history admission failed: '+code)

class HistoryPrefixGuard:
    def __init__(self):
        self._lock=threading.RLock();self._message_sha256s=();self._last_messages_sha256=None
        self._observations=0;self._receipts=[];self._poison=None;self._pending=None
        self._roles=[];self._tool_starts=[];self._protected_starts=[];self._system_hashes=[]
        self._tool_ids_by_start={}
    def _reject(self,code,details):
        if self._poison is None:self._poison={'code':code,'details':deepcopy(details)}
        raise HistoryContinuityError(code,details) from None
    def _ref(self,ref):
        if type(ref) is not dict or type(ref.get('path')) is not str or type(ref.get('sha256')) is not str:
            self._reject('COMPRESSION_REFERENCE_INVALID',{'transport_authorized':False})
        try:
            raw=Path(ref['path']).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=ref['sha256']:raise ValueError()
            return json.loads(raw)
        except Exception:self._reject('COMPRESSION_REFERENCE_INVALID',{'transport_authorized':False})
    @staticmethod
    def _has_exact_text(value, text):
        if isinstance(value, str):
            if value == text: return True
            try: return HistoryPrefixGuard._has_exact_text(json.loads(value), text)
            except (ValueError, TypeError): return False
        if isinstance(value, list): return any(HistoryPrefixGuard._has_exact_text(item,text) for item in value)
        if isinstance(value, dict): return any(HistoryPrefixGuard._has_exact_text(item,text) for item in value.values())
        return False
    @staticmethod
    def _wire_text(message):
        """Exact native OpenAI text shape; never treat images or extra parts as text."""
        content=message.get('content')
        if isinstance(content,str):return content
        if (isinstance(content,list) and content and all(type(part) is dict and
                set(part)=={'type','text'} and part.get('type')=='text' and
                isinstance(part.get('text'),str) for part in content)):
            return ''.join(part['text'] for part in content)
        return None
    def authorize_compression(self,transition,*,summary_text,exact_progress_text):
        """One-use permission tied to archived native memory and actual summary.

        Caller owns the native memory swap. Independent wire collectors may use
        this same contract after verifying those exact durable references.
        This is not permission for arbitrary model-generated history edits.
        """
        with self._lock:
            if self._poison is not None or self._pending is not None or self._observations<1:
                self._reject('COMPRESSION_AUTHORIZATION_STATE_INVALID',{'transport_authorized':False})
            if (type(transition) is not dict or transition.get('schema')!='cityagent-native-compression-transition-v1'
                    or transition.get('old_messages_sha256')!=self._last_messages_sha256
                    or transition.get('same_agent') is not True or transition.get('same_context') is not True
                    or type(transition.get('index')) is not int or transition['index']<1
                    or type(summary_text) is not str or not summary_text.strip()
                    or type(exact_progress_text) is not str or not exact_progress_text.strip()
                    or hashlib.sha256(summary_text.encode()).hexdigest()!=transition.get('summary_sha256')
                    or hashlib.sha256(exact_progress_text.encode()).hexdigest()!=transition.get('exact_progress_sha256')):
                self._reject('COMPRESSION_AUTHORIZATION_INVALID',{'transport_authorized':False})
            old=self._ref(transition.get('archived_history_ref'));new=self._ref(transition.get('replacement_memory_ref'))
            summary=self._ref(transition.get('summary_ref'));progress=self._ref(transition.get('exact_progress_ref'))
            if (summary.get('text')!=summary_text or progress.get('text')!=exact_progress_text
                    or not self._has_exact_text(new,summary_text) or not self._has_exact_text(new,exact_progress_text)
                    or type(old) is not list or type(new) is not list or type(transition.get('retained_native_start')) is not int
                    or not 0<transition['retained_native_start']<len(old)
                    or new[-(len(old)-transition['retained_native_start']):]!=old[transition['retained_native_start']:]):
                self._reject('COMPRESSION_NATIVE_SUFFIX_CHANGED',{'transport_authorized':False})
            starts=self._tool_starts[-2:]+self._protected_starts
            start=min(starts) if starts else 0
            if start<1:self._reject('COMPRESSION_NO_CLOSED_PREFIX',{'transport_authorized':False})
            self._pending={'transition':deepcopy(transition),'summary_text':summary_text,'progress_text':exact_progress_text,
                'retained_hashes':self._message_sha256s[start:],'retained_start':start}
    @staticmethod
    def _tool_structure(messages):
        calls={};results={};starts=[];protected=[]
        for index,message in enumerate(messages):
            for call in message.get('tool_calls') or []:
                key=call.get('id');name=(call.get('function') or {}).get('name')
                if not isinstance(key,str) or not key or key in calls:raise ValueError('call')
                calls[key]=(index,name);starts.append(index)
                if name=='inspect_delivery':protected.append(index)
            if message.get('role')=='tool':
                key=message.get('tool_call_id')
                if key not in calls or key in results:raise ValueError('result')
                results[key]=index
        if any(key not in results for key in calls):raise ValueError('unanswered')
        return sorted(set(starts)),sorted(set(protected))
    def compression_required_tool_call_ids(self):
        """Read-only native navigation for the exact suffix already protected.

        Rejected calls may be represented as ordinary text on the actual wire.
        Their native batches must not displace the recent real tool batches.
        Return every sibling ID at a protected batch start, without exposing or
        changing its content or weakening the subsequent exact hash check.
        """
        with self._lock:
            if self._poison is not None:
                self._reject('HISTORY_GUARD_POISONED',{'previous_failure_code':self._poison['code'],'transport_authorized':False})
            starts=set(self._tool_starts[-2:]+self._protected_starts)
            return tuple(key for start in sorted(starts) for key in self._tool_ids_by_start[start])
    def observe_complete_request(self,payload):
        with self._lock:
            if self._poison is not None:self._reject('HISTORY_GUARD_POISONED',{'previous_failure_code':self._poison['code'],'transport_authorized':False})
            if type(payload) is not dict or type(payload.get('messages')) is not list or not payload['messages'] or any(type(m) is not dict for m in payload['messages']):
                self._reject('HISTORY_MESSAGES_MALFORMED',{'transport_authorized':False})
            messages=payload['messages']
            try:current=tuple(_digest(m) for m in messages);current_sha=_digest(messages)
            except (TypeError,ValueError,OverflowError,RecursionError):self._reject('HISTORY_MESSAGES_NOT_CANONICAL_JSON',{'transport_authorized':False})
            old=self._message_sha256s;authorized=False;compression_index=None
            if self._pending is None:
                if current[:len(old)]!=old:
                    index=next((i for i,(a,b) in enumerate(zip(old,current)) if a!=b),min(len(old),len(current)))
                    self._reject('HISTORY_PREFIX_CHANGED',{'previous_message_count':len(old),'current_message_count':len(current),'first_changed_or_missing_index':index,'transport_authorized':False})
            else:
                pending=self._pending;retained=pending['retained_hashes']
                positions=[i for i in range(len(current)-len(retained)+1) if current[i:i+len(retained)]==retained]
                if len(positions)!=1:self._reject('COMPRESSION_ACTUAL_SUFFIX_CHANGED',{'transport_authorized':False})
                position=positions[0]
                old_start=pending['retained_start']
                # Native memory may retain one additional completed cycle added
                # since the previous wire. Preserve more literal old history,
                # never allow edits to those extra old messages.
                while position>0 and old_start>0 and current[position-1]==old[old_start-1]:
                    position-=1;old_start-=1
                before=messages[:position]
                if (not all(h in current[:positions[0]] for h in self._system_hashes)
                    or not any(self._wire_text(m)==pending['summary_text'] for m in before)
                    or not any(self._wire_text(m)==pending['progress_text'] for m in before)
                    or any(m.get('role') not in ('system','user') for m in before)
                    or any(_digest(m) not in self._system_hashes and
                        self._wire_text(m) not in (pending['summary_text'],pending['progress_text']) for m in before)):
                    self._reject('COMPRESSION_ACTUAL_PREFIX_NOT_BOUND',{'transport_authorized':False})
                authorized=True;compression_index=pending['transition']['index'];self._pending=None
            try:starts,protected=self._tool_structure(messages)
            except (TypeError,ValueError,KeyError):self._reject('HISTORY_TOOL_PAIR_INVALID',{'transport_authorized':False})
            receipt={'policy':HISTORY_POLICY,'observation_index':self._observations+1,'previous_message_count':len(old),
                'current_message_count':len(current),'appended_message_count':len(current)-len(old),
                'previous_messages_sha256':self._last_messages_sha256,'current_messages_sha256':current_sha,
                'prefix_preserved':not authorized,'authorized_compression':authorized,'compression_index':compression_index,
                'model_request_emitted':False,'semantic_verdict':False}
            self._message_sha256s=current;self._last_messages_sha256=current_sha;self._observations+=1
            self._tool_starts=starts;self._protected_starts=protected
            self._tool_ids_by_start={index:tuple(call['id'] for call in message.get('tool_calls') or [])
                for index,message in enumerate(messages) if message.get('tool_calls')}
            if not self._system_hashes:self._system_hashes=[_digest(m) for m in messages if m.get('role')=='system']
            self._receipts.append(deepcopy(receipt));return receipt
    @property
    def receipts(self):
        with self._lock:return deepcopy(self._receipts)
    def snapshot(self):
        with self._lock:return {'policy':HISTORY_POLICY,'observations':self._observations,'last_message_count':len(self._message_sha256s),
            'last_messages_sha256':self._last_messages_sha256,'poisoned':self._poison is not None,'failure':deepcopy(self._poison),
            'compression_authorization_pending':self._pending is not None,'contains_source_text':False}
