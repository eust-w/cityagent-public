"""Native TongAgents compression with durable exact state and atomic memory swap.

Model summary is semantic navigation, never authoritative tool state or evidence.
SDK imports are lazy; importing this module executes no SDK/model/HTTP operation.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from enum import Enum
import hashlib
import importlib
import json
from pathlib import Path
import time


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def plain(value):
    if isinstance(value, Enum): return value.value
    if value is None or type(value) in (str, int, float, bool): return value
    if isinstance(value, (list, tuple)): return [plain(item) for item in value]
    if isinstance(value, dict): return {str(key): plain(item) for key, item in value.items()}
    if is_dataclass(value): return plain(asdict(value))
    if callable(getattr(value, 'model_dump', None)): return plain(value.model_dump(mode='json'))
    if callable(getattr(value, 'dump', None)):
        dumped = value.dump()
        return {'native_type': type(value).__name__, 'native_dump': plain(dumped)}
    raise TypeError('Native message cannot be archived losslessly')


def save(path, value):
    from .document_workspace import atomic_json
    path = Path(path)
    atomic_json(path, value)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def load_native_api():
    memory = importlib.import_module('tongagents.memory.memory')
    messages = importlib.import_module('tongagents.agents.llm.messages')
    compression = importlib.import_module('tongagents.agents.llm_agent.context_compression')
    return {'memory': memory, 'messages': messages, 'compression': compression}


def serialize_prefix(api, messages):
    """Installed SDK V3 contract: exact native Serializable.dump wrappers."""
    memory = api['memory']
    result = []
    for message in messages:
        chunks = []
        for chunk in message.chunks:
            value = chunk.content
            if isinstance(value, str): item = {'format':'exact_text','content':value}
            elif callable(getattr(value, 'dump', None)):
                dumped = value.dump()
                if not isinstance(dumped, str): raise TypeError('Native Serializable.dump must return text')
                item = {'format':'native_serializable_dump','type':type(value).__name__,'content':dumped}
            else: raise TypeError('Unsupported native chunk cannot be silently dropped')
            item.update(require_upload=bool(getattr(chunk,'require_upload',False)), media_id=getattr(chunk,'media_id',None))
            chunks.append(item)
        item = {'original_message_id':message.id,'original_role':getattr(message.role,'value',str(message.role)),
            'timestamp':getattr(message,'timestamp',None),'meta_data':dict(message.meta_data),'chunks':chunks}
        result.append(memory.Message(chunks=[memory.MessageChunk(content=canonical(item).decode())],role=memory.MessageRole.USER))
    return result


def native_items(message):
    return [chunk.content for chunk in message.chunks]


def native_tool_pairs(messages):
    calls, answers = {}, {}
    for index, message in enumerate(messages):
        for content in native_items(message):
            for call in (getattr(content, 'tool_call_messages', None) or []):
                key = call.tool_call_id
                if not isinstance(key, str) or not key or key in calls:
                    raise ValueError('Native tool call ID is missing or reused')
                calls[key] = (index, call.tool_name)
            if type(content).__name__ == 'ToolResultMessage':
                key = content.tool_call_id
                if key in answers or key not in calls or content.name != calls[key][1]:
                    raise ValueError('Native tool result has no matching preceding call')
                answers[key] = index
    return calls, answers


def retained_boundary(messages, *, protected_tools=('inspect_delivery',), recent_cycles=3,
                      required_tool_call_ids=()):
    """A contiguous suffix preserving complete recent cycles and open calls.

    All export inspection pairs are retained verbatim, including stale ones.
    They are small and must not be replaced by a summary claiming they were read.
    """
    calls, answers = native_tool_pairs(messages)
    if (type(required_tool_call_ids) not in (list, tuple)
            or any(type(key) is not str or not key for key in required_tool_call_ids)
            or len(set(required_tool_call_ids)) != len(required_tool_call_ids)):
        raise ValueError('Actual wire protected tool identities are invalid')
    if any(key not in calls or key not in answers for key in required_tool_call_ids):
        raise ValueError('Actual wire protected tool pair is missing from native memory')
    completed = sorted(set(index for key, (index, _) in calls.items() if key in answers))
    starts = completed[-recent_cycles:]
    starts += [calls[key][0] for key in required_tool_call_ids]
    starts += [index for key, (index, name) in calls.items()
               if key not in answers or name in protected_tools]
    if not starts: return 0
    return min(starts)


def private_memory(api, messages, *, agent_id, session_id, directory):
    """Dedicated native StoreEngine; never clear/delete the old memory."""
    memory_api = api['memory']
    class PrivateStore(memory_api.StoreEngine):
        def __init__(self): self.rows = []
        def write(self, rows, *args, **kwargs):
            candidate = [*self.rows, *list(rows)]
            save(Path(directory) / 'active-native-history.json', plain(candidate))
            self.rows = candidate
            return True
        def read(self, *args, **kwargs):
            # The dedicated engine contains only this one document/Agent/session.
            if len(args) >= 2 and (args[0] != agent_id or args[1] != session_id):
                raise ValueError('Native memory query crosses document identity')
            recent = args[2] if len(args) > 2 else kwargs.get('recent_n')
            if recent is not None and (type(recent) is not int or recent < 1):
                raise ValueError('Invalid native memory recent_n')
            return self.rows[:] if recent is None else self.rows[-recent:]
        def clear(self): raise RuntimeError('Native history clearing is prohibited')
        def delete(self, *args, **kwargs): raise RuntimeError('Native history deletion is prohibited')
    engine = PrivateStore()
    memory = memory_api.PersistenceMemory(agent_id=agent_id, session_id=session_id, engine=engine)
    memory.add_messages(messages)
    if [plain(message) for message in memory.get_messages()] != [plain(message) for message in messages]:
        raise ValueError('New native memory did not read back exactly')
    return memory


class NativeContextCompression:
    def __init__(self, directory, *, summary_client, exact_progress, before_model,
                 refresh_progress, history_guard, trigger_tokens=196608, native_api=None, deadline=None, identity=None):
        self.directory = Path(directory)
        self.summary_client, self.exact_progress = summary_client, exact_progress
        self.before_model, self.refresh_progress = before_model, refresh_progress
        self.history_guard, self.trigger_tokens = history_guard, trigger_tokens
        self.native_api = native_api
        self.deadline = deadline
        self.identity = deepcopy(identity or {})
        self.agent_id = "cityagent-" + digest(self.identity)
        self.session_id = "document-" + digest({"directory":str(self.directory), "identity":self.identity})
        self.pending = False
        self.last_input_tokens = None
        self.transitions = []
        self.skips = []
        self.failure = None
        self._summary_calls_seen = 0

    def initial_memory(self):
        api = self.native_api or load_native_api()
        return private_memory(api, [], agent_id=self.agent_id, session_id=self.session_id, directory=self.directory / "initial-memory")

    def observe_complete_request(self, payload, actual_input_tokens):
        if type(actual_input_tokens) is not int or actual_input_tokens < 0:
            raise ValueError('Compression requires actual complete-request token admission')
        self.last_input_tokens = actual_input_tokens
        self.pending = actual_input_tokens >= self.trigger_tokens

    def _summary(self, native_prompt_messages, api):
        self._callback_invocations += 1
        if self._callback_invocations != 1:
            raise RuntimeError('Native compressor attempted another summary callback')
        prompt = []
        for item in native_prompt_messages:
            content = getattr(item, 'content', None)
            if isinstance(content, str): text = content
            elif isinstance(content, list):
                text = '\n'.join(part.text for part in content if isinstance(getattr(part, 'text', None), str))
                if len([part for part in content if isinstance(getattr(part, 'text', None), str)]) != len(content):
                    raise ValueError('Unsupported native compressor prompt content')
            else: raise ValueError('Native compressor prompt must be actual text')
            prompt.append({'role': 'user', 'content': text})
        self.before_model()
        start = len(self.summary_client.calls)
        timeout = self.summary_client.timeout
        try:
            if self.deadline is not None:
                remaining = self.deadline - time.monotonic()
                if remaining <= 0: raise TimeoutError("Summary shares the document deadline")
                self.summary_client.timeout = min(timeout, remaining)
            output = self.summary_client.generate([{'role': 'system', 'content':
                '你为同一个桥梁报告Agent压缩语义上下文。保留已理解的对象、年份、左右幅关联、未决事项与必要原文导航。'
                '只总结提供的历史，不补病害、原因、风险或建议；程序准确状态另行保存，摘要不可修改工具结果、事实ID、版本或续读位置。'
                '明确这只是语义笔记，不是原文证据，也不表示文件检查完成。'}, *prompt])
        finally:
            self.summary_client.timeout = timeout
            for call in self.summary_client.calls[start:]:
                call.update(model_role='summary', purpose='context_compression_summary')
        self.before_model(include_model=False)
        if output.get('status') != 'success' or output.get('finish_reason') != 'stop' or not output.get('text', '').strip():
            raise ValueError('Own summary model did not return complete nonempty native summary')
        self._callback_summary = output['text']
        self._callback_call_id = output.get('call_id')
        return api['messages'].ModelOutputMessage(
            model_text_message=api['messages'].ModelTextMessage(content=output['text']))

    def before_native_call(self, agent, supplied_messages, *, context=None):
        if self.failure is not None: raise RuntimeError('Prior native compression failed; original history retained')
        if not self.pending: return supplied_messages
        required_tool_call_ids = self.history_guard.compression_required_tool_call_ids()
        if not required_tool_call_ids:
            # A native rejected batch can exist while the actual wire contains
            # only ordinary text. The guard has no closed suffix to authorize
            # yet; preserve all memory and wait for a real observed tool cycle.
            skipped = {'schema': 'native-compression-skip-v1',
                'reason': 'no_authorizable_actual_wire_tool_prefix',
                'prior_actual_input_tokens': self.last_input_tokens,
                'trigger_tokens': self.trigger_tokens,
                'actual_messages_sha256': self.history_guard.snapshot()['last_messages_sha256'],
                'summary_requested': False, 'memory_rebound': False}
            save(self.directory / f'skipped-{len(self.skips) + 1:04d}.json', skipped)
            self.skips.append(skipped)
            self.pending = False
            return supplied_messages
        api = self.native_api or load_native_api()
        if context is None:
            context = agent.run_context  # only a caller-supplied native-context double may use this convenience
        old_memory = context.memory
        old = list(old_memory.get_messages())
        boundary = retained_boundary(old,
            required_tool_call_ids=required_tool_call_ids)
        if boundary < 1:
            # No safe closed prefix; do not erase source or leave a tool orphan.
            self.pending = False
            return supplied_messages
        index = len(self.transitions) + 1
        base = self.directory / f'transition-{index:04d}'
        archive = save(base / 'original-native-history.json', plain(old))
        self.before_model(include_model=False)
        self.refresh_progress()
        progress = self.exact_progress.snapshot(page_size=16)
        full_state = progress.get('full_state')
        if full_state is not None:
            state_source = Path(full_state['path'])
            raw_state = state_source.read_bytes()
            if hashlib.sha256(raw_state).hexdigest() != full_state['sha256']:
                raise ValueError('Exact progress snapshot changed before compression archive')
            state_archive = base / 'exact-authoritative-state.json'
            state_archive.parent.mkdir(parents=True, exist_ok=True)
            state_archive.write_bytes(raw_state)
            progress['full_state'] = {'path':str(state_archive), 'sha256':full_state['sha256']}

        progress_text = ('CITYAGENT_EXACT_SAVED_PROGRESS_V1\n' + canonical(progress).decode() +
            '\n这是程序准确保存状态和分页导航；不是模型推断。需要详细事实、原文或草稿时使用原工具按需回读。')
        progress_ref = save(base / 'exact-progress-payload.json', {'text': progress_text, 'snapshot': progress})
        stable_system = [message for message in old[:boundary] if any(
            type(content).__name__ == 'SystemPromptMessage' for content in native_items(message))]
        prefix = [message for message in old[:boundary] if message not in stable_system]
        if not prefix:
            self.pending = False
            return supplied_messages
        original_digest = digest(plain(old))
        self._callback_summary = None
        self._callback_invocations = 0
        try:
            # preserve=0: the selected removed prefix must ALL reach the summary model.
            # Exact native callback coverage is checked by focused installed SDK probe.
            # SDK default conversion omits native tool arguments. Preserve every
            # field using the same lossless archive representation, then send it
            # through the ACTUAL native compressor (installed V3 probe verified).
            memory_api = api['memory']
            serialization = serialize_prefix(api, prefix)
            save(base / 'compressor-input.json', plain(serialization))
            native_summary = api['compression'].ContextCompressor(min_messages_preserve=0).compress(
                serialization, llm_callable=lambda items: self._summary(items, api))
            summary_text = '\n'.join(content for content in native_items(native_summary) if isinstance(content, str))
            if (self._callback_invocations != 1 or self._callback_summary is None or summary_text != self._callback_summary
                    or getattr(native_summary.role, 'value', native_summary.role) != 'sys_summary'
                    or digest(plain(old)) != original_digest):
                raise ValueError('SDK summary fallback/mutation rejected; original history retained')
            messages_api, memory_api = api['messages'], api['memory']
            # Keep the actual native compressor result unchanged as evidence.
            # The Qwen chat template accepts only the original leading SYSTEM;
            # a native SYS_SUMMARY would be rendered as a second SYSTEM by SDK.
            native_summary_ref = save(base / 'native-compressor-summary.json', plain(native_summary))
            summary_message = memory_api.Message(chunks=[memory_api.MessageChunk(content=
                messages_api.UserPromptMessage(content=[messages_api.PromptTextContent(text=summary_text)]))],
                role=memory_api.MessageRole.USER)
            exact_message = memory_api.Message(chunks=[memory_api.MessageChunk(content=
                messages_api.UserPromptMessage(content=[messages_api.PromptTextContent(text=progress_text)]))],
                role=memory_api.MessageRole.USER)
            replacement = [*deepcopy(stable_system), summary_message, exact_message, *deepcopy(old[boundary:])]
            native_tool_pairs(replacement)

            summary_ref = save(base / 'semantic-summary.json', {'text': summary_text,
                'model_call_id': self._callback_call_id, 'source': 'own_summary_model', 'is_evidence': False})
            aid, sid = self.agent_id, self.session_id
            if not isinstance(aid, str) or not isinstance(sid, str):
                raise ValueError('Native persisted Agent/session identity missing')
            new_memory = private_memory(api, replacement, agent_id=aid, session_id=sid, directory=base / 'new-memory')
            replacement_ref = save(base / 'replacement-native-history.json', plain(new_memory.get_messages()))
            transition = {'schema': 'cityagent-native-compression-transition-v1', 'index': index,
                'old_messages_sha256': self.history_guard.snapshot()['last_messages_sha256'],
                'archived_history_ref': archive, 'replacement_memory_ref': replacement_ref,
                'summary_ref': summary_ref, 'exact_progress_ref': progress_ref,
                'native_summary_ref': native_summary_ref,
                'active_summary_message_role': 'user',
                'summary_sha256': hashlib.sha256(summary_text.encode()).hexdigest(),
                'exact_progress_sha256': hashlib.sha256(progress_text.encode()).hexdigest(),
                'summary_call_id': self._callback_call_id, 'retained_native_start': boundary,
                'retained_native_count': len(old) - boundary, 'prior_actual_input_tokens': self.last_input_tokens,
                'trigger_tokens': self.trigger_tokens, 'same_agent': True, 'same_context': True,
                'semantic_verdict': None, 'memory_rebound': False, 'new_messages_sha256': None}
            # Durable candidate/authorization precede changing the actual context.
            save(base / 'transition.json', transition)
            self.history_guard.authorize_compression(transition, summary_text=summary_text, exact_progress_text=progress_text)
            context.memory = new_memory
            if context.memory is not new_memory: raise ValueError('Native active memory replacement failed')
            new_input = agent._get_msg_from_memory()
            transition['memory_rebound'] = True
            transition['active_native_sha256'] = digest(plain(new_memory.get_messages()))
            self.transitions.append(transition)
            save(base / 'transition.json', transition)
            self.pending = False
            return new_input
        except Exception as error:
            # Never clear old memory or pretend a swallowed SDK fallback succeeded.
            context.memory = old_memory
            self.failure = type(error).__name__
            save(base / 'failed.json', {'schema': 'native-compression-failure-v1', 'error_type': self.failure,
                'original_history_ref': archive, 'original_memory_retained': context.memory is old_memory,
                'completed': False})
            raise

    def on_admitted_transition(self, receipt):
        if receipt.get('authorized_compression'):
            index = receipt['compression_index']
            if not 1 <= index <= len(self.transitions): raise ValueError('Compression index differs')
            transition = self.transitions[index - 1]
            transition['new_messages_sha256'] = receipt['current_messages_sha256']
            transition['wire_admission_receipt_sha256'] = digest(receipt)
            save(self.directory / f'transition-{index:04d}' / 'transition.json', transition)
