"""Exact complete-request output allocation, opt-in and independent of SDK estimates.

A token-count receipt is never semantic evidence. It binds an immutable full
chat request and preserves all text, tools, history and template parameters.
"""
from __future__ import annotations
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import inspect
import json
import threading

POLICY = "complete_request_remaining_context_v1"
CAPPED_POLICY = "complete_request_remaining_context_capped_v1"
_HOOK_LOCK = threading.RLock()

def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

def request_input_hash(payload):
    value = deepcopy(payload)
    value.pop("max_tokens", None)
    value.pop("max_completion_tokens", None)
    return hashlib.sha256(canonical(value)).hexdigest()

def remaining_output(context, input_tokens, output_cap=None):
    if type(context) is not int or context < 1 or type(input_tokens) is not int or input_tokens < 0:
        raise ValueError("Context and exact complete input count must be integers")
    if input_tokens >= context:
        raise ValueError("Complete input occupies the context; no clipping or fallback permitted")
    if output_cap is not None and (type(output_cap) is not int or output_cap < 1):
        raise ValueError("Per-request output cap must be a positive integer")
    return min(context - input_tokens, output_cap) if output_cap is not None else context - input_tokens

def tokenize_chat_payload(payload):
    """Project the installed vLLM0.19.1 TokenizeChatRequest rendering fields.

    The installed chat and tokenize paths share preprocess_chat. Sampler/cache
    metadata does not render prompt tokens. Unimplemented prompt transforms are
    rejected, rather than silently counted with another template.
    """
    if type(payload) is not dict or not isinstance(payload.get("messages"), list) or not payload["messages"]:
        raise ValueError("A complete chat request with messages is required")
    if payload.get("truncate_prompt_tokens") is not None or payload.get("prompt_embeds") is not None:
        raise ValueError("Prompt truncation/embeddings are unsupported by exact text allocation")
    choice = payload.get("tool_choice")
    tools = payload.get("tools") or []
    # Installed vLLM 0.19.1 render/serving.py renders messages + tool_dicts
    # before the parser applies the required/named JSON generation constraint.
    # Therefore these choices use the same tokenize input. Keep the full wire
    # choice in request_input_hash; never omit it from actual generation.
    if isinstance(choice, dict):
        if (set(choice) != {"type", "function"} or choice.get("type") != "function"
                or type(choice.get("function")) is not dict
                or set(choice["function"]) != {"name"}
                or choice["function"]["name"] not in {
                    t.get("function", {}).get("name") for t in tools}):
            raise ValueError("Named tool choice must identify a supplied function")
    elif choice == "required":
        if not tools:
            raise ValueError("Required tool choice needs supplied tools")
    elif choice == "none":
        if tools:
            raise ValueError("None with tools depends on server exclusion policy; remove tools explicitly")
    elif choice not in (None, "auto"):
        raise ValueError("Unsupported tool choice for exact text allocation")
    for message in payload["messages"]:
        if type(message) is not dict:
            raise ValueError("Invalid complete chat message")
        content = message.get("content")
        if isinstance(content, list) and any(type(p) is not dict or p.get("type") != "text" for p in content):
            raise ValueError("Multimodal allocation needs exact native processor admission")
    value = {"model": payload["model"], "messages": deepcopy(payload["messages"]),
             "add_generation_prompt": payload.get("add_generation_prompt", True),
             "continue_final_message": payload.get("continue_final_message", False),
             "add_special_tokens": payload.get("add_special_tokens", False)}
    for key in ("tools", "chat_template", "chat_template_kwargs", "media_io_kwargs", "mm_processor_kwargs"):
        if key in payload:
            value[key] = deepcopy(payload[key])
    if any(type(value[k]) is not bool for k in ("add_generation_prompt", "continue_final_message", "add_special_tokens")):
        raise ValueError("Chat rendering flags must be explicit booleans when supplied")
    if value["add_generation_prompt"] and value["continue_final_message"]:
        raise ValueError("Generation prompt and continuation cannot both be enabled")
    return value

@contextmanager
def allocate_sdk_wire_requests(base_url, prepare_request, project_request=None):
    """Apply allocation after actual OpenAI JSON/extra_body construction.

    SDK ModelConfig.max_tokens=None removes its static generation limit. This
    scoped, capability-checked hook fills every actual SDK HTTP generation,
    including SDK compression calls, before any transport. Gateway admission
    independently recounts the immutable body. Private SDK token estimates
    never authorize an output budget. Caller owns the native session lock.
    """
    if not callable(prepare_request):
        raise ValueError("Dynamic SDK requests require an allocation callback")
    if project_request is not None and not callable(project_request):
        raise ValueError("Protocol history projection must be callable")
    import httpx
    from openai._base_client import BaseClient
    original = BaseClient._build_request
    if tuple(inspect.signature(original).parameters) != ("self", "options", "retries_taken"):
        raise RuntimeError("Installed OpenAI request builder is unsupported; no fallback")
    target = base_url.rstrip("/") + "/chat/completions"
    def build(client, options, *, retries_taken=0):
        request = original(client, options, retries_taken=retries_taken)
        if request.url.path.endswith("/chat/completions"):
            if str(request.url) != target or request.method != "POST":
                raise ValueError("Dynamic SDK generation must use the pinned audited gateway")
            payload = json.loads(request.read())
            # A separately proved, rejected native tool pair can be represented
            # as ordinary text for chat templates that parse historical JSON.
            # Projection precedes BOTH history admission and exact counting;
            # allocation itself must still leave that complete input immutable.
            if project_request is not None:
                payload = project_request(deepcopy(payload))
                if type(payload) is not dict:
                    raise ValueError("Protocol history projection must return a complete request")
            prepared = prepare_request(deepcopy(payload))
            if (type(prepared) is not dict or request_input_hash(prepared) != request_input_hash(payload)
                    or "max_completion_tokens" in prepared or type(prepared.get("max_tokens")) is not int):
                raise ValueError("Dynamic allocation changed the complete model input")
            headers = request.headers.copy()
            headers.pop("content-length", None)
            request = httpx.Request(request.method, request.url, headers=headers,
                                    content=canonical(prepared), extensions=request.extensions)
        return request
    with _HOOK_LOCK:
        BaseClient._build_request = build
        try:
            yield
        finally:
            BaseClient._build_request = original
