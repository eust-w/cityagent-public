"""Independent output observers. No bridge-fact rules and no business execution.

Input must be raw generated text, BEFORE the tool parser. This module does not
trim, repair or reinterpret tool arguments. An alarm invalidates the response.
"""
from collections import deque
from dataclasses import dataclass, asdict
import codecs
import json
import hashlib
from pathlib import Path
from types import MappingProxyType


@dataclass(frozen=True)
class Policy:
    fast_chars: int = 128
    slow_chars: int = 1024
    check_stride: int = 32
    fast_blank_fraction: float = .98
    slow_blank_fraction: float = .995
    fast_repeat_fraction: float = .90
    slow_repeat_fraction: float = .85
    ngram_chars: int = 4


class DualWindowGuard:
    def __init__(self, policy=None):
        self.policy = policy or Policy()
        if not self.policy.slow_chars >= self.policy.fast_chars >= 8:
            raise ValueError("Invalid dual-window policy")
        self.text = deque(maxlen=self.policy.slow_chars)
        self.blank = deque(maxlen=self.policy.slow_chars)
        self.position = 0
        self.in_string = False
        self.escaped = False
        self.depth = 0
        self.last_container_end = 0
        self.completed_containers = 0
        self.alarm = None

    @staticmethod
    def density(text, n):
        grams = [text[i:i+n] for i in range(len(text)-n+1)]
        return 1 - len(set(grams)) / len(grams) if grams else 0.0

    def feed(self, text):
        if self.alarm:
            return self.alarm
        p = self.policy
        for char in text:
            self.position += 1
            outside = not self.in_string
            self.blank.append(outside and char.isspace())
            self.text.append(char)
            if self.in_string:
                if self.escaped:
                    self.escaped = False
                elif char == '\\':
                    self.escaped = True
                elif char == '"':
                    self.in_string = False
            elif char == '"':
                self.in_string = True
            elif char in '[{':
                self.depth += 1
            elif char in ']}' and self.depth:
                self.depth -= 1
                self.completed_containers += 1
                self.last_container_end = self.position
            if len(self.text) < p.slow_chars or self.position % p.check_stride:
                continue
            flags = list(self.blank)
            fast_blank = sum(flags[-p.fast_chars:]) / p.fast_chars
            slow_blank = sum(flags) / p.slow_chars
            reason = None
            details = {'fast_blank_fraction': fast_blank,
                       'slow_blank_fraction': slow_blank}
            if fast_blank >= p.fast_blank_fraction and slow_blank >= p.slow_blank_fraction:
                reason = 'sustained_whitespace_outside_string'
            elif self.position - self.last_container_end >= p.slow_chars:
                slow = ''.join(self.text)
                # Pure whitespace inside a string is not declared meaningless.
                # Native N-gram detection has no such distinction; audit it separately.
                if any(c.isalnum() for c in slow[-p.fast_chars:]):
                    fast_density = self.density(slow[-p.fast_chars:], p.ngram_chars)
                    if fast_density >= p.fast_repeat_fraction:
                        slow_density = self.density(slow, p.ngram_chars)
                        details.update(fast_repeat_fraction=fast_density,
                                       slow_repeat_fraction=slow_density)
                        if slow_density >= p.slow_repeat_fraction:
                            reason = 'sustained_repetition_without_container_completion'
            if reason:
                self.alarm = {'reason': reason, 'character_offset': self.position,
                              'in_string': self.in_string, 'json_depth': self.depth,
                              'completed_containers': self.completed_containers,
                              'window_details': details, 'policy': asdict(p),
                              'tail': ''.join(self.text)[-240:]}
                return self.alarm
        return None


class RawTokenDecoder:
    """Incremental UTF-8 decoding of this model's actual ByteLevel BPE vocabulary.

    No per-token .decode() replacement characters and no cumulative O(n^2)
    decoding. Added special tokens retain their literal spelling.
    """
    def __init__(self, tokenizer_json=None, *, tokens=None):
        if tokens is not None:
            self.tokens = tokens
            self.reset()
            return
        data = json.loads(tokenizer_json)
        if data.get('decoder', {}).get('type') != 'ByteLevel' or data.get('model', {}).get('type') != 'BPE':
            raise ValueError("Raw generation protection requires a verified ByteLevel BPE tokenizer")
        # Construct the byte-to-unicode alphabet with an explicit counter.
        values = list(range(33,127)) + list(range(161,173)) + list(range(174,256))
        chars = values[:]
        missing = 0
        for b in range(256):
            if b not in values:
                values.append(b); chars.append(256 + missing); missing += 1
        inverse = {chr(c): b for b, c in zip(values, chars)}
        self.tokens = {i: bytes(inverse[c] for c in text)
                       for text, i in data['model']['vocab'].items()}
        for token in data.get('added_tokens', []):
            self.tokens[token['id']] = token['content'].encode('utf-8')
        self.reset()

    def reset(self):
        self.decoder = codecs.getincrementaldecoder('utf-8')(errors='strict')

    def feed(self, ids):
        return self.decoder.decode(b''.join(self.tokens[i] for i in ids), final=False)

    def finish(self):
        # A cancelled stream can legitimately end in the middle of a UTF-8 code point.
        pending, _ = self.decoder.getstate()
        return {'pending_utf8_bytes': len(pending)}



NATIVE_REPETITION_POLICY = {"min_pattern_size": 1, "max_pattern_size": 64, "min_count": 16}


class RawGenerationGuard:
    """Per-response observer of generated IDs, before server tool-parser projection.

    These IDs arrive over SSE after server parsing. Decoding reconstructs the
    original generated text; prompt_token_ids and parsed text are never mixed in.
    No parsed tool argument is repaired, and no business handler is called here.
    """
    def __init__(self, tokens, tokenizer_sha256):
        self.decoder = RawTokenDecoder(tokens=tokens)
        self.guard = DualWindowGuard()
        self.tokenizer_sha256 = tokenizer_sha256
        self.generated_tokens = 0
        self.decoded_characters = 0
        self.generated_sha256 = hashlib.sha256()
        self.tool_contract = None

    def configure_tool_parameters(self, tools):
        from .raw_tool_contract import RawToolContractGuard
        self.tool_contract = RawToolContractGuard(tools)

    def feed(self, token_ids):
        if type(token_ids) is not list or any(type(t) is not int or t < 0 for t in token_ids):
            raise ValueError("Generated token IDs must be a list of nonnegative integers")
        try:
            text = self.decoder.feed(token_ids)
        except (KeyError, UnicodeError) as exc:
            raise ValueError("Generated IDs do not match the pinned raw tokenizer") from exc
        self.generated_tokens += len(token_ids)
        self.decoded_characters += len(text)
        self.generated_sha256.update(text.encode('utf-8'))
        contract = self.tool_contract.feed(text) if self.tool_contract is not None else None
        if contract is not None:
            return {**contract, "channel": "raw_generated_tokens", "offset_unit": "unicode_code_points"}
        alarm = self.guard.feed(text)
        if alarm is None:
            return None
        # Full generated text stays in the private raw SSE capture; receipts
        # expose structural evidence and a hash instead of the generated tail.
        public = {k: v for k, v in alarm.items() if k != 'tail'}
        public.update(channel='raw_generated_tokens', offset_unit='unicode_code_points',
                      tail_sha256=hashlib.sha256(alarm['tail'].encode('utf-8')).hexdigest())
        return public

    def evidence(self):
        return {'tokenizer_sha256': self.tokenizer_sha256,
                'generated_tokens_observed': self.generated_tokens,
                'decoded_characters_observed': self.decoded_characters,
                'decoded_text_sha256': self.generated_sha256.hexdigest(),
                'policy': asdict(self.guard.policy), **self.decoder.finish()}

    def validate_complete(self, usage):
        if self.decoder.finish()['pending_utf8_bytes']:
            raise ValueError("Completed generation ended inside a UTF-8 code point")
        if self.generated_tokens != usage.get('completion_tokens'):
            raise ValueError("Generated token IDs do not cover the complete output usage")


class RawGenerationGuardFactory:
    """Validate/hash vocabulary once, allocate fresh decoding/window state per request."""
    def __init__(self, tokenizer_path, expected_sha256):
        if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
                or any(c not in '0123456789abcdef' for c in expected_sha256)):
            raise ValueError("A lowercase SHA256 for the serving model tokenizer is required")
        raw = Path(tokenizer_path).read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if actual != expected_sha256:
            raise ValueError("Raw guard tokenizer file SHA256 does not match the pinned tokenizer")
        decoder = RawTokenDecoder(raw)
        self.tokens = MappingProxyType(decoder.tokens)
        self.tokenizer_sha256 = actual

    def __call__(self):
        return RawGenerationGuard(self.tokens, self.tokenizer_sha256)
