"""Observe the pinned Qwen XML tool stream before parsed argument deltas.

This is a format budget, not a semantic/gibberish classifier. An alarm rejects
an entire response; this observer never repairs or executes partial parameters.
"""
import re


class StringCounter:
    """Count decoded JSON string code points across fragments/escape sequences."""
    def __init__(self, limit):
        self.limit = limit
        self.in_string = False
        self.escape = False
        self.unicode_left = 0
        self.count = 0
        self.maximum = 0
        self.pending_high = False

    def feed(self, text):
        for c in text:
            if not self.in_string:
                if c == '"':
                    self.in_string, self.count = True, 0
            elif self.unicode_left:
                self.unicode_value += c
                self.unicode_left -= 1
                if not self.unicode_left:
                    try:
                        value = int(self.unicode_value, 16)
                    except ValueError:
                        value = 0  # Invalid JSON is rejected independently.
                    if not (self.pending_high and 0xDC00 <= value <= 0xDFFF):
                        self.count += 1
                    self.pending_high = 0xD800 <= value <= 0xDBFF
            elif self.escape:
                self.escape = False
                if c == 'u':
                    self.unicode_left, self.unicode_value = 4, ''
                else:
                    self.count += 1
                    self.pending_high = False
            elif c == '\\':
                self.escape = True
            elif c == '"':
                self.in_string = False
            else:
                self.count += 1
                self.pending_high = False
            self.maximum = max(self.maximum, self.count)
            if self.count > self.limit:
                return True
        return False


class EncodedContainer:
    """Observe a JSON-string-encoded top-level container without executing it.

    The adapter supports this lossless representation. Its enclosing quotes
    must not cause the whole batch to be counted as one business text field.
    """
    def __init__(self):
        self.started = self.closed = self.escape = False
        self.hex_digits = None
        self.high = None

    def feed(self, text):
        out = []
        def emit(char):
            value = ord(char)
            if self.high is not None:
                if 0xDC00 <= value <= 0xDFFF:
                    out.append(chr(0x10000 + ((self.high-0xD800)<<10) + value-0xDC00))
                    self.high = None
                    return
                out.append(chr(self.high)); self.high = None
            if 0xD800 <= value <= 0xDBFF:
                self.high = value
            else:
                out.append(char)
        for char in text:
            if not self.started:
                if char == '"':
                    self.started = True
                continue
            if self.closed:
                continue
            if self.hex_digits is not None:
                self.hex_digits += char
                if len(self.hex_digits) == 4:
                    try:
                        emit(chr(int(self.hex_digits, 16)))
                    except ValueError:
                        pass  # Invalid JSON is never saved by the real parser.
                    self.hex_digits = None
            elif self.escape:
                self.escape = False
                if char == 'u':
                    self.hex_digits = ''
                else:
                    emit({'n': '\n', 'r': '\r', 't': '\t', 'b': '\b', 'f': '\f'}.get(char, char))
            elif char == '\\':
                self.escape = True
            elif char == '"':
                self.closed = True
                if self.high is not None:
                    out.append(chr(self.high)); self.high = None
            else:
                emit(char)
        return ''.join(out)


def max_string_limit(schema):
    limits = []
    if 'maxLength' in schema:
        limits.append(schema['maxLength'])
    if schema.get('type') == 'string' and not limits:
        return None
    for child in list(schema.get('properties', {}).values()) + ([schema['items']] if isinstance(schema.get('items'), dict) else []):
        limit = max_string_limit(child)
        if limit is None:
            return None
        limits.append(limit)
    # Property names are also JSON strings; they are permitted by the schema.
    limits.extend(len(k) for k in schema.get('properties', {}))
    return max(limits, default=0)


class RawToolContractGuard:
    def __init__(self, tools):
        self.schemas = {t['function']['name']: t['function']['parameters'] for t in tools}
        self.buffer = ''
        self.tool = self.parameter = None
        self.limit = None
        self.count = 0
        self.offset = 0
        self.alarm = None

    def feed(self, text):
        if self.alarm:
            return self.alarm
        self.offset += len(text)
        self.buffer += text
        while self.buffer:
            if self.parameter is None:
                match = re.search(r'<function=([^>\s]+)>|<parameter=([^>\s]+)>|</function>', self.buffer)
                if not match:
                    self.buffer = self.buffer[-128:]
                    break
                self.buffer = self.buffer[match.end():]
                if match.group(1):
                    self.tool = match.group(1)
                elif match.group(2):
                    self.parameter = match.group(2)
                    schema = self.schemas.get(self.tool, {}).get('properties', {}).get(self.parameter, {})
                    self.limit = max_string_limit(schema)
                    self.scalar = schema.get('type') == 'string'
                    self.mode = None
                    self.encoded = None
                    self.counter = StringCounter(self.limit or 10**12)
                    self.count = 0
                else:
                    self.tool = None
                continue
            closing = '</parameter>'
            end = self.buffer.find(closing)
            # Hold a possible fragmented closing tag outside the body counter.
            size = end if end >= 0 else max(0, len(self.buffer) - len(closing) + 1)
            body, self.buffer = self.buffer[:size], self.buffer[size:]
            if self.limit is not None and self.limit > 0 and body:
                if self.mode is None:
                    body = body.lstrip()
                    if body:
                        self.mode = 'json' if body.startswith('"') else 'plain'
                        if not self.scalar and self.mode == 'json':
                            self.encoded = EncodedContainer()
                exceeded = False
                if self.scalar and self.mode == 'plain':
                    self.count += len(body)
                    # Whitespace before the end marker is not part of JSON value.
                    # A small margin prevents cancelling legal framing whitespace.
                    exceeded = self.count > self.limit + 128
                else:
                    exceeded = self.counter.feed(self.encoded.feed(body) if self.encoded is not None else body)
                    self.count = self.counter.maximum
                if exceeded:
                    self.alarm = dict(kind='tool_argument_contract', reason='declared_text_field_budget_exceeded',
                        tool=self.tool, parameter=self.parameter, max_string_characters=self.limit,
                        observed_string_characters=self.count, raw_stream_characters_observed=self.offset,
                        action='Reject the complete generation; no partial arguments may execute')
                    return self.alarm
            if end >= 0:
                self.buffer = self.buffer[len(closing):]
                self.parameter = None
            else:
                break
        return None
