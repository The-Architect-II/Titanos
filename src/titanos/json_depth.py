"""JSON nesting bound for request bodies. Standard library only.

The server's Guard refuses a JSON body nested deeper than JSON_MAX_DEPTH with a typed 400 before any route parses it.
CPython's own parser gives up at a platform-dependent depth (about 3000 on Windows and 10000 on Linux for 3.12, far less
on 3.11 inside a deep request stack), so without a fixed bound the same body was a RecursionError 500 on one OS and
accepted on another."""
import itertools
import re

JSON_MAX_DEPTH = 256
_STRING = re.compile(rb'"[^"\\]*(?:\\.[^"\\]*)*"', re.S)
_SPACE = re.compile(rb'[ \t\r\n]*')
_NOT_BRACKETS = bytes(b for b in range(256) if b not in b'[]{}')
_STEP = [0] * 256
for _byte, _delta in ((ord('['), 1), (ord('{'), 1), (ord(']'), -1), (ord('}'), -1)):
    _STEP[_byte] = _delta


def json_too_deep(body, limit=JSON_MAX_DEPTH):
    """True when body (bytes or bytearray) is a JSON document (first non-space byte '[' or '{') whose array/object
    nesting exceeds limit. Brackets inside strings do not count. Bodies with no more than limit opening brackets are
    never scanned."""
    start = _SPACE.match(body).end()
    if body[start:start + 1] not in (b'[', b'{') or body.count(b'[') + body.count(b'{') <= limit:
        return False
    brackets = _STRING.sub(b'', body).translate(None, _NOT_BRACKETS)
    return max(itertools.accumulate(map(_STEP.__getitem__, brackets)), default=0) > limit


__all__ = ['JSON_MAX_DEPTH', 'json_too_deep']
