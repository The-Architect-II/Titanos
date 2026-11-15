"""Bounded reader for the key-value header of a GGUF file (standard library only, no model load).

Admission and context sizing need a few facts before llama-server starts: the trained context length, the layer and
attention-head counts that size the KV cache, and the bytes of the weights. The GGUF format (version 2 and 3, little
endian) stores them as typed key-value pairs after a fixed preamble: magic b'GGUF', uint32 version, uint64 tensor count,
uint64 key-value count, then each pair as a length-prefixed UTF-8 key, a uint32 value type and the value. Arrays are
skipped unless they hold at most MAX_KEPT_ARRAY numbers (per-layer head counts); strings longer than MAX_STRING_BYTES,
more than MAX_KEYS pairs or a header past MAX_HEADER_BYTES are refused rather than read.
"""
import mmap
import os
import struct
from pathlib import Path

MAGIC = b'GGUF'
MAX_KEYS = 65536
MAX_STRING_BYTES = 1 << 20
MAX_ARRAY_ITEMS = 1 << 24
MAX_KEPT_ARRAY = 4096
MAX_HEADER_BYTES = 256 * 1024 * 1024
# value type -> struct format (little endian) for the fixed-size scalars
_SCALARS = {0: '<B', 1: '<b', 2: '<H', 3: '<h', 4: '<I', 5: '<i', 6: '<f', 7: '<?', 10: '<Q', 11: '<q', 12: '<d'}
_STRING, _ARRAY = 8, 9
# bytes per KV-cache element for llama.cpp's --cache-type-k/-v values (block formats: block bytes / 32 elements)
CACHE_TYPE_BYTES = {'f32': 4.0, 'f16': 2.0, 'bf16': 2.0, 'q8_0': 34 / 32, 'q4_0': 18 / 32, 'q4_1': 20 / 32,
                    'iq4_nl': 18 / 32, 'q5_0': 22 / 32, 'q5_1': 24 / 32}


class _Reader:
    """Sequential little-endian reads over a read-only memory map, bounded by limit bytes from the start."""

    def __init__(self, view, limit):
        self.view, self.limit, self.offset = view, limit, 0

    def _advance(self, count):
        if count < 0 or self.offset + count > self.limit:
            raise ValueError('GGUF header is truncated or exceeds its bounded size')
        start, self.offset = self.offset, self.offset + count
        return start

    def take(self, count):
        start = self._advance(count)
        return bytes(self.view[start:start + count])

    def unpack(self, fmt):
        return struct.unpack_from(fmt, self.view, self._advance(struct.calcsize(fmt)))[0]

    def string(self, keep=True):
        size = self.unpack('<Q')
        if size > MAX_STRING_BYTES:
            raise ValueError('GGUF string exceeds its bounded size')
        if not keep:
            self._advance(size)
            return None
        return self.take(size).decode('utf-8', errors='replace')

    def value(self, kind):
        if kind in _SCALARS:
            return self.unpack(_SCALARS[kind])
        if kind == _STRING:
            return self.string()
        if kind != _ARRAY:
            raise ValueError(f'Unknown GGUF value type {kind}')
        item, count = self.unpack('<I'), self.unpack('<Q')
        if count > MAX_ARRAY_ITEMS:
            raise ValueError('GGUF array exceeds its bounded size')
        if item in _SCALARS:
            width = struct.calcsize(_SCALARS[item])
            start = self._advance(width * count)
            if count <= MAX_KEPT_ARRAY:
                return list(struct.unpack_from('<' + _SCALARS[item][1] * count, self.view, start))
            return None
        if item == _STRING:
            for _ in range(count):
                self.string(keep=False)
            return None
        raise ValueError('Nested or unknown GGUF array type')


def read_metadata(path, *, max_header_bytes=MAX_HEADER_BYTES):
    """{'version', 'tensor_count', 'metadata': {key: value}} from a GGUF file's header. Strings and numbers are kept,
    numeric arrays only up to MAX_KEPT_ARRAY items (longer arrays and string arrays are None). Raises ValueError for a
    file that is not a little-endian GGUF version 2 or 3, or whose header breaks a bound."""
    path = Path(path)
    with path.open('rb') as stream:
        size = os.fstat(stream.fileno()).st_size
        if size < 24:
            raise ValueError('Not a GGUF file')
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as view:
            reader = _Reader(view, min(max_header_bytes, size))
            if reader.take(4) != MAGIC:
                raise ValueError('Not a GGUF file')
            version = reader.unpack('<I')
            if version not in (2, 3):
                raise ValueError(f'Unsupported GGUF version or byte order ({version})')
            tensors, keys = reader.unpack('<Q'), reader.unpack('<Q')
            if keys > MAX_KEYS:
                raise ValueError('GGUF header has too many keys')
            metadata = {}
            for _ in range(keys):
                name = reader.string()
                metadata[name] = reader.value(reader.unpack('<I'))
    return {'version': version, 'tensor_count': tensors, 'metadata': metadata}


def _positive(value):
    return value if type(value) is int and value > 0 else None


def model_facts(path, *, files=()):
    """The facts admission and context sizing use, from one GGUF header: architecture, context_length (the trained
    context), block_count, embedding_length, head_count, head_count_kv (an int, or a per-layer list), key_length,
    value_length, and weight_bytes (the size of path plus any extra shard or projector files). A fact the header
    does not state is None."""
    header = read_metadata(path)['metadata']
    arch = header.get('general.architecture') if isinstance(header.get('general.architecture'), str) else None
    def fact(name):
        return header.get(f'{arch}.{name}') if arch else None
    head_count = fact('attention.head_count')
    head_count = _positive(head_count) or (head_count if isinstance(head_count, list) and head_count else None)
    kv = fact('attention.head_count_kv')
    if isinstance(kv, list):
        kv = [value if type(value) is int and value >= 0 else None for value in kv]
        kv = kv if kv and None not in kv else None
    else:
        kv = _positive(kv)
    sizes = {Path(path).resolve(): Path(path).stat().st_size}
    for extra in files:
        sizes[Path(extra).resolve()] = Path(extra).stat().st_size
    return {'architecture': arch, 'context_length': _positive(fact('context_length')),
            'block_count': _positive(fact('block_count')), 'embedding_length': _positive(fact('embedding_length')),
            'head_count': head_count, 'head_count_kv': kv if kv is not None else head_count,
            'key_length': _positive(fact('attention.key_length')), 'value_length': _positive(fact('attention.value_length')),
            'weight_bytes': sum(sizes.values())}


_FACTS = {}


def cached_model_facts(path, *, files=()):
    """model_facts keyed by each file's resolved path, size and modification time, so a request path reads a header
    once per file version. Raises what model_facts raises (OSError, ValueError)."""
    key = tuple((str(Path(item).resolve()), stat.st_size, stat.st_mtime_ns)
                for item in (path, *files) for stat in (Path(item).stat(),))
    if key not in _FACTS:
        if len(_FACTS) >= 64:
            _FACTS.clear()
        _FACTS[key] = model_facts(path, files=files)
    return dict(_FACTS[key])


def kv_cache_bytes(facts, tokens, cache_type_k=None, cache_type_v=None):
    """Bytes of a llama.cpp KV cache holding tokens positions across all layers (f16 unless a cache type is given), or
    None when the header lacks the counts to say. Per layer and position it stores head_count_kv x key_length keys and
    head_count_kv x value_length values; key and value length default to embedding_length / head_count. This is the
    standard attention cache only: sliding-window, recurrent and latent-attention layouts are estimated as full
    attention, which over-states them (the safe direction for admission)."""
    layers, embedding = facts.get('block_count'), facts.get('embedding_length')
    heads, kv = facts.get('head_count'), facts.get('head_count_kv')
    if not layers or not embedding or not heads or not kv:
        return None
    heads = max(heads) if isinstance(heads, list) else heads
    if not heads:
        return None
    key = facts.get('key_length') or embedding // heads
    value = facts.get('value_length') or embedding // heads
    per_layer = kv if isinstance(kv, list) else [kv] * layers
    width = CACHE_TYPE_BYTES[cache_type_k or 'f16'] * key + CACHE_TYPE_BYTES[cache_type_v or 'f16'] * value
    return int(sum(per_layer[:layers]) * width * tokens + .999)


__all__ = ['CACHE_TYPE_BYTES', 'cached_model_facts', 'kv_cache_bytes', 'model_facts', 'read_metadata']
