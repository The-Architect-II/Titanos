"""titanos.gguf_header over synthetic GGUF files written by this script: typed scalars, strings and arrays, per-layer KV
head counts, the trained context length, weight bytes across shards, the KV-cache estimate per cache type, the cache keyed
by file version, and refusals for a wrong magic, an unsupported version, a truncated header and an oversized string. The
files hold a header and a few padding bytes, never tensors. No model, process, GPU or network."""
import os
import struct
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from titanos import gguf_header  # noqa: E402
from titanos.gguf_header import cached_model_facts, kv_cache_bytes, model_facts, read_metadata  # noqa: E402

U32, I32, F32, BOOL, STRING, ARRAY, U64 = 4, 5, 6, 7, 8, 9, 10


def text(value):
    data = value.encode('utf-8')
    return struct.pack('<Q', len(data)) + data


def typed(kind, value):
    if kind == STRING:
        return text(value)
    if kind == ARRAY:
        item, values = value
        body = b''.join(text(v) if item == STRING else struct.pack({U32: '<I', I32: '<i', F32: '<f'}[item], v) for v in values)
        return struct.pack('<IQ', item, len(values)) + body
    return struct.pack({U32: '<I', I32: '<i', F32: '<f', BOOL: '<?', U64: '<Q'}[kind], value)


def gguf(path, pairs, *, magic=b'GGUF', version=3, tail=b'\0' * 32):
    body = b''.join(text(key) + struct.pack('<I', kind) + typed(kind, value) for key, kind, value in pairs)
    Path(path).write_bytes(magic + struct.pack('<IQQ', version, 7, len(pairs)) + body + tail)
    return Path(path)


QWEN_LIKE = [('general.architecture', STRING, 'qwen2'), ('general.name', STRING, 'synthetic'),
             ('qwen2.context_length', U32, 32768), ('qwen2.block_count', U32, 24),
             ('qwen2.embedding_length', U32, 896), ('qwen2.attention.head_count', U32, 14),
             ('qwen2.attention.head_count_kv', U32, 2), ('qwen2.rope.freq_base', F32, 1e6),
             ('tokenizer.ggml.add_bos_token', BOOL, False),
             ('tokenizer.ggml.tokens', ARRAY, (STRING, ['a', 'bb', 'ccc'] * 2000)),
             ('tokenizer.ggml.token_type', ARRAY, (I32, [1] * 5000))]


def check_metadata(folder):
    path = gguf(folder / 'qwen.gguf', QWEN_LIKE)
    header = read_metadata(path)
    assert header['version'] == 3 and header['tensor_count'] == 7, header
    meta = header['metadata']
    assert meta['general.architecture'] == 'qwen2' and meta['qwen2.context_length'] == 32768
    assert meta['tokenizer.ggml.add_bos_token'] is False and abs(meta['qwen2.rope.freq_base'] - 1e6) < 1
    # a string array and a numeric array above MAX_KEPT_ARRAY are skipped, not kept
    assert meta['tokenizer.ggml.tokens'] is None and meta['tokenizer.ggml.token_type'] is None
    facts = model_facts(path)
    assert facts == {'architecture': 'qwen2', 'context_length': 32768, 'block_count': 24, 'embedding_length': 896,
                     'head_count': 14, 'head_count_kv': 2, 'key_length': None, 'value_length': None,
                     'weight_bytes': path.stat().st_size}, facts
    # Qwen2.5-0.5B shape: 24 layers x 2 KV heads x (64 + 64) x 2 bytes = 12 KiB per position
    assert kv_cache_bytes(facts, 1) == 12288 and kv_cache_bytes(facts, 32768) == 384 * 1024 * 1024
    assert kv_cache_bytes(facts, 1024, 'q8_0', 'q8_0') == 24 * 2 * 128 * 34 // 32 * 1024
    assert kv_cache_bytes(facts, 1024, 'f32', None) == 24 * 2 * (64 * 4 + 64 * 2) * 1024
    shard = folder / 'qwen-00002.gguf'
    shard.write_bytes(b'\0' * 1000)
    assert model_facts(path, files=[shard])['weight_bytes'] == path.stat().st_size + 1000
    assert model_facts(path, files=[path])['weight_bytes'] == path.stat().st_size  # a file named twice counts once


def check_layer_lists(folder):
    pairs = [('general.architecture', STRING, 'gemma3'), ('gemma3.block_count', U32, 4),
             ('gemma3.embedding_length', U32, 1024), ('gemma3.attention.head_count', U32, 8),
             ('gemma3.attention.head_count_kv', ARRAY, (U32, [4, 4, 0, 2])), ('gemma3.attention.key_length', U32, 256),
             ('gemma3.attention.value_length', U32, 128)]
    facts = model_facts(gguf(folder / 'layers.gguf', pairs))
    assert facts['head_count_kv'] == [4, 4, 0, 2] and facts['context_length'] is None
    assert kv_cache_bytes(facts, 10) == (4 + 4 + 0 + 2) * (256 * 2 + 128 * 2) * 10
    # no head counts: the estimate says it cannot tell rather than guessing
    facts = model_facts(gguf(folder / 'bare.gguf', [('general.architecture', STRING, 'mamba'), ('mamba.block_count', U32, 4)]))
    assert facts['head_count'] is None and kv_cache_bytes(facts, 4096) is None
    # missing kv count falls back to the attention head count (multi-head attention)
    facts = model_facts(gguf(folder / 'mha.gguf', [('general.architecture', STRING, 'llama'), ('llama.block_count', U32, 2),
                                                   ('llama.embedding_length', U32, 64), ('llama.attention.head_count', U32, 4)]))
    assert facts['head_count_kv'] == 4 and kv_cache_bytes(facts, 1) == 2 * 4 * (16 + 16) * 2  # head size 64 / 4


def refused(path, fragment):
    try:
        read_metadata(path)
    except ValueError as exc:
        assert fragment in str(exc), (fragment, str(exc))
        return
    raise AssertionError(f'{path.name} was read')


def check_refusals(folder):
    refused(gguf(folder / 'magic.gguf', QWEN_LIKE, magic=b'GGML'), 'Not a GGUF file')
    refused(gguf(folder / 'v1.gguf', QWEN_LIKE, version=1), 'Unsupported GGUF version')
    refused(gguf(folder / 'big-endian.gguf', QWEN_LIKE, version=3 << 24), 'Unsupported GGUF version')
    whole = gguf(folder / 'whole.gguf', QWEN_LIKE, tail=b'').read_bytes()
    cut = folder / 'cut.gguf'
    cut.write_bytes(whole[:len(whole) // 2])
    refused(cut, 'truncated')
    long_string = folder / 'long.gguf'
    long_string.write_bytes(b'GGUF' + struct.pack('<IQQ', 3, 0, 1) + struct.pack('<Q', gguf_header.MAX_STRING_BYTES + 1)
                            + b'x' * 64)
    refused(long_string, 'string exceeds')
    tiny = folder / 'tiny.gguf'
    tiny.write_bytes(b'GGUF')
    refused(tiny, 'Not a GGUF file')
    try:
        read_metadata(gguf(folder / 'bounded.gguf', QWEN_LIKE), max_header_bytes=200)
    except ValueError as exc:
        assert 'bounded size' in str(exc)
    else:
        raise AssertionError('a header past max_header_bytes was read')


def check_cache(folder):
    path = gguf(folder / 'cached.gguf', QWEN_LIKE)
    first = cached_model_facts(path)
    first['context_length'] = 1  # a caller's copy never changes the cache
    assert cached_model_facts(path)['context_length'] == 32768
    pairs = [(k, kind, 8192 if k == 'qwen2.context_length' else v) for k, kind, v in QWEN_LIKE]
    gguf(path, pairs, tail=b'\0' * 64)  # a new file version (size and mtime) is read again
    os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 10**9))
    assert cached_model_facts(path)['context_length'] == 8192


def main():
    with tempfile.TemporaryDirectory(prefix='titanos-gguf-') as name:
        folder = Path(name)
        check_metadata(folder)
        check_layer_lists(folder)
        check_refusals(folder)
        check_cache(folder)
    print('{"check": "gguf_header", "result": "pass", "scope": "synthetic GGUF headers; no model, process, GPU or network"}')


if __name__ == '__main__':
    main()
