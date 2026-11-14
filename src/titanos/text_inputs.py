"""Explicit token budgets shared by text encoders and cross-encoders."""
import sys
import hashlib


def check_cancel(cancel):
    if cancel is not None and cancel.is_set():
        from .engine import EngineError
        raise EngineError('Inference cancelled', 499, 'cancelled')


def input_limit(tokenizer, requested, model=None, fixed_length=None):
    if isinstance(requested, bool) or not isinstance(requested, int) or requested < 1:
        raise ValueError('Input token limit must be a positive integer')
    limits = [requested]
    for value in (getattr(tokenizer, 'model_max_length', None), fixed_length,
                  getattr(getattr(model, 'config', None), 'max_position_embeddings', None)):
        if isinstance(value, int) and not isinstance(value, bool) and 0 < value < sys.maxsize:
            limits.append(value)
    if model is not None:
        for name, module in model.named_modules():
            size, padding = getattr(module, 'num_embeddings', None), getattr(module, 'padding_idx', None)
            if name.rsplit('.', 1)[-1] == 'position_embeddings' and isinstance(size, int) and size > 0:
                size = size - padding - 1 if isinstance(padding, int) and padding >= 0 else size
                if size < 1:
                    raise ValueError('Model has no usable input positions')
                limits.append(size)
    return min(limits)


def truncation_policy(value, defaults, process_default):
    if value is None:
        value = defaults.get('truncate')
    if value is None:
        value = process_default
    if not isinstance(value, bool):
        raise ValueError('truncate must be a boolean')
    return value


def token_budget(value, defaults, key, process_default):
    if value is None:
        value = defaults.get(key)
    return process_default if value is None else value


def prepare_inputs(tokenizer, texts, limit, truncate, *, pairs=None, tensor_type='pt', fixed_length=None, cancel=None,
                   pair_strategy='only_second', include_token_hashes=False):
    """Validate every untruncated sequence before producing any model input."""
    check_cancel(cancel)
    if pair_strategy not in {'only_first', 'only_second'}:
        raise ValueError('Pair truncation must preserve one complete sequence')
    if not texts or any(not isinstance(text, str) or not text for text in texts):
        raise ValueError('Text inputs must be nonempty strings')
    if pairs is not None and (len(pairs) != len(texts) or any(not isinstance(text, str) or not text for text in pairs)):
        raise ValueError('Every query requires one nonempty document')
    kwargs = {'truncation': False, 'padding': False, 'return_attention_mask': True}
    encoded = tokenizer(texts, text_pair=pairs, **kwargs) if pairs is not None else tokenizer(texts, **kwargs)
    original = [len(ids) for ids in encoded['input_ids']]
    if include_token_hashes:
        import json
        def token_hash(ids):
            if any(type(value) is not int or value < 0 for value in ids):
                raise ValueError('Tokenizer IDs must be nonnegative integers')
            return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
        original_hashes = [token_hash(ids) for ids in encoded['input_ids']]
    if len(original) != len(texts) or any(count < 1 for count in original):
        raise ValueError('Tokenizer returned invalid input sequences')
    oversized = [index for index, count in enumerate(original) if count > limit]
    if oversized and not truncate:
        index = oversized[0]
        raise ValueError(f'Input {index} has {original[index]} tokens, exceeding max_tokens={limit}; set truncate=true to permit truncation')
    if oversized:
        # Preserve the first sequence for retrieval, or the hypothesis for pair classification.
        if pairs is not None:
            preserved = texts if pair_strategy == 'only_second' else pairs
            query_ids = tokenizer(preserved, truncation=False, padding=False, add_special_tokens=False)['input_ids']
            overhead = tokenizer.num_special_tokens_to_add(pair=True)
            if any(len(ids) + overhead >= limit for ids in query_ids):
                if pair_strategy == 'only_second':
                    raise ValueError('Query exceeds the pair budget; query truncation is not permitted')
                raise ValueError('Hypothesis exceeds the pair budget; hypothesis truncation is not permitted')
        kwargs.update(truncation=pair_strategy if pairs is not None else True, max_length=limit)
        encoded = tokenizer(texts, text_pair=pairs, **kwargs) if pairs is not None else tokenizer(texts, **kwargs)
    processed = [len(ids) for ids in encoded['input_ids']]
    if len(processed) != len(original) or any(count < 1 or count > limit or count > before for count, before in zip(processed, original)):
        raise ValueError('Tokenizer did not respect the requested input token budget')
    processed_hashes = [token_hash(ids) for ids in encoded['input_ids']] if include_token_hashes else None
    check_cancel(cancel)
    padded = tokenizer.pad(encoded, padding='max_length' if fixed_length else True,
                           max_length=fixed_length, return_tensors=tensor_type)
    details = [{'index': index, 'input_tokens': before, 'processed_tokens': after, 'truncated': after < before,
                'input_sha256': hashlib.sha256((pairs[index] if pairs is not None else texts[index]).encode('utf-8')).hexdigest(),
                **({'query_sha256': hashlib.sha256(texts[index].encode('utf-8')).hexdigest()} if pairs is not None else {})}
               for index, (before, after) in enumerate(zip(original, processed))]
    if include_token_hashes:
        for detail, before, after in zip(details, original_hashes, processed_hashes):
            detail.update(original_token_ids_sha256=before, processed_token_ids_sha256=after)
    return padded, details, {'enabled': truncate, 'side': getattr(tokenizer, 'truncation_side', 'right'),
                            'strategy': pair_strategy if pairs is not None else 'single_sequence', 'max_tokens': limit}
