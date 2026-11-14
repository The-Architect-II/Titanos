"""Strict local, single-label sentiment classification without score calibration."""

import hashlib
import json
import math
from pathlib import Path
import sys


_CANONICAL = {'negative', 'neutral', 'positive'}


def resolve_label_mapping(config, explicit=None):
    count = getattr(config, 'num_labels', None)
    if isinstance(count, bool) or not isinstance(count, int) or count not in (2, 3):
        raise ValueError('Sentiment requires a two- or three-class classifier')
    if getattr(config, 'problem_type', None) not in (None, 'single_label_classification'):
        raise ValueError('Sentiment requires single-label classification, not regression or multilabel output')
    native = getattr(config, 'id2label', None) or {}
    if not isinstance(native, dict):
        raise ValueError('Classifier id2label must be a mapping')
    labels = [native.get(i, native.get(str(i))) for i in range(count)]
    indices = {str(i) for i in range(count)}
    if explicit is None:
        values = [label.strip().casefold() if isinstance(label, str) else None for label in labels]
        if any(value not in _CANONICAL for value in values):
            raise ValueError('Unknown sentiment labels; configure defaults.label_mapping for every native class')
    else:
        if not isinstance(explicit, dict) or len(explicit) != count:
            raise ValueError('label_mapping must map every native class by index or label')
        normalized = {}
        for key, value in explicit.items():
            if isinstance(key, bool) or not isinstance(key, (str, int)) or not isinstance(value, str):
                raise ValueError('label_mapping requires native index/label keys and canonical string values')
            key = str(key).strip().casefold()
            if key in normalized:
                raise ValueError('label_mapping contains ambiguous duplicate keys')
            normalized[key] = value.strip().casefold()
        if set(normalized) == indices:
            values = [normalized[str(i)] for i in range(count)]
        else:
            names = [label.strip().casefold() if isinstance(label, str) else None for label in labels]
            if len(set(names)) != count or set(normalized) != set(names):
                raise ValueError('label_mapping must cover all native indices or all native labels without extra keys')
            values = [normalized[name] for name in names]
    if any(value not in _CANONICAL for value in values) or len(set(values)) != count:
        raise ValueError('Sentiment classes must map one-to-one to negative, neutral or positive')
    return {str(i): value for i, value in enumerate(values)}


def checkpoint_fingerprint(directory, *, purpose='Sentiment'):
    """Hash local weights/config/tokenizer bytes and relative names, not model IDs."""
    root = Path(directory).resolve()
    if not root.is_dir() or not (root / 'config.json').is_file():
        raise ValueError(f'{purpose} requires a local checkpoint directory with config.json')
    suffixes = {'.safetensors', '.json', '.txt', '.model', '.vocab', '.merges', '.bpe', '.spm', '.tiktoken'}
    files = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in suffixes)
    if not any(p.suffix == '.safetensors' for p in files):
        raise ValueError(f'{purpose} requires local safetensors weights')
    records = []
    for path in files:
        if not path.resolve().is_relative_to(root):
            raise ValueError('Checkpoint files must remain inside their model directory')
        digest = hashlib.sha256()
        size = 0
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
                size += len(chunk)
        records.append({'path': path.relative_to(root).as_posix(), 'bytes': size, 'sha256': digest.hexdigest()})
        if path.name.endswith('.safetensors.index.json'):
            weights = json.loads(path.read_text(encoding='utf-8')).get('weight_map')
            if not isinstance(weights, dict) or not weights:
                raise ValueError('Invalid safetensors shard index')
            for name in weights.values():
                if not isinstance(name, str) or Path(name).is_absolute() or Path(name).suffix != '.safetensors':
                    raise ValueError('Shard index must reference relative safetensors files')
                shard = (path.parent / name).resolve()
                if not shard.is_relative_to(root) or not shard.is_file():
                    raise ValueError('Shard index references a missing or external weight file')
    encoded = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(b'titanos-sentiment-checkpoint-v1\n' + encoded).hexdigest()


class SentimentClassifier:
    def __init__(self, spec, device, load_kwargs):
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

        self.spec = spec
        local = {'local_files_only': True, 'trust_remote_code': False}
        before = checkpoint_fingerprint(spec.path)
        config = AutoConfig.from_pretrained(spec.path, **local)
        self.label_mapping = resolve_label_mapping(config, spec.defaults.get('label_mapping'))
        kwargs = dict(load_kwargs, **local, use_safetensors=True, config=config, output_loading_info=True)
        self.model, info = AutoModelForSequenceClassification.from_pretrained(spec.path, **kwargs)
        if info.get('missing_keys') or info.get('mismatched_keys') or info.get('error_msgs'):
            raise ValueError('Sentiment checkpoint has missing or incompatible classification weights')
        if resolve_label_mapping(self.model.config, spec.defaults.get('label_mapping')) != self.label_mapping:
            raise ValueError('Loaded classifier labels differ from checkpoint metadata')
        self.processor = AutoTokenizer.from_pretrained(spec.path, **local)
        self.model.to(device).eval()
        devices = {str(value.device) for value in self.model.parameters()}
        devices.update(str(value.device) for value in self.model.buffers())
        if len(devices) != 1:
            raise ValueError('Sentiment classifier must reside on exactly one device')
        self.device = devices.pop()
        import torch
        requested, actual = torch.device(device), torch.device(self.device)
        if requested.type != actual.type or (requested.index is not None and requested.index != actual.index):
            raise ValueError('Sentiment classifier did not load on its requested device')
        after = checkpoint_fingerprint(spec.path)
        if before != after:
            raise ValueError('Checkpoint files changed while the sentiment classifier was loading')
        self.provenance = {'repository': spec.source_repo, 'revision': spec.source_revision,
                           'checkpoint_digest': after, 'label_mapping': dict(self.label_mapping),
                           'backend': 'transformers', 'device': self.device}
        # Hugging Face uses huge sentinel values when tokenizer context is unknown.
        self.context_limits = [value for value in (getattr(self.processor, 'model_max_length', None),
                                                   getattr(config, 'max_position_embeddings', None))
                               if isinstance(value, int) and not isinstance(value, bool) and 0 < value < sys.maxsize]
        # RoBERTa-family position tables reserve indices through padding_idx.
        # Inspect the actual embedding rather than treating 514 slots as 514 tokens.
        for name, module in self.model.named_modules():
            size, padding = getattr(module, 'num_embeddings', None), getattr(module, 'padding_idx', None)
            if name.rsplit('.', 1)[-1] == 'position_embeddings' and isinstance(size, int) and size > 0:
                usable = size - padding - 1 if isinstance(padding, int) and padding >= 0 else size
                if usable < 1:
                    raise ValueError('Classifier position embeddings have no usable input positions')
                self.context_limits.append(usable)

    def _input_limit(self, value=None):
        value = self.spec.defaults.get('max_input_tokens') if value is None else value
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            raise ValueError('max_input_tokens must be a positive integer')
        limits = self.context_limits + ([value] if value is not None else [])
        if not limits:
            raise ValueError('This checkpoint has no finite input limit; configure max_input_tokens')
        return min(limits)

    def details(self):
        return {**self.provenance, 'label_mapping': dict(self.label_mapping), 'labels': list(self.label_mapping.values()),
                'model_input_limit': min(self.context_limits) if self.context_limits else None}

    def infer(self, task, data, cancel=None):
        import torch
        from .engine import EngineError

        if task != 'sentiment' or self.spec.task != 'sentiment':
            raise ValueError('This checkpoint is registered for sentiment classification')
        if not isinstance(data, dict) or set(data) - {'text', 'max_input_tokens'}:
            raise ValueError('Sentiment accepts text and optional max_input_tokens only')
        text = data.get('text')
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Sentiment text must be a nonempty string')

        def check_cancel():
            if cancel is not None and cancel.is_set():
                raise EngineError('Sentiment inference cancelled', 499, 'cancelled')

        check_cancel()
        limit = self._input_limit(data.get('max_input_tokens'))
        inputs = self.processor(text, return_tensors='pt', truncation=False, padding=False)
        ids = inputs.get('input_ids')
        if ids is None or ids.ndim != 2 or ids.shape[0] != 1:
            raise ValueError('Sentiment tokenizer must return one input sequence')
        count = int(ids.shape[-1])
        if count < 1:
            raise ValueError('Sentiment tokenizer returned no input tokens')
        if count > limit:
            raise ValueError(f'Text has {count} tokens, exceeding max_input_tokens={limit}; truncation is disabled')
        check_cancel()
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = self.model(**inputs, return_dict=True)
        check_cancel()
        logits = output.logits
        if tuple(logits.shape) != (1, len(self.label_mapping)) or not bool(torch.isfinite(logits).all().item()):
            raise ValueError('Classifier returned invalid or non-finite sentiment logits')
        values = torch.softmax(logits.float(), dim=-1)[0].detach().cpu().tolist()
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values) or not math.isclose(sum(values), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError('Classifier returned an invalid sentiment score vector')
        check_cancel()
        return {'scores': {self.label_mapping[str(i)]: float(value) for i, value in enumerate(values)},
                'score_kind': 'softmax_uncalibrated', 'input_tokens': count,
                'provenance': {**self.provenance, 'label_mapping': dict(self.label_mapping)}}
