"""A local native-label sequence classifier over ordered premise/hypothesis pairs."""
import math
from .pair_contract import pair_hash, softmax, validate_pairs
from .sentiment import checkpoint_fingerprint
from .settings import current_settings
from .text_inputs import check_cancel, input_limit, prepare_inputs, token_budget, truncation_policy


def native_labels(config, maximum):
    count = getattr(config, 'num_labels', None)
    if type(count) is not int or not 2 <= count <= maximum:
        raise ValueError('Pair classification requires a categorical head with 2+ labels within TITANOS_PAIR_MAX_LABELS')
    if getattr(config, 'problem_type', None) not in (None, 'single_label_classification'):
        raise ValueError('Pair classification does not accept regression or multilabel heads')
    mapping = getattr(config, 'id2label', None)
    if not isinstance(mapping, dict) or len(mapping) != count:
        raise ValueError('Native id2label must name every classifier output index')
    normalized = {}
    indices = {str(i) for i in range(count)}
    for key, value in mapping.items():
        if isinstance(key, bool) or not isinstance(key, (int, str)) or str(key) not in indices:
            raise ValueError('Native label indices must be contiguous nonnegative integers')
        index = int(key)
        if index in normalized or not isinstance(value, str) or not value.strip():
            raise ValueError('Native label indices/names must be unambiguous and nonblank')
        try:value.encode('utf-8')
        except UnicodeEncodeError as exc:raise ValueError('Native labels must be Unicode scalar text') from exc
        normalized[index] = value
    labels = [normalized[i] for i in range(count)]
    if len(set(labels)) != count:raise ValueError('Native class names must be unique; labels are never remapped')
    return labels


class PairClassifier:
    def __init__(self, spec, device, load_kwargs):
        import torch
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
        self.spec = spec
        local = {'local_files_only':True, 'trust_remote_code':False}
        before = checkpoint_fingerprint(spec.path, purpose='Pair classification')
        config = AutoConfig.from_pretrained(spec.path, **local)
        self.labels = native_labels(config, current_settings().pair_max_labels)
        self.model, info = AutoModelForSequenceClassification.from_pretrained(spec.path,
            **dict(load_kwargs, **local, config=config, use_safetensors=True, output_loading_info=True))
        if info.get('missing_keys') or info.get('mismatched_keys') or info.get('error_msgs'):
            raise ValueError('Pair classifier has missing or incompatible classification weights')
        if native_labels(self.model.config, current_settings().pair_max_labels) != self.labels:
            raise ValueError('Loaded classifier label mapping differs from its checkpoint')
        self.processor = AutoTokenizer.from_pretrained(spec.path, **local)
        self.model.to(device).eval()
        devices = {str(value.device) for value in self.model.parameters()} | {str(value.device) for value in self.model.buffers()}
        if len(devices) != 1:raise ValueError('Pair classifier must reside on exactly one device')
        self.device = devices.pop()
        requested, actual = torch.device(device), torch.device(self.device)
        if requested.type != actual.type or (requested.index is not None and requested.index != actual.index):
            raise ValueError('Pair classifier did not load on the requested device')
        after = checkpoint_fingerprint(spec.path, purpose='Pair classification')
        if before != after:raise ValueError('Checkpoint changed while the pair classifier was loading')
        self.provenance = {'repository':spec.source_repo, 'revision':spec.source_revision, 'checkpoint_digest':after,
            'backend':'transformers', 'device':self.device, 'model_class':type(self.model).__name__,
            'model_type':config.model_type, 'label_source':'loaded_config_id2label'}

    def details(self):
        return {**self.provenance, 'labels':[{'index':i,'label':label} for i,label in enumerate(self.labels)],
                'score_kind':'softmax_uncalibrated','calibrated':False,'input_pair':['premise','hypothesis'],
                'semantic_label_mapping_applied':False}

    def infer(self, task, data, cancel=None):
        import torch
        from .engine import EngineError
        if task != 'pair-classification' or self.spec.task != task:
            raise ValueError('This checkpoint is registered for pair classification')
        if set(data)-{'pairs','max_input_tokens','truncate'}:
            raise ValueError('Unsupported pair-classification controls')
        check_cancel(cancel)
        pairs, ids = validate_pairs(data.get('pairs'))
        settings = current_settings()
        maximum = token_budget(data.get('max_input_tokens'), self.spec.defaults, 'max_input_tokens', settings.pair_max_input_tokens)
        if type(maximum) is not int or not 1 <= maximum <= settings.pair_max_input_tokens:
            raise ValueError('max_input_tokens exceeds TITANOS_PAIR_MAX_INPUT_TOKENS')
        limit = input_limit(self.processor, maximum, self.model)
        truncate = truncation_policy(data.get('truncate'), self.spec.defaults, settings.pair_truncate)
        batch, details, policy = prepare_inputs(self.processor, [pair.premise for pair in pairs], limit, truncate,
            pairs=[pair.hypothesis for pair in pairs], pair_strategy='only_first', include_token_hashes=True, cancel=cancel)
        results = []
        # Reuse the cross-encoder pattern: all pairs checked first, one forward per pair.
        with torch.inference_mode():
            for index, (pair, detail) in enumerate(zip(pairs, details)):
                check_cancel(cancel)
                inputs = {name:value[index:index+1].to(self.device) for name,value in batch.items()}
                logits = self.model(**inputs, return_dict=True).logits
                check_cancel(cancel)
                if tuple(logits.shape) != (1, len(self.labels)):
                    raise EngineError('Classifier logits do not match the native categorical head',502,'invalid_backend_response')
                values = logits.cpu().tolist()[0]
                if any(isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) for value in values):
                    raise EngineError('Classifier returned non-finite or nonnumeric logits',502,'invalid_backend_response')
                check_cancel(cancel)
                winner = max(range(len(values)), key=values.__getitem__)
                results.append({'index':index,'id':ids[index],'id_source':'caller' if pair.id is not None else 'index_default',
                    'label_index':winner,'label':self.labels[winner],'logits':values,'scores':softmax(values),
                    'input':{'premise_sha256':detail['query_sha256'],'hypothesis_sha256':detail['input_sha256'],
                        'pair_sha256':pair_hash(pair.premise,pair.hypothesis),
                        **{name:detail[name] for name in ['input_tokens','processed_tokens','truncated','original_token_ids_sha256','processed_token_ids_sha256']}}})
        tokens = sum(row['processed_tokens'] for row in details)
        return {'labels':[{'index':i,'label':label} for i,label in enumerate(self.labels)],'results':results,
                'truncation':policy,'usage':{'prompt_tokens':tokens,'total_tokens':tokens},'provenance':dict(self.provenance)}
