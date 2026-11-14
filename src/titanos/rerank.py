"""Joint query/document scoring with a local trained scalar cross-encoder."""
import math

from .sentiment import checkpoint_fingerprint
from .settings import current_settings
from .text_inputs import check_cancel, input_limit, prepare_inputs, truncation_policy


class Reranker:
    def __init__(self, spec, device, load_kwargs):
        import torch
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

        self.spec = spec
        local = {'local_files_only': True, 'trust_remote_code': False}
        before = checkpoint_fingerprint(spec.path)
        config = AutoConfig.from_pretrained(spec.path, **local)
        if type(getattr(config, 'num_labels', None)) is not int or config.num_labels != 1:
            raise ValueError('Reranking requires a trained scalar cross-encoder with num_labels=1')
        kwargs = dict(load_kwargs, **local, config=config, use_safetensors=True, output_loading_info=True)
        self.model, info = AutoModelForSequenceClassification.from_pretrained(spec.path, **kwargs)
        if info.get('missing_keys') or info.get('mismatched_keys') or info.get('error_msgs'):
            raise ValueError('Reranker checkpoint has missing or incompatible scoring weights')
        if type(getattr(self.model.config, 'num_labels', None)) is not int or self.model.config.num_labels != 1:
            raise ValueError('Loaded reranker does not have a scalar scoring head')
        self.processor = AutoTokenizer.from_pretrained(spec.path, **local)
        self.model.to(device).eval()
        devices = {str(p.device) for p in self.model.parameters()} | {str(p.device) for p in self.model.buffers()}
        if len(devices) != 1:
            raise ValueError('Reranker must reside on exactly one device')
        self.device = devices.pop()
        requested, actual = torch.device(device), torch.device(self.device)
        if requested.type != actual.type or (requested.index is not None and requested.index != actual.index):
            raise ValueError('Reranker did not load on the requested device')
        after = checkpoint_fingerprint(spec.path)
        if before != after:
            raise ValueError('Checkpoint changed while the reranker was loading')
        self.provenance = {'repository': spec.source_repo, 'revision': spec.source_revision,
                           'checkpoint_digest': after, 'backend': 'transformers', 'device': self.device}

    def details(self):
        return {**self.provenance, 'score_kind': 'cross_encoder_logit', 'higher_is_more_relevant': True,
                'calibrated': False, 'input_pair': ['query', 'document']}

    def infer(self, task, data, cancel=None):
        import torch
        from .engine import EngineError

        if task != 'rerank' or self.spec.task != 'rerank':
            raise ValueError('This model is registered for reranking')
        if set(data) - {'query', 'documents', 'max_input_tokens', 'truncate'}:
            raise ValueError('Unsupported reranking controls')
        query, documents = data.get('query'), data.get('documents')
        if not isinstance(query, str) or not query.strip() or not isinstance(documents, list) or not documents:
            raise ValueError('Reranking requires a nonblank query and a document list')
        settings = current_settings()
        if len(query) > settings.max_prompt_chars or len(documents) > settings.rerank_max_documents:
            raise ValueError('Reranking query or document count exceeds configured limits')
        if any(not isinstance(doc, str) or not doc.strip() or len(doc) > settings.max_text_chars for doc in documents):
            raise ValueError('Documents must be nonblank strings within the configured text limit')
        maximum = data.get('max_input_tokens')
        if maximum is None:
            maximum = self.spec.defaults.get('max_input_tokens')
        if maximum is None:
            maximum = settings.rerank_max_input_tokens
        if isinstance(maximum, bool) or not isinstance(maximum, int) or not 1 <= maximum <= settings.rerank_max_input_tokens:
            raise ValueError('max_input_tokens exceeds the configured reranking budget')
        limit = input_limit(self.processor, maximum, self.model)
        truncate = truncation_policy(data.get('truncate'), self.spec.defaults, settings.rerank_truncate)
        batch, details, policy = prepare_inputs(self.processor, [query] * len(documents), limit, truncate,
                                                pairs=documents, cancel=cancel)
        results = []
        # Batch=1 keeps per-document cancellation bounded and scores independent
        # of configurable batch boundaries. Every pair is checked before forward.
        with torch.inference_mode():
            for index in range(len(documents)):
                check_cancel(cancel)
                inputs = {name: value[index:index + 1].to(self.device) for name, value in batch.items()}
                logits = self.model(**inputs, return_dict=True).logits
                check_cancel(cancel)
                if tuple(logits.shape) != (1, 1):
                    raise EngineError('Reranker must return one scalar score per pair', 502, 'invalid_backend_response')
                score = float(logits.float().item())
                if not math.isfinite(score):
                    raise EngineError('Reranker produced a non-finite relevance score', 502, 'invalid_backend_response')
                results.append({'index': index, 'relevance_score': score})
        results.sort(key=lambda row: (-row['relevance_score'], row['index']))
        tokens = sum(row['processed_tokens'] for row in details)
        return {'results': results, 'score_kind': 'cross_encoder_logit', 'input_details': details,
                'truncation': policy, 'usage': {'prompt_tokens': tokens, 'total_tokens': tokens},
                'provenance': dict(self.provenance)}
