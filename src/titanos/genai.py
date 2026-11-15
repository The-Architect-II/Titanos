"""OpenVINO GenAI's native local LLM pipeline behind the existing chat API."""

import json
from pathlib import Path

from .settings import current_settings


TOKEN_ALIASES = ('max_tokens', 'max_completion_tokens', 'max_output_tokens', 'max_new_tokens')
PORTABLE_OPTIONS = {'temperature', 'top_p', 'top_k', 'min_p', 'repeat_penalty', 'seed', 'stop'}
WRAPPER_OPTIONS = {'n', 'stream', 'stream_options', 'min_score', 'min_margin',
                   'sentiment_prompt_version', 'sentiment_decoding', 'retrieval_prompt_version'}


def generation_options(defaults, data, native_config):
    """Resolve portable controls once; keep the one-result/text streamer contract."""
    request_fields = set(TOKEN_ALIASES) | PORTABLE_OPTIONS | {'model', 'worker', 'prompt', 'messages', 'n', 'stream', 'stream_options'}
    unsupported = {key for key, value in data.items() if value is not None and key not in request_fields}
    if unsupported:
        raise ValueError('OpenVINO GenAI does not support request fields: ' + ', '.join(sorted(unsupported)))
    kwargs = {key: value for key, value in defaults.items()
              if value is not None and key not in set(TOKEN_ALIASES) | PORTABLE_OPTIONS | WRAPPER_OPTIONS}

    def value(name, fallback=None):
        return next((source[name] for source in (data, defaults) if source.get(name) is not None), fallback)

    if value('n', 1) != 1:
        raise ValueError('OpenVINO GenAI chat returns exactly one sequence (n=1)')
    settings = current_settings()
    limit = next((source[key] for source in (data, defaults) for key in TOKEN_ALIASES
                  if source.get(key) is not None), settings.default_max_tokens)
    temperature = value('temperature', settings.default_temperature)
    stops = value('stop', []) or []
    stops = [stops] if isinstance(stops, str) else stops
    if value('stop') is not None:
        kwargs.pop('stop_strings', None)  # The portable stop override belongs to the streamer below.
    kwargs.update(max_new_tokens=limit, do_sample=temperature > 0,
                  apply_chat_template=False, num_return_sequences=1, echo=False)
    for portable, native in [('top_k', 'top_k'), ('min_p', 'min_p'),
                             ('repeat_penalty', 'repetition_penalty'), ('seed', 'rng_seed')]:
        setting = value(portable)
        if setting is not None:
            kwargs[native] = setting
    if temperature > 0:
        kwargs.update(temperature=temperature, top_p=value('top_p', settings.default_top_p))
    missing = object()
    unsupported = []
    for key in kwargs:
        option = getattr(native_config, key, missing)
        if key.startswith('_') or option is missing or callable(option):
            unsupported.append(key)
    if unsupported:
        raise ValueError('Unsupported OpenVINO GenAI generation options: ' + ', '.join(sorted(unsupported)))
    return kwargs, stops


class GenAIModel:
    def __init__(self, spec):
        import openvino as ov
        import openvino_genai as genai
        self.spec = spec
        from .compiled import compiled_device
        self.device = compiled_device(spec.backend, spec.device)
        if self.device not in ov.Core().available_devices:
            raise ValueError(f"OpenVINO device {self.device!r} is unavailable")
        root = Path(spec.path)
        for filename in ['openvino_model.xml', 'openvino_model.bin', 'openvino_tokenizer.xml', 'openvino_detokenizer.xml']:
            if not (root / filename).is_file():
                raise ValueError(f"GenAI requires a prepared model directory containing {filename}")
        options = dict(spec.load_options)
        if self.device == 'CPU':
            options.setdefault('INFERENCE_PRECISION_HINT', 'f32')
            options.setdefault('INFERENCE_NUM_THREADS', current_settings().threads)
        self.model = genai.LLMPipeline(str(root), self.device, **options)
        self.tokenizer = self.model.get_tokenizer()
        cfg = root / 'config.json'
        self.context = json.loads(cfg.read_text()).get('max_position_embeddings') if cfg.exists() else None

    def details(self):
        return {'backend': 'openvino-genai', 'device': self.device,
                'placement': 'explicit OpenVINO LLM device; tokenization and orchestration use CPU'}

    def infer(self, task, data, emit=None, cancel=None):
        import openvino_genai as genai
        kwargs, stops = generation_options(self.spec.defaults, data, genai.GenerationConfig())
        if 'prompt' in data:
            prompt = data['prompt']
        else:
            messages = []
            for item in data['messages']:
                content = item['content']
                if isinstance(content, list):
                    if any(p['type'] != 'text' for p in content):
                        raise ValueError('This OpenVINO GenAI adapter accepts text only')
                    content = '\n'.join(p['text'] for p in content)
                messages.append({'role': item['role'], 'content': content})
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        limit = kwargs['max_new_tokens']
        input_count = self.tokenizer.encode(prompt).input_ids.shape[-1]
        if self.context and input_count + limit > self.context:
            raise ValueError(f'Prompt plus max_tokens exceeds {self.context}-token context')
        pending, stopped = '', False

        def stream(text):
            nonlocal pending, stopped
            if cancel and cancel.is_set():
                return genai.StreamingStatus.CANCEL
            pending += text
            cuts = [pending.find(s) for s in stops if s in pending]
            if cuts:
                pending, stopped = pending[:min(cuts)], True
            hold = max((i for s in stops for i in range(1, min(len(s), len(pending)) + 1) if pending.endswith(s[:i])), default=0) if not stopped else 0
            ready, pending = (pending[:-hold], pending[-hold:]) if hold else (pending, '')
            if emit and ready:
                emit(ready)
            return genai.StreamingStatus.STOP if stopped else genai.StreamingStatus.RUNNING

        result = self.model.generate([prompt], streamer=stream, **kwargs)
        text = result.texts[0]
        for stop in stops:
            text = text.split(stop, 1)[0]
        if emit and pending and not stopped and not (cancel and cancel.is_set()):
            emit(pending)
        metrics = result.perf_metrics
        inputs, outputs = metrics.get_num_input_tokens(), metrics.get_num_generated_tokens()
        return {'text': text, 'finish_reason': 'length' if result.finish_reasons[0] == genai.GenerationFinishReason.LENGTH and not stopped else 'stop',
                'usage': {'prompt_tokens': inputs, 'completion_tokens': outputs, 'total_tokens': inputs + outputs},
                'provenance': {'repository': getattr(self.spec, 'source_repo', None), 'revision': getattr(self.spec, 'source_revision', None),
                               'backend': 'openvino-genai', 'device': getattr(self, 'device', None)}}
