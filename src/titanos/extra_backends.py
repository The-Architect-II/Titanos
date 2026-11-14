"""Local checkpoint adapters for the API Plus text and image extensions."""

import re
import sys
import time
from .settings import current_settings as load_settings
from .sentiment import checkpoint_fingerprint
from .translation_contract import NMTProvenance, text_digest


class ExtraBackend:
    def __init__(self, spec, device, load_kwargs):
        from transformers import (AutoImageProcessor, AutoModelForObjectDetection,
                                  AutoModelForSeq2SeqLM, AutoModelForVision2Seq,
                                  AutoProcessor, AutoTokenizer)
        self.spec, self.device = spec, device
        kwargs = dict(load_kwargs, trust_remote_code=spec.trust_remote_code)
        local = {"local_files_only": True, "trust_remote_code": spec.trust_remote_code}
        self.nmt_provenance = None
        self.nmt_provenance_unavailable = 'Attribution requires a safe local single-device checkpoint'
        before = None
        if spec.task == "translation":
            if not spec.allow_pickle and not spec.trust_remote_code and spec.parallelism == 'single':
                try:
                    before = checkpoint_fingerprint(spec.path, purpose='Translation')
                except (ValueError, OSError) as exc:
                    # HF cache snapshots can validly link weights outside the snapshot.
                    # Ordinary inference retains its loader path; attribution stays unavailable.
                    self.nmt_provenance_unavailable = str(exc)
            if before is not None:
                self.model, info = AutoModelForSeq2SeqLM.from_pretrained(spec.path, **dict(kwargs, output_loading_info=True))
                if info.get('missing_keys') or info.get('mismatched_keys') or info.get('error_msgs'):
                    raise ValueError('Translation checkpoint has missing or incompatible model weights')
            else:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(spec.path, **kwargs)
            self.processor = AutoTokenizer.from_pretrained(spec.path, **local)
        elif spec.task == "image-to-text":
            self.model = AutoModelForVision2Seq.from_pretrained(spec.path, **kwargs)
            if self.model.config.model_type == "vision-encoder-decoder":
                # AutoProcessor can select either a tokenizer or an image processor
                # for older TrOCR metadata, depending on installed optional packages.
                self.processor = AutoImageProcessor.from_pretrained(spec.path, **local)
                self.decoder = AutoTokenizer.from_pretrained(spec.path, use_fast=False, **local)
            else:
                self.processor = AutoProcessor.from_pretrained(spec.path, **local)
                self.decoder = getattr(self.processor, "tokenizer", None)
                if self.decoder is None:
                    self.decoder = self.processor if hasattr(self.processor, "batch_decode") else AutoTokenizer.from_pretrained(spec.path, use_fast=False, **local)
        elif spec.task == "object-detection":
            self.model = AutoModelForObjectDetection.from_pretrained(spec.path, **kwargs)
            self.processor = AutoImageProcessor.from_pretrained(spec.path, **local)
            if not hasattr(self.processor, "post_process_object_detection"):
                raise ValueError("This image processor does not expose typed object detection postprocessing")
        else:
            raise ValueError(f"No extension adapter for {spec.task}")
        self.model.to(device).eval()
        if before is not None:
            devices = {str(value.device) for value in self.model.parameters()}
            devices.update(str(value.device) for value in self.model.buffers())
            if len(devices) != 1:
                raise ValueError('Attributed translation requires exactly one loaded device')
            actual = devices.pop()
            if actual.split(':')[0] != device.split(':')[0] or (':' in device and actual != device):
                raise ValueError('Translation model did not load on the requested device')
            after = checkpoint_fingerprint(spec.path, purpose='Translation')
            if before != after:
                raise ValueError('Checkpoint files changed while translation was loading')
            self.device = actual
            self.nmt_provenance = NMTProvenance(repository=spec.source_repo, revision=spec.source_revision,
                checkpoint_digest=after, backend='transformers', device=actual,
                model_class=type(self.model).__name__, tokenizer_class=type(self.processor).__name__,
                model_type=self.model.config.model_type).model_dump()

    def _generation(self, data, cancel):
        from transformers import StoppingCriteria, StoppingCriteriaList
        defaults = self.spec.defaults
        settings = load_settings()
        allowed = {"num_beams", "do_sample", "temperature", "top_p", "top_k", "repetition_penalty",
                   "length_penalty", "early_stopping", "no_repeat_ngram_size"}
        kwargs = {k: defaults[k] for k in allowed if k in defaults}
        kwargs.update(defaults.get("generation_kwargs", {}))
        kwargs.update({k: data[k] for k in allowed if data.get(k) is not None})
        if kwargs.get("temperature") is None:
            kwargs["temperature"] = settings.default_temperature
        if kwargs.get("top_p") is None:
            kwargs["top_p"] = settings.default_top_p
        limit = data.get("max_tokens") or defaults.get("max_tokens") or defaults.get("max_new_tokens") or settings.default_max_tokens
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= settings.max_tokens:
            raise ValueError("max_tokens must be positive and within TITANOS_MAX_TOKENS")
        kwargs["max_new_tokens"] = limit
        if kwargs.get("temperature", 0) > 0:
            kwargs.setdefault("do_sample", True)
        elif kwargs.get("temperature") == 0:
            kwargs["do_sample"] = False
            kwargs.pop("temperature")
            kwargs.pop("top_p", None)
        if cancel is not None:
            class Cancel(StoppingCriteria):
                def __call__(self, input_ids, scores, **unused):
                    return cancel.is_set()
            kwargs["stopping_criteria"] = StoppingCriteriaList([Cancel()])
        return kwargs

    def _inputs(self, inputs):
        return {key: value.to(device=self.device, dtype=self.model.dtype) if value.is_floating_point()
                else value.to(self.device) for key, value in inputs.items()}

    def infer(self, task, data, cancel=None):
        from .engine import EngineError
        if task != self.spec.task:
            raise ValueError(f"Checkpoint is registered for {self.spec.task}, not {task}")
        if cancel is not None and cancel.is_set():
            raise EngineError("Inference cancelled", 499, "cancelled")
        if task == "translation":
            result = self.translate(data, cancel)
        elif task == "image-to-text":
            from .tesseract import CONTROL_NAMES
            if any(data.get(key) is not None for key in CONTROL_NAMES):
                raise ValueError('Native OCR controls require a Tesseract backend')
            result = self.image_to_text(data, cancel)
        else:
            result = self.detect(data, cancel)
        if cancel is not None and cancel.is_set():
            raise EngineError("Inference cancelled", 499, "cancelled")
        return result

    def translate(self, data, cancel):
        from .engine import EngineError
        tokenizer = self.processor
        source, target = data["source_language"], data["target_language"]
        if source == target:
            return {"text": data["text"], "translated": False}
        if data.get('include_provenance') and self.nmt_provenance is None:
            raise EngineError('Translation provenance unavailable: ' + self.nmt_provenance_unavailable,
                              400, 'unsupported_capability')
        started = time.monotonic()
        kwargs = self._generation(data, cancel)
        vocab = tokenizer.get_vocab()
        codes = getattr(tokenizer, 'lang_code_to_id', {}) or {
            name:vocab[name] for name in getattr(tokenizer, 'additional_special_tokens', [])
            if name in vocab and re.fullmatch(r'[a-z]{3}_[A-Z][a-z]{3}', name)}
        if hasattr(tokenizer, 'src_lang') and codes:
            if source not in codes or target not in codes:
                raise ValueError('Unsupported source or target multilingual tokenizer language ID')
            token = codes[target]
            if type(token) is not int or token < 0:
                raise ValueError('Multilingual tokenizer returned an invalid target token ID')
            kwargs['forced_bos_token_id'] = token
            tokenizer.src_lang = source
            selection = {'mode':'multilingual', 'tokenizer_source_language':tokenizer.src_lang,
                         'tokenizer_target_language':target, 'forced_bos_token_id':token}
        else:
            # Fixed-direction checkpoints (e.g. Marian/T5) need a declared pair.
            pair = self.spec.defaults
            if pair.get("source_language") != source or pair.get("target_language") != target:
                raise ValueError("This checkpoint needs matching source_language and target_language in model defaults, or supported multilingual tokenizer language IDs")
            actual_source = getattr(tokenizer, 'source_lang', None)
            actual_target = getattr(tokenizer, 'target_lang', None)
            if actual_source and actual_source != source or actual_target and actual_target != target:
                raise ValueError('Declared fixed language IDs differ from the loaded tokenizer pair')
            selection = {'mode':'fixed_pair', 'tokenizer_source_language':actual_source,
                         'tokenizer_target_language':actual_target,
                         'forced_bos_token_id':kwargs.get('forced_bos_token_id', getattr(getattr(self.model, 'generation_config', None), 'forced_bos_token_id', None))}
        text = self.spec.defaults.get("translation_prefix", "") + data["text"]
        inputs = tokenizer(text, return_tensors="pt", truncation=False)
        # Reject truncation rather than silently translating only part of a document.
        configured = self.spec.defaults.get("max_input_tokens", load_settings().embedding_max_tokens)
        if type(configured) is not int or configured < 1:
            raise ValueError('Translation max_input_tokens must be a positive integer')
        limits = [value for value in (getattr(tokenizer, 'model_max_length', None), getattr(self.model.config, 'max_position_embeddings', None))
                  if type(value) is int and 0 < value < sys.maxsize]
        configured = min([configured, *limits])
        if inputs["input_ids"].shape[-1] > configured:
            raise ValueError("Text exceeds this model's max_input_tokens; split the document or raise its configured limit")
        if cancel is not None and cancel.is_set():
            raise EngineError('Translation cancelled before generation', 499, 'cancelled')
        output = self.model.generate(**self._inputs(inputs), **kwargs)
        if cancel is not None and cancel.is_set():
            raise EngineError('Translation cancelled after generation', 499, 'cancelled')
        translated = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        return {"text": translated, "translated": True, 'input_sha256':text_digest(data['text']),
                'output_sha256':text_digest(translated), 'provenance':self.nmt_provenance,
                'language_selection':{**selection, 'source_language':source, 'target_language':target,
                                      'registered_source_language':self.spec.defaults.get('source_language'),
                                      'registered_target_language':self.spec.defaults.get('target_language'),
                                      'translation_prefix':self.spec.defaults.get('translation_prefix', '')},
                'measurements':{'processed_input_sha256':text_digest(text), 'input_tokens':int(inputs['input_ids'].shape[-1]),
                                'input_token_limit':configured,
                                'output_sequence_tokens':int(output.shape[-1]),
                                'inference_seconds':time.monotonic()-started}}

    def image_to_text(self, data, cancel):
        processor_args = {"images": data["image"], "return_tensors": "pt"}
        prompt = data.get("prompt") if data.get("prompt") is not None else self.spec.defaults.get("prompt")
        if prompt:
            if getattr(self.model.config, "model_type", "") == "vision-encoder-decoder":
                raise ValueError("This OCR checkpoint does not accept prompts; omit prompt or select a prompt-capable image-to-text model")
            processor_args["text"] = prompt
        inputs = self.processor(**processor_args)
        generation = self._generation(data, cancel)
        self._ocr_position_budget(inputs, generation)
        if cancel is not None and cancel.is_set():
            from .engine import EngineError
            raise EngineError('OCR cancelled before generation', 499, 'cancelled')
        output = self.model.generate(**self._inputs(inputs), **generation)
        return {"text": self.decoder.batch_decode(output, skip_special_tokens=True)[0]}

    def _ocr_position_budget(self, inputs, generation):
        # The admitted TrOCR family has a learned, finite decoder position table.
        # Other image-to-text architectures retain their existing generation path.
        if self.model.config.model_type != 'vision-encoder-decoder':
            return
        decoder = self.model.get_decoder()
        if decoder.config.model_type != 'trocr' or not decoder.config.use_learned_position_embeddings:
            # Sinusoidal TrOCR tables grow; their initial length is not a hard cap.
            return
        capacity = decoder.config.max_position_embeddings
        if type(capacity) is not int or capacity < 1:
            raise ValueError('TrOCR decoder position capacity must be a positive integer')
        positions = decoder.model.decoder.embed_positions
        rows, offset = positions.weight.shape[0], positions.offset
        if type(offset) is not int or not 0 <= offset < rows:
            raise ValueError('TrOCR decoder position table is invalid')
        capacity = min(capacity, rows - offset)
        config = self.model.generation_config
        start = generation.get('decoder_start_token_id', config.decoder_start_token_id)
        if start is None:
            start = generation.get('bos_token_id', config.bos_token_id)
        if type(start) is not int or start < 0:
            raise ValueError('TrOCR needs one valid decoder start token')
        prefix = generation.get('decoder_input_ids', inputs.get('decoder_input_ids', inputs.get('input_ids')))
        prefix_length = 1
        if prefix is not None:
            shape = getattr(prefix, 'shape', ())
            if len(shape) != 2 or shape[0] != 1 or shape[1] < 1:
                raise ValueError('TrOCR decoder prefix must be one nonempty token sequence')
            # Transformers prepends the start token only when it is absent.
            prefix_length = shape[1] + int(prefix[0, 0].item() != start)
        if prefix_length + generation['max_new_tokens'] > capacity:
            raise ValueError(f"OCR decoder prefix plus max_tokens exceeds this model's {capacity}-position capacity")

    def detect(self, data, cancel=None):
        import torch
        from .detection import options, suppress
        policy = options(data, self.spec.defaults)
        threshold = policy['threshold']
        image = data["image"]
        inputs = self._inputs(self.processor(images=image, return_tensors="pt"))
        if cancel is not None and cancel.is_set():
            from .engine import EngineError
            raise EngineError('Detection cancelled before forward', 499, 'cancelled')
        output = self.model(**inputs)
        sizes = torch.tensor([[image.height, image.width]], device=self.device)
        results = self.processor.post_process_object_detection(output, threshold=threshold, target_sizes=sizes)[0]
        objects = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            label_id = int(label)
            objects.append({"label": self.model.config.id2label.get(label_id, str(label_id)), "label_id": label_id,
                            "score": float(score), "box": dict(zip(("xmin", "ymin", "xmax", "ymax"), map(float, box.tolist())))})
        result = {"objects": objects, "width": image.width, "height": image.height, "threshold": threshold}
        if policy['nms']:
            result['objects'], result['postprocessing'] = suppress(objects, policy['nms_iou'], cancel)
        return result
