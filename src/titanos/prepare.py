"""Explicit local export, numerical check and atomic publication of embedding models."""

import hashlib
import json
import os
import tempfile
from importlib.metadata import version
from pathlib import Path

from .config import Config, ModelSpec, read_config, update_config


def prepare(path, args):
    cfg = read_config(path)
    if args.source not in cfg.models:
        raise ValueError("Source must be a registered local model")
    if args.name in cfg.models:
        raise ValueError("Destination model name already exists; choose a new name")
    source = cfg.models[args.source]
    if source.backend != 'transformers' or source.task != 'embeddings':
        raise ValueError("Built-in preparation supports Transformers embedding encoders; use vendor exporters for other architectures")
    from .settings import current_settings
    if not 8 <= args.sequence_length <= current_settings().embedding_max_tokens:
        raise ValueError("sequence-length must be at least 8 and no greater than TITANOS_EMBEDDING_MAX_TOKENS")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise ValueError("Preparation output already exists; choose a new directory")
    spec = ModelSpec(path=str(output), task='embeddings', backend=args.backend, device=args.device,
                     source_repo=source.source_repo, source_revision=source.source_revision)
    Config(models={args.name: spec})
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='titanos-prepare-', dir=output.parent) as tmp:
        staging = Path(tmp) / 'model'
        staging.mkdir()
        import numpy as np
        import torch
        from .backends import LoadedModel
        from .compiled import CompiledModel
        model = LoadedModel(source.model_copy(update={'device': 'cpu', 'dtype': 'float32'}))
        names = list(model.processor('Titanos model preparation', padding='max_length', truncation=True,
                                     max_length=args.sequence_length, return_tensors='pt'))
        sample = model.processor('Titanos model preparation', padding='max_length', truncation=True,
                                 max_length=args.sequence_length, return_tensors='pt')

        class Encoder(torch.nn.Module):
            def __init__(self, encoder):
                super().__init__()
                self.encoder = encoder

            def forward(self, *values):
                return self.encoder(**dict(zip(names, values))).last_hidden_state

        with torch.inference_mode():
            torch.onnx.export(Encoder(model.model).eval(), tuple(sample.values()), str(staging / 'model.onnx'),
                              input_names=names, output_names=['last_hidden_state'], opset_version=17, dynamo=False)
        model.processor.save_pretrained(staging)
        model.model.config.save_pretrained(staging)
        import onnx
        onnx.checker.check_model(str(staging / 'model.onnx'))
        if args.backend == 'openvino':
            import openvino as ov
            ov.save_model(ov.convert_model(str(staging / 'model.onnx')), staging / 'openvino_model.xml', compress_to_fp16=False)
            (staging / 'model.onnx').unlink()
        # Publication requires comparison against the original CPU model, on the requested device.
        runtime = CompiledModel(spec.model_copy(update={'path': str(staging)}))
        texts = ['The quick brown fox jumps over the lazy dog.', 'A different sentence about local inference.', 'Café, 日本語, and numbers 123.']
        feed = model.processor(texts, padding=True, truncation=True, max_length=args.sequence_length, return_tensors='pt')
        with torch.inference_mode():
            hidden = model.model(**feed).last_hidden_state
            mask = feed['attention_mask'].unsqueeze(-1)
            vectors = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            reference = torch.nn.functional.normalize(vectors.float(), dim=1).tolist()
        actual = runtime.infer('embeddings', {'input': texts})['vectors']
        try:
            np.testing.assert_allclose(actual, reference, rtol=1e-4, atol=1e-4)
        except AssertionError as exc:
            raise ValueError("Export failed numerical validation; no model was published") from exc
        receipt = {'schema_version': 1, 'source': source.model_dump(), 'sequence_length': args.sequence_length,
                   'backend': args.backend, 'runtime_version': version('openvino' if args.backend == 'openvino' else 'onnxruntime'),
                   'validation': {'passed': True, 'samples': texts, 'rtol': 1e-4, 'atol': 1e-4,
                                  'execution': runtime.details(), 'scope': 'export parity, not task quality or all inputs'},
                   'files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in staging.iterdir() if p.is_file()}}
        (staging / 'titanos-preparation.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
        del runtime, model
        if output.exists():
            raise ValueError("Preparation output appeared during export; refusing to overwrite")
        os.rename(staging, output)
    def register(latest):
        if args.name in latest.models:
            raise ValueError(f"Destination name was registered during export; prepared files remain at {output}")
        latest.models[args.name] = spec
    update_config(path, register)
    return {'model': args.name, 'path': str(output), 'validation': receipt['validation']}
