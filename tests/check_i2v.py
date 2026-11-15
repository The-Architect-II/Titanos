"""I2V mapping/cancellation checks with explicit synthetic pipeline fixtures."""
import base64
import io
import json
from pathlib import Path
import tempfile
from contextlib import nullcontext
import threading
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from titanos.engine import EngineError
from titanos.i2v import capabilities, generate, load_video_pipeline


def rejected(call, code=None):
    try:
        call()
    except EngineError as exc:
        if code:
            assert exc.code == code, exc.code
        return
    raise AssertionError('Expected an explicit rejected request')


def check():
    image = Image.new('RGB', (8, 8), 'red')
    stream = io.BytesIO()
    image.save(stream, 'PNG')
    inline = 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode()
    data = {'input_reference': inline, 'prompt': 'A red object moves', 'width': 64, 'height': 64,
            'steps': 2, 'frames': 5, 'fps': 8, 'seed': 7}
    class Pipeline:
        called = False
        cancel_during = None
        def __call__(self, image, prompt=None, width=512, height=512, num_inference_steps=25, num_frames=17,
                     guidance_scale=5, guidance_scale_2=None, generator=None, output_type='np', return_dict=True,
                     callback_on_step_end=None, max_sequence_length=512):
            self.called = True
            self.observed = {'pixel': image.getpixel((0, 0)), 'prompt': prompt, 'width': width, 'height': height,
                             'steps': num_inference_steps, 'frames': num_frames, 'guidance': guidance_scale,
                             'guidance_2': guidance_scale_2, 'seed': generator.initial_seed() if generator else None,
                             'max_sequence_length': max_sequence_length}
            if self.cancel_during:
                self.cancel_during.set()
            if callback_on_step_end:
                assert callback_on_step_end(self, 0, 1, {'latents': 'fixture'}) == {'latents': 'fixture'}
            return SimpleNamespace(frames=[[object() for _ in range(num_frames)]])
    pipeline = Pipeline()
    defaults = {'guidance_scale': 4, 'max_sequence_length': 128, 'frames': 9, 'response_format': 'url'}
    result = generate(pipeline, {**data, 'guidance_scale': 6, 'pipeline_options': {'guidance_scale_2': 3}}, defaults, 'cpu')
    assert len(result) == 5
    assert pipeline.observed == {'pixel': (255, 0, 0), 'prompt': data['prompt'], 'width': 64, 'height': 64,
                                 'steps': 2, 'frames': 5, 'guidance': 6, 'guidance_2': 3, 'seed': 7, 'max_sequence_length': 128}
    nested = {**defaults, 'pipeline_options': {'guidance_scale_2': 2, 'max_sequence_length': 64}}
    generate(pipeline, {**data, 'pipeline_options': {'guidance_scale_2': 3}}, nested, 'cpu')
    assert pipeline.observed['guidance_2'] == 3 and pipeline.observed['max_sequence_length'] == 64
    assert capabilities(pipeline)['image_conditioning'] and capabilities(pipeline)['cancellation']
    for url in ('https://example.invalid/image.png', 'file:///private.png', 'local.png'):
        rejected(lambda: generate(pipeline, {**data, 'input_reference': url}, {}, 'cpu'))
    rejected(lambda: generate(pipeline, {**data, 'pipeline_options': {'callback_on_step_end': 'fake'}}, {}, 'cpu'), 'unsupported_parameter')
    for invalid in ({'max_sequence_length': 'a local path'}, {'guidance_scale_2': float('nan')}, {'attention_kwargs': {'path': 'file'}}):
        rejected(lambda: generate(pipeline, {**data, 'pipeline_options': invalid}, {}, 'cpu'))
    cancelled = threading.Event()
    cancelled.set()
    fresh = Pipeline()
    rejected(lambda: generate(fresh, data, {}, 'cpu', cancelled), 'cancelled')
    assert not fresh.called
    cancelled.clear()
    fresh.cancel_during = cancelled
    rejected(lambda: generate(fresh, data, {}, 'cpu', cancelled), 'cancelled')
    class TextOnly:
        def __call__(self, prompt):
            raise AssertionError('Text-only pipeline must not execute an I2V request')
    rejected(lambda: generate(TextOnly(), data, {}, 'cpu'), 'unsupported_capability')
    class ImageOnly:
        def __call__(self, image, width, height, num_frames, num_inference_steps, fps, generator=None):
            assert fps == 8
            return SimpleNamespace(frames=[[object() for _ in range(num_frames)]])
    rejected(lambda: generate(ImageOnly(), data, {}, 'cpu'), 'unsupported_parameter')
    assert len(generate(ImageOnly(), {**data, 'prompt': ''}, {}, 'cpu')) == 5
    class Scheduler:
        def step(self, sample, *, eta=0, generator=None):
            if self.cancel_during:
                self.cancel_during.set()
            return sample + 1
    class I2VGenXLPipeline:
        def __init__(self):
            self.scheduler = Scheduler()
            self.scheduler.cancel_during = None
        def __call__(self, image, prompt, width, height, num_frames, num_inference_steps, generator=None):
            import inspect
            assert 'eta' in inspect.signature(self.scheduler.step).parameters
            assert self.scheduler.step(1, eta=.2, generator=generator) == 2
            return SimpleNamespace(frames=[[object() for _ in range(num_frames)]])
    legacy = I2VGenXLPipeline()
    original_step = legacy.scheduler.step
    event = threading.Event()
    assert len(generate(legacy, data, {}, 'cpu', event)) == 5
    assert 'step' not in vars(legacy.scheduler) and legacy.scheduler.step == original_step
    assert capabilities(legacy)['cancellation']
    legacy.scheduler.cancel_during = event
    rejected(lambda: generate(legacy, data, {}, 'cpu', event), 'cancelled')
    assert 'step' not in vars(legacy.scheduler) and legacy.scheduler.step == original_step
    event.clear()
    def broken_step(sample, *, eta=0, generator=None):
        raise RuntimeError('fixture failure')
    legacy.scheduler.step = broken_step
    instance_step = legacy.scheduler.step
    try:
        generate(legacy, data, {}, 'cpu', event)
    except (RuntimeError, AssertionError):
        pass
    else:
        raise AssertionError('Fixture must fail')
    assert legacy.scheduler.step is instance_step
    with tempfile.TemporaryDirectory() as folder:
        model = Path(folder)
        (model / 'model_index.json').write_text(json.dumps({'_class_name': 'WanImageToVideoPipeline'}))
        with patch('torch.cuda.device', return_value=nullcontext()), patch('diffusers.DiffusionPipeline.from_pretrained', return_value=pipeline) as loader:
            loaded, dispatched = load_video_pipeline(str(model), 'cuda:0', {'local_files_only': True})
            assert loaded is pipeline and dispatched
            assert loader.call_args.kwargs['device_map'] == 'cuda' and loader.call_args.kwargs['low_cpu_mem_usage']
            assert loader.call_args.kwargs['local_files_only']
    from titanos.i2v import _load_wan_component
    from transformers import UMT5Config, UMT5EncoderModel
    import torch
    with tempfile.TemporaryDirectory() as folder:
        original = UMT5EncoderModel(UMT5Config(vocab_size=32, d_model=8, d_ff=16, d_kv=4, num_heads=2, num_layers=1)).eval()
        original.save_pretrained(folder)
        restored = _load_wan_component(folder, UMT5EncoderModel, torch.float32, 'cpu')
        tokens = torch.tensor([[1, 2, 3]])
        with torch.inference_mode():
            assert torch.equal(original(tokens).last_hidden_state, restored(tokens).last_hidden_state)
        assert restored.shared.weight is restored.encoder.embed_tokens.weight
        assert all(not p.is_meta for p in restored.parameters())
    from diffusers import WanTransformer3DModel
    with tempfile.TemporaryDirectory() as folder:
        original = WanTransformer3DModel(num_attention_heads=2, attention_head_dim=8, in_channels=4, out_channels=4,
                                        text_dim=8, freq_dim=16, ffn_dim=32, num_layers=1)
        original.save_pretrained(folder)
        standard = WanTransformer3DModel.from_pretrained(folder, local_files_only=True, torch_dtype=torch.bfloat16)
        restored = _load_wan_component(folder, WanTransformer3DModel, torch.bfloat16, 'cpu')
        for name, value in standard.state_dict().items():
            actual = restored.state_dict()[name]
            assert actual.dtype == value.dtype and torch.equal(actual, value), name
        assert all(not buffer.is_meta for buffer in restored.buffers())
        standard.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
        compressed = _load_wan_component(folder, WanTransformer3DModel, torch.bfloat16, 'cpu', torch.float8_e4m3fn)
        for name, value in standard.state_dict().items():
            actual = compressed.state_dict()[name]
            assert actual.dtype == value.dtype and torch.equal(actual.float(), value.float()), name
        from safetensors.torch import load_file, save_file
        file = Path(folder) / 'diffusion_pytorch_model.safetensors'
        state = load_file(file)
        state['blocks.0.attn2.norm_added_q.weight'] = torch.ones(16)
        save_file(state, file, metadata={'format': 'pt'})
        legacy = _load_wan_component(folder, WanTransformer3DModel, torch.bfloat16, 'cpu')
        assert legacy._titanos_checkpoint_compatibility['ignored_unexpected_keys'] == ['blocks.0.attn2.norm_added_q.weight']
        state['blocks.0.attn2.unknown_parameter.weight'] = torch.ones(16)
        save_file(state, file, metadata={'format': 'pt'})
        try:
            _load_wan_component(folder, WanTransformer3DModel, torch.bfloat16, 'cpu')
        except ValueError as exc:
            assert 'unknown_parameter' in str(exc)
        else:
            raise AssertionError('Unrecognized checkpoint keys must not be dropped')
    from titanos.i2v import _StagedWan
    with tempfile.TemporaryDirectory() as folder:
        originals = {}
        for name in ('text_encoder', 'other'):
            model = UMT5EncoderModel(UMT5Config(vocab_size=32, d_model=8, d_ff=16, d_kv=4, num_heads=2, num_layers=1)).eval()
            model.save_pretrained(Path(folder) / name)
            originals[name] = model
        stage = _StagedWan(folder, 'cpu', {'default': torch.float32}, {'text_encoder': UMT5EncoderModel, 'other': UMT5EncoderModel})
        for name in ('text_encoder', 'other', 'text_encoder'):
            with torch.inference_mode():
                output = stage.components[name](tokens).last_hidden_state
                assert torch.equal(output, originals[name](tokens).last_hidden_state)
            assert stage.active == name
            assert all(p.is_meta for key, module in stage.components.items() if key != name for p in module.parameters())
        stage.release()
        assert stage.active is None and all(p.is_meta for module in stage.components.values() for p in module.parameters())
    print('PASS: inline image forwarding, signature controls, request precedence, seeds, cancellation, unsupported behavior and exact tiny UMT5 pread restoration (fixtures; no production video inference)')


if __name__ == '__main__':
    check()
