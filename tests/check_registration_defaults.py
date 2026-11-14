"""Public model defaults survive registration without enabling executable loaders."""
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from fastapi.testclient import TestClient
from titanos.config import Config, ModelSpec, read_config, write_config
from titanos.server import create_app


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        checkpoint = root / 'checkpoint'
        checkpoint.mkdir()
        instruction = root / 'instruction'
        (instruction / 'unet').mkdir(parents=True)
        (instruction / 'unet/config.json').write_text('{"in_channels":8}', encoding='utf-8')
        # Metadata-only builtin instruction fixture: registration must not load weights.
        (instruction / 'model_index.json').write_text(json.dumps({
            '_class_name': 'StableDiffusionInstructPix2PixPipeline',
            'scheduler': ['diffusers', 'EulerAncestralDiscreteScheduler'],
            'unet': ['diffusers', 'UNet2DConditionModel'], 'vae': ['diffusers', 'AutoencoderKL'],
            'text_encoder': ['transformers', 'CLIPTextModel'], 'tokenizer': ['transformers', 'CLIPTokenizer'],
            'safety_checker': ['stable_diffusion', 'StableDiffusionSafetyChecker'],
            'feature_extractor': ['transformers', 'CLIPImageProcessor'], 'requires_safety_checker': False,
        }), encoding='utf-8')
        registry = root / 'models.json'
        cfg = Config()
        write_config(registry, cfg)
        assert ModelSpec(path=str(checkpoint),task='image',backend='openvino-genai',device='GPU.0').device=='GPU.0'
        assert ModelSpec(path=str(checkpoint),task='image',backend='openvino-genai',device='CPU').device=='CPU'
        assert ModelSpec(path=str(checkpoint),task='image',backend='openvino-genai',device='OV-gpu-uuid').device=='OV-gpu-uuid'
        for invalid_device in ('GPU','NPU'):
            try:ModelSpec(path=str(checkpoint),task='image',backend='openvino-genai',device=invalid_device)
            except ValueError:pass
            else:raise AssertionError('GenAI image accepted invalid suggested device '+invalid_device)
        app = create_app(cfg, output_dir=root / 'outputs', registry_path=registry)
        cases = [
            ('sentiment', {'max_input_tokens': 256, 'min_score': 0, 'min_margin': 0,
                           'label_mapping': {'0': 'negative', '1': 'positive'}}),
            ('chat', {'sentiment_prompt_version': 'label-v2', 'sentiment_decoding': 'label_choices',
                      'max_output_tokens': 64}),
            ('embeddings', {'max_tokens': 128, 'truncate': False}),
            ('rerank', {'max_input_tokens': 128, 'truncate': True}),
            ('translation', {'source_language': 'eng_Latn', 'target_language': 'fra_Latn',
                             'translation_prefix': 'translate English to French: '}),
            ('object-detection', {'threshold': 0, 'nms': False, 'nms_iou': 0}),
            ('image-segmentation', {'multimask': False, 'mask_threshold': -0.5}),
            ('transcription', {'prompt':'Names and vocabulary'}),
        ]
        with patch('titanos.engine.Engine.load', side_effect=AssertionError('Registration loaded a model')) as model_load, TestClient(app, base_url='http://localhost') as client:
            for index, (task, defaults) in enumerate(cases):
                name = 'model-' + str(index)
                result = client.post('/api/models/register', json={'name': name, 'spec': {
                    'path': str(checkpoint), 'task': task, 'device': 'cpu', 'defaults': defaults}})
                assert result.status_code == 200, result.text
                assert read_config(registry).models[name].defaults == defaults
            for name, extra, status in [('library', {'defaults': {'generation_kwargs': {'custom_processor': 'x'}}}, 400),
                                       ('loader', {'load_options': {'device_map': 'auto'}}, 422),
                                       ('code', {'trust_remote_code': True}, 400), ('pickle', {'allow_pickle': True}, 400)]:
                before = registry.read_bytes()
                result = client.post('/api/models/register', json={'name': name, 'spec': {
                    'path': str(checkpoint), 'task': 'chat', 'device': 'cpu', **extra}})
                assert result.status_code == status, (name, result.text)
                assert registry.read_bytes() == before
            variant_cases = [
                ('instruction-fp16', 'image', instruction, {'variant': 'fp16'}),
                ('instruction-non-ema', 'image', instruction, {'variant': 'non_ema'}),
                ('instruction-hyphen', 'image', instruction, {'variant': 'fp16-test'}),
                ('instruction-default', 'image', instruction, None),
                ('video-fp16', 'video', checkpoint, {'variant': 'fp16'}),
                ('video-non-ema', 'video', checkpoint, {'variant': 'non_ema'}),
                ('video-hyphen', 'video', checkpoint, {'variant': 'fp16-test'}),
            ]
            for name, task, path, options in variant_cases:
                spec = {'path': str(path), 'task': task, 'backend': 'diffusers', 'device': 'cpu'}
                if options is not None: spec['load_options'] = options
                result = client.post('/api/models/register', json={'name': name, 'spec': spec})
                assert result.status_code == 200, result.text
                assert cfg.models[name].load_options == (options or {})
                assert read_config(registry).models[name].load_options == (options or {})
            assert client.post('/api/models/reload').status_code == 200
            assert cfg.models['instruction-fp16'].load_options == {'variant': 'fp16'}
            assert cfg.models['instruction-default'].load_options == {}
            for name, _, _, options in variant_cases:
                assert cfg.models[name].load_options == (options or {})
            caps = client.post('/api/models/capabilities', json={'model': 'instruction-fp16'})
            assert caps.status_code == 200 and caps.json()['instruction_edit']['requires_input_image']
            assert client.get('/api/models/loaded').json()['data'] == []
            rejected = [
                ({'backend': 'transformers', 'task': 'chat'}, {'variant': 'fp16'}),
                ({'backend': 'tesseract', 'task': 'image-to-text'}, {'variant': 'fp16'}),
                ({'task': 'chat'}, {'variant': 'fp16'}),
                ({}, {'variant': 'fp16', 'low_cpu_mem_usage': True}),
                ({}, {'variant': 'fp16', 'custom_pipeline': 'external'}),
                ({}, {'variant': 'fp16', 'safety_checker': None}),
                *[({}, {'variant': value}) for value in ('', ' ', ' fp16', 'fp16\n', 'fp16\r', 'fp16\r\n', '../fp16', 'fp16/x', 'fp16\\x', 'fp16.bin', 'é', True, 16, None, [])],
            ]
            for index, (fields, options) in enumerate(rejected):
                before = registry.read_bytes()
                result = client.post('/api/models/register', json={'name': 'invalid-variant-' + str(index), 'spec': {
                    'path': str(checkpoint), 'task': 'image', 'backend': 'diffusers', 'device': 'cpu',
                    'load_options': options, **fields}})
                assert result.status_code in (400, 422), (fields, options, result.text)
                assert registry.read_bytes() == before
            ocr_defaults = {'language':'eng','psm':7,'dpi':300,'include_geometry':False,
                            'preserve_interword_spaces':False,'char_whitelist':'ABC','char_blacklist':'xyz','max_output_bytes':1024}
            result = client.post('/api/models/register', json={'name':'native-ocr','spec':{
                'path':str(checkpoint),'task':'image-to-text','backend':'tesseract','defaults':ocr_defaults}})
            assert result.status_code == 200, result.text
            assert read_config(registry).models['native-ocr'].defaults == ocr_defaults
            assert client.post('/api/models/reload').status_code == 200
            assert cfg.models['native-ocr'].defaults == ocr_defaults
            assert cfg.models['native-ocr'].defaults['include_geometry'] is False
            ocr_rejected = [
                ('tesseract', {'psm':2}), ('tesseract', {'psm':'7'}),
                ('tesseract', {'include_geometry':1}), ('tesseract', {'dpi':69}),
                ('tesseract', {'max_output_bytes':0}), ('tesseract', {'unknown_ocr_option':True}),
                ('tesseract', {'top_p':.5}), ('tesseract', {'char_whitelist':'A\nB'}),
                ('transformers', {'psm':7}), ('transformers', {'include_geometry':False}),
            ]
            for index, (backend, defaults) in enumerate(ocr_rejected):
                before = registry.read_bytes()
                result = client.post('/api/models/register', json={'name':'invalid-ocr-'+str(index),'spec':{
                    'path':str(checkpoint),'task':'image-to-text','backend':backend,'defaults':defaults}})
                assert result.status_code in (400,422), (backend, defaults, result.text)
                assert registry.read_bytes() == before
            assert client.get('/api/models/loaded').json()['data'] == []
            model_load.assert_not_called()
        print(json.dumps({'passed': True, 'public_default_groups': len(cases),
                          'protected_loader_rejections': 4, 'variant_registration_reload_cases': len(variant_cases),
                          'variant_rejections': len(rejected), 'ocr_default_registration_reload': True,
                          'ocr_default_rejections': len(ocr_rejected), 'model_inference': False}))


if __name__ == '__main__':
    main()
