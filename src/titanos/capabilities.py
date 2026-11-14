"""Describe implemented execution modes without mistaking a plan for a test."""


def execution_modes(spec):
    """Passive adapter/platform facts; driver and collective probes are separate."""
    import sys
    native = spec.backend == 'llama-cpp'
    torch_tp = spec.backend == 'transformers' and spec.task in {'chat', 'vision', 'embeddings'}
    sharding = native or spec.backend == 'diffusers' or (spec.backend == 'transformers' and
        spec.task in {'chat', 'vision', 'embeddings', 'transcription', 'speech'})
    backend = 'llama.cpp' if native else 'pytorch' if spec.backend in {'transformers', 'diffusers', 'xtts', 'qwen3-tts'} else spec.backend
    def mode(implemented):
        return {'implemented': implemented, 'available': None if implemented else False,
                'runtime_available': None if implemented else False,
                'reason': 'runtime_unverified' if implemented else 'unsupported_task',
                'execution_backend': backend, 'qualified': None if implemented else False}
    result = {'single': mode(True), 'replicas': mode(True), 'model_sharding': mode(sharding),
              'tensor_parallel': mode(native or torch_tp)}
    if native:
        result['single']['split_modes'] = ['none']
        result['model_sharding'].update(split_modes=['layer', 'row'], device_families=['cuda'])
        result['tensor_parallel'].update(split_modes=['tensor'], experimental=True,
            device_families=['cuda'],
            meaning='One llama.cpp process splits weights and KV; this is not PyTorch distributed tensor parallelism')
    elif torch_tp:
        result['tensor_parallel']['execution_backend'] = 'torch.distributed'
        result['tensor_parallel']['collectives'] = {
            family: {'backend': collective, 'available': False if family == 'cuda' and sys.platform == 'win32' else None,
                     'reason': 'nccl_unavailable' if family == 'cuda' and sys.platform == 'win32' else 'runtime_unverified'}
            for family, collective in [('cuda', 'nccl'), ('rocm', 'nccl'), ('xpu', 'xccl'), ('cpu', 'gloo')]}
    return result

def describe(spec):
    from .model_inspection import inspect_setup
    metadata = {}
    setup = inspect_setup(spec, metadata=metadata)
    result = _describe(spec, setup, metadata)
    modes = execution_modes(spec)
    for name, value in result.get('modes', {}).items():
        modes[name] = {**modes.get(name, {}), **value}
        if not modes[name].get('implemented'):
            modes[name].update(available=False, runtime_available=False, reason='unsupported_task')
    result['modes'] = modes
    result['setup'] = setup
    return result


def _describe(spec, setup, metadata):
    if spec.task == 'diarization':
        from .diarization import CONTROL_NAMES, prepare
        return {'backend': 'sherpa-onnx', 'task': 'diarization', 'device': 'cpu',
                'implemented': ['diarization', 'replicas'], 'stream': False,
                'controls': sorted(CONTROL_NAMES), 'defaults': prepare(spec, {}),
                'sample_rates': [16000], 'anonymous_speakers': 'first appearance; reset per request',
                'overlap': 'exact intersections of estimated per-speaker intervals; no audio separation',
                'model_limits': {'local_speakers_per_10s_chunk': 3, 'powerset_simultaneous_speakers': 2},
                'short_audio': 'At most 10 seconds: local segmentation only; explicit speaker count or nondefault clustering threshold is unsupported',
                'cancellation': 'Decoder checks, embedding callback exceptions and after native return; no checks during native segmentation or clustering',
                'source': {'repository': spec.source_repo, 'revision': spec.source_revision},
                'qualified': None, 'qualification_note': 'Anonymous interval estimates are not identity, calibrated confidence or target-native quality qualification'}
    if spec.backend=='ace-step':
        from .music_backend import inspect_bundle,ACE_SUPPORTED_VARIANTS,PLANNER_FLAGS
        try:
            from pathlib import Path
            if 'titanos-music.json' not in metadata:
                raise ValueError('Music metadata was unavailable within the passive inspection limits; inspect setup issues')
            root = Path(spec.path).resolve()
            def bounded_json(path):
                name = Path(path).relative_to(root).as_posix()
                if name not in metadata:
                    raise ValueError('Music metadata was unavailable within the passive inspection limits; inspect setup issues')
                return metadata[name]
            bundle=inspect_bundle(spec.path,verify=False,read_json=bounded_json)
            if bundle['manifest']['variant'] not in ACE_SUPPORTED_VARIANTS:
                raise ValueError('This music variant is a discovery candidate; the current adapter implements XL SFT only')
            declared={**bundle['manifest']['capabilities'],'seed':True,'guidance_scale':True,'steps':True,**dict.fromkeys(PLANNER_FLAGS,True)};error=None
        except (OSError,ValueError,TypeError,KeyError) as exc:declared={};error=str(exc)
        return {'backend':'ace-step','task':'music','device':spec.device,'implemented':['music','asset_jobs','replicas'],
                'modes':{'single':{'implemented':True,'qualified':None},'replicas':{'implemented':True,'qualified':None},
                         'model_sharding':{'implemented':False,'qualified':False},'tensor_parallel':{'implemented':False,'qualified':False}},
                'music':{'implemented':error is None,'family':'ace-step-1.5','controls':declared,
                         'output':'channel-preserving PCM16 WAV artifact','progress':'request-bound native stage events',
                         'runtime':'owned isolated receipt','qualification':None,'configuration_error':error},
                'source':{'repository':spec.source_repo,'revision':spec.source_revision},
                'qualified':None,'qualification_note':'Bundle/runtime validation is not listening-quality or target-device qualification'}
    if spec.backend == 'sherpa-onnx':
        return {'backend':'sherpa-onnx','task':'speech','device':'cpu',
                'implemented':['speech','asset_jobs'],'stream':False,
                'voices':'default or numeric speaker ID','output':'PCM WAV',
                'source':{'repository':spec.source_repo,'revision':spec.source_revision},
                'qualified':None,
                'qualification_note':'Provider archive compatibility and runtime load do not establish voice quality'}
    if spec.backend == "onnx-diffusers":
        from .onnx_images import DEFAULTS
        return {"backend": spec.backend, "task": "image", "device": spec.device,
                "implemented": ["image_generation", "replicas"], "image_edit": False,
                "instruction_edit": None, "image_mask": None,
                "family": "sd1-onnx-pndm", "controls": sorted(DEFAULTS),
                "load_controls": {"disable_metacommands": {"type": "boolean", "device": "dml",
                    "required": False, "omitted": "Do not send the option; retain the installed ORT default",
                    "purpose": "Software diagnostic; not a qualified workaround"}},
                "cpu_ep_fallback_allowed": spec.allow_cpu_fallback, "qualified": None,
                "device_policy": "Explicit CPU or DXGI adapter index plus matching LUID",
                "cpu_support": "Tokenization, image preprocessing, NumPy noise and Torch CPU PNDM scheduler",
                "qualification_note": "Provider names alone do not prove physical GPU execution; inspect component operator profiles and owned process device activity"}
    if spec.backend == 'tesseract':
        from .tesseract import CONTROL_NAMES, DECODE_NAMES, controls
        return {'backend': 'tesseract', 'task': 'image-to-text', 'device': 'cpu',
                'implemented': ['ocr', 'replicas'], 'stream': False, 'qualified': None,
                'controls': sorted(CONTROL_NAMES), 'unsupported_controls': sorted(DECODE_NAMES),
                'defaults': controls({}, spec.defaults), 'oem': 1,
                'model_format': 'external LSTM .traineddata', 'process_lifetime': 'one owned subprocess per request',
                'geometry': 'optional TSV hierarchy in input pixels; native confidence is uncalibrated',
                'qualification_note': 'Native CLI quality does not qualify this API build; run exact model/runtime/device checks'}
    from .instruction_edit import capabilities as instruction_capabilities, PIPELINE
    instruction = spec.backend == 'diffusers' and spec.task == 'image' and setup['architecture'].get('pipeline_class') == PIPELINE
    genai_images = {'implemented': spec.backend == 'openvino-genai' and spec.task == 'image',
                    'families': ['exported SD', 'exported SDXL'], 'image_edit_requires': 'vae_encoder component',
                    'inpainting': 'alpha-binary-v1 requires vae_encoder; exact native behavior unqualified', 'device_policy': 'auto selects CPU; otherwise one explicit CPU, GPU.N or OV- identity',
                    'safety_checker_invoked_by_adapter': False,
                    'output': 'PNG from copied uint8 NHWC tensors', 'qualification': None}
    image_mask = {'implemented':spec.task=='image' and spec.backend in {'diffusers','openvino-genai'} and not instruction,
                  'contract':'alpha-binary-v1','families':['SD','SDXL'],'mask':'one binary-alpha RGBA PNG; 0 edit, 255 keep',
                  'geometry':'exact source/mask/output dimensions; no resizing',
                  'preserve_unmasked':'request/model/process boolean; optional server RGB copying',
                  'qualification':None} if spec.task=='image' else None
    pair = {'implemented':spec.task=='pair-classification' and spec.backend=='transformers',
            'input_order':['premise','hypothesis'], 'labels':'native loaded config id2label; no semantic remapping',
            'score_kind':'softmax_uncalibrated', 'calibrated':False, 'truncate_opt_in':'premise only; hypothesis preserved',
            'qualification':None} if spec.task=='pair-classification' else None
    audio = {'asr': {'response_formats': ['json', 'text', 'verbose_json'],
                     'timestamps': 'architecture-dependent backend alignments; explicit unsupported requests fail',
                     'quality_fields': 'unavailable when not exposed by the pipeline'} if spec.task == 'transcription' else None,
             'vad': {'implemented': spec.task == 'vad', 'sample_rates': [8000, 16000], 'state_reset': 'per_request',
                     'backend': 'onnx', 'device': 'cpu', 'calibrated': False, 'qualification': None} if spec.task == 'vad' else None}
    if spec.task == 'transcription':
        from .audio_gating import resolve_gate
        from .settings import current_settings
        try:
            gate = resolve_gate({}, spec.defaults, current_settings())
            audio['asr']['vad_gating'] = {**gate, 'implemented':True, 'registered_model_required':True,
                'skip':'Empty transcript when configured VAD reports no regions; ASR is not executed',
                'speech':'Full original audio is transcribed; no cropping or timestamp rewriting',
                'explicit_disable':'vad_enabled=false', 'qualification':None}
        except ValueError as exc:
            audio['asr']['vad_gating'] = {'implemented':True, 'configuration_error':str(exc), 'qualification':None}
    retrieval = {'rerank': {'implemented': spec.task == 'rerank' and spec.backend == 'transformers',
                            'method': 'trained_scalar_cross_encoder', 'score_kind': 'cross_encoder_logit',
                            'calibrated': False, 'qualification': None},
                 'embedding_input_policy': {'default': 'reject', 'truncate_opt_in': True,
                                            'per_input_token_reporting': True} if spec.task == 'embeddings' else None}
    role = 'embedding' if spec.task=='embeddings' and spec.backend=='transformers' else 'rerank' if spec.task=='rerank' and spec.backend=='transformers' else 'llm' if spec.task=='chat' and spec.backend=='llama-cpp' else None
    retrieval['answer_workflow'] = {'role':role,'eligible':role is not None and spec.parallelism=='single' and not spec.allow_pickle and not spec.trust_remote_code,
        'stream':False,'corpus':'inline passages','citation_integrity':'literal source quotes','semantic_entailment':'not_automatically_verified','qualification':None}
    sentiment_supported = spec.task == 'sentiment' or (spec.task == 'chat' and spec.backend in {'transformers', 'openvino-genai', 'llama-cpp'})
    sentiment = {'implemented': sentiment_supported,
                 'method': ('dedicated_classifier' if spec.task == 'sentiment' else 'llm') if sentiment_supported else None,
                 'inputs': ['text', 'asr_transcript'] if sentiment_supported else [], 'calibrated': False, 'audio_affect': False,
                 'qualification': None}
    if sentiment_supported:
        sentiment.update(composition_policy='require_agreement',
            composition_role='classifier' if spec.task == 'sentiment' else 'llm',
            prompt_versions=['weights-v1', 'label-v2', 'label-v3', 'facets-v1'] if spec.task == 'chat' else [],
            decoding=(['validate','native_json_schema'] if spec.backend=='llama-cpp' else ['validate', 'label_choices', 'facet_choices'] if spec.backend == 'transformers' else ['validate']) if spec.task=='chat' else [],
            categorical_scores='absent; no generated confidence is invented',
            facets='optional independent evaluation flags; neutral_statement does not verify factual truth')
    responses = {'implemented': spec.task in {'chat', 'vision'}, 'store': False,
                 'inputs': ['text', 'inline_image'] if spec.task == 'vision' else ['text'] if spec.task == 'chat' else [],
                 'tools': spec.backend == 'llama-cpp', 'structured_output': spec.backend == 'llama-cpp',
                 'qualification': None}
    if spec.backend == 'llama-cpp':
        return {'backend':'llama-cpp','task':spec.task,'implemented':['chat','completion','tools','json_schema','responses','continuous_batching','replicas'], 'responses': responses, 'sentiment': sentiment, 'retrieval':retrieval, 'native_options':spec.native.model_dump(), 'device':spec.device, 'qualified':None, 'qualification_note':'Backend capability is not model quality or target-hardware qualification'}
    native_plan, error, rank_compatibility = None, None, None
    # Framework defaults/TP plans require a real runtime. Do not import a
    # framework or run checkpoint config machinery during passive inspection.
    tp_adapter = spec.backend == 'transformers' and spec.task in {'chat', 'vision', 'embeddings'}
    return {'backend': spec.backend, 'task': spec.task, 'configured_mode': spec.parallelism, 'responses': responses, 'sentiment': sentiment, 'retrieval': retrieval, 'audio': audio, 'pair_classification':pair, 'genai_images':genai_images, 'image_mask':image_mask,
        'instruction_edit':instruction_capabilities() if instruction else None,
        'depth': {'implemented': spec.task == 'depth-estimation', 'families': ['relative Depth Anything'],
                  'semantics': 'relative_inverse_depth', 'units': None, 'larger_values': 'nearer',
                  'encoding': 'base64_float32_le', 'normalization': 'none', 'qualification': None},
        'segmentation': {'implemented':spec.task=='image-segmentation','family':'builtin SAM',
                         'prompt':'one box or one positive/negative point set','output':'image-aligned binary RGBA edit masks',
                         'alternatives':'native multimask order; no ranking or class labels','scores':'predicted_iou_uncalibrated',
                         'automatic_mask_generator':False,'qualification':None},
        'source': {'repository': spec.source_repo, 'revision': spec.source_revision},
        'modes': {
            'single': {'implemented': True, 'qualified': None},
            'replicas': {'implemented': True, 'qualified': None,
                'meaning': 'Independent model copies or different models in separate worker processes'},
            'model_sharding': {'implemented': (spec.backend == 'diffusers' and not instruction) or (spec.backend == 'transformers' and spec.task not in {'translation','image-to-text','object-detection','sentiment','rerank','pair-classification','depth-estimation','image-segmentation'}), 'qualified': None,
                'meaning': 'Modules or pipeline components placed across assigned CUDA/ROCm GPUs',
                'requires': 'At least two GPUs, a supported device map, sufficient memory, and an inference smoke check'},
            'tensor_parallel': {'implemented': tp_adapter, 'native_plan_present': bool(native_plan) if native_plan is not None else None,
                'configured_rank_count': spec.tensor_parallel_size, 'attention_dimensions_compatible': rank_compatibility,
                'qualified': None, 'meaning': 'Native Transformers operations split across cooperating distributed ranks',
                'requires': 'Architecture TP plan, compatible tensor dimensions, and NCCL/RCCL, XCCL, or Gloo runtime'}},
        'native_tp_plan': native_plan, 'inspection_error': error,
        'qualification_note': 'null means untested for this exact model/runtime/device assignment. Use the supplied smoke checks; placement alone is not execution evidence.'}
