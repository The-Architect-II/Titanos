"""Passive local setup hints, never checkpoint code, tensor loading or admission.

File presence is deliberately weaker than successful runtime validation. Links,
unfamiliar layouts and exhausted inspection budgets remain unknown, not rejected.
"""
import hashlib
from importlib.util import find_spec
import json
import os
from pathlib import Path, PurePosixPath
import stat

from .link_policy import os_alias
from .settings import current_settings


def _link(info):
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, 'st_file_attributes', 0) & 0x400)


class _Inspection:
    def __init__(self, directory, settings):
        self.root = Path(os.path.abspath(os.path.expanduser(directory)))
        self.settings = settings
        self.files, self.directories, self.links, self.issues, self.metadata = {}, set(), set(), [], {}
        self.complete, self.entries, self.read_bytes = True, 0, 0

    def issue(self, code, message, files, action, severity='error'):
        row = dict(code=code, severity=severity, message=message, files=files, next_action=action)
        if row not in self.issues:
            self.issues.append(row)

    def scan(self):
        try:
            for parent in reversed((self.root, *self.root.parents)):
                info = parent.lstat()
                if _link(info) and not (parent != self.root and os_alias(parent, info)):  # macOS /var -> /private/var
                    self.complete = False
                    self.issue('linked_model_path', 'A linked model directory was not traversed.', [],
                               'Inspect an ordinary local copy, or validate the linked model explicitly at load time.', 'warning')
                    return
            if not self.root.is_dir():
                self.issue('model_directory_missing', 'The configured model directory is absent or is not a directory.', [],
                           'Set path to the existing local model directory. Model weights stay external to Titanos.')
                return
            pending = [self.root]
            while pending:
                directory = pending.pop()
                with os.scandir(directory) as stream:
                    for entry in stream:
                        if self.entries >= self.settings.model_inspection_max_files:
                            self.complete = False
                            self.issue('inspection_limit', 'The directory inventory reached its configured entry limit.', [],
                                       'Raise TITANOS_MODEL_INSPECTION_MAX_FILES or inspect a narrower model directory.', 'warning')
                            return
                        self.entries += 1
                        # DirEntry.stat can omit Windows volume/file identity.
                        info = Path(entry.path).lstat()
                        name = Path(entry.path).relative_to(self.root).as_posix()
                        if _link(info):
                            self.links.add(name)
                        elif stat.S_ISDIR(info.st_mode):
                            self.directories.add(name)
                            if len(PurePosixPath(name).parts) < self.settings.model_inspection_max_depth:
                                pending.append(Path(entry.path))
                            else:
                                self.complete = False
                                self.issue('inspection_depth', 'A nested directory exceeded the passive depth limit.', [name],
                                           'Raise TITANOS_MODEL_INSPECTION_MAX_DEPTH or inspect the specific component directory separately.', 'warning')
                        elif stat.S_ISREG(info.st_mode):
                            self.files[name] = info
        except FileNotFoundError:
            self.issue('model_directory_missing', 'The configured directory or an entry disappeared during inspection.', [],
                       'Check the local path and retry after acquisition or copying finishes.')
            self.complete = False
        except OSError:
            self.complete = False
            self.issue('model_directory_unreadable', 'The local directory could not be fully inspected.', [],
                       'Check read access and retry; no model was loaded.', 'warning')

    def uncertain(self, name):
        return not self.complete or any(name == link or name.startswith(link + '/') for link in self.links)

    def require(self, name, purpose='model file'):
        if name in self.files:
            if self.files[name].st_size == 0:
                self.issue('empty_file', 'A required ' + purpose + ' is empty.', [name], 'Restore this file from the same pinned model revision.')
                return False
            return True
        unknown = self.uncertain(name)
        self.issue('file_uninspected' if unknown else 'missing_file',
                   'A required ' + purpose + (' could not be inspected.' if unknown else ' is missing.'), [name],
                   'Check this file in the complete local model/export; keep tokenizer and preprocessing files from the same revision.',
                   'warning' if unknown else 'error')
        return False

    def json(self, name, required=True):
        if name in self.metadata:
            return self.metadata[name]['value']
        if name not in self.files:
            if required:
                self.require(name, 'JSON metadata file')
            return None
        info = self.files[name]
        remaining = self.settings.model_inspection_max_metadata_bytes - self.read_bytes
        if info.st_size > remaining:
            self.issue('metadata_limit', 'JSON metadata exceeds the remaining inspection byte budget.', [name],
                       'Raise TITANOS_MODEL_INSPECTION_MAX_METADATA_BYTES only for a trusted larger metadata file.', 'warning')
            return None
        try:
            # No tensor bytes are read. Reject a file replaced during this metadata read.
            with (self.root / name).open('rb') as stream:
                actual = os.fstat(stream.fileno())
                if (actual.st_dev, actual.st_ino, actual.st_size, actual.st_mtime_ns) != (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns):
                    raise OSError('Metadata changed')
                raw = stream.read(remaining + 1)
            self.read_bytes += len(raw)
            if len(raw) > remaining or _link((self.root / name).lstat()):
                raise OSError('Metadata changed')
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError('Expected object')
            self.metadata[name] = {'value': value, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
            return value
        except (ValueError, UnicodeError, RecursionError):
            self.issue('invalid_metadata', 'Expected valid JSON object metadata.', [name],
                       'Restore the original metadata from the model revision; do not execute repository code to repair it.')
        except OSError:
            self.issue('metadata_unreadable', 'Metadata changed or could not be read safely.', [name],
                       'Finish copying the model and retry inspection.', 'warning')
        return None

    def weights(self, prefix='', allow_pickle=False, variant=None):
        suffix = ('.' + variant if isinstance(variant, str) and variant else '')
        names = [name for name in self.files if name.startswith(prefix) and '/' not in name[len(prefix):]]
        indexes = [name for name in names if name.endswith('.safetensors.index.json') or allow_pickle and name.endswith('.bin.index.json')]
        weights = [name for name in names if name.endswith(suffix + '.safetensors') or allow_pickle and name.endswith(suffix + '.bin')]
        if not indexes and not weights:
            linked = any(name.startswith(prefix) for name in self.links)
            self.issue('weights_uninspected' if linked or not self.complete else 'missing_weights',
                       'No inspectable safe weight file or shard index was found for this component.', [prefix or '.'],
                       'Acquire the complete safetensors checkpoint (including its shard index), or review explicit trusted loader options.',
                       'warning' if linked or not self.complete else 'error')
        for name in weights:
            self.require(name, 'weight file')
        for name in indexes:
            value = self.json(name)
            if value is None:
                continue
            mapping = value.get('weight_map')
            if not isinstance(mapping, dict) or not mapping or not all(isinstance(v, str) for v in mapping.values()):
                self.issue('invalid_shard_index', 'The weight index has no valid weight_map.', [name], 'Restore the shard index from the same model revision.')
                continue
            for shard in sorted(set(mapping.values())):
                if not safe_relative(shard):
                    self.issue('invalid_shard_path', 'A shard index references an absolute or escaping path.', [name], 'Use a checkpoint whose shard files are contained in its directory.')
                else:
                    self.require(prefix + shard, 'checkpoint shard')

    def tokenizer(self, prefix=''):
        options = ('tokenizer.json', 'tokenizer.model', 'spiece.model', 'sentencepiece.bpe.model', 'vocab.txt', 'vocab.json')
        present = [prefix + name for name in options if prefix + name in self.files]
        for name in present:
            self.require(name, 'tokenizer vocabulary')
        if not present:
            # Custom tokenizer layouts exist; config presence cannot prove them invalid.
            unknown = self.uncertain(prefix) or prefix + 'tokenizer_config.json' in self.files or bool(self.links)
            self.issue('tokenizer_unverified' if unknown else 'missing_tokenizer', 'No recognized local tokenizer vocabulary was found.', [prefix or '.'],
                       'Copy the tokenizer assets from the same revision, or validate the documented custom tokenizer layout explicitly.',
                       'warning' if unknown else 'error')
        self.json(prefix + 'tokenizer_config.json', required=False)


def safe_relative(value):
    return isinstance(value, str) and bool(value) and not any(c in value for c in ('\\', ':', '\x00')) and not PurePosixPath(value).is_absolute() and '..' not in PurePosixPath(value).parts


# Passive advisories that describe exactly what a successful load verifies, and the runtime-details fields that show it.
# Backends whose loaded runtime reports what the passive advisory could not verify: (advisory code, revision key, bindings key,
# observed message template). Only qwen3-tts exposes wrapper_revision/reference_bindings; the generic LoadedModel details of
# other speech runtimes (xtts) carry no voice bindings, so their advisories are never reconciled.
RUNTIME_OBSERVABLE = {'qwen3-tts': ('voice_wrapper_unverified', 'wrapper_revision', 'reference_bindings', 'streaming wrapper revision {revision} and {count} registered PCM16 reference voice{plural} ({voices}) are bound')}


def reconcile_setup_with_runtime(setup, backend, details):
    """Reconcile a passive setup inspection with the observed runtime details of the same, currently loaded model.

    Only the advisory row that says the wrapper/reference voices 'were not loaded' is replaced, by an 'observed' row that
    quotes what the loaded runtime actually reports. Errors and unrelated warnings stay exactly as inspected; nothing is
    suppressed. Returns a new dict, or the original setup when the model is not loaded or reports no bindings."""
    if not isinstance(setup, dict) or not isinstance(details, dict) or details.get('status') == 'unavailable':
        return setup
    rule = RUNTIME_OBSERVABLE.get(backend)
    if rule is None:
        return setup
    code, revision_key, bindings_key, template = rule
    bindings = details.get(bindings_key)
    revision = details.get(revision_key) if revision_key else None
    if not isinstance(bindings, dict) or not bindings or (revision_key and not isinstance(revision, str)):
        return setup
    voices = sorted(str(name) for name in bindings)
    observed_row = None
    issues = []
    for row in setup.get('issues', []):
        if isinstance(row, dict) and row.get('code') == code:
            message = 'Loaded: ' + template.format(revision=revision[:12] if revision else '', count=len(voices), plural='' if len(voices) == 1 else 's', voices=', '.join(voices)) + '.'
            observed_row = dict(code=code.replace('_unverified', '_observed'), severity='info', message=message, files=[], next_action=None,
                                observed={'wrapper_revision': revision, 'voices': voices, 'voice_mode': details.get('voice_mode'), 'device': details.get('device'), 'dtype': details.get('dtype')})
            issues.append(observed_row)
        else:
            issues.append(row)
    if observed_row is None:
        return setup
    severities = {row.get('severity') for row in issues if isinstance(row, dict)}
    status = 'blocked' if 'error' in severities else 'needs_review' if 'warning' in severities else 'verified_at_load'
    return {**setup, 'status': status, 'issues': issues, 'model_loaded': True,
            'observed_at_load': {'scope': 'runtime details of the currently loaded model; passive file checks unchanged', 'backend': details.get('backend'), 'device': details.get('device'), 'reconciled_codes': [observed_row['code']]}}


def inspect_setup(spec, settings=None, *, metadata=None):
    """Return advisory structure through existing HTTP/MCP/CLI capabilities."""
    settings = settings or current_settings()
    check = _Inspection(spec.path, settings)
    check.scan()
    backend, task = spec.backend, spec.task
    architecture = {'compatibility': 'unverified'}
    runtime = {'status': 'not_checked', 'scope': 'module discoverability only; no imports, device checks or load'}
    module = {'transformers': 'transformers', 'diffusers': 'diffusers', 'xtts': 'TTS', 'qwen3-tts': 'qwen_tts',
              'onnx': 'onnxruntime', 'openvino': 'openvino', 'openvino-genai': 'openvino_genai',
              'onnx-diffusers': 'onnxruntime', 'sherpa-onnx': 'sherpa_onnx'}.get(backend)
    if module:
        try:
            modules = [module] + (['torch'] if backend in {'transformers', 'diffusers', 'xtts', 'qwen3-tts'} else [])
            found = {name: find_spec(name) is not None for name in modules}
            missing = [name for name, present in found.items() if not present]
            runtime.update(module=module, modules=found, status='missing' if missing else 'present_unverified')
            if missing:
                check.issue('runtime_missing', 'Required backend modules are absent: ' + ', '.join(missing) + '.', [],
                            'Use a Titanos distribution containing this backend, or prepare its documented Python runtime. Driver and model compatibility still require a load check.')
        except (ImportError, ValueError, AttributeError):
            runtime.update(module=module, status='unknown')

    if backend == 'llama-cpp':
        for name in dict.fromkeys([spec.native.model_file, *spec.native.model_files]):
            check.require(name, 'GGUF model/shard')
        if task == 'vision' and not spec.native.mmproj:
            check.issue('missing_projector', 'This vision adapter needs its matching GGUF projector.', [], 'Set native.mmproj to the matching local projector; a text-only GGUF is insufficient.')
        if spec.native.mmproj:
            check.require(spec.native.mmproj, 'vision projector')
        check.issue('native_runtime_unverified', 'The selected llama.cpp executable and GGUF architecture were not executed or parsed.', [],
                    'Inspect native runtime status, then perform an explicit load check with the chosen executable and device.', 'warning')
    elif task == 'diarization':
        value = check.json('diarization.json')
        if value and isinstance(value.get('components'), dict):
            for item in value['components'].values():
                if isinstance(item, dict) and safe_relative(item.get('path')):
                    check.require(item['path'], 'diarization component')
        check.issue('profile_verification_required', 'Component hashes and the trusted diarization profile are verified only at load.', ['diarization.json'],
                    'Use the pinned diarization provider bundle; an arbitrary ONNX pair is not compatible by filename.', 'warning')
    elif backend == 'sherpa-onnx':
        for extension, label in (('.onnx', 'ONNX model'), ('tokens.txt', 'tokens.txt')):
            matches = [name for name in check.files if name.endswith(extension) and (extension != 'tokens.txt' or PurePosixPath(name).name == extension)]
            if len(matches) != 1:
                check.issue('ambiguous_speech_files' if matches else 'missing_speech_files', 'The VITS speech adapter requires exactly one ' + label + '.', matches,
                            'Use a complete supported Sherpa VITS/Piper archive with one ONNX model and tokens.txt.',
                            'warning' if not check.complete or check.links else 'error')
        check.issue('speech_abi_unverified', 'The ONNX VITS ABI, phoneme data and voice IDs are not verified by filenames.', [],
                    'Keep the archive phoneme/lexicon assets and perform an explicit load check.', 'warning')
    elif backend in {'onnx', 'openvino'}:
        graph = spec.graph or ('silero_vad.onnx' if task == 'vad' else 'openvino_model.xml' if backend == 'openvino' else 'model.onnx')
        check.require(graph, 'prepared graph')
        if backend == 'openvino' and PurePosixPath(graph).suffix == '.xml':
            weights = str(PurePosixPath(graph).with_suffix('.bin'))
            if weights not in check.files:
                check.issue('openvino_weights_unverified', 'No paired IR weight file was found; a weightless IR can still be valid.', [weights],
                            'Keep the export BIN if the XML references weights; verify weightless or other input formats with the runtime.', 'warning')
        if task == 'embeddings':
            check.tokenizer()
        check.issue('graph_contract_unverified', 'Graph operators, external tensor data and named input/output shapes have not been parsed.', [graph],
                    'Use the documented prepared graph for this task; validate its actual runtime inputs and device on load.', 'warning')
    elif backend in {'diffusers', 'onnx-diffusers'} or backend == 'openvino-genai' and task == 'image':
        index = check.json('model_index.json')
        if index:
            kind = index.get('_class_name')
            architecture['pipeline_class'] = kind if isinstance(kind, str) else None
            if backend == 'openvino-genai':
                if not isinstance(kind, str) or kind not in {'StableDiffusionPipeline', 'StableDiffusionXLPipeline'}:
                    check.issue('unsupported_pipeline', 'This GenAI image adapter supports exported SD and SDXL pipelines only.', ['model_index.json'], 'Choose a supported SD/SDXL export or another appropriate backend.')
                components = ['unet', 'vae_decoder', 'text_encoder'] + (['text_encoder_2'] if kind == 'StableDiffusionXLPipeline' else [])
                for component in components:
                    for name in ('openvino_model.xml', 'openvino_model.bin', 'config.json'):
                        check.require(component + '/' + name, 'export component')
                for component in ['tokenizer'] + (['tokenizer_2'] if kind == 'StableDiffusionXLPipeline' else []):
                    for suffix in ('xml', 'bin'):
                        check.require(component + '/openvino_tokenizer.' + suffix, 'exported tokenizer')
                check.require('scheduler/scheduler_config.json', 'scheduler configuration')
            elif backend == 'onnx-diffusers':
                if not isinstance(kind, str) or kind not in {'OnnxStableDiffusionPipeline', 'StableDiffusionOnnxPipeline'}:
                    check.issue('unsupported_pipeline', 'The ONNX image adapter supports its prepared SD1 pipeline only.', ['model_index.json'], 'Use the documented SD1 ONNX export or select a matching backend.')
                for component in ('unet', 'text_encoder', 'vae_decoder', 'safety_checker'):
                    check.require(component + '/model.onnx', 'export component')
                for name in ('tokenizer/vocab.json', 'tokenizer/merges.txt', 'tokenizer/tokenizer_config.json', 'scheduler/scheduler_config.json', 'feature_extractor/preprocessor_config.json'):
                    check.require(name)
            else:
                for component, declaration in index.items():
                    if component.startswith('_') or not isinstance(declaration, list) or len(declaration) != 2 or declaration == [None, None]:
                        continue
                    if not safe_relative(component) or '/' in component:
                        check.issue('invalid_component_path', 'A pipeline component name is not a contained directory.', ['model_index.json'], 'Restore the original builtin pipeline metadata.')
                        continue
                    prefix = component + '/'
                    if 'tokenizer' in component:
                        check.tokenizer(prefix)
                    elif component == 'scheduler':
                        check.json(prefix + 'scheduler_config.json')
                    elif component in {'feature_extractor', 'processor'}:
                        check.json(prefix + 'preprocessor_config.json')
                    else:
                        check.json(prefix + 'config.json')
                        check.weights(prefix, spec.allow_pickle, spec.load_options.get('variant'))
                check.issue('pipeline_unverified', 'Declared pipeline classes and component interfaces were not imported or executed.', ['model_index.json'],
                            'Use builtin local components and explicitly validate the pipeline at load; unfamiliar classes remain unverified.', 'warning')
    elif backend == 'openvino-genai':
        for name in ('openvino_model.xml', 'openvino_model.bin', 'openvino_tokenizer.xml', 'openvino_tokenizer.bin', 'openvino_detokenizer.xml', 'openvino_detokenizer.bin'):
            check.require(name, 'complete GenAI export file')
    elif backend == 'qwen3-tts':
        for name in ('config.json', 'model.safetensors', 'speech_tokenizer/config.json', 'speech_tokenizer/model.safetensors'):
            check.require(name, 'Qwen3-TTS Base component')
        config = check.json('config.json')
        if config and config.get('tts_model_type') not in (None, 'base'):
            check.issue('unsupported_speech_variant', 'This Qwen3-TTS adapter implements Base voice cloning only.', ['config.json'], 'Use the Base checkpoint and pinned streaming wrapper.')
        check.issue('voice_wrapper_unverified', 'The streaming wrapper revision and PCM16 reference voices were not loaded.', [],
                    'Use the documented pinned Qwen streaming wrapper and registered PCM16 reference audio.', 'warning')
    elif backend == 'xtts':
        for name in ('config.json', 'model.pth', 'vocab.json'):
            check.require(name, 'XTTS component')
        check.json('config.json')
        check.issue('voice_reference_unverified', 'XTTS reference audio and checkpoint compatibility were not loaded.', [],
                    'Use the complete XTTS checkpoint and a registered reference voice with the explicit trusted-pickle setting.', 'warning')
    elif backend == 'transformers':
        prefix = ''
        subfolder = spec.load_options.get('subfolder')
        if subfolder:
            if safe_relative(subfolder):
                prefix = subfolder.rstrip('/') + '/'
            else:
                check.issue('external_loader_path', 'The loader selects a path outside this passive inventory.', [], 'Inspect the actual local component separately; explicit loader controls remain unchanged.', 'warning')
        config = None if 'config' in spec.load_options else check.json(prefix + 'config.json')
        if config:
            architecture.update({key: config[key] for key in ('model_type', 'torch_dtype', 'dtype') if isinstance(config.get(key), str)})
            names = config.get('architectures')
            if isinstance(names, list) and len(names) <= 64 and all(isinstance(name, str) for name in names):
                architecture['architectures'] = names
            if task == 'image-segmentation' and config.get('model_type') != 'sam':
                check.issue('unsupported_architecture', 'This prompted segmentation adapter requires builtin SAM (not SAM2).', [prefix + 'config.json'], 'Use a SAM checkpoint with its original processor.')
            if task == 'depth-estimation' and config.get('model_type') != 'depth_anything':
                check.issue('unsupported_architecture', 'This relative-depth adapter requires Depth Anything.', [prefix + 'config.json'], 'Choose a supported relative Depth Anything checkpoint.')
            if task == 'depth-estimation' and (config.get('depth_estimation_type') not in (None, 'relative') or config.get('backbone') is not None or config.get('use_pretrained_backbone') or config.get('use_timm_backbone')):
                check.issue('unsupported_depth_contract', 'This adapter cannot use metric depth or an external/pretrained backbone.', [prefix + 'config.json'], 'Use relative Depth Anything with its embedded DINOv2 backbone.')
            if config.get('auto_map'):
                check.issue('custom_code_unverified', 'Metadata declares custom code; passive inspection never imports it.', [prefix + 'config.json'], 'Prefer a builtin adapter, or review explicit trust controls before a separate load check.', 'warning')
        if not any(name in spec.load_options for name in ('state_dict', 'from_tf', 'from_flax')):
            check.weights(prefix, spec.allow_pickle, spec.load_options.get('variant'))
        if task not in {'object-detection', 'depth-estimation', 'image-segmentation'}:
            check.tokenizer()
        if task in {'vision', 'transcription', 'image-to-text', 'object-detection', 'depth-estimation', 'image-segmentation'}:
            if prefix + 'preprocessor_config.json' in check.files:
                check.json(prefix + 'preprocessor_config.json')
            elif prefix + 'processor_config.json' in check.files:
                check.json(prefix + 'processor_config.json')
                check.issue('processor_layout_unverified', 'Processor metadata exists, but its image/audio preprocessing assets were not recognized.', [prefix + 'processor_config.json'], 'Verify the complete processor assets from the same revision before loading.', 'warning')
            else:
                check.require(prefix + 'preprocessor_config.json', 'image/audio preprocessing configuration')
        if spec.load_options:
            check.issue('loader_options_unverified', 'Explicit loader options can select a different component or weight layout.', [], 'Check the selected variant/subfolder and overrides during the explicit load check.', 'warning')
    elif backend == 'ace-step':
        music = check.json('titanos-music.json')
        if music:
            architecture['variant'] = music.get('variant') if isinstance(music.get('variant'), str) else None
            if music.get('variant') != 'xl-sft':
                check.issue('unsupported_music_variant', 'The current ACE-Step runner implements XL-SFT only.', ['titanos-music.json'],
                            'Select the supported XL-SFT bundle; base and turbo are discovery candidates only.')
            components = music.get('components')
            if isinstance(components, dict):
                for component in components.values():
                    if not isinstance(component, dict) or not safe_relative(component.get('path')):
                        check.issue('invalid_component_path', 'Music component metadata does not name a contained file.', ['titanos-music.json'], 'Use the original pinned provider bundle.')
                        continue
                    inventory = check.json(component['path'])
                    if inventory and isinstance(inventory.get('files'), list):
                        for item in inventory['files']:
                            if isinstance(item, dict) and safe_relative(item.get('path')):
                                check.require(item['path'], 'music weight/configuration file')
        check.issue('music_runtime_unverified', 'Music needs its supported bundle and separate trusted portable runtime.', [],
                    'Use XL-SFT and a complete ACE-Step runtime distribution; inspect runtime status before downloading or loading weights.', 'warning')
    elif backend == 'tesseract':
        for language in str(spec.defaults.get('language', settings.ocr_language)).split('+'):
            check.require(language + '.traineddata', 'OCR language weights')
        check.issue('ocr_runtime_unverified', 'The external Tesseract executable was not run.', [], 'Use the documented native Tesseract runtime and verify its version when loading.', 'warning')
    else:
        check.issue('layout_unverified', 'This adapter has no complete passive layout check.', [], 'Consult the model adapter reference and explicitly validate the documented local bundle.', 'warning')
    if metadata is not None:
        metadata.update({name: row['value'] for name, row in check.metadata.items()})
    status = 'blocked' if any(row['severity'] == 'error' for row in check.issues) else 'needs_review' if check.issues else 'ready_for_load_check'
    return {'schema_version': 1, 'status': status, 'scope': 'passive_local_files_and_json', 'advisory': True,
            'model_loaded': False, 'qualified': None, 'architecture': architecture, 'runtime': runtime,
            'issues': check.issues, 'inventory': {'entries': check.entries, 'regular_files': len(check.files), 'links_skipped': len(check.links), 'complete': check.complete},
            'metadata': {name: {key: row[key] for key in ('bytes', 'sha256')} for name, row in check.metadata.items()},
            'limits': {'max_files': settings.model_inspection_max_files, 'max_metadata_bytes': settings.model_inspection_max_metadata_bytes, 'max_depth': settings.model_inspection_max_depth},
            'qualification_note': 'Presence and metadata do not prove weight validity, architecture/runtime compatibility, preprocessing correctness, device placement or quality. Registration and explicit load controls are unchanged.'}
