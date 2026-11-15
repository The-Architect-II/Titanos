"""Anonymous CPU diarization with reviewed, external Sherpa ONNX components.

API behavior follows sherpa-onnx 1.13.8 (11afbd009a7f8c08f4bcf2fc1b265d0df4670fbf).
Its pyannote implementation preserves multilabel segments, but only calls Python
back during embedding extraction (and after a single segmentation chunk).
"""
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import stat

from .settings import current_settings
from .text_inputs import check_cancel


CONTROL_NAMES = {'num_speakers', 'cluster_threshold', 'min_duration_on', 'min_duration_off'}
SAMPLE_RATE = 16000
WINDOW_SAMPLES = 160000
SHIFT_SAMPLES = 16000
RUNTIME_VERSION = '1.13.8'
CATALOG = Path(__file__).with_name('data') / 'diarization-profiles.json'
SEGMENTATION_ABI = {'segmentation_model_type': 'pyannote-segmentation-3.0',
                    'window_size': WINDOW_SAMPLES, 'receptive_field_size': 991,
                    'receptive_field_shift': 270, 'num_speakers': 3,
                    'num_classes': 7, 'powerset_max_classes': 2, 'sample_rate': SAMPLE_RATE}


def prepare(spec, options, settings=None, *, duration=None):
    """Validate and resolve controls without reading models or importing a runtime."""
    settings = settings or current_settings()
    if (spec.task != 'diarization' or spec.backend != 'sherpa-onnx'
            or spec.device not in {'auto', 'cpu'} or spec.dtype != 'auto'
            or spec.parallelism != 'single' or spec.graph or spec.native
            or spec.trust_remote_code or spec.allow_pickle or spec.allow_cpu_fallback
            or spec.load_options or spec.voices):
        raise ValueError('Diarization requires a reviewed local Sherpa ONNX bundle on one CPU, dtype=auto and no loader overrides')
    if not isinstance(options, dict) or set(options) - (CONTROL_NAMES | {'audio', 'worker'}):
        raise ValueError('Unsupported diarization controls')
    if set(spec.defaults) - CONTROL_NAMES:
        raise ValueError('Unsupported diarization model defaults')
    values = {}
    for key in CONTROL_NAMES:
        value = options.get(key)
        if value is None:
            value = spec.defaults.get(key)
        if value is None:
            value = getattr(settings, 'diarization_' + key)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0 or value > 3.4028234663852886e38):
            raise ValueError(key + ' must be finite and nonnegative within the native float range')
        if key == 'num_speakers':
            if not isinstance(value, int) or value > settings.diarization_max_speakers:
                raise ValueError('num_speakers must be integer 0 (auto) or within TITANOS_DIARIZATION_MAX_SPEAKERS')
        elif key.startswith('min_duration') and value > settings.max_audio_seconds:
            raise ValueError(key + ' exceeds TITANOS_MAX_AUDIO_SECONDS')
        values[key] = value if key == 'num_speakers' else float(value)
    if duration is not None:
        validate_duration_controls(values, duration)
    return values


def validate_duration_controls(values, duration):
    if not isinstance(duration, (float, int)) or isinstance(duration, bool) or not math.isfinite(duration) or duration <= 0:
        raise ValueError('Diarization duration must be finite and positive')
    if duration <= 10 and (values['num_speakers'] != 0 or values['cluster_threshold'] != .5):
        raise ValueError('Sherpa clips at most 10 seconds bypass clustering: use num_speakers=0 and cluster_threshold=0.5, or provide audio longer than 10 seconds')


def _ordinary(path):
    path = path.expanduser().absolute()
    for part in (*reversed(path.parents), path):
        from .link_policy import refused_link  # OS-owned ancestor aliases (macOS /var -> /private/var) are followed
        if refused_link(part, ancestor=part != path):
            raise ValueError('Diarization bundle paths cannot contain links or reparse points')
    return path


def _json(path):
    if path.stat().st_size > 1024 * 1024:
        raise ValueError('Diarization metadata exceeds 1 MiB')
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError('Duplicate diarization metadata field')
            out[key] = value
        return out
    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=unique)


def _digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inspect_bundle(directory):
    """Admission uses product-reviewed component hashes, not a self-signed manifest."""
    root = _ordinary(Path(directory))
    if not root.is_dir():
        raise ValueError('Diarization requires an ordinary local bundle directory')
    manifest_path = _ordinary(root / 'diarization.json')
    manifest = _json(manifest_path)
    catalog = _json(CATALOG)
    if manifest.get('schema_version') != 1 or catalog.get('schema_version') != 1:
        raise ValueError('Unsupported diarization bundle schema')
    profiles = [p for p in catalog['profiles'] if p['id'] == manifest.get('profile_id')]
    if len(profiles) != 1:
        raise ValueError('Diarization bundle profile is not in the trusted product catalog')
    profile = profiles[0]
    if (profile['sample_rate'] != SAMPLE_RATE or profile['max_speakers_per_frame'] != 2
            or profile.get('abi') != SEGMENTATION_ABI
            or profile.get('runtime_pin') != {'sherpa-onnx': RUNTIME_VERSION, 'sherpa-onnx-core': RUNTIME_VERSION}
            or set(profile['components']) != {'segmentation', 'embedding'}
            or manifest.get('components') != profile['components']):
        raise ValueError('Diarization component ABI or identity differs from the reviewed profile')
    paths, files = {}, {}
    for role, pin in profile['components'].items():
        if pin['path'] != role + '.onnx':
            raise ValueError('Diarization component path must use its fixed local ONNX filename')
        path = _ordinary(root / pin['path'])
        if (not path.is_file() or path.stat().st_size != pin['bytes'] or _digest(path) != pin['sha256']):
            raise ValueError('Diarization component bytes differ from the trusted profile: ' + role)
        paths[role] = path
        files[role] = dict(pin)
    return {'paths': paths, 'files': files, 'profile': profile,
            'manifest_sha256': _digest(manifest_path)}


def _invalid(message):
    from .engine import EngineError
    return EngineError(message, 502, 'invalid_backend_response')


def segments_from_native(rows, count, max_speakers):
    """Quantize native boundaries to input samples, preserving each speaker's spans."""
    # Native frame-center timestamps can extend by one receptive field at EOF.
    # Reject arbitrary excursions; crop only that bounded frame-edge extension.
    tolerance = SEGMENTATION_ABI['receptive_field_size']
    spans, clipped = [], 0
    for row in rows:
        speaker, start, end = row.speaker, row.start, row.end
        if (isinstance(speaker, bool) or not isinstance(speaker, int) or speaker < 0
                or isinstance(start, bool) or isinstance(end, bool)
                or not isinstance(start, (float, int)) or not isinstance(end, (float, int))
                or not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end):
            raise _invalid('Diarization returned invalid speaker/time values')
        a, b = round(start * SAMPLE_RATE), round(end * SAMPLE_RATE)
        if a >= count or b > count + tolerance or a >= b:
            raise _invalid('Diarization returned intervals outside the decoded input')
        if b > count:
            clipped += 1
            b = count
        spans.append((a, b, speaker))
    spans.sort()
    mapping, previous, result = {}, {}, []
    for start, end, speaker in spans:
        if speaker not in mapping:
            if len(mapping) >= max_speakers:
                raise _invalid('Diarization exceeded TITANOS_DIARIZATION_MAX_SPEAKERS')
            mapping[speaker] = 'speaker_' + str(len(mapping))
        if start < previous.get(speaker, -1):
            raise _invalid('Diarization returned overlapping intervals for the same speaker')
        previous[speaker] = end
        result.append({'speaker': mapping[speaker], 'start': start / SAMPLE_RATE,
                       'end': end / SAMPLE_RATE, 'start_sample': start, 'end_sample': end})
    return [{'id': name} for name in mapping.values()], result, clipped


class SherpaDiarizationModel:
    def __init__(self, spec):
        values = prepare(spec, {})
        self.spec, self.device = spec, 'cpu'
        self.layout = inspect_bundle(spec.path)
        self.runtime_versions = {name: version(name) for name in ('sherpa-onnx', 'sherpa-onnx-core')}
        if set(self.runtime_versions.values()) != {RUNTIME_VERSION}:
            raise ValueError('Diarization requires the reviewed Sherpa ONNX 1.13.8 Python/native runtime')
        import sherpa_onnx
        if sherpa_onnx.__version__ != RUNTIME_VERSION:
            raise ValueError('Imported Sherpa module differs from its reviewed distribution version')
        self.sherpa = sherpa_onnx
        self.runtime_versions['sherpa_bundled_onnxruntime'] = str(sherpa_onnx.onnxruntime_version)
        self.runtime = None
        self._values = None
        self._threads = None
        self.last_request = None
        self._configure(values)

    def _configure(self, values):
        threads = current_settings().threads
        if self.runtime is not None and self._values == values and self._threads == threads:
            return
        if self.runtime is not None:
            self.layout = inspect_bundle(self.spec.path)
        s = self.sherpa
        config = s.OfflineSpeakerDiarizationConfig(
            segmentation=s.OfflineSpeakerSegmentationModelConfig(
                pyannote=s.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=str(self.layout['paths']['segmentation']), window_shift_ratio=.1),
                num_threads=threads, debug=False, provider='cpu'),
            embedding=s.SpeakerEmbeddingExtractorConfig(model=str(self.layout['paths']['embedding']),
                num_threads=threads, debug=False, provider='cpu'),
            clustering=s.FastClusteringConfig(num_clusters=values['num_speakers'] or -1,
                threshold=values['cluster_threshold']),
            min_duration_on=values['min_duration_on'], min_duration_off=values['min_duration_off'])
        if not config.validate():
            raise ValueError('Sherpa rejected the diarization component configuration')
        # set_config updates clustering only in 1.13.8. Reconstruct to apply all
        # controls; keep no cross-request speaker labels or clustering state.
        runtime = s.OfflineSpeakerDiarization(config)
        if runtime.sample_rate != SAMPLE_RATE:
            raise ValueError('Diarization native sample rate differs from the reviewed ABI')
        for role, path in self.layout['paths'].items():
            if _digest(_ordinary(path)) != self.layout['files'][role]['sha256']:
                raise ValueError('Diarization checkpoint changed during load')
        self.runtime, self._values, self._threads = runtime, dict(values), threads

    def infer(self, task, data, emit=None, cancel=None):
        if task != 'diarization':
            raise ValueError('Diarization model only supports task=diarization')
        values = prepare(self.spec, data)
        check_cancel(cancel)
        from .media import read_audio
        audio = read_audio(data['audio'], SAMPLE_RATE, cancel)
        settings = current_settings()
        count = len(audio)
        if count > 2**31 - 1:
            raise ValueError('Decoded audio exceeds the native signed 32-bit sample-count ABI')
        validate_duration_controls(values, count / SAMPLE_RATE)
        chunks = 1 + max(0, math.ceil((count - WINDOW_SAMPLES) / SHIFT_SAMPLES))
        if chunks > settings.diarization_max_chunks:
            raise ValueError('Audio exceeds TITANOS_DIARIZATION_MAX_CHUNKS (10-second windows at 1-second stride)')
        check_cancel(cancel)
        self._configure(values)
        check_cancel(cancel)
        def progress(processed, total):
            # 1.13.8 ignores the return value. Python exceptions unwind the
            # callback; native segmentation/clustering have no such boundary.
            check_cancel(cancel)
            return 0
        native = self.runtime.process(audio, callback=progress)
        check_cancel(cancel)
        speakers, segments, clipped = segments_from_native(
            native.sort_by_start_time(), count, settings.diarization_max_speakers)
        from .audio_contract import DiarizationResult, speaker_overlaps
        parameters = {**values, 'clustering_applied': chunks > 1, 'window_shift_ratio': .1,
                      'segmentation_chunks': chunks, 'threads': self._threads,
                      'speaker_limit': settings.diarization_max_speakers,
                      'clipped_output_segments': clipped,
                      'cancellation': 'embedding_callback_exception_or_after_native_return'}
        result = DiarizationResult(model='', duration=count / SAMPLE_RATE, decoded_samples=count,
            input_sha256=hashlib.sha256(data['audio']).hexdigest(), speakers=speakers,
            segments=segments, overlaps=speaker_overlaps(segments, SAMPLE_RATE),
            parameters=parameters, provenance=self.details()).model_dump()
        check_cancel(cancel)
        self.last_request = {'decoded_samples': count, 'speakers': len(speakers), 'parameters': parameters}
        return result

    def details(self):
        return {'backend': 'sherpa-onnx', 'task': 'diarization', 'device': 'cpu',
                'providers': ['cpu'], 'threads': self._threads, 'runtime_versions': self.runtime_versions,
                'profile_id': self.layout['profile']['id'], 'model_files': self.layout['files'],
                'bundle_manifest_sha256': self.layout['manifest_sha256'],
                'source': self.layout['profile']['provenance'],
                'model_license': self.layout['profile']['license'],
                'segmentation_abi': dict(SEGMENTATION_ABI),
                'speaker_labels': 'anonymous_first_appearance_reset_per_request',
                'cancellation': 'Callback exception boundaries and after native return; no check during segmentation/clustering',
                'qualification': None}

    def close(self):
        self.runtime = None
