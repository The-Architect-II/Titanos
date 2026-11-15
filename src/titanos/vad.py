"""Silero ONNX frame ABI and a bounded, request-local speech interval detector.

Frame/context/state semantics follow snakers4/silero-vad v6.2.1 OnnxWrapper
(MIT), commit 7e30209a3e901f9842f81b225f3e93d8199902b1. No Torch dependency.
"""
import hashlib
import math
from pathlib import Path

import numpy as np

from .engine import EngineError
from .media import read_audio
from .settings import current_settings
from .text_inputs import check_cancel


def parameters(data, defaults):
    settings = current_settings()
    choices = {'sample_rate': settings.vad_sample_rate, 'threshold': settings.vad_threshold,
               'min_speech_ms': settings.vad_min_speech_ms, 'min_silence_ms': settings.vad_min_silence_ms,
               'speech_pad_ms': settings.vad_speech_pad_ms}
    for key, process in choices.items():
        value = data.get(key) if data.get(key) is not None else defaults.get(key, process)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or (isinstance(value, float) and not math.isfinite(value)):
            raise ValueError(key + ' must be a finite number')
        choices[key] = value
    if choices['sample_rate'] not in {8000, 16000} or not isinstance(choices['sample_rate'], int):
        raise ValueError('VAD sample_rate must be 8000 or 16000')
    if not 0 < choices['threshold'] < 1:
        raise ValueError('VAD threshold must be between zero and one')
    if choices['min_speech_ms'] < 0 or choices['speech_pad_ms'] < 0 or choices['min_silence_ms'] <= 0:
        raise ValueError('VAD minimum speech/padding must be nonnegative and minimum silence positive')
    if any(choices[k] > settings.max_audio_seconds * 1000 for k in ('min_speech_ms', 'min_silence_ms', 'speech_pad_ms')):
        raise ValueError('VAD timing controls exceed the configured audio duration limit')
    if not isinstance(data.get('return_frames', False), bool):
        raise ValueError('return_frames must be a boolean')
    choices['negative_threshold'] = max(0.0, choices['threshold'] - .15)
    return choices


def intervals(scores, count, rate, width, values):
    minimum = math.ceil(values['min_speech_ms'] * rate / 1000)
    silence = math.ceil(values['min_silence_ms'] * rate / 1000)
    padding = math.ceil(values['speech_pad_ms'] * rate / 1000)
    spans, start, quiet = [], None, None
    for i, score in enumerate(scores):
        at, end = i * width, min((i + 1) * width, count)
        if score >= values['threshold']:
            quiet = None
            if start is None:
                start = at
        elif start is not None and score <= values['negative_threshold']:
            if quiet is None:
                quiet = at
            if end - quiet >= silence:
                if quiet - start >= minimum:
                    spans.append((start, quiet))
                start, quiet = None, None
    if start is not None and count - start >= minimum:
        spans.append((start, count))
    padded = []
    for start, end in spans:
        start, end = max(0, start - padding), min(count, end + padding)
        if padded and start <= padded[-1][1]:
            padded[-1] = (padded[-1][0], max(end, padded[-1][1]))
        else:
            padded.append((start, end))
    return [{'start': start / rate, 'end': end / rate, 'start_sample': start, 'end_sample': end} for start, end in padded]


class SileroVAD:
    def __init__(self, spec):
        from .compiled import onnx_session
        self.spec = spec
        parameters({}, spec.defaults)
        root = Path(spec.path).resolve()
        path = (root / (spec.graph or 'silero_vad.onnx')).resolve()
        if not path.is_relative_to(root) or not path.is_file() or path.suffix != '.onnx':
            raise ValueError('VAD requires a local ONNX graph inside its model directory')
        with path.open('rb') as stream:
            self.digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        self.session = onnx_session(path, 'cpu')
        inputs = {p.name: p.type for p in self.session.get_inputs()}
        if inputs != {'input': 'tensor(float)', 'state': 'tensor(float)', 'sr': 'tensor(int64)'} or len(self.session.get_outputs()) != 2:
            raise ValueError('Graph does not implement the Silero input/state/sample-rate ABI')
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != self.digest:
                raise ValueError('VAD checkpoint changed during load')

    def details(self):
        return {'backend': 'onnx', 'device': 'cpu', 'providers': self.session.get_providers(),
                'repository': self.spec.source_repo, 'revision': self.spec.source_revision,
                'checkpoint_digest': self.digest, 'state_reset': 'per_request', 'sample_rates': [8000, 16000]}

    def infer(self, task, data, emit=None, cancel=None):
        if task != 'vad':
            raise ValueError('VAD model only supports task=vad')
        if set(data) - {'audio', 'worker', 'sample_rate', 'threshold', 'min_speech_ms', 'min_silence_ms', 'speech_pad_ms', 'return_frames'}:
            raise ValueError('Unsupported VAD controls')
        values = parameters(data, self.spec.defaults)
        rate = values['sample_rate']
        wave = read_audio(data['audio'], rate, cancel)
        width, context_width = (512, 64) if rate == 16000 else (256, 32)
        count = len(wave)
        if math.ceil(count / width) > current_settings().vad_max_frames:
            raise ValueError('Audio exceeds TITANOS_VAD_MAX_FRAMES')
        state = np.zeros((2, 1, 128), dtype=np.float32)
        context = np.zeros((1, context_width), dtype=np.float32)
        scores, frames = [], []
        for at in range(0, count, width):
            check_cancel(cancel)
            frame = wave[at:at + width]
            padded = width - len(frame)
            if padded:
                frame = np.pad(frame, (0, padded))
            samples = np.concatenate((context, frame.reshape(1, width)), axis=1)
            output, next_state = self.session.run(None, {'input': samples, 'state': state, 'sr': np.array(rate, dtype=np.int64)})
            check_cancel(cancel)
            if np.asarray(output).shape != (1, 1) or np.asarray(next_state).shape != (2, 1, 128) or np.asarray(output).dtype != np.float32 or np.asarray(next_state).dtype != np.float32 or not np.isfinite(next_state).all():
                raise EngineError('VAD returned an invalid state or output shape', 502, 'invalid_backend_response')
            score = float(output[0, 0])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise EngineError('VAD returned an invalid speech probability', 502, 'invalid_backend_response')
            state, context = np.asarray(next_state, dtype=np.float32), samples[:, -context_width:].copy()
            scores.append(score)
            if data.get('return_frames', False):
                end = min(at + width, count)
                frames.append({'start': at / rate, 'end': end / rate, 'start_sample': at, 'end_sample': end,
                               'speech_probability': score, 'padded_samples': padded})
        check_cancel(cancel)
        return {'model': '', 'task': 'vad', 'sample_rate': rate, 'duration': count / rate, 'decoded_samples': count,
                'frame_samples': width, 'context_samples': context_width, 'state_reset': 'per_request',
                'score_kind': 'speech_probability_uncalibrated', 'segments': intervals(scores, count, rate, width, values),
                'frames': frames if data.get('return_frames', False) else None, 'parameters': values,
                'input_sha256': hashlib.sha256(data['audio']).hexdigest(), 'provenance': self.details()}

    def close(self):
        self.session = None
