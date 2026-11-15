"""Native printed OCR with external traineddata and one owned process per call."""
import csv
import hashlib
import io
import os
from pathlib import Path
import sys
import tempfile
import time

from pydantic import BaseModel, ConfigDict, Field
from typing import Literal
from .engine import EngineError
from .settings import current_settings

PSM = Literal[3, 4, 5, 6, 7, 8, 9, 10, 11, 13]


class OCRControls(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    language: str | None = Field(default=None, max_length=128, pattern=r'^[A-Za-z0-9_]+(?:\+[A-Za-z0-9_]+)*$')
    psm: PSM | None = None
    dpi: int | None = Field(default=None, strict=True, ge=70, le=2400)
    preserve_interword_spaces: bool | None = Field(default=None, strict=True)
    char_whitelist: str | None = Field(default=None, max_length=4096)
    char_blacklist: str | None = Field(default=None, max_length=4096)
    include_geometry: bool | None = Field(default=None, strict=True)
    max_output_bytes: int | None = Field(default=None, strict=True, gt=0)


CONTROL_NAMES = set(OCRControls.model_fields)
DECODE_NAMES = {'prompt', 'max_tokens', 'max_new_tokens', 'temperature', 'top_p', 'top_k', 'seed', 'stop'}


def controls(values, defaults=None):
    data = {**(defaults or {}), **{k: v for k, v in values.items() if v is not None}}
    if any(data.get(k) is not None for k in DECODE_NAMES):
        raise ValueError('Tesseract has no prompt or token sampling decoder; omit generation controls')
    result = OCRControls.model_validate({k: v for k, v in data.items() if k in CONTROL_NAMES}).model_dump(exclude_none=True)
    settings = current_settings()
    result.setdefault('language', settings.ocr_language)
    result.setdefault('psm', settings.ocr_psm)
    result.setdefault('include_geometry', False)
    result.setdefault('max_output_bytes', settings.ocr_max_output_bytes)
    if result['max_output_bytes'] > settings.ocr_max_output_bytes:
        raise ValueError('max_output_bytes exceeds TITANOS_OCR_MAX_OUTPUT_BYTES')
    for key in ('char_whitelist', 'char_blacklist'):
        text = result.get(key, '')
        if any(c in text for c in '\x00\r\n'):
            raise ValueError('OCR character lists cannot contain NUL or line breaks')
        text.encode('utf-8', errors='strict')
    return result


def validate_registration(spec):
    if (spec.task != 'image-to-text' or spec.device not in {'auto', 'cpu'} or spec.dtype != 'auto'
            or spec.parallelism != 'single' or spec.native is not None or spec.graph is not None
            or spec.trust_remote_code or spec.allow_pickle or spec.allow_cpu_fallback or spec.load_options or spec.voices):
        raise ValueError('Tesseract requires one CPU image-to-text worker and external traineddata, without framework/native-GGUF options')
    if set(spec.defaults) - CONTROL_NAMES:
        raise ValueError('Unsupported Tesseract model defaults; token decoding controls do not apply')
    controls({}, spec.defaults)
    spec.device = 'cpu'


def executable_path(model_root):
    model_root = Path(model_root).resolve()
    configured = current_settings().tesseract_executable
    if configured:
        candidate = Path(configured).expanduser()
        if not candidate.is_absolute():
            raise ValueError('TITANOS_TESSERACT_EXECUTABLE must be an absolute trusted executable path')
    else:
        if not getattr(sys, 'frozen', False):
            raise ValueError('Set TITANOS_TESSERACT_EXECUTABLE or use a bundle containing native Tesseract')
        candidate = Path(sys._MEIPASS) / 'native' / 'tesseract' / ('tesseract.exe' if os.name == 'nt' else 'tesseract')
    candidate = candidate.resolve()
    if candidate.is_relative_to(model_root) or not candidate.is_file():
        raise ValueError('Trusted Tesseract executable must exist outside the external model directory')
    return candidate


def digest(path, check=lambda: None):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        while data := stream.read(1024 * 1024):
            check()
            result.update(data)
    return result.hexdigest()


def parse_tsv(raw, width, height):
    expected = ['level', 'page_num', 'block_num', 'par_num', 'line_num', 'word_num', 'left', 'top', 'width', 'height', 'conf', 'text']
    reader = csv.DictReader(io.StringIO(raw), delimiter='\t', quoting=csv.QUOTE_NONE)
    if reader.fieldnames != expected:
        raise ValueError('Unexpected native OCR TSV header')
    rows = []
    for row in reader:
        if len(rows) >= current_settings().ocr_max_tsv_rows or None in row or any(v is None for v in row.values()):
            raise ValueError('Native OCR TSV exceeds its row budget or has malformed columns')
        values = {k: int(row[k]) for k in expected[:10]}
        confidence = float(row['conf'])
        if (values['level'] not in range(1, 6) or any(v < 0 for v in values.values())
                or values['left'] + values['width'] > width or values['top'] + values['height'] > height
                or not (confidence == -1 or 0 <= confidence <= 100)):
            raise ValueError('Native OCR TSV contains invalid geometry or confidence')
        rows.append({**values, 'conf': confidence, 'text': row['text']})
    return rows


class TesseractModel:
    def __init__(self, spec):
        validate_registration(spec)
        self.spec, self.device = spec, 'cpu'
        self.root = Path(spec.path).resolve()
        if not self.root.is_dir():
            raise ValueError('Tesseract model path must be an external traineddata directory')
        self.executable = executable_path(self.root)
        self.executable_sha256 = digest(self.executable)
        self.model_files(controls({}, spec.defaults)['language'], hashes=False)

    def model_files(self, language, check=lambda: None, hashes=True):
        files = []
        total = 0
        for name in language.split('+'):
            check()
            path = (self.root / (name + '.traineddata')).resolve()
            if not path.is_relative_to(self.root) or not path.is_file():
                raise ValueError('Selected OCR language lacks local traineddata: ' + name)
            size = path.stat().st_size
            total += size
            if size == 0 or total > current_settings().ocr_max_model_bytes:
                raise ValueError('Selected OCR data exceeds TITANOS_OCR_MAX_MODEL_BYTES or is empty')
            files.append({'language': name, 'bytes': size, 'sha256': digest(path, check) if hashes else None})
        return files

    def details(self):
        values = controls({}, self.spec.defaults)
        return {'backend': 'tesseract', 'device': 'cpu', 'executable_sha256': self.executable_sha256,
                'source': {'repository': self.spec.source_repo, 'revision': self.spec.source_revision},
                'traineddata': self.model_files(values['language']), 'defaults': values, 'resident_model': False,
                'placement': 'Native CPU subprocess per request; external LSTM traineddata, OEM1'}

    def infer(self, task, data, emit=None, cancel=None):
        # Replica workers provide a callback for every task. OCR returns its final
        # dictionary through the worker's done message and never emits chunks.
        if task != 'image-to-text' or data.get('stream'):
            raise ValueError('Tesseract supports nonstreaming image-to-text only')
        options = controls(data, self.spec.defaults)
        image = data['image']
        settings = current_settings()
        if image.width * image.height > settings.max_input_image_pixels:
            raise ValueError('Image exceeds TITANOS_MAX_INPUT_IMAGE_PIXELS')
        timeout = min(data.get('timeout', settings.client_timeout * 1000), settings.client_timeout * 1000) / 1000
        deadline = time.monotonic() + timeout
        def check():
            if cancel is not None and cancel.is_set():
                raise EngineError('OCR cancelled', 499, 'cancelled')
            if time.monotonic() >= deadline:
                raise EngineError('OCR exceeded its deadline', 504, 'deadline_exceeded')
        check()
        before = self.model_files(options['language'], check)
        if digest(self.executable, check) != self.executable_sha256:
            raise ValueError('Native OCR executable changed after registration')
        from .owned_command import run_owned
        with tempfile.TemporaryDirectory(prefix='titanos-ocr-') as directory:
            root = Path(directory)
            source, output = root / 'input.png', root / 'output'
            with source.open('wb') as stream:
                image.save(stream, format='PNG')
            if source.stat().st_size > settings.max_input_image_bytes:
                raise ValueError('Decoded PNG exceeds TITANOS_MAX_INPUT_IMAGE_BYTES')
            check()
            argv = [str(self.executable), str(source), str(output), '--tessdata-dir', str(self.root),
                    '-l', options['language'], '--oem', '1', '--psm', str(options['psm']),
                    '-c', 'tessedit_create_txt=1']
            if options['include_geometry']:
                argv += ['-c', 'tessedit_create_tsv=1']
            if 'dpi' in options:
                argv += ['--dpi', str(options['dpi'])]
            for key, native in [('preserve_interword_spaces', 'preserve_interword_spaces'),
                                ('char_whitelist', 'tessedit_char_whitelist'), ('char_blacklist', 'tessedit_char_blacklist')]:
                if key in options:
                    value = int(options[key]) if isinstance(options[key], bool) else options[key]
                    argv += ['-c', native + '=' + str(value)]
            win = os.environ.get('SystemRoot', 'C:/Windows')
            env = {'PATH': str(Path(win) / 'System32') + os.pathsep + win, 'SystemRoot': win, 'WINDIR': win} if os.name == 'nt' else {'PATH': '/usr/bin:/bin'}
            env.update(TEMP=directory, TMP=directory, TMPDIR=directory, OMP_NUM_THREADS='1')
            run_owned(argv, root, env, check, options['max_output_bytes'])
            check()
            raw = output.with_suffix('.txt').read_bytes()
            text = raw.decode('utf-8', errors='strict')
            framed = text[:-2] if text.endswith('\r\n') else text[:-1] if text.endswith('\n') else text
            result = {'text': framed, 'raw_text': text, 'ocr': {'backend': 'tesseract', 'engine': 'LSTM', 'oem': 1,
                      'device': 'cpu', 'language': options['language'], 'psm': options['psm'], 'traineddata': before,
                      'executable_sha256': self.executable_sha256, 'framing': 'one terminal CRLF or LF removed; raw_text preserved'}}
            if options['include_geometry']:
                tsv = output.with_suffix('.tsv').read_bytes().decode('utf-8', errors='strict')
                result['geometry'] = {'units': 'input_pixels', 'width': image.width, 'height': image.height,
                                      'confidence': 'native 0..100 uncalibrated; -1 structural', 'rows': parse_tsv(tsv, image.width, image.height)}
            if before != self.model_files(options['language'], check):
                raise ValueError('Native OCR model data changed during inference')
            check()
            return result
