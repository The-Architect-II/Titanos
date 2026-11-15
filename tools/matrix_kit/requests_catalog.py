"""One real request per Titanos task, with output verification, for the matrix kit.

Every function takes (client, model_name, entry) where client is an httpx.Client bound to a running Titanos server and entry is
the manifest entry (task, backend, spec, request overrides). It returns a small JSON-serialisable summary of the verified output
and raises on any failure (HTTP status, missing field, empty or malformed output). Verification is an engine-contract check:
the model answered with the right kind of output on the requested device. It is not an accuracy score; quality observations
are returned under 'quality' for the maintainer to read, never used to decide pass/fail here.

run(task, client, name, entry, guard=None): `guard` is the cell's active-monitor callable (run_cells: STOP and the resource budgets).
Every wait loop here calls guard(**owned_work) on each iteration when one is given; it returns nothing or raises to abort the request
(the kit then cancels only the owned work it is told about: music passes its Titanos job id as music_job=<id>, and attaches the same
id to any exception it raises). Without a guard the behaviour is unchanged.
"""
import base64
import hashlib
import inspect
import io
import math
import struct
import time
import wave

def red_square(size=32):
    """A real PNG written by Pillow (a hand-typed base64 constant was refused by the server as an invalid image)."""
    from PIL import Image
    buf = io.BytesIO()
    Image.new('RGB', (size, size), (220, 30, 30)).save(buf, format='PNG')
    return buf.getvalue()


def data_url(png_bytes):
    return 'data:image/png;base64,' + base64.b64encode(png_bytes).decode()


def tone_wav(seconds=1.0, hz=440, rate=16000):
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        n = int(seconds * rate)
        w.writeframes(b''.join(struct.pack('<h', int(8000 * math.sin(2 * math.pi * hz * i / rate))) for i in range(n)))
    return buf.getvalue()


def text_image(text='HELLO WORLD'):
    from PIL import Image, ImageDraw, ImageFont
    image = Image.new('RGB', (640, 96), 'white')
    draw = ImageDraw.Draw(image)
    font = None
    for candidate in ['C:/Windows/Fonts/arial.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', '/System/Library/Fonts/Supplemental/Arial.ttf']:
        try:
            font = ImageFont.truetype(candidate, 54)
            break
        except OSError:
            continue
    draw.text((25, 14), text, fill='black', font=font or ImageFont.load_default(size=54))
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    return buf.getvalue()


def scene_image():
    """A 128x128 synthetic scene (coloured rectangles) for detection/depth/segmentation: the contract is a well-formed response."""
    from PIL import Image, ImageDraw
    image = Image.new('RGB', (128, 128), (200, 220, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle([20, 60, 100, 120], fill=(120, 60, 20))
    draw.ellipse([60, 10, 110, 60], fill=(250, 200, 40))
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    return buf.getvalue()


def ok(r, what):
    if r.status_code != 200:
        raise RuntimeError(f'{what}: HTTP {r.status_code}: {r.text[:1500]}')
    return r


def chat(c, name, entry):
    r = ok(c.post('/v1/chat/completions', json={'model': name, 'messages': [{'role': 'user', 'content': 'Say hello in one short sentence.'}], 'temperature': 0, 'max_tokens': 24}), 'chat')
    text = r.json()['choices'][0]['message']['content']
    if not (text or '').strip():
        raise RuntimeError('empty chat completion')
    return {'text': text[:200]}


def vision(c, name, entry):
    r = ok(c.post('/v1/chat/completions', json={'model': name, 'messages': [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': data_url(red_square())}}, {'type': 'text', 'text': 'What color is this image?'}]}], 'max_tokens': 16, 'temperature': 0}), 'vision')
    text = r.json()['choices'][0]['message']['content']
    if not (text or '').strip():
        raise RuntimeError('empty vision answer')
    return {'text': text[:200], 'quality': {'criterion': 'mentions red', 'met': 'red' in text.lower()}}


def embeddings(c, name, entry):
    r = ok(c.post('/v1/embeddings', json={'model': name, 'input': ['a cat', 'a dog']}), 'embeddings')
    data = r.json()['data']
    if len(data) != 2 or not data[0]['embedding']:
        raise RuntimeError('embeddings missing or wrong count')
    return {'count': 2, 'dim': len(data[0]['embedding'])}


def speech(c, name, entry):
    voice = (entry.get('request') or {}).get('voice', 'default')
    body = {'model': name, 'input': 'Hello world. The quick brown fox jumps over the lazy dog.', 'voice': voice, 'response_format': 'wav'}
    lang = (entry.get('request') or {}).get('language')
    if lang:
        body['language'] = lang
    r = ok(c.post('/v1/audio/speech', json=body), 'speech')
    wav = r.content
    if wav[:4] != b'RIFF' or len(wav) < 1000:
        raise RuntimeError(f'not a WAV of useful length ({len(wav)} bytes)')
    return {'wav_bytes': len(wav), 'wav_sha256': hashlib.sha256(wav).hexdigest()}


def transcription(c, name, entry):
    r = ok(c.post('/v1/audio/transcriptions', data={'model': name}, files={'file': ('tone.wav', tone_wav(), 'audio/wav')}), 'transcription')
    body = r.json()
    if 'text' not in body:
        raise RuntimeError('no text field in transcription')
    bad = c.post('/v1/audio/transcriptions', data={'model': name}, files={'file': ('bad.wav', b'invalid', 'audio/wav')})
    if bad.status_code != 400:
        raise RuntimeError(f'invalid audio must be refused with 400, got {bad.status_code}')
    return {'text': body['text'][:200], 'invalid_audio_refused': True, 'quality': {'criterion': 'synthetic tone; text content not scored', 'met': None}}


def image(c, name, entry):
    req = entry.get('request') or {}
    size, steps = req.get('size', '64x64'), req.get('steps', 2)
    if req.get('edit_only'):
        # instruction-edit families (InstructPix2Pix) have no text-to-image path: the edit is the real request
        e = ok(c.post('/v1/images/edits', data={'model': name, 'prompt': 'make the cube blue', 'size': size, 'steps': max(steps, 4)}, files={'image': ('image.png', scene_image(), 'image/png')}), 'image edit')
        edit = base64.b64decode(e.json()['data'][0]['b64_json'])
        if edit[:8] != b'\x89PNG\r\n\x1a\n':
            raise RuntimeError('edit is not a PNG')
        return {'edit_png_bytes': len(edit), 'generation': 'not offered by an edit-only family'}
    r = ok(c.post('/v1/images/generations', json={'model': name, 'prompt': 'A red cube', 'size': size, 'steps': steps, 'seed': 1}), 'image generation')
    png = base64.b64decode(r.json()['data'][0]['b64_json'])
    if png[:8] != b'\x89PNG\r\n\x1a\n':
        raise RuntimeError('generation is not a PNG')
    out = {'generation_png_bytes': len(png)}
    e = c.post('/v1/images/edits', data={'model': name, 'prompt': 'a blue cube', 'size': size, 'steps': max(steps, 4), 'strength': 0.5}, files={'image': ('image.png', png, 'image/png')})
    if e.status_code == 200:
        edit = base64.b64decode(e.json()['data'][0]['b64_json'])
        if edit[:8] != b'\x89PNG\r\n\x1a\n':
            raise RuntimeError('edit is not a PNG')
        out['edit'] = 'ok'
    else:
        out['edit'] = f'not offered by this backend: HTTP {e.status_code} {e.text[:200]}'
    return out


def video(c, name, entry):
    req = entry.get('request') or {}
    body = {'model': name, 'prompt': 'A red cube', 'size': req.get('size', '64x64'), 'steps': req.get('steps', 2), 'frames': req.get('frames', 5), 'seed': 1}
    if req.get('image_to_video'):
        body['input_reference'] = data_url(scene_image())  # image-to-video families need a reference frame
    r = ok(c.post('/v1/videos/generations', json=body), 'video')
    result = r.json()
    url = result.get('url')
    if not url:
        raise RuntimeError(f'video response has no url: {str(result)[:300]}')
    mp4 = ok(c.get(url), 'video file')
    if b'ftyp' not in mp4.content[:32]:
        raise RuntimeError('video file is not an MP4 (no ftyp box)')
    if result.get('frames') != body['frames']:
        raise RuntimeError(f"frames {result.get('frames')} != requested {body['frames']}")
    return {'mp4_bytes': len(mp4.content), 'frames': result.get('frames'), 'mp4_sha256': hashlib.sha256(mp4.content).hexdigest()}


def music(c, name, entry, guard=None):
    r = c.post('/v1/music/generations', json={'model': name, 'prompt': 'a short calm piano phrase', 'duration': 10, 'steps': (entry.get('request') or {}).get('steps', 8), 'seed': 1})
    if r.status_code not in (200, 202):
        raise RuntimeError(f'music: HTTP {r.status_code}: {r.text[:1500]}')
    job = r.json()
    ident = job.get('id') or job.get('job_id') or job.get('identifier')
    if not ident:
        raise RuntimeError(f'music job without id: {job}')
    deadline = time.monotonic() + float((entry.get('request') or {}).get('job_seconds', 3600))
    try:
        # the guard (STOP, resource budgets) runs on every poll while this job is active; it knows the job id so a trip cancels exactly this job
        while True:
            if guard is not None:
                guard(music_job=ident)
            s = ok(c.get(f'/api/jobs/{ident}'), 'music job status').json()
            status = str(s.get('status') or s.get('state') or '').lower()
            if status in {'done', 'completed', 'succeeded', 'finished'}:
                return {'job': ident, 'status': status, 'keys': sorted(s.keys())}
            if status in {'failed', 'error', 'cancelled', 'canceled'}:
                # the job's own error text first and within its own bound (the adapter's placement_refused_before_load refusal must
                # survive whatever precedes 'error' in the record); the truncated record follows
                error = s.get('error')
                detail = (f'error: {str(error)[:1500]}; ' if error else '') + f'record: {str(s)[:800]}'
                raise RuntimeError(f'music job {ident} {status}: {detail}')
            if time.monotonic() > deadline:
                raise RuntimeError(f'music job {ident} still {status!r} after the request budget')
            time.sleep(2)
    except BaseException as exc:  # noqa: BLE001  (the raised data names the owned job so the cell can cancel it)
        exc.music_job = ident
        raise


def tensor(c, name, entry):
    details = ok(c.post('/api/models/details', json={'model': name}), 'tensor details').json()
    inputs = {}
    for iname, item in (details.get('inputs') or {}).items():
        shape = [int(d) if isinstance(d, int) and d > 0 else 1 for d in item['shape']]
        count = 1
        for d in shape:
            count *= d
        inputs[iname] = {'dtype': item['dtype'], 'shape': shape, 'data': [1] * count}
    if not inputs:
        raise RuntimeError(f'tensor model reports no inputs: {str(details)[:500]}')
    r = ok(c.post('/api/infer', json={'model': name, 'inputs': inputs}), 'tensor infer')
    body = r.json()
    if not isinstance(body, dict) or not body:
        raise RuntimeError('empty tensor result')
    return {'inputs': list(inputs), 'output_keys': sorted(body.keys())[:10]}


def translation(c, name, entry):
    defaults = ((entry.get('spec') or {}).get('defaults') or {})
    target = (entry.get('request') or {}).get('target_language') or defaults.get('target_language') or 'fra_Latn'
    body = {'model': name, 'text': 'The house is wonderful.', 'target_language': target}
    # a fixed-direction checkpoint (Marian, T5) is served only when the request pair equals the declared defaults pair, so send the
    # declared source too; multilingual checkpoints keep Titanos's request default (eng_Latn)
    source = (entry.get('request') or {}).get('source_language') or defaults.get('source_language')
    if source:
        body['source_language'] = source
    r = ok(c.post('/v1/translations', json=body), 'translation')
    text = r.json().get('text', '')
    if not text.strip():
        raise RuntimeError('empty translation')
    return {'text': text[:200], 'target_language': target}


def image_to_text(c, name, entry):
    r = ok(c.post('/v1/extensions/infer', json={'operation': 'ocr', 'model': name, 'image_data_url': data_url(text_image())}), 'ocr')
    text = r.json().get('text', '')
    if not text.strip():
        raise RuntimeError('empty OCR text')
    return {'text': text[:200], 'quality': {'criterion': 'recovers HELLO and WORLD', 'met': all(w in text.upper() for w in ('HELLO', 'WORLD'))}}


def object_detection(c, name, entry):
    r = ok(c.post('/v1/images/detect', json={'model': name, 'image_data_url': data_url(scene_image()), 'threshold': 0.1}), 'detection')
    objects = r.json().get('objects')
    if not isinstance(objects, list):
        raise RuntimeError('detection response has no objects list')
    for o in objects:
        if not (0 <= o['score'] <= 1 and o['box']['xmax'] >= o['box']['xmin']):
            raise RuntimeError(f'malformed detection {o}')
    return {'objects': len(objects), 'labels': sorted({o.get('label') for o in objects})[:10]}


def depth_estimation(c, name, entry):
    r = ok(c.post('/v1/extensions/infer', json={'operation': 'depth', 'model': name, 'image_data_url': data_url(scene_image())}), 'depth')
    body = r.json()
    if not isinstance(body, dict) or not body:
        raise RuntimeError('empty depth response')
    return {'keys': sorted(body.keys())[:12]}


def image_segmentation(c, name, entry):
    r = ok(c.post('/v1/extensions/infer', json={'operation': 'segmentation', 'model': name, 'image_data_url': data_url(scene_image()), 'points': [[64, 90]], 'point_labels': [1]}), 'segmentation')
    body = r.json()
    if not isinstance(body, dict) or not body:
        raise RuntimeError('empty segmentation response')
    return {'keys': sorted(body.keys())[:12]}


def sentiment(c, name, entry):
    r = ok(c.post('/v1/extensions/infer', json={'operation': 'sentiment', 'model': name, 'text': 'I really enjoyed this; it was wonderful.'}), 'sentiment')
    body = r.json()
    if not isinstance(body, dict) or not body:
        raise RuntimeError('empty sentiment response')
    return {'keys': sorted(body.keys())[:12], 'label': body.get('label')}


def rerank(c, name, entry):
    r = ok(c.post('/v1/rerank', json={'model': name, 'query': 'What is the capital of France?', 'documents': ['Paris is the capital of France.', 'Bananas are yellow.']}), 'rerank')
    body = r.json()
    results = body.get('results') or body.get('data')
    if not results:
        raise RuntimeError(f'no rerank results: {str(body)[:300]}')
    return {'results': len(results)}


def pair_classification(c, name, entry):
    r = ok(c.post('/v1/extensions/infer', json={'operation': 'pair_classification', 'model': name, 'pairs': [{'premise': 'A man is playing a guitar.', 'hypothesis': 'A person plays music.'}]}), 'pair classification')
    body = r.json()
    if not isinstance(body, dict) or not body:
        raise RuntimeError('empty pair-classification response')
    return {'keys': sorted(body.keys())[:12]}


def vad(c, name, entry):
    r = ok(c.post('/v1/audio/vad', json={'model': name, 'audio_base64': base64.b64encode(tone_wav(2.0)).decode(), 'sample_rate': 16000}), 'vad')
    body = r.json()
    if not isinstance(body, dict) or not body:
        raise RuntimeError('empty VAD response')
    return {'keys': sorted(body.keys())[:12]}


def diarization(c, name, entry):
    r = ok(c.post('/v1/extensions/infer', json={'operation': 'diarization', 'model': name, 'audio_base64': base64.b64encode(tone_wav(3.0)).decode()}), 'diarization')
    body = r.json()
    if not isinstance(body, dict) or not body:
        raise RuntimeError('empty diarization response')
    return {'keys': sorted(body.keys())[:12]}


CATALOG = {'chat': chat, 'vision': vision, 'embeddings': embeddings, 'speech': speech, 'transcription': transcription, 'image': image, 'video': video,
           'music': music, 'tensor': tensor, 'translation': translation, 'image-to-text': image_to_text, 'object-detection': object_detection,
           'depth-estimation': depth_estimation, 'image-segmentation': image_segmentation, 'sentiment': sentiment, 'rerank': rerank,
           'pair-classification': pair_classification, 'vad': vad, 'diarization': diarization}


def run(task, client, name, entry, guard=None):
    """One verified request for the task. `guard` (optional) is called on every iteration of the request's wait loops, see the module
    docstring; requests without a wait loop run exactly as before (the caller wraps them under its own monitor)."""
    if task not in CATALOG:
        raise RuntimeError(f'no request in the catalogue for task {task!r}')
    fn = CATALOG[task]
    if guard is not None and 'guard' in inspect.signature(fn).parameters:
        return fn(client, name, entry, guard=guard)
    return fn(client, name, entry)
