"""Native-passthrough endpoints (R32) added to the WP4 mocks on demand, so the typed mocks keep their pinned surfaces.

install_openai_native(mock): /v1/files (multipart create, list, get, delete, streamed content with content-disposition
and a set-cookie the broker must drop), /v1/batches (create, get), /v1/responses/compact and /v1/echo-json (the received
JSON echoed back, to prove the native lane forwards every argument untouched), GET /v1/chat/completions (list).
install_anthropic_native(mock): /v1/messages/batches (create, get) and its JSONL results stream.
install_gemini_native(mock): the resumable upload (POST /upload/v1beta/files: start answers an X-Goog-Upload-URL on
the mock's own origin; upload, finalize answers {file:{name}}), GET /v1beta/files/{id}, POST /v1beta/models/{m}:predict.
install_elevenlabs_native(mock): GET /v2/voices and POST /v1/sound-generation (binary audio).
install_openai_videos(mock): OpenAI's Videos API (create from JSON or multipart, retrieve, content, delete).

Every route keeps the mock's credential check (auth=True) and is recorded in mock.hits with surface 'native:<name>'.
State lives on the mock instance (mock.native_files, mock.native_batches) and is cleared by clear_native(mock).
"""
from __future__ import annotations

import json

from .fixtures import dumps
from .server import MockRequest, Reply, json_reply

AUDIO = b'ID3\x04\x00\x00\x00\x00\x00\x00' + b'\xff\xfb\x90\x00' * 64


def clear_native(mock) -> None:
    mock.native_files = {}
    mock.native_batches = {}
    mock.videos = {}


def _state(mock):
    if not hasattr(mock, 'native_files') or not hasattr(mock, 'videos'):
        clear_native(mock)


def _not_found(mock, what: str) -> Reply:
    return mock.error_reply(404, f'No such {what}', 'not_found')


def install_openai_native(mock) -> None:
    _state(mock)
    prefix = mock.PREFIX

    async def file_create(request: MockRequest) -> Reply:
        form = await request.form()
        upload = form.get('file')
        data = await upload.read() if upload is not None else b''
        number = len(mock.native_files) + 1
        file_id = f'file-native{number:04d}'
        record = {'id': file_id, 'object': 'file', 'bytes': len(data), 'created_at': 1767225600,
                  'filename': getattr(upload, 'filename', None) or 'upload.bin', 'purpose': form.get('purpose')}
        mock.native_files[file_id] = (record, data)
        return json_reply(record)

    async def file_list(request: MockRequest) -> Reply:
        return json_reply({'object': 'list', 'data': [record for record, _ in mock.native_files.values()]})

    async def file_get(request: MockRequest) -> Reply:
        found = mock.native_files.get(request.path_params['file_id'])
        return json_reply(found[0]) if found else _not_found(mock, 'file')

    async def file_delete(request: MockRequest) -> Reply:
        found = mock.native_files.pop(request.path_params['file_id'], None)
        if found is None:
            return _not_found(mock, 'file')
        return json_reply({'id': found[0]['id'], 'object': 'file', 'deleted': True})

    async def file_content(request: MockRequest) -> Reply:
        found = mock.native_files.get(request.path_params['file_id'])
        if found is None:
            return _not_found(mock, 'file')
        data = found[1]
        frames = [data[i:i + 7] for i in range(0, len(data), 7)] or [b'']
        return Reply(200, [('content-type', 'application/octet-stream'),
                           ('content-disposition', f'attachment; filename="{found[0]["filename"]}"'),
                           ('set-cookie', 'mock_session=native-0001; Path=/')], frames=frames, stream_kind='binary')

    async def batch_create(request: MockRequest) -> Reply:
        body = request.json() or {}
        number = len(mock.native_batches) + 1
        batch = {'id': f'batch_native{number:04d}', 'object': 'batch', 'endpoint': body.get('endpoint'),
                 'input_file_id': body.get('input_file_id'), 'status': 'validating'}
        mock.native_batches[batch['id']] = batch
        return json_reply(batch)

    async def batch_get(request: MockRequest) -> Reply:
        found = mock.native_batches.get(request.path_params['batch_id'])
        return json_reply(found) if found else _not_found(mock, 'batch')

    async def echo_json(request: MockRequest) -> Reply:
        return json_reply({'object': 'echo', 'received': request.json(), 'query': request.query,
                           'content_type': request.headers.get('content-type'),
                           'openai_beta': request.headers.get('openai-beta')})

    async def stored_list(request: MockRequest) -> Reply:
        return json_reply({'object': 'list', 'data': [], 'has_more': False})

    mock.route('POST', prefix + r'/files', file_create, 'native:files')
    mock.route('GET', prefix + r'/files', file_list, 'native:files')
    mock.route('GET', prefix + r'/files/(?P<file_id>[^/]+)/content', file_content, 'native:files')
    mock.route('GET', prefix + r'/files/(?P<file_id>[^/]+)', file_get, 'native:files')
    mock.route('DELETE', prefix + r'/files/(?P<file_id>[^/]+)', file_delete, 'native:files')
    mock.route('POST', prefix + r'/batches', batch_create, 'native:batches')
    mock.route('GET', prefix + r'/batches/(?P<batch_id>[^/]+)', batch_get, 'native:batches')
    mock.route('POST', prefix + r'/responses/compact', echo_json, 'native:responses')
    mock.route('POST', prefix + r'/responses/input_tokens', echo_json, 'native:responses')
    mock.route('GET', prefix + r'/chat/completions', stored_list, 'native:chat_completions_stored')


def install_anthropic_native(mock) -> None:
    _state(mock)

    async def batch_create(request: MockRequest) -> Reply:
        body = request.json() or {}
        number = len(mock.native_batches) + 1
        batch = {'id': f'msgbatch_native{number:04d}', 'type': 'message_batch', 'processing_status': 'in_progress',
                 'request_counts': {'processing': len(body.get('requests') or [])}}
        mock.native_batches[batch['id']] = (batch, body.get('requests') or [])
        return json_reply(batch)

    async def batch_get(request: MockRequest) -> Reply:
        found = mock.native_batches.get(request.path_params['batch_id'])
        return json_reply(found[0]) if found else _not_found(mock, 'batch')

    async def batch_results(request: MockRequest) -> Reply:
        found = mock.native_batches.get(request.path_params['batch_id'])
        if found is None:
            return _not_found(mock, 'batch')
        lines = [(dumps({'custom_id': item.get('custom_id'), 'result': {'type': 'succeeded'}}) + '\n').encode('utf-8')
                 for item in found[1]]
        return Reply(200, [('content-type', 'application/binary')], frames=lines or [b''], stream_kind='binary')

    mock.route('POST', r'/v1/messages/batches', batch_create, 'native:batches')
    mock.route('GET', r'/v1/messages/batches/(?P<batch_id>[^/]+)/results', batch_results, 'native:batches')
    mock.route('GET', r'/v1/messages/batches/(?P<batch_id>[^/]+)', batch_get, 'native:batches')


def install_gemini_native(mock) -> None:
    _state(mock)

    async def upload(request: MockRequest) -> Reply:
        command = (request.headers.get('x-goog-upload-command') or '').lower()
        if command == 'start':
            url = f'{mock.base_url}/upload/v1beta/files?upload_id=native-upload-0001&upload_protocol=resumable'
            return Reply(200, [('content-type', 'application/json'), ('x-goog-upload-url', url),
                               ('x-goog-upload-status', 'active'), ('x-goog-upload-chunk-granularity', '8388608')],
                         b'')
        if 'finalize' in command:
            if request.q('upload_id') != 'native-upload-0001':
                return mock.error_reply(400, 'Unknown upload id', 'INVALID_ARGUMENT')
            number = len(mock.native_files) + 1
            name = f'files/native{number:04d}'
            record = {'name': name, 'sizeBytes': str(len(request.body)), 'mimeType': 'audio/wav', 'state': 'ACTIVE',
                      'uri': f'{mock.api_base}/{name}'}
            mock.native_files[name.split('/', 1)[1]] = (record, request.body)
            return Reply(200, [('content-type', 'application/json'), ('x-goog-upload-status', 'final')],
                         dumps({'file': record}).encode('utf-8'))
        return mock.error_reply(400, 'Unknown upload command', 'INVALID_ARGUMENT')

    async def file_get(request: MockRequest) -> Reply:
        found = mock.native_files.get(request.path_params['file_id'])
        return json_reply(found[0]) if found else _not_found(mock, 'file')

    async def predict(request: MockRequest) -> Reply:
        body = request.json() or {}
        return json_reply({'predictions': [{'mimeType': 'image/png', 'bytesBase64Encoded': 'iVBORw0KGgo='}],
                           'model': request.path_params['model'], 'instances': len(body.get('instances') or [])})

    mock.route('POST', r'/upload/v1beta/files', upload, 'native:files')
    mock.route('GET', r'/v1beta/files/(?P<file_id>[^/]+)', file_get, 'native:files')
    mock.route('POST', r'/v1beta/models/(?P<model>[^/:]+):predict', predict, 'native:imagen')


def install_elevenlabs_native(mock) -> None:
    _state(mock)

    async def voices_v2(request: MockRequest) -> Reply:
        return json_reply({'voices': [{'voice_id': 'nativeVoice0001', 'name': 'Native'}], 'has_more': False,
                           'total_count': 1, 'next_page_token': None, 'query': request.query})

    async def sound(request: MockRequest) -> Reply:
        body = request.json() or {}
        if not body.get('text'):
            return mock.error_reply(422, 'text is required', 'invalid_request')
        frames = [AUDIO[i:i + 64] for i in range(0, len(AUDIO), 64)]
        return Reply(200, [('content-type', 'audio/mpeg'), ('history-item-id', 'hist-native-0001'),
                           ('character-cost', str(len(body['text'])))], frames=frames, stream_kind='binary')

    mock.route('GET', r'/v2/voices', voices_v2, 'native:voices')
    mock.route('POST', r'/v1/sound-generation', sound, 'native:sound_effects')


VIDEO_MP4 = b'\x00\x00\x00\x18ftypmp42' + b'\x00' * 64


def install_openai_videos(mock) -> None:
    """OpenAI's Videos API on the mock: POST /v1/videos (JSON, or multipart with an input_reference file: the answer's
    `received` echoes the fields and the file's byte count), GET /v1/videos/{id}, GET /v1/videos/{id}/content (video/mp4,
    VIDEO_MP4) and DELETE /v1/videos/{id}. State: mock.videos (cleared by clear_native)."""
    _state(mock)

    async def create(request: MockRequest) -> Reply:
        if 'multipart/form-data' in (request.headers.get('content-type') or ''):
            form = await request.form()
            reference = form.get('input_reference')
            fields = {k: v for k, v in form.items() if k != 'input_reference'}
            fields['reference_bytes'] = len(await reference.read()) if reference is not None else 0
        else:
            fields = request.json() or {}
        video_id = f'video_mock{len(mock.videos) + 1:04d}'
        record = {'id': video_id, 'object': 'video', 'model': fields.get('model'), 'status': 'queued', 'progress': 0,
                  'created_at': 1767225600, 'seconds': str(fields.get('seconds', '4')),
                  'size': fields.get('size', '720x1280'), 'received': fields}
        mock.videos[video_id] = record
        return json_reply(record)

    async def get(request: MockRequest) -> Reply:
        found = mock.videos.get(request.path_params['video_id'])
        return json_reply(found) if found else _not_found(mock, 'video')

    async def content(request: MockRequest) -> Reply:
        if request.path_params['video_id'] not in mock.videos:
            return _not_found(mock, 'video')
        return Reply(200, [('content-type', 'video/mp4')], frames=[VIDEO_MP4], stream_kind='binary')

    async def delete(request: MockRequest) -> Reply:
        found = mock.videos.pop(request.path_params['video_id'], None)
        if found is None:
            return _not_found(mock, 'video')
        return json_reply({'id': found['id'], 'object': 'video.deleted', 'deleted': True})

    prefix = mock.PREFIX
    mock.route('POST', prefix + r'/videos', create, 'native:videos')
    mock.route('GET', prefix + r'/videos/(?P<video_id>[^/]+)/content', content, 'native:videos')
    mock.route('GET', prefix + r'/videos/(?P<video_id>[^/]+)', get, 'native:videos')
    mock.route('DELETE', prefix + r'/videos/(?P<video_id>[^/]+)', delete, 'native:videos')


def received(hit) -> dict:
    """The JSON document a native echo route received (decoded from the recorded body)."""
    return json.loads(hit.body.decode('utf-8')) if hit.body else {}


__all__ = ['AUDIO', 'VIDEO_MP4', 'clear_native', 'install_anthropic_native', 'install_elevenlabs_native',
           'install_gemini_native', 'install_openai_native', 'install_openai_videos', 'received']
