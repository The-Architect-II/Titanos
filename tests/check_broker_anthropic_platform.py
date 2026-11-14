"""Anthropic platform over the integrated app (create_app: KeylessHostGuard, Guard + Authenticator, BrokerDispatch) against
the WP4 Anthropic mock (Files, Message Batches, models with the model-object fields, per-model request rules). The
netguard is installed and every test ends with zero non-loopback connection attempts. Synthetic sentinel credentials only.

- Files API through the native passthrough with the stock SDK's headers (x-api-key = the caller's tk_ key,
  anthropic-version): the multipart upload reaches the provider with the upstream credential and binds the file to the
  caller; GET /v1/files/{id} reaches the provider (not Titanos' artifact route) for the Anthropic SDK and stays the
  artifact route for an OpenAI-style client; another key gets 404; a non-downloadable file's content answers the
  provider's 400; a generated (downloadable) file streams with content-disposition; DELETE answers file_deleted;
- Messages that name provider-side state: a file source or a container without cloud:store is 403 scope_denied before
  egress; another key's uploaded file is 404 not_found before egress; the owner's own file reaches the provider;
- Message Batches: create binds the batch; the results_url of a retrieved batch is rewritten to this server's
  /native/<upstream>/ path and following it with the same x-api-key streams the JSONL results; a batch naming another
  key's file is 404 before egress; listing batches needs broker:admin; deleting an unfinished batch answers the
  provider's 400;
- the Anthropic-shaped catalog: GET /v1/models and /v1/models/{id} with anthropic-version (x-api-key accepted) answer
  {data, has_more, first_id, last_id} with the provider's display_name, max_input_tokens and capabilities from the live
  catalog, paged with limit and after_id; an unknown id is 404 in Anthropic's envelope; other clients keep the OpenAI
  shape;
- an OpenAI chat client reaching Claude Opus 5.5: reasoning_effort becomes output_config.effort, temperature is dropped
  and named in X-Titanos-Dropped, tool_choice required degrades to auto, and the mock (which applies the provider's model
  rules) answers 200; the omitted-display thinking block comes back as reasoning_signature and is replayed on the next
  turn;
- the stock Anthropic SDK's bare model id (claude-opus-5-5, no @upstream) resolves on the single Anthropic upstream;
- Managed Agents rows are broker:admin only and the Admin API is reachable under /native/<upstream>/ only.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import broker_wp7_support as W  # noqa: E402  (installs the netguard)
from broker_mocks import AnthropicMock, netguard  # noqa: E402

BOOT = 'anthropic-check-bootstrap-0001'
MOCKS: dict = {}
NATIVE = ('native', 'cloud', 'cloud:store', 'inference', 'models:read')
VERSION = {'anthropic-version': '2023-06-01'}
MODELS = ['claude-sonnet-4-5-20250929', 'claude-opus-5-5']


def setUpModule():
    MOCKS['anthropic'] = AnthropicMock()
    MOCKS['anthropic'].start()


def tearDownModule():
    MOCKS['anthropic'].stop()


def document(*, allow_store: bool = True) -> dict:
    return {'policy': {'cloud_policy': 'explicit'}, 'upstreams': {
        'ant': W.S.upstream(MOCKS['anthropic'], 'anthropic', credential='ant-key', models=MODELS, catalog='live',
                            data_policy={'allow_store': allow_store})}}


def sdk(token: str, **extra) -> dict:
    return {'x-api-key': token, **VERSION, **extra}


class Base(unittest.TestCase):
    def setUp(self):
        netguard.reset_attempts()
        self.mock = MOCKS['anthropic']
        self.mock.reset()
        self.full = W.FullApp(document(), api_key=BOOT)
        self.http = self.full.http

    def tearDown(self):
        self.full.close()
        self.assertEqual(netguard.attempts(), [], 'a non-loopback connect was attempted')

    def key(self, *scopes, name='k') -> str:
        return self.full.mint(*scopes, name=name)[1]

    def upload(self, token, content=b'%PDF-1.4\n', name='report.pdf'):
        return self.http.post('/v1/files', files={'file': (name, content, 'application/pdf')}, headers=sdk(token))

    def message(self, token, content, **extra):
        body = {'model': 'claude-sonnet-4-5-20250929@ant', 'max_tokens': 32,
                'messages': [{'role': 'user', 'content': content}], **extra}
        return self.http.post('/v1/messages', json=body, headers=sdk(token))


class Files(Base):
    def test_upload_metadata_ownership_content_and_delete(self):
        owner, other = self.key(*NATIVE, name='owner'), self.key(*NATIVE, name='other')
        answer = self.upload(owner)
        self.assertEqual(answer.status_code, 200, answer.text)
        file_id = answer.json()['id']
        self.assertEqual(answer.json()['type'], 'file')
        hit = self.mock.hits[-1]
        self.assertEqual((hit.method, hit.path, hit.credential), ('POST', '/v1/files', '<sentinel:anthropic>'))
        self.assertIn(b'%PDF-1.4', hit.body)
        self.assertFalse(hit.presented(owner))
        self.assertFalse(any(name.startswith('x-titanos') for name in hit.headers))
        meta = self.http.get(f'/v1/files/{file_id}', headers=sdk(owner))
        self.assertEqual((meta.status_code, meta.json().get('id')), (200, file_id), meta.text)
        # the id is native-bound to this key, so its owner reaches the provider from any client (Authorization too);
        # an unknown id from an OpenAI-style client keeps Titanos' artifact route
        own = self.http.get(f'/v1/files/{file_id}', headers={'Authorization': 'Bearer ' + owner})
        self.assertEqual(own.status_code, 200, own.text)
        local = self.http.get('/v1/files/file-not-uploaded-here', headers={'Authorization': 'Bearer ' + owner})
        self.assertEqual(local.status_code, 404)
        before = len(self.mock.hits)
        foreign = self.http.get(f'/v1/files/{file_id}', headers=sdk(other))
        self.assertEqual((foreign.status_code, foreign.json()['error']['type']), (404, 'not_found_error'))
        self.assertEqual(len(self.mock.hits), before)
        content = self.http.get(f'/v1/files/{file_id}/content', headers=sdk(owner))
        self.assertEqual(content.status_code, 400)
        self.assertIn('not downloadable', content.json()['error']['message'])
        generated = self.mock.add_generated_file()
        out = self.http.get(f'/v1/files/{generated}/content', headers=sdk(BOOT))
        self.assertEqual((out.status_code, out.content), (200, b'x,y\n1,2\n'), out.text)
        self.assertIn('output.csv', out.headers.get('content-disposition', ''))
        gone = self.http.delete(f'/v1/files/{file_id}', headers=sdk(owner))
        self.assertEqual((gone.status_code, gone.json()), (200, {'id': file_id, 'type': 'file_deleted'}))
        listing = self.http.get('/v1/files', headers=sdk(owner))
        self.assertEqual(listing.status_code, 403)  # the whole account's files: broker:admin only
        listing = self.http.get('/v1/files', headers=sdk(BOOT))
        self.assertEqual((listing.status_code, set(listing.json())), (200, {'data', 'next_page'}))


class ProviderState(Base):
    def test_file_sources_and_containers_are_store_gated_and_owner_bound(self):
        owner, other = self.key(*NATIVE, name='owner'), self.key(*NATIVE, name='other')
        plain = self.key('inference', 'cloud', name='plain')
        file_id = self.upload(owner).json()['id']
        document_block = [{'type': 'document', 'source': {'type': 'file', 'file_id': file_id}},
                          {'type': 'text', 'text': 'Summarize.'}]
        before = len(self.mock.hits)
        refused = self.message(plain, document_block)
        self.assertEqual((refused.status_code, refused.json()['error']['type']), (403, 'permission_error'))
        self.assertIn('cloud:store', refused.json()['error']['message'])
        self.assertEqual(refused.headers.get('x-titanos-error-origin'), 'policy')
        refused = self.message(plain, 'hi', container='container_mock0001')
        self.assertEqual(refused.status_code, 403)
        foreign = self.message(other, document_block)
        self.assertEqual(foreign.status_code, 404, foreign.text)
        self.assertEqual(len(self.mock.hits), before)
        allowed = self.message(owner, document_block)
        self.assertEqual(allowed.status_code, 200, allowed.text)
        sent = self.mock.hits[-1].json()
        self.assertEqual(sent['messages'][0]['content'][0]['source']['file_id'], file_id)
        # anthropic skills are not account state; plain inference passes
        self.assertEqual(self.message(plain, 'hi', container={'skills': [
            {'type': 'anthropic', 'skill_id': 'pptx', 'version': 'latest'}]}).status_code, 200)


class Batches(Base):
    def test_results_url_is_rewritten_and_followed(self):
        token = self.key(*NATIVE)
        body = {'requests': [{'custom_id': 'r1', 'params': {'model': 'claude-opus-5-5', 'max_tokens': 16,
                                                              'messages': [{'role': 'user', 'content': 'hi'}]}},
                             {'custom_id': 'r2', 'params': {'model': 'claude-opus-5-5', 'max_tokens': 16,
                                                              'temperature': 0.2,
                                                              'messages': [{'role': 'user', 'content': 'hi'}]}}]}
        created = self.http.post('/v1/messages/batches', json=body, headers=sdk(token))
        self.assertEqual(created.status_code, 200, created.text)
        batch_id = created.json()['id']
        self.assertEqual(created.json()['processing_status'], 'in_progress')
        got = self.http.get(f'/v1/messages/batches/{batch_id}', headers=sdk(token))
        self.assertEqual(got.status_code, 200, got.text)
        url = got.json()['results_url']
        self.assertEqual(url, f'http://127.0.0.1:8000/native/ant/v1/messages/batches/{batch_id}/results')
        self.assertNotIn(str(self.mock.server.port), got.text)
        results = self.http.get(url.split('8000', 1)[1], headers=sdk(token))
        self.assertEqual(results.status_code, 200, results.text)
        lines = {json.loads(line)['custom_id']: json.loads(line)['result'] for line in results.text.splitlines()}
        self.assertEqual(lines['r1']['type'], 'succeeded')
        self.assertEqual(lines['r2']['type'], 'errored')  # the provider's model rule (no sampling on Opus 5.5)
        listed = self.http.get('/v1/messages/batches', headers=sdk(token))
        self.assertEqual(listed.status_code, 403)
        second = self.http.post('/v1/messages/batches', json=body, headers=sdk(token)).json()['id']
        self.mock.batches[second]['processing_status'] = 'in_progress'
        busy = self.http.delete(f'/v1/messages/batches/{second}', headers=sdk(token))
        self.assertEqual(busy.status_code, 400)
        done = self.http.delete(f'/v1/messages/batches/{batch_id}', headers=sdk(token))
        self.assertEqual(done.json(), {'id': batch_id, 'type': 'message_batch_deleted'})

    def test_batch_naming_another_keys_file_is_refused(self):
        owner, other = self.key(*NATIVE, name='owner'), self.key(*NATIVE, name='other')
        file_id = self.upload(owner).json()['id']
        body = {'requests': [{'custom_id': 'r1', 'params': {'model': 'claude-opus-5-5', 'max_tokens': 16, 'messages': [
            {'role': 'user', 'content': [{'type': 'document', 'source': {'type': 'file', 'file_id': file_id}}]}]}}]}
        before = len(self.mock.hits)
        refused = self.http.post('/v1/messages/batches', json=body, headers=sdk(other))
        self.assertEqual(refused.status_code, 404, refused.text)
        self.assertEqual(len(self.mock.hits), before)
        self.assertEqual(self.http.post('/v1/messages/batches', json=body, headers=sdk(owner)).status_code, 200)


class Catalog(Base):
    def test_anthropic_shaped_models(self):
        token = self.key(*NATIVE)
        listing = self.http.get('/v1/models', headers=sdk(token))
        self.assertEqual(listing.status_code, 200, listing.text)
        document = listing.json()
        self.assertEqual(set(document), {'data', 'has_more', 'first_id', 'last_id'})
        rows = {row['id']: row for row in document['data']}
        # the single Anthropic upstream's models are listed by the provider's bare ids (bare_model_ids resolves them)
        sonnet = rows['claude-sonnet-4-5-20250929']
        self.assertEqual((sonnet['type'], sonnet['display_name'], sonnet['max_input_tokens']),
                         ('model', 'Claude Sonnet 4.5', 200000))
        self.assertTrue(sonnet['capabilities']['image_input']['supported'])
        self.assertEqual(rows['claude-opus-5-5']['display_name'], 'claude-opus-5-5')
        self.assertFalse(any('@' in name for name in rows), sorted(rows))
        bare = self.http.get('/v1/models/claude-sonnet-4-5-20250929', headers=sdk(token))
        self.assertEqual((bare.status_code, bare.json()['id'], bare.json()['max_tokens']),
                         (200, 'claude-sonnet-4-5-20250929', 64000), bare.text)
        first = self.http.get('/v1/models?limit=1', headers=sdk(token)).json()
        self.assertEqual((len(first['data']), first['has_more']), (1, True))
        rest = self.http.get(f'/v1/models?limit=1000&after_id={first["last_id"]}', headers=sdk(token)).json()
        self.assertNotIn(first['last_id'], [row['id'] for row in rest['data']])
        one = self.http.get('/v1/models/claude-sonnet-4-5-20250929@ant', headers=sdk(token))
        self.assertEqual((one.status_code, one.json()['max_tokens']), (200, 64000))
        missing = self.http.get('/v1/models/no-such-model', headers=sdk(token))
        self.assertEqual((missing.status_code, missing.json()['type'], missing.json()['error']['type']),
                         (404, 'error', 'not_found_error'))
        openai = self.http.get('/v1/models', headers={'Authorization': 'Bearer ' + token})
        self.assertEqual(openai.json()['object'], 'list')


class Translated(Base):
    TOOLS = [{'type': 'function', 'function': {'name': 'get_weather', 'parameters': {
        'type': 'object', 'properties': {'location': {'type': 'string'}}}}}]

    def test_openai_client_to_opus_5_5(self):
        token = self.key('inference', 'cloud')
        body = {'model': 'claude-opus-5-5@ant', 'reasoning_effort': 'high', 'temperature': 0.5,
                'tools': self.TOOLS, 'tool_choice': 'required',
                'messages': [{'role': 'user', 'content': 'Weather in Paris?'}]}
        answer = self.http.post('/v1/chat/completions', json=body, headers={'Authorization': 'Bearer ' + token})
        self.assertEqual(answer.status_code, 200, answer.text)
        sent = self.mock.hits[-1].json()
        self.assertEqual(sent['model'], 'claude-opus-5-5')
        self.assertEqual(sent['output_config'], {'effort': 'high'})
        self.assertEqual(sent['thinking'], {'type': 'adaptive', 'display': 'summarized'})
        self.assertEqual(sent['tool_choice'], {'type': 'auto'})
        self.assertNotIn('temperature', sent)
        self.assertGreaterEqual(sent['max_tokens'], 16000)
        self.assertIn('temperature', answer.headers.get('x-titanos-dropped', ''))
        # without reasoning_effort the model thinks with its default display (omitted): the signed empty block
        body = {k: v for k, v in body.items() if k != 'reasoning_effort'}
        answer = self.http.post('/v1/chat/completions', json=body, headers={'Authorization': 'Bearer ' + token})
        self.assertEqual(answer.status_code, 200, answer.text)
        self.assertNotIn('thinking', self.mock.hits[-1].json())
        message = answer.json()['choices'][0]['message']
        self.assertEqual(message['reasoning_signature'], 'mock-signature-omitted-0001')
        follow = dict(body, messages=body['messages'] + [message, {
            'role': 'tool', 'tool_call_id': message['tool_calls'][0]['id'], 'content': 'sunny'}])
        again = self.http.post('/v1/chat/completions', json=follow, headers={'Authorization': 'Bearer ' + token})
        self.assertEqual(again.status_code, 200, again.text)
        replayed = self.mock.hits[-1].json()['messages'][1]['content'][0]
        self.assertEqual(replayed, {'type': 'thinking', 'thinking': '', 'signature': 'mock-signature-omitted-0001'})

    def test_bare_model_id_from_the_anthropic_sdk(self):
        token = self.key('inference', 'cloud')
        answer = self.http.post('/v1/messages', json={'model': 'claude-opus-5-5', 'max_tokens': 32, 'messages': [
            {'role': 'user', 'content': 'hi'}]}, headers=sdk(token))
        self.assertEqual(answer.status_code, 200, answer.text)
        self.assertEqual(answer.headers.get('x-titanos-served-by'), 'cloud:ant')
        self.assertEqual(self.mock.hits[-1].json()['model'], 'claude-opus-5-5')


class AdminRows(Base):
    def test_managed_agents_and_admin_api(self):
        token = self.key(*NATIVE)
        refused = self.http.get('/v1/agents', headers=sdk(token))
        self.assertEqual(refused.status_code, 403)
        before = len(self.mock.hits)
        reached = self.http.get('/v1/agents', headers=sdk(BOOT))
        self.assertEqual(len(self.mock.hits), before + 1)
        self.assertEqual((reached.status_code, self.mock.hits[-1].path), (404, '/v1/agents'))
        admin = self.http.get('/native/ant/v1/organizations/me', headers={'Authorization': 'Bearer ' + BOOT})
        self.assertEqual(self.mock.hits[-1].path, '/v1/organizations/me')
        self.assertEqual(admin.status_code, 404)


if __name__ == '__main__':
    unittest.main()
