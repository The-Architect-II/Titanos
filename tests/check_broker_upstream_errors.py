"""Broker upstream error mapping (WP5): translate/upstream_errors and the ratified R10 credential marking.

- the status-to-code table (400/404/413/422 passthrough, 401 and a key-rejecting 403 -> 502 upstream_auth_failed, a plain
  403 in the provider's own envelope passed through as 403, 408 -> 504, 429 -> 429 with
  the upstream Retry-After, 402 and 5xx -> 503, 3xx -> 502 upstream_redirect_refused, other 4xx -> 502
  upstream_bad_response; a Gemini 400 API_KEY_INVALID is an auth failure);
- Retry-After preserved (delta-seconds, HTTP date, retry-after-ms, Gemini RetryInfo);
- 4xx passthrough bodies are scrubbed of registered secrets and the leak sentinel, rendered deterministically in every
  inbound envelope (OpenAI keeps the provider's plain error code);
- x-should-retry: false on 502 upstream_auth_failed and 508 loop_detected (and never on retryable codes);
- in-stream error classification (error-first streams) and httpx network errors;
- R10: through the real EgressClient and the WP4 mocks, a credential is marked failed:upstream_auth on every 401, on a 403
  only when the provider body carries its auth error code (OpenAI-compatible error.code invalid_api_key, Anthropic
  authentication_error, Gemini API_KEY_INVALID, ElevenLabs detail.status invalid_api_key) and on Gemini's 400
  API_KEY_INVALID; a plain 403 never takes the credential offline (streamed calls included);
- live lane follow-up: a provider 404 or 410 (a 410 Gone for a retired model) is a passthrough
  client error answering 404, 404 model_not_found through model_failure with a message naming what the provider
  reported, in every envelope, and never marks the credential;
- a 404 becomes model_not_found only when it is about the model (OpenAI code model_not_found, Azure DeploymentNotFound,
  Anthropic not_found_error naming the model, Gemini NOT_FOUND naming models/<id>, OpenAI-compatible
  messages naming the model or its id, a function-not-found sentence); a 404 about a voice
  (ElevenLabs voice_not_found), a file, a response id, or a bare 404 stays not_found; a 410 on a model call always is.
Loopback mocks only (netguard installed); synthetic sentinels only.
"""
import json
import unittest
from email.utils import formatdate
import time

import broker_adapter_support as S  # installs the netguard first
from broker_mocks import SENTINELS, netguard

import httpx

from titanos.broker.errors import BrokerError, error_headers
from titanos.broker.adapters.base import RawRequest
from titanos.broker.surfaces import SURFACES
from titanos.broker.translate import upstream_errors as ue
from titanos.broker.translate.upstream_errors import UpstreamFailure, classify
from titanos.broker.types import UpstreamError
from titanos.broker.vault import Secret

Secret(SENTINELS['leak'])  # registered like every resolved secret, so scrub() removes it wherever it appears
Secret(SENTINELS['openai'])


def body(adapter, status, message='upstream message', code=None):
    if adapter == 'anthropic':
        return json.dumps({'type': 'error', 'error': {'type': code or 'invalid_request_error', 'message': message}}).encode()
    if adapter == 'gemini':
        return json.dumps({'error': {'code': status, 'message': message, 'status': code or 'INVALID_ARGUMENT'}}).encode()
    if adapter == 'elevenlabs':
        return json.dumps({'detail': {'status': code or 'error', 'message': message}}).encode()
    if adapter == 'ollama_native':
        return json.dumps({'error': message}).encode()
    return json.dumps({'error': {'message': message, 'type': 'invalid_request_error', 'param': None,
                                 'code': code}}).encode()


GEMINI_KEY_INVALID = json.dumps({'error': {'code': 400, 'message': 'API key not valid. Please pass a valid API key.',
                                           'status': 'INVALID_ARGUMENT', 'details': [
                                               {'@type': 'type.googleapis.com/google.rpc.ErrorInfo',
                                                'reason': 'API_KEY_INVALID', 'domain': 'googleapis.com'}]}}).encode()


class Table(unittest.TestCase):
    def setUp(self):
        netguard.reset_attempts()

    def tearDown(self):
        self.assertEqual(netguard.attempts(), [])

    def test_status_to_code_table(self):
        cases = [
            (400, 'openai_compat', None, 'invalid_request', 400, 'client_error'),
            (404, 'openai_compat', 'model_not_found', 'model_not_found', 404, 'client_error'),
            (404, 'ollama_native', None, 'not_found', 404, 'client_error'),
            (413, 'anthropic', 'request_too_large', 'request_too_large', 413, 'client_error'),
            (422, 'elevenlabs', None, 'invalid_request', 422, 'client_error'),
            (401, 'openai_compat', 'invalid_api_key', 'upstream_auth_failed', 502, 'auth'),
            (403, 'anthropic', 'permission_error', 'invalid_request', 403, 'client_error'),  # a plain 403: passed through
            (403, 'anthropic', 'authentication_error', 'upstream_auth_failed', 502, 'auth'),
            (408, 'openai_compat', None, 'upstream_timeout', 504, 'timeout'),
            (429, 'gemini', 'RESOURCE_EXHAUSTED', 'upstream_rate_limited', 429, 'rate_limited'),
            (402, 'openai_compat', None, 'upstream_unavailable', 503, 'server_error'),
            (307, 'openai_compat', None, 'upstream_redirect_refused', 502, 'client_error'),
            (409, 'openai_compat', None, 'upstream_bad_response', 502, 'client_error'),
            (405, 'anthropic', None, 'upstream_bad_response', 502, 'client_error'),
        ] + [(status, adapter, None, 'upstream_unavailable', 503, 'server_error')
             for status in (500, 502, 503, 504, 529) for adapter in ue.ADAPTERS]
        for status, adapter, code, expected, http, kind in cases:
            with self.subTest(status=status, adapter=adapter):
                err = classify(status, {}, body(adapter, status, code=code), adapter=adapter)
                self.assertEqual(err.kind, kind)
                self.assertEqual(ue.broker_code(err), (expected, http))
                failure = UpstreamFailure(err)
                self.assertEqual((failure.code, failure.status, failure.origin), (expected, http, 'upstream'))
                self.assertEqual(failure.passthrough, status in (400, 404, 413, 422) or (status, kind) == (403, 'client_error'))
        gemini_key = classify(400, {}, GEMINI_KEY_INVALID, adapter='gemini')
        self.assertEqual((gemini_key.kind, ue.broker_code(gemini_key)), ('auth', ('upstream_auth_failed', 502)))
        self.assertEqual(gemini_key.provider_code, 'API_KEY_INVALID')

    def test_retry_after_is_preserved(self):
        self.assertEqual(ue.retry_after({'retry-after': '7'}), 7.0)
        self.assertEqual(ue.retry_after(httpx.Headers({'Retry-After': '2'})), 2.0)
        self.assertEqual(ue.retry_after({'retry-after-ms': '1500'}), 1.5)
        when = formatdate(time.time() + 30, usegmt=True)
        self.assertAlmostEqual(ue.retry_after({'retry-after': when}), 30, delta=2)
        retry_info = json.dumps({'error': {'code': 429, 'message': 'quota', 'status': 'RESOURCE_EXHAUSTED', 'details': [
            {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '37s'}]}}).encode()
        self.assertEqual(ue.retry_after({}, retry_info), 37.0)
        err = classify(429, {'retry-after': '11'}, body('openai_compat', 429), adapter='openai_compat')
        failure = UpstreamFailure(err)
        self.assertEqual(failure.retry_after, 11.0)
        self.assertEqual(failure.response_headers()['Retry-After'], '11')
        self.assertNotIn('x-should-retry', failure.response_headers())
        self.assertIsNone(ue.retry_after({'retry-after': 'soon'}))

    def test_passthrough_bodies_are_scrubbed(self):
        leak, key = SENTINELS['leak'], SENTINELS['openai']
        for adapter in ue.ADAPTERS:
            with self.subTest(adapter=adapter):
                raw = body(adapter, 400, f'bad field; provided key {key} rejected; trace {leak}; Bearer {"z" * 24}')
                err = classify(400, {}, raw, adapter=adapter)
                for protocol in ('openai', 'anthropic', 'gemini', 'elevenlabs', 'ollama', 'responses'):
                    response = ue.passthrough_4xx(err, protocol)
                    text = response.body.decode('utf-8')
                    self.assertNotIn(leak, text)
                    self.assertNotIn(key, text)
                    self.assertNotIn('z' * 24, text)
                    self.assertIn('bad field', text)
                    self.assertEqual(response.status_code, 400)
                    self.assertEqual(response.headers['x-titanos-error-origin'], 'upstream')
        long = classify(400, {}, body('openai_compat', 400, 'x' * 5000), adapter='openai_compat')
        self.assertLessEqual(len(long.message), ue.MESSAGE_MAX + 3)

    def test_per_protocol_envelopes(self):
        err = classify(400, {}, body('openai_compat', 400, 'too long', 'context_length_exceeded'),
                       adapter='openai_compat')
        shapes = {protocol: json.loads(ue.passthrough_4xx(err, protocol).body) for protocol in
                  ('openai', 'responses', 'anthropic', 'gemini', 'elevenlabs', 'ollama')}
        # OpenAI-shaped clients keep the provider's error.type (and error.param when it names one)
        self.assertEqual(shapes['openai'], {'error': {'message': 'too long', 'type': 'invalid_request_error',
                                                      'param': None, 'code': 'context_length_exceeded'}})
        self.assertEqual(shapes['responses'], shapes['openai'])
        self.assertEqual(shapes['anthropic'], {'type': 'error', 'error': {'type': 'invalid_request_error',
                                                                          'message': 'too long'}})
        self.assertEqual(shapes['gemini'], {'error': {'code': 400, 'message': 'too long', 'status': 'INVALID_ARGUMENT'}})
        self.assertEqual(shapes['elevenlabs'], {'detail': {'status': 'invalid_request', 'message': 'too long'}})
        self.assertEqual(shapes['ollama'], {'error': 'too long'})
        missing = classify(404, {}, body('anthropic', 404, 'model: x', 'not_found_error'), adapter='anthropic')
        self.assertEqual(json.loads(ue.passthrough_4xx(missing, 'anthropic').body)['error']['type'], 'not_found_error')
        self.assertEqual(json.loads(ue.passthrough_4xx(missing, 'gemini').body)['error']['status'], 'NOT_FOUND')
        with self.assertRaises(ValueError):
            ue.passthrough_4xx(classify(503, {}, b'', adapter='openai_compat'), 'openai')
        self.assertIsInstance(ue.to_broker_error(err, 'anthropic'), BrokerError)
        with self.assertRaises(ValueError):
            ue.to_broker_error(err, 'smtp')

    def test_no_retry_headers(self):
        auth = UpstreamFailure(classify(401, {}, body('openai_compat', 401), adapter='openai_compat'))
        headers = auth.response_headers()
        self.assertEqual(headers['x-should-retry'], 'false')
        self.assertEqual(headers['X-Titanos-Error-Origin'], 'upstream')
        self.assertEqual(error_headers('loop_detected')['x-should-retry'], 'false')
        self.assertEqual(BrokerError('loop', 'loop_detected').status, 508)
        for retryable in ('upstream_unavailable', 'upstream_timeout', 'upstream_rate_limited', 'upstream_bad_response'):
            self.assertNotIn('x-should-retry', error_headers(retryable), retryable)
        self.assertEqual(error_headers('upstream_redirect_refused')['x-should-retry'], 'false')

    def test_stream_errors(self):
        cases = [
            ({'type': 'error', 'error': {'type': 'overloaded_error', 'message': 'Overloaded'}}, 'anthropic',
             'upstream_unavailable'),
            ({'type': 'error', 'error': {'type': 'api_error', 'message': 'boom'}}, 'anthropic', 'upstream_unavailable'),
            ({'type': 'error', 'error': {'type': 'rate_limit_error', 'message': 'slow'}}, 'anthropic',
             'upstream_rate_limited'),
            ({'error': {'code': 503, 'message': 'overloaded', 'status': 'UNAVAILABLE'}}, 'gemini', 'upstream_unavailable'),
            ({'error': {'message': 'sorry', 'type': 'server_error', 'param': None, 'code': None}}, 'openai_compat',
             'upstream_unavailable'),
            ({'error': {'message': 'unauthorised', 'type': 'authentication_error'}}, 'anthropic', 'upstream_auth_failed'),
            ({'type': 'response.failed', 'response': {'error': {'code': 'server_error', 'message': 'x'}}},
             'openai_compat', 'upstream_unavailable'),
            ({'type': 'response.failed', 'response': {'error': {'code': 'rate_limit_exceeded', 'message': 'x'}}},
             'openai_compat', 'upstream_rate_limited'),
            # the Responses 'error' event carries code and message at the top level
            ({'type': 'error', 'code': 'rate_limit_exceeded', 'message': 'Slow down', 'param': None, 'sequence_number': 1},
             'openai_compat', 'upstream_rate_limited'),
            ({'type': 'error', 'code': 'server_error', 'message': 'boom', 'param': None}, 'openai_compat',
             'upstream_unavailable'),
        ]
        for payload, adapter, expected in cases:
            with self.subTest(payload=payload):
                err = ue.classify_stream_error(json.dumps(payload), adapter=adapter)
                self.assertEqual(ue.broker_code(err)[0], expected)
        top = ue.classify_stream_error({'type': 'error', 'code': 'rate_limit_exceeded', 'message': 'Slow down'},
                                       adapter='openai_compat')
        self.assertEqual((top.status, top.provider_code, top.message), (429, 'rate_limit_exceeded', 'Slow down'))

    def test_provider_codes_carrying_secrets_are_dropped(self):
        key = SENTINELS['openai']
        for code in (key, 'sk-test-' + 'a1b2' * 8, 'nvapi-' + 'Q' * 40):
            with self.subTest(code=code[:6]):
                err = classify(400, {}, body('openai_compat', 400, 'bad', code), adapter='openai_compat')
                self.assertIsNone(err.provider_code)
                for protocol in ('openai', 'responses', 'anthropic', 'gemini'):
                    text = ue.passthrough_4xx(err, protocol).body.decode('utf-8')
                    self.assertNotIn(code, text)
                self.assertEqual(json.loads(ue.passthrough_4xx(err, 'openai').body)['error']['code'], 'invalid_request')
                stream = ue.classify_stream_error({'error': {'code': code, 'message': 'x'}}, adapter='openai_compat')
                self.assertIsNone(stream.provider_code)
        plain = classify(400, {}, body('openai_compat', 400, 'bad', 'context_length_exceeded'), adapter='openai_compat')
        self.assertEqual(plain.provider_code, 'context_length_exceeded')
        self.assertIsNone(ue.safe_code(True))
        self.assertEqual(ue.safe_code(404), '404')

    def test_network_errors(self):
        request = httpx.Request('GET', 'http://127.0.0.1:9/')
        refused = ue.network_error(httpx.ConnectError('refused', request=request))
        self.assertEqual((refused.kind, refused.request_bytes_sent, ue.broker_code(refused)[0]),
                         ('network', False, 'upstream_unavailable'))
        timeout = ue.network_error(httpx.ReadTimeout('slow', request=request))
        self.assertEqual((timeout.kind, timeout.request_bytes_sent, ue.broker_code(timeout)[0]),
                         ('timeout', True, 'upstream_timeout'))
        connect_timeout = ue.network_error(httpx.ConnectTimeout('slow', request=request))
        self.assertFalse(connect_timeout.request_bytes_sent)
        self.assertEqual(ue.broker_code(UpstreamError('network', 503, None, 'x'))[0], 'upstream_unavailable')

    def test_credential_failure_rules(self):
        self.assertTrue(ue.credential_failure(401, b'', adapter='openai_compat'))
        self.assertTrue(ue.credential_failure(401, b'not json', adapter=None))
        self.assertFalse(ue.credential_failure(403, body('openai_compat', 403, 'region'), adapter='openai_compat'))
        self.assertTrue(ue.credential_failure(403, body('openai_compat', 403, code='invalid_api_key'),
                                              adapter='openai_compat'))
        self.assertTrue(ue.credential_failure(403, body('anthropic', 403, code='authentication_error'), adapter='anthropic'))
        self.assertFalse(ue.credential_failure(403, body('anthropic', 403, code='permission_error'), adapter='anthropic'))
        self.assertTrue(ue.credential_failure(400, GEMINI_KEY_INVALID, adapter='gemini'))
        self.assertTrue(ue.credential_failure(403, GEMINI_KEY_INVALID.replace(b'"code": 400', b'"code": 403'),
                                              adapter='gemini'))
        self.assertFalse(ue.credential_failure(403, body('gemini', 403, code='PERMISSION_DENIED'), adapter='gemini'))
        self.assertFalse(ue.credential_failure(400, body('openai_compat', 400, code='invalid_api_key'),
                                               adapter='openai_compat'))
        self.assertTrue(ue.credential_failure(403, body('elevenlabs', 403, code='invalid_api_key'), adapter='elevenlabs'))
        self.assertFalse(ue.credential_failure(403, body('elevenlabs', 403, code='quota_exceeded'), adapter='elevenlabs'))
        self.assertFalse(ue.credential_failure(500, b'', adapter='openai_compat'))

    def test_key_rejection_codes_and_scope_refusals(self):
        """key_rejected: the provider's own key-rejection code (an inference call marks at once only then); a restricted
        key's per-endpoint 401 (OpenAI 'insufficient permissions ... Missing scopes') is never a credential failure."""
        scopes = body('openai_compat', 401, 'You have insufficient permissions for this operation. Missing scopes: '
                                            'api.model.request. Check that you have the correct role.')
        self.assertFalse(ue.credential_failure(401, scopes, adapter='openai_compat'))
        self.assertFalse(ue.credential_failure(401, scopes, adapter=None))
        self.assertTrue(ue.scope_refusal(scopes, adapter='openai_compat'))
        self.assertFalse(ue.key_rejected(401, scopes, adapter='openai_compat'))
        self.assertTrue(ue.key_rejected(401, body('openai_compat', 401, 'Incorrect API key provided', 'invalid_api_key'),
                                        adapter='openai_compat'))
        self.assertTrue(ue.key_rejected(401, body('anthropic', 401, code='authentication_error'), adapter='anthropic'))
        self.assertTrue(ue.key_rejected(400, GEMINI_KEY_INVALID, adapter='gemini'))
        self.assertTrue(ue.key_rejected(401, body('elevenlabs', 401, code='invalid_api_key'), adapter='elevenlabs'))
        self.assertTrue(ue.key_rejected(401, b'{"error": "unauthorized"}', adapter='ollama_native'))
        for status, payload, adapter in ((401, b'', 'openai_compat'), (401, b'not json', None),
                                         (401, b'{"message": "Unauthorized", "request_id": "r"}', 'openai_compat'),
                                         (401, body('elevenlabs', 401, code='missing_permissions'), 'elevenlabs'),
                                         (403, body('anthropic', 403, code='permission_error'), 'anthropic'),
                                         (500, body('openai_compat', 500, code='invalid_api_key'), 'openai_compat')):
            with self.subTest(status=status, adapter=adapter, payload=payload[:40]):
                self.assertFalse(ue.key_rejected(status, payload, adapter=adapter))
        # a bare 401 still counts as a credential failure (an inference call has the probe endpoint confirm it)
        self.assertTrue(ue.credential_failure(401, b'{"message": "Unauthorized"}', adapter='openai_compat'))


class FakeVault:
    def __init__(self):
        self.marked = []

    def mark_validated(self, id, ok, code, fp):
        self.marked.append((id, ok, code))

    def touch_due(self, id):
        return False

    def touch(self, id):  # pragma: no cover - never due
        raise AssertionError


class MockErrors(unittest.TestCase):
    """Real provider error shapes from the WP4 mocks, through the adapters and the EgressClient."""

    @classmethod
    def setUpClass(cls):
        cls.targets = S.start_targets(('OAI', 'ANT', 'GEM', 'XI', 'OLC'))

    @classmethod
    def tearDownClass(cls):
        S.stop_targets(cls.targets)

    def setUp(self):
        netguard.reset_attempts()
        for target in self.targets.values():
            target.mock.reset()

    def tearDown(self):
        self.assertEqual(netguard.attempts(), [])

    def call(self, column, *, sentinel=None, stream=False, vault=None, surface=None, body=None):
        target = self.targets[column]
        credential = S.handle(target.origin, target.profile_id, sentinel or target.sentinel, credential_id='cred')
        grant = S.grant(target.upstream, target.profile_id, target.model, credential)
        surface = SURFACES[surface or {'OAI': 'openai.chat', 'ANT': 'anthropic.messages', 'GEM':
                                       'gemini.generateContent', 'XI': 'elevenlabs.tts', 'OLC': 'ollama.chat'}[column]]
        payload = body or {
            'OAI': {'model': 'm', 'messages': [{'role': 'user', 'content': 'hi'}], 'stream': stream},
            'ANT': {'model': 'm', 'max_tokens': 16, 'messages': [{'role': 'user', 'content': 'hi'}], 'stream': stream},
            'GEM': {'contents': [{'role': 'user', 'parts': [{'text': 'hi'}]}]},
            'XI': {'text': 'hello', 'model_id': 'm'},
            'OLC': {'model': 'm', 'messages': [{'role': 'user', 'content': 'hi'}], 'stream': stream}}[column]
        raw = RawRequest(method='POST', path_params={'voice_id': '21m00Tcm4TlvDq8ikWAM'} if column == 'XI' else {},
                         json=payload)

        async def go():
            async with S.Egress(vault=vault) as egress:
                result = await target.adapter.passthrough(grant, egress, surface, raw)
                chunks = [c async for c in result.body]
                await result.aclose()
                return result, b''.join(chunks)
        return S.run(go())

    def failure(self, *args, **kw) -> UpstreamFailure:
        with self.assertRaises(UpstreamFailure) as ctx:
            self.call(*args, **kw)
        return ctx.exception

    def test_rate_limit_and_overload_shapes(self):
        self.targets['OAI'].mock.faults.push('status', status=429, retry_after=9)
        failure = self.failure('OAI')
        self.assertEqual((failure.code, failure.status, failure.retry_after), ('upstream_rate_limited', 429, 9.0))
        self.targets['ANT'].mock.faults.push('status', status=529)
        self.assertEqual(self.failure('ANT').code, 'upstream_unavailable')
        self.targets['GEM'].mock.faults.push('status', status=503)
        self.assertEqual(self.failure('GEM').code, 'upstream_unavailable')
        self.targets['OLC'].mock.faults.push('status', status=500)
        self.assertEqual(self.failure('OLC').code, 'upstream_unavailable')

    def test_client_errors_pass_through_scrubbed(self):
        self.targets['OAI'].mock.faults.push('status', status=400)  # the mock echoes the key and the leak sentinel
        failure = self.failure('OAI')
        self.assertEqual((failure.code, failure.status), ('invalid_request', 400))
        self.assertTrue(failure.passthrough)
        rendered = ue.passthrough_4xx(failure.upstream, 'openai').body.decode()
        self.assertNotIn(SENTINELS['leak'], rendered)
        self.assertNotIn(SENTINELS['openai'], rendered)
        self.assertNotIn(SENTINELS['leak'], str(failure))

    def test_error_first_stream_is_a_pre_commit_failure(self):
        self.targets['ANT'].mock.respond_with('anthropic.messages', 'messages_stream_error_first')
        failure = self.failure('ANT', stream=True)
        self.assertEqual(failure.code, 'upstream_unavailable')
        self.assertEqual(failure.upstream.provider_code, 'overloaded_error')
        self.targets['OAI'].mock.faults.push('error_first_event')
        self.assertEqual(self.failure('OAI', stream=True).code, 'upstream_unavailable')
        self.targets['OLC'].mock.faults.push('error_first_event')
        self.assertEqual(self.failure('OLC', stream=True).code, 'upstream_unavailable')

    # R10 -------------------------------------------------------------------------------------------------------------
    def assertMarked(self, vault, marked):
        self.assertEqual(vault.marked, [('cred', False, 'upstream_auth')] if marked else [])

    def test_r10_openai_compatible_shapes(self):
        vault = FakeVault()
        failure = self.failure('OAI', sentinel='wrong', vault=vault)  # 401 invalid_api_key
        self.assertEqual((failure.code, failure.status), ('upstream_auth_failed', 502))
        self.assertMarked(vault, True)
        vault = FakeVault()
        self.targets['OAI'].mock.faults.push('status', status=403, message='Country, region, or territory not supported',
                                             code='unsupported_country_region_territory')
        # a plain OpenAI-compatible 403 is the provider refusing this request for the account, not the key: it passes
        # through as 403 with the provider's code (tests/check_broker_openai_completeness.py covers the envelope)
        region = self.failure('OAI', vault=vault)
        self.assertEqual((region.code, region.status, region.passthrough), ('invalid_request', 403, True))
        self.assertEqual(region.upstream.provider_code, 'unsupported_country_region_territory')
        self.assertMarked(vault, False)
        vault = FakeVault()
        self.targets['OAI'].mock.faults.push('status', status=403, code='invalid_api_key')
        self.failure('OAI', vault=vault)
        self.assertMarked(vault, True)
        vault = FakeVault()  # streamed calls read the 403 body too
        self.targets['OAI'].mock.faults.push('status', status=403, code='invalid_api_key')
        self.failure('OAI', vault=vault, stream=True)
        self.assertMarked(vault, True)
        vault = FakeVault()
        self.targets['OAI'].mock.faults.push('status', status=403, message='flagged by moderation')
        self.failure('OAI', vault=vault, stream=True)
        self.assertMarked(vault, False)

    def test_r10_anthropic_shapes(self):
        vault = FakeVault()
        self.failure('ANT', sentinel='wrong', vault=vault)
        self.assertMarked(vault, True)
        vault = FakeVault()
        self.targets['ANT'].mock.faults.push('status', status=403, code='authentication_error')
        self.failure('ANT', vault=vault)
        self.assertMarked(vault, True)
        vault = FakeVault()
        self.targets['ANT'].mock.faults.push('status', status=403, code='permission_error')
        self.assertEqual(self.failure('ANT', vault=vault).code, 'invalid_request')  # the provider's 403, passed through
        self.assertMarked(vault, False)

    def test_r10_gemini_shapes(self):
        vault = FakeVault()
        failure = self.failure('GEM', sentinel='wrong', vault=vault)  # 400 INVALID_ARGUMENT, reason API_KEY_INVALID
        self.assertEqual((failure.code, failure.status, failure.upstream.status), ('upstream_auth_failed', 502, 400))
        self.assertMarked(vault, True)
        vault = FakeVault()
        self.targets['GEM'].mock.faults.push('status', status=403)  # PERMISSION_DENIED without a key reason
        self.assertEqual(self.failure('GEM', vault=vault).code, 'invalid_request')  # passed through as the 403 it is
        self.assertMarked(vault, False)
        vault = FakeVault()
        self.targets['GEM'].mock.faults.push('status', status=400, message='bad schema')
        self.assertEqual(self.failure('GEM', vault=vault).code, 'invalid_request')
        self.assertMarked(vault, False)

    def test_r10_elevenlabs_shapes(self):
        vault = FakeVault()
        self.failure('XI', sentinel='wrong', vault=vault)
        self.assertMarked(vault, True)
        vault = FakeVault()
        self.targets['XI'].mock.faults.push('status', status=403, code='invalid_api_key')
        self.failure('XI', vault=vault)
        self.assertMarked(vault, True)
        vault = FakeVault()
        self.targets['XI'].mock.faults.push('status', status=403, code='voice_access_denied')
        self.failure('XI', vault=vault)
        self.assertMarked(vault, False)

    def test_r10_ollama_401_marks(self):
        vault = FakeVault()
        self.failure('OLC', sentinel='wrong', vault=vault)
        self.assertMarked(vault, True)

    def test_gone_never_marks_the_credential(self):
        """404 and 410 from every provider family's mock, through the adapters and the real EgressClient: a passthrough
        client error answering 404, model_not_found through model_failure, never a credential mark."""
        for column in ('OAI', 'ANT', 'GEM', 'XI', 'OLC'):
            for status in (404, 410):
                with self.subTest(column=column, status=status):
                    vault = FakeVault()
                    self.targets[column].mock.faults.push('status', status=status, message='model is gone')
                    failure = self.failure(column, vault=vault)
                    self.assertEqual(failure.status, 404)
                    self.assertTrue(failure.passthrough)
                    self.assertEqual(failure.upstream.kind, 'client_error')
                    self.assertMarked(vault, False)
                    self.assertEqual(ue.model_failure(failure, 'm').code, 'model_not_found')


class ModelGone(unittest.TestCase):
    """Live lane follow-up: a provider that answers 410 Gone for retired models (was 502 upstream_bad_response).
    A provider 404 or 410 is a passthrough client error answering 404, and model_failure makes it 404 model_not_found
    with a message that says what the provider reported; neither status ever marks the credential (R10)."""

    def setUp(self):
        netguard.reset_attempts()

    def tearDown(self):
        self.assertEqual(netguard.attempts(), [])

    def test_410_and_404_mapping(self):
        gone = json.dumps({'status': 410, 'title': 'Gone',
                             'detail': "Model 'meta/llama-3.1-8b-instruct' has reached its end of life"}).encode()
        err = classify(410, {}, gone, adapter='openai_compat')
        self.assertEqual((err.kind, err.status), ('client_error', 410))
        self.assertEqual(ue.broker_code(err), ('not_found', 404))
        self.assertTrue(UpstreamFailure(err).passthrough)
        mapped = ue.model_failure(UpstreamFailure(err), 'meta/llama-3.1-8b-instruct@custom_openai')
        self.assertEqual((mapped.code, mapped.status, mapped.origin), ('model_not_found', 404, 'upstream'))
        self.assertEqual(mapped.upstream.kind, 'client_error')
        self.assertEqual(str(mapped), "The provider reports model 'meta/llama-3.1-8b-instruct@custom_openai' is "
                                      "retired (HTTP 410 Gone): Model 'meta/llama-3.1-8b-instruct' has reached its "
                                      "end of life")
        bare = classify(404, {}, b'', adapter='openai_compat')
        self.assertIs(ue.model_unavailable(bare), bare)  # a bare 404 says nothing about the model
        self.assertEqual(ue.broker_code(bare), ('not_found', 404))
        missing = ue.model_unavailable(classify(404, {}, json.dumps({'error': {
            'message': 'The model `gpt-x` does not exist or you do not have access to it.', 'type': 'invalid_request_error',
            'param': None, 'code': 'model_not_found'}}).encode(), adapter='openai_compat'))
        self.assertTrue(missing.message.startswith('The provider reports the requested model is not available (HTTP 404)'))
        self.assertEqual(ue.broker_code(missing), ('model_not_found', 404))
        for protocol, status_field in (('openai', None), ('anthropic', 'not_found_error'), ('gemini', 'NOT_FOUND'),
                                       ('ollama', None), ('elevenlabs', 'model_not_found')):
            with self.subTest(protocol):
                response = ue.passthrough_4xx(mapped.upstream, protocol)
                self.assertEqual(response.status_code, 404)
                document = json.loads(response.body)
                text = json.dumps(document)
                self.assertIn('retired (HTTP 410 Gone)', text)
                if protocol == 'openai':
                    self.assertEqual(document['error']['code'], 'model_not_found')
                elif status_field:
                    self.assertIn(status_field, text)
        for status in (404, 410):
            self.assertFalse(ue.credential_failure(status, gone, adapter='openai_compat'))


class ModelNotFoundOnlyForTheModel(unittest.TestCase):
    """model_unavailable / model_failure rewrite a provider 404 as 404 model_not_found only when the provider code or
    message is about the model; a 410 Gone on a model call always is; a 404 about anything else keeps not_found."""

    def setUp(self):
        netguard.reset_attempts()

    def tearDown(self):
        self.assertEqual(netguard.attempts(), [])

    @staticmethod
    def err(status, document, adapter):
        return classify(status, {}, json.dumps(document).encode(), adapter=adapter)

    def assertModel(self, err, model=None, upstream_model=None):
        failure = UpstreamFailure(err)
        mapped = ue.model_failure(failure, model, upstream_model=upstream_model)
        self.assertIsNot(mapped, failure)
        self.assertEqual((mapped.code, mapped.status, mapped.origin), ('model_not_found', 404, 'upstream'))
        self.assertTrue(mapped.passthrough)
        self.assertEqual(mapped.upstream.kind, 'client_error')
        return mapped

    def assertOther(self, err, model=None, upstream_model=None):
        failure = UpstreamFailure(err)
        self.assertIs(ue.model_failure(failure, model, upstream_model=upstream_model), failure)
        self.assertIs(ue.model_unavailable(err, model, upstream_model=upstream_model), err)
        self.assertEqual((failure.code, failure.status), ('not_found', 404))
        self.assertEqual(ue.broker_code(err), ('not_found', 404))

    def test_each_providers_model_not_found(self):
        cases = {
            'openai': (self.err(404, {'error': {'message': 'The model `gpt-9` does not exist or you do not have access '
                                                           'to it.', 'type': 'invalid_request_error', 'param': None,
                                                'code': 'model_not_found'}}, 'openai_compat'), 'gpt-9@openai'),
            'azure_openai': (self.err(404, {'error': {'code': 'DeploymentNotFound', 'message': 'The API deployment for '
                                                      'this resource does not exist.'}}, 'openai_compat'),
                             'gpt-4o-mini@azure'),
            'anthropic': (self.err(404, {'type': 'error', 'error': {'type': 'not_found_error',
                                                                    'message': 'model: claude-9'}}, 'anthropic'),
                          'claude-9@anthropic'),
            'gemini': (self.err(404, {'error': {'code': 404, 'status': 'NOT_FOUND', 'message': 'models/gemini-9 is not '
                                                'found for API version v1beta, or is not supported for '
                                                'generateContent.'}}, 'gemini'), 'gemini-9@gemini'),
            'function_not_found': (self.err(404, {'status': 404, 'title': 'Not Found', 'detail': "Function 'f0e1d2c3': "
                                               "Not found for account 'acct'"}, 'openai_compat'),
                                'meta/llama-9@custom_openai'),
            'compatible_model_word': (self.err(404, {'error': {'message': "Model meta/llama-9 not found",
                                                          'type': 'invalid_request_error'}}, 'openai_compat'),
                                  'meta/llama-9@custom_openai'),
            'compatible_model_id': (self.err(404, {'detail': 'meta/llama-9 is not served by this endpoint'},
                                             'openai_compat'), None),
        }
        for name, (err, model) in cases.items():
            with self.subTest(name):
                self.assertEqual((err.kind, err.status), ('client_error', 404))
                upstream = 'meta/llama-9' if name == 'compatible_model_id' else None
                mapped = self.assertModel(err, model, upstream)
                self.assertIn('is not available (HTTP 404)', str(mapped))
                self.assertFalse(ue.credential_failure(404, b'', adapter='openai_compat'))
        # the bare model id counts only when it is the requested or upstream model
        self.assertOther(cases['compatible_model_id'][0], 'other-model@x', upstream_model='other-model')

    def test_voice_file_response_and_bare_404_stay_not_found(self):
        voice = self.err(404, {'detail': {'status': 'voice_not_found',
                                          'message': 'A voice with the voice_id voice1 was not found.'}}, 'elevenlabs')
        self.assertEqual(voice.provider_code, 'voice_not_found')
        self.assertOther(voice, 'eleven_multilingual_v2@elevenlabs', upstream_model='eleven_multilingual_v2')
        # a code naming another resource wins even when the message mentions the model
        voice_model = self.err(404, {'detail': {'status': 'voice_not_found', 'message': 'Voice voice1 is not available '
                                                'for model eleven_multilingual_v2.'}}, 'elevenlabs')
        self.assertOther(voice_model, 'eleven_multilingual_v2@elevenlabs', upstream_model='eleven_multilingual_v2')
        document = json.loads(ue.passthrough_4xx(voice, 'elevenlabs').body)
        self.assertEqual(document['detail']['status'], 'not_found')
        self.assertIn('voice_id voice1', document['detail']['message'])
        others = [
            self.err(404, {'error': {'message': 'No such File object: file-abc123', 'type': 'invalid_request_error',
                                     'param': 'id', 'code': None}}, 'openai_compat'),
            self.err(404, {'error': {'message': "Response with id 'resp_abc123' not found.",
                                     'type': 'invalid_request_error', 'param': None, 'code': None}}, 'openai_compat'),
            self.err(404, {'type': 'error', 'error': {'type': 'not_found_error', 'message': 'File not found: file_01'}},
                     'anthropic'),
            self.err(404, {'error': {'code': 404, 'status': 'NOT_FOUND', 'message': 'Requested entity was not found.'}},
                     'gemini'),
            self.err(404, {'error': {'code': 'file_not_found', 'message': 'file-abc123'}}, 'openai_compat'),
            classify(404, {}, b'', adapter='openai_compat'),
            classify(404, {}, b'<html>404 page not found</html>', adapter='openai_compat'),
        ]
        for err in others:
            with self.subTest(code=err.provider_code, message=err.message):
                self.assertOther(err, 'gpt-4o-mini@openai', upstream_model='gpt-4o-mini')

    def test_410_on_a_model_call_is_always_model_not_found(self):
        for err in (self.err(410, {'error': {'message': 'Gone'}}, 'openai_compat'),
                    self.err(410, {'status': 410, 'title': 'Gone', 'detail': 'end of life'}, 'openai_compat'),
                    classify(410, {}, b'', adapter='openai_compat')):
            with self.subTest(message=err.message):
                self.assertEqual(ue.broker_code(err), ('not_found', 404))  # unmapped (handle-bound routes)
                mapped = self.assertModel(err, 'm@u')
                self.assertIn('retired (HTTP 410 Gone)', str(mapped))

    def test_a_code_naming_another_resource_is_never_model_not_found(self):
        """A provider code that names a voice, file, response, thread, assistant, batch, vector store or upload (and not
        the model) stays not_found on 404 and on 410, even when the message names the model or the word model."""
        message = 'The model gpt-4o-mini cannot use this resource: model gpt-4o-mini'
        codes = ('voice_not_found', 'voice_unavailable', 'file_expired', 'FileNotFound', 'response_not_found',
                 'thread_gone', 'assistant_deleted', 'batch_expired', 'vector_store_expired', 'VectorStoreNotFound',
                 'upload_expired', 'invalid_file')
        for status in (404, 410):
            for code in codes:
                with self.subTest(status=status, code=code):
                    adapter = 'elevenlabs' if code.startswith('voice') else 'openai_compat'
                    document = ({'detail': {'status': code, 'message': message}} if adapter == 'elevenlabs'
                                else {'error': {'message': message, 'type': 'invalid_request_error', 'code': code}})
                    err = self.err(status, document, adapter)
                    self.assertEqual(err.provider_code, code)
                    self.assertTrue(ue.names_other_resource(code))
                    self.assertFalse(ue.names_model(err, 'gpt-4o-mini@openai', 'gpt-4o-mini'))
                    failure = UpstreamFailure(err)
                    self.assertIs(ue.model_failure(failure, 'gpt-4o-mini@openai', upstream_model='gpt-4o-mini'), failure)
                    self.assertIs(ue.model_unavailable(err, 'gpt-4o-mini@openai', upstream_model='gpt-4o-mini'), err)
                    self.assertEqual(ue.broker_code(err), ('not_found', 404))
                    self.assertEqual((failure.code, failure.status), ('not_found', 404))
        # a code that names the model (or a generic code) is not another resource
        for code in ('model_not_found', 'DeploymentNotFound', 'model_file_missing', 'not_found', 'NOT_FOUND', '', None):
            with self.subTest(code=code):
                self.assertFalse(ue.names_other_resource(code))
        model_file = self.err(410, {'error': {'message': 'gone', 'code': 'model_file_missing'}}, 'openai_compat')
        self.assertIn('retired (HTTP 410 Gone)', str(self.assertModel(model_file, 'm@u')))


if __name__ == '__main__':
    S.main()
