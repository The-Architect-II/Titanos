"""Broker core contracts (WP1): profile rows, surfaces, route scopes, the role switch table, parent headers and serve
preconditions, the error table and protocol envelopes, the adapter ABC, frozen IR and shared types, the vault interface,
request ContextVars, an I/O-free package import, and the provider-hostname source scan. In-process only: the app built for
route coverage uses the route-contract-matrix generator's hermetic build (temporary TITANOS_HOME, device probes stubbed);
no server, model or network."""
import asyncio
import copy
import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import httpx  # noqa: E402
from titanos.broker import context, envelopes, errors, profiles, role, scopes, surfaces, types  # noqa: E402
from titanos.broker.adapters import base  # noqa: E402
from titanos.broker.translate import ir  # noqa: E402
from titanos.broker import vault  # noqa: E402
from titanos.settings import Settings  # noqa: E402

PACKAGE = ROOT / 'src' / 'titanos' / 'broker'
PROFILE_IDS = {'openai', 'azure_openai', 'xai', 'groq', 'together', 'mistral', 'deepseek', 'fireworks', 'perplexity',
               'nvidia_api_catalog', 'nim_custom', 'cerebras', 'sambanova', 'openrouter', 'hf_router', 'litellm',
               'hivemind_gateway', 'pair_facade', 'custom_openai', 'anthropic', 'gemini', 'elevenlabs', 'ollama_cloud',
               'pending_contract', 'switchyard', 'azure_openai_v1', 'fireworks_audio', 'nvidia_ai_endpoints'}
INTERCEPT_SURFACES = {'openai.rerank', 'openai.chat', 'openai.completions', 'openai.responses', 'openai.responses.retrieve',
                      'openai.responses.delete', 'openai.responses.cancel', 'openai.responses.input_items',
                      'openai.embeddings', 'openai.audio.speech', 'openai.audio.transcriptions', 'openai.audio.translations',
                      'openai.images.generations', 'openai.images.edits', 'openai.moderations', 'anthropic.messages',
                      'anthropic.count_tokens', 'gemini.generateContent', 'gemini.streamGenerateContent',
                      'gemini.embedContent', 'gemini.batchEmbedContents', 'gemini.countTokens', 'ollama.chat',
                      'ollama.generate', 'ollama.embed', 'ollama.embeddings', 'elevenlabs.tts', 'elevenlabs.tts_stream',
                      'elevenlabs.stt'}
NO_RETRY = {'budget_exceeded', 'budget_unpriced_target', 'scope_denied', 'cloud_disabled', 'role_forbids',
            'delegation_required', 'transport_insecure', 'keyless_mode_refused', 'data_policy_denied', 'broker_name_conflict',
            'advisor_profile_unconfigured', 'last_key_required', 'upstream_auth_failed', 'egress_refused',
            'upstream_redirect_refused', 'upstream_response_too_large', 'credential_unavailable', 'vault_unavailable',
            'vault_sealed', 'classifier_unavailable', 'upstream_unconfigured', 'loop_detected'}
STATUSES = {
    400: {'model_required', 'invalid_request', 'ambiguous_credentials', 'location_unsatisfiable',
          'broker_capability_unavailable', 'provenance_unavailable_for_cloud', 'broker_config_contains_secret',
          'idempotency_key_required', 'budget_required'},
    401: {'authentication_error', 'key_expired', 'key_revoked', 'query_credential_refused'},
    402: {'budget_exceeded', 'budget_unpriced_target'},
    403: {'scope_denied', 'cloud_disabled', 'role_forbids', 'delegation_required', 'transport_insecure',
          'keyless_mode_refused', 'data_policy_denied'},
    404: {'model_not_found', 'response_not_found', 'not_found'},
    409: {'broker_name_conflict', 'advisor_profile_unconfigured', 'last_key_required', 'registry_busy'},
    413: {'request_too_large'}, 422: {'credential_validation_failed'},
    429: {'rate_limit_exceeded', 'concurrency_exceeded', 'upstream_rate_limited', 'auth_throttled'}, 499: {'cancelled'},
    502: {'upstream_auth_failed', 'upstream_bad_response', 'egress_refused', 'upstream_redirect_refused',
          'upstream_response_too_large', 'stream_interrupted'},
    503: {'upstream_unavailable', 'breaker_open', 'credential_unavailable', 'vault_unavailable', 'vault_sealed',
          'classifier_unavailable', 'upstream_unconfigured'},
    504: {'upstream_timeout', 'deadline_exceeded'}, 508: {'loop_detected'},
}
SWITCH_FIELDS = ('cloud_egress', 'cloud_fallback', 'router_upstreams', 'advisors', 'health_probes', 'live_catalogs',
                 'anonymous_cloud_allowed', 'requires_keys', 'requires_tls_off_loopback', 'vault_sealed', 'delegation',
                 'parent_headers_authoritative')
# the roles spec switch table, field by field (managed rows by managed_delegation)
SWITCH_TABLE = {
    ('standalone', 'off'): (1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0),
    ('broker', 'off'): (1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0),
    ('managed', 'off'): (0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1),
    ('managed', 'parent'): (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1),
}


def principal(scopes_, kind='key', **kw):
    return types.Principal(id='kfixture01', kind=kind, scopes=frozenset(scopes_), **kw)


class Profiles(unittest.TestCase):
    def test_rows_are_consistent(self):
        self.assertEqual(set(profiles.PROFILES), PROFILE_IDS)
        native = {'openai_compat': 'openai.', 'anthropic': 'anthropic.', 'gemini': 'gemini.', 'elevenlabs': 'elevenlabs.',
                  'ollama_native': 'ollama.'}
        for row in profiles.PROFILES.values():
            with self.subTest(profile=row.id):
                self.assertIn(row.adapter, profiles.ADAPTERS)
                self.assertIn(row.kind, profiles.PROFILE_KINDS)
                self.assertIn(row.auth, profiles.AUTH_TEMPLATES)
                self.assertLessEqual(set(row.surfaces), set(surfaces.SURFACES))
                self.assertLessEqual(row.data_modalities, profiles.DATA_MODALITIES)
                if row.kind == 'slot':
                    self.assertEqual((row.adapter, row.auth, dict(row.surfaces), row.default_base_url),
                                     ('none', 'none', {}, None))
                    continue
                self.assertTrue(row.surfaces)
                split = dict(row.extras.get('surface_adapters') or {})  # switchyard: anthropic.* through 'anthropic'
                self.assertTrue(all(name.startswith(native[split.get(name, row.adapter)]) for name in row.surfaces),
                                row.surfaces)
                self.assertTrue(all(path.startswith('/') for path in row.surfaces.values()))
                if row.default_base_url is not None:
                    local_router = row.kind == 'router' and row.default_base_url.startswith('http://127.0.0.1:')
                    self.assertTrue(row.default_base_url.startswith('https://') or local_router)
                self.assertEqual(row.allow_any_model_default, row.kind == 'router')
                with self.assertRaises(dataclasses.FrozenInstanceError):
                    row.auth = 'none'
        self.assertEqual(profiles.OPERATOR_SURFACE_PROFILES,
                         {'custom_openai', 'nim_custom', 'litellm', 'hivemind_gateway', 'pair_facade', 'switchyard',
                          'azure_openai_v1'})
        self.assertEqual({r.id for r in profiles.PROFILES.values() if r.hop_header},
                         {'custom_openai', 'nim_custom', 'litellm', 'hivemind_gateway', 'pair_facade', 'switchyard'})
        self.assertEqual({r.id for r in profiles.PROFILES.values() if r.kind == 'router'},
                         {'openrouter', 'hf_router', 'litellm', 'hivemind_gateway', 'pair_facade', 'switchyard'})
        self.assertEqual(profiles.TITANOS_ONLY_FIELDS, frozenset({'worker', 'location', 'timeout', 'require_loaded',
                                                                  'include_provenance', 'vad_enabled', 'vad_model',
                                                                  'vad_options'}))
        self.assertEqual(profiles.INBOUND_HEADER_ALLOW, frozenset({'anthropic-beta', 'openai-beta', 'accept'}))
        self.assertNotIn('png', profiles.profile('xai').image_formats)
        self.assertEqual(profiles.profile('mistral').params.tool_id_rule, 'alnum9')
        self.assertIn('n>1', profiles.profile('groq').params.refuse['*'])
        self.assertEqual(profiles.profile('anthropic').extras['default_anthropic_version'], '2023-06-01')
        self.assertEqual(profiles.profile('elevenlabs').usage_headers, ('character-cost',))
        self.assertEqual(profiles.AUTH_TEMPLATES['bearer'], ('Authorization', 'Bearer {secret}'))
        self.assertIsNone(profiles.AUTH_TEMPLATES['none'])
        self.assertEqual(profiles.profile('openai').static_headers, ('OpenAI-Organization', 'OpenAI-Project'))
        self.assertEqual(profiles.profile('openrouter').static_headers, ('HTTP-Referer', 'X-Title'))
        with self.assertRaises(KeyError):
            profiles.profile('bedrock')

    def test_provider_hostnames_only_in_profiles(self):
        names = profiles.hostnames()
        self.assertIn('api.openai.com', names)
        self.assertIn('openai.azure.com', names)
        self.assertNotIn('127.0.0.1', names)  # a local router's loopback default is an address, not a provider name
        offenders = []
        for path in PACKAGE.rglob('*.py'):
            text = path.read_text(encoding='utf-8')
            if path.name == 'profiles.py' and path.parent == PACKAGE:
                continue
            offenders.extend((path.relative_to(PACKAGE).as_posix(), name) for name in names if name in text)
        self.assertEqual(offenders, [])


class Surfaces(unittest.TestCase):
    def test_table(self):
        self.assertLessEqual(INTERCEPT_SURFACES, {s.name for s in surfaces.by_kind('intercept')})
        self.assertEqual({s.name for s in surfaces.by_kind('intercept')}, INTERCEPT_SURFACES)
        for row in surfaces.SURFACES.values():
            with self.subTest(surface=row.name):
                self.assertIn(row.protocol, surfaces.PROTOCOLS)
                self.assertIn(row.model_from, surfaces.MODEL_FROM)
                self.assertIn(row.stream_from, surfaces.STREAM_FROM)
                self.assertIn(row.model_absent, ('refuse', 'local_passthrough'))
                self.assertTrue(scopes.valid_scope(row.scope))
                concrete = row.path.replace('{model:path}', 'models-x/y').replace('{voice_id}', 'v1').replace(
                    '{response_id}', 'resp_1').replace('{name:path}', 'n/m')
                self.assertEqual(surfaces.match(row.method, concrete)[0], row)
                self.assertEqual(scopes.required_scope(row.method, concrete), row.scope)
                if row.handle_bound:
                    self.assertFalse(row.replay_safe)
        self.assertEqual(surfaces.SURFACES['elevenlabs.tts'].model_absent, 'local_passthrough')
        self.assertEqual({s.name for s in surfaces.SURFACES.values() if s.model_absent == 'local_passthrough'},
                         {'elevenlabs.tts'})
        for name in ('openai.images.edits', 'openai.audio.transcriptions', 'elevenlabs.stt', 'openai.images.generations'):
            self.assertFalse(surfaces.SURFACES[name].replay_safe, name)
        self.assertTrue(surfaces.SURFACES['ollama.chat'].stream_default)

    def test_match(self):
        found = surfaces.match('POST', '/v1beta/models/gemini-2.0-flash:streamGenerateContent')
        self.assertEqual((found[0].name, found[1]), ('gemini.streamGenerateContent', {'model': 'gemini-2.0-flash'}))
        self.assertEqual(surfaces.match('POST', '/v1beta/models/a/b:c:embedContent')[1], {'model': 'a/b:c'})
        self.assertIsNone(surfaces.match('POST', '/v1beta/models/m:unknownAction'))
        self.assertEqual(surfaces.match('POST', '/v1/text-to-speech/voice9/stream')[0].name, 'elevenlabs.tts_stream')
        self.assertEqual(surfaces.match('POST', '/v1/text-to-speech/voice9')[1], {'voice_id': 'voice9'})
        self.assertEqual(surfaces.match('GET', '/v1/responses/resp_1/input_items')[0].name, 'openai.responses.input_items')
        self.assertEqual(surfaces.match('post', '/v1/chat/completions')[0].name, 'openai.chat')
        self.assertIsNone(surfaces.match('GET', '/v1/chat/completions'))
        self.assertIsNone(surfaces.match('POST', '/v1/chat/completions/'))
        self.assertEqual(surfaces.match('GET', '/v1/models/org/model')[1], {'name': 'org/model'})


class Scopes(unittest.TestCase):
    def test_required_scope(self):
        for path in ('/healthz', '/', '/docs', '/openapi.json', '/ui/app.js', '/v1/health/ready', '/healthz/'):
            self.assertIsNone(scopes.required_scope('GET', path), path)
        for method in ('POST', 'HEAD', 'OPTIONS'):  # public framing (405) decides these, never a 401/403
            self.assertIsNone(scopes.required_scope(method, '/healthz'), method)
        self.assertEqual(scopes.required_scope('OPTIONS', '/v1/chat/completions'), 'admin')
        self.assertEqual(scopes.required_scope('GET', '/api/version'), 'authenticated')
        self.assertIsNone(scopes.required_scope('GET', '/api/version',
                                                public_routes=role.public_routes(Settings(role='managed'))))
        for method, path, expected in (
                ('POST', '/mcp', 'mcp'), ('GET', '/mcp/', 'mcp'), ('DELETE', '/mcp/session', 'mcp'),
                ('GET', '/api/keys/self', 'authenticated'), ('GET', '/api/keys/abcdefgh23', 'keys:admin'),
                ('POST', '/v1beta/models/text-embedding-004:embedContent', 'inference:embeddings'),
                ('POST', '/v1beta/models/m:batchEmbedContents', 'inference:embeddings'),
                ('POST', '/v1beta/models/m:countTokens', 'inference:chat'),
                ('POST', '/v1beta/models/m:unknown', 'inference:chat'),
                ('GET', '/api/broker/credentials', 'credentials:write|broker:admin'),
                ('PUT', '/api/broker/models/org/alias', 'broker:admin'), ('GET', '/ui/missing.js', 'authenticated'),
                ('GET', '/docs/assets/other.js', 'authenticated'), ('GET', '/api/info', 'authenticated'),
                ('POST', '/api/models/reload', 'models:write'), ('GET', '/v1/files/abc.png', 'authenticated'),
                ('POST', '/api/unknown-route', 'admin'), ('PATCH', '/v1/chat/completions', 'admin'),
                ('DELETE', '/api/models/registrations/org/name', 'models:write')):
            with self.subTest(path=path, method=method):
                self.assertEqual(scopes.required_scope(method, path), expected)
        for scope in set(scopes.ROUTE_SCOPES.values()):
            for part in scope.split('|'):
                self.assertTrue(part == 'authenticated' or scopes.valid_scope(part), part)

    def test_public_set_mirrors_server(self):
        from titanos import server
        self.assertLessEqual(server.PUBLIC_ROUTES, scopes.BASE_PUBLIC_ROUTES)
        self.assertLessEqual(scopes.BASE_PUBLIC_ROUTES - server.PUBLIC_ROUTES, {'/v1/health/ready'})
        self.assertEqual(role.public_routes(Settings()), scopes.BASE_PUBLIC_ROUTES)
        self.assertEqual(role.public_routes(Settings(role='broker')), scopes.BASE_PUBLIC_ROUTES)
        self.assertEqual(role.public_routes(Settings(role='managed')) - scopes.BASE_PUBLIC_ROUTES, {'/api/version'})

    def test_every_app_route_is_listed_or_public(self):
        spec = importlib.util.spec_from_file_location('route_matrix_builder', ROOT / 'tools' / 'build_route_contract_matrix.py')
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        from fastapi.routing import APIRoute
        from starlette.routing import Mount, Route
        with tempfile.TemporaryDirectory(prefix='titanos-broker-scopes-') as state:
            app = builder.build_app(state)
            try:
                missing = []
                for route in app.routes:
                    if isinstance(route, Mount):
                        if scopes.scope_for_template('*', route.path) is None:
                            missing.append(('*', route.path))
                        continue
                    if not isinstance(route, (APIRoute, Route)):
                        continue
                    for method in sorted(route.methods or ()):
                        if method == 'HEAD':
                            continue
                        if route.path in scopes.BASE_PUBLIC_ROUTES and method == 'GET':
                            continue
                        if scopes.scope_for_template(method, route.path) is None:
                            missing.append((method, route.path))
                for path, operations in app.openapi()['paths'].items():
                    for method in operations:
                        if scopes.scope_for_template(method, path) is None and path not in scopes.BASE_PUBLIC_ROUTES:
                            missing.append((method.upper(), path))
                self.assertEqual(missing, [])
            finally:
                try:
                    asyncio.run(app.state.native.close())
                    app.state.engine.target.close()
                except Exception:
                    pass

    def test_allows_and_umbrellas(self):
        admin = principal({'admin'}, kind='bootstrap')
        self.assertTrue(all(scopes.allows(admin, s) for s in scopes.SCOPES | {'cloud:openai'}))
        p = principal({'inference', 'cloud', 'broker:admin', 'models:write'})
        for scope in ('inference:chat', 'inference:tensor', 'cloud:openai', 'broker:read', 'models:read',
                      'credentials:write|broker:admin', 'authenticated', None):
            self.assertTrue(scopes.allows(p, scope), scope)
        for scope in ('cloud:store', 'credentials:write', 'keys:admin', 'jobs', 'admin', 'route:explicit'):
            self.assertFalse(scopes.allows(p, scope), scope)
        narrow = principal({'inference:chat', 'cloud:groq'})
        self.assertTrue(scopes.allows(narrow, 'inference:chat'))
        self.assertFalse(scopes.allows(narrow, 'inference:embeddings'))
        self.assertFalse(scopes.allows(narrow, 'inference'))
        self.assertFalse(scopes.allows(narrow, 'cloud:openai'))
        self.assertFalse(scopes.allows(None, 'authenticated'))
        self.assertTrue(scopes.valid_scope('cloud:ollama_cloud'))
        for bad in ('cloud:Bad', 'cloud:', 'inference:everything', 'root'):
            self.assertFalse(scopes.valid_scope(bad), bad)

    def test_keyless_and_bootstrap_scopes(self):
        self.assertEqual(scopes.BOOTSTRAP_SCOPES, frozenset({'admin'}))
        self.assertEqual(scopes.ANONYMOUS_LOOPBACK_SCOPES & scopes.KEYLESS_NEVER_SCOPES, frozenset())
        self.assertNotIn('cloud', scopes.ANONYMOUS_LOOPBACK_SCOPES)
        self.assertIn('cloud', scopes.anonymous_scopes(anonymous_cloud=True, role='standalone'))
        self.assertNotIn('cloud', scopes.anonymous_scopes(anonymous_cloud=True, role='broker'))
        self.assertNotIn('cloud', scopes.anonymous_scopes(anonymous_cloud=True, role='managed'))
        self.assertNotIn('cloud', scopes.anonymous_scopes(anonymous_cloud=False, role='standalone'))

    def test_can_use_target(self):
        local = types.Target(ref='qwen3-8b', klass='local', local_name='qwen3-8b')
        cloud = types.Target(ref='gpt-4.1-mini@openai', klass='cloud', upstream_id='openai', upstream_model='gpt-4.1-mini')
        router = types.Target(ref='a/b@openrouter', klass='router', upstream_id='openrouter', upstream_model='a/b')
        full = principal({'inference', 'cloud'})
        self.assertTrue(all(scopes.can_use_target(full, t) for t in (local, cloud, router)))
        no_cloud = principal({'inference'})
        self.assertEqual([scopes.can_use_target(no_cloud, t) for t in (local, cloud, router)], [True, False, False])
        per_upstream = principal({'inference', 'cloud:openai'})
        self.assertEqual([scopes.can_use_target(per_upstream, t) for t in (local, cloud, router)], [True, True, False])
        constrained = principal({'inference', 'cloud'}, constraints=types.KeyConstraints(
            models=('gpt-*@openai', 'qwen3-*'), upstreams=('openai',), target_classes=frozenset({'local', 'cloud'})))
        self.assertEqual([scopes.can_use_target(constrained, t) for t in (local, cloud, router)], [True, True, False])
        only_cloud = principal({'cloud'}, constraints=types.KeyConstraints(target_classes=frozenset({'cloud'})))
        self.assertFalse(scopes.can_use_target(only_cloud, local))
        self.assertFalse(scopes.can_use_target(None, local))


class Roles(unittest.TestCase):
    def test_switch_table(self):
        for (role_name, delegation), expected in SWITCH_TABLE.items():
            with self.subTest(role=role_name, delegation=delegation):
                switches = role.switches_for(Settings(role=role_name, managed_delegation=delegation))
                self.assertEqual(tuple(int(getattr(switches, f)) for f in SWITCH_FIELDS), expected)
                self.assertEqual(tuple(f.name for f in dataclasses.fields(switches)), SWITCH_FIELDS)

    def test_purpose_allowed(self):
        for (role_name, delegation), _ in SWITCH_TABLE.items():
            switches = role.switches_for(Settings(role=role_name, managed_delegation=delegation))
            for purpose in types.GRANT_PURPOSES:
                for delegated in (False, True):
                    with self.subTest(role=role_name, delegation=delegation, purpose=purpose, delegated=delegated):
                        allowed = role.purpose_allowed(switches, purpose, delegated=delegated)
                        if role_name != 'managed':
                            self.assertTrue(allowed)
                        else:
                            self.assertEqual(allowed, purpose == 'inference' and delegated and delegation == 'parent')
        with self.assertRaises(ValueError):
            role.purpose_allowed(role.switches_for(Settings()), 'gossip', delegated=False)

    def test_parent_headers(self):
        managed = Settings(role='managed', managed_delegation='parent')
        parent = principal({'delegate', 'inference'}, kind='parent')
        hints = role.parse_parent_headers({'X-Request-ID': 'req-1', 'X-Titanos-Delegate': 'cloud',
                                           'x-hivemind-placement': 'local_only', 'X-Titanos-Location': 'cloud_only',
                                           'traceparent': '00-abc-def-01', 'X-HiveMind-Attempt': '2',
                                           'X-Titanos-Hop': 'abcdef0123456789', 'X-Titanos-Deadline-Ms': '1500',
                                           'X-Titanos-Target': 'gpt-4.1-mini@openai'}, managed, parent)
        self.assertEqual((hints.request_id, hints.location, hints.delegated, hints.attempt, hints.trace_id, hints.hop,
                          hints.hop_value, hints.target_header),
                         ('req-1', 'local_only', True, '2', 'req-1', True, 'abcdef0123456789',
                          'gpt-4.1-mini@openai'))  # x-request-id is first in managed_trace_headers
        traced = role.parse_parent_headers({'X-Request-ID': 'req-1', 'traceparent': '00-abc-def-01'},
                                           Settings(managed_trace_headers='traceparent,x-request-id'), None)
        self.assertEqual(traced.trace_id, '00-abc-def-01')
        self.assertLessEqual(hints.deadline - time.monotonic(), 1.5)
        self.assertEqual(role.parse_parent_headers({'x-request-id': 'req-1'}, managed, parent).trace_id, 'req-1')
        for settings_, who in ((Settings(role='managed'), parent), (Settings(), parent), (managed, principal({'inference'})),
                               (managed, principal({'delegate'}, kind='anonymous_loopback')), (managed, None)):
            self.assertFalse(role.parse_parent_headers({'x-titanos-delegate': 'cloud'}, settings_, who).delegated)
        plain = role.parse_parent_headers([(b'x-request-id', b'bad id with spaces')], Settings(), None)
        self.assertEqual(len(plain.request_id), 32)
        self.assertEqual((plain.location, plain.hop, plain.delegated), (None, False, False))
        self.assertAlmostEqual(plain.deadline - time.monotonic(), Settings().client_timeout, delta=5)
        self.assertEqual(role.parse_parent_headers({'x-titanos-location': 'prefer_local'}, Settings(), None).location,
                         'prefer_local')
        self.assertEqual(role.parse_parent_headers({'x-hivemind-placement': 'somewhere'}, Settings(), None).location, None)
        for bad in ({'x-titanos-location': 'mars'}, {'x-titanos-delegate': 'yes'}, {'x-titanos-deadline-ms': '-5'},
                    {'x-titanos-deadline-ms': 'soon'}, {'x-titanos-target': 'm' * 257}):
            with self.subTest(bad=bad):
                with self.assertRaises(errors.BrokerError) as caught:
                    role.parse_parent_headers(bad, Settings(), None)
                self.assertEqual(caught.exception.code, 'invalid_request')

    def test_serve_preconditions(self):
        none = types.KeyStoreSummary()
        keyed = types.KeyStoreSummary(has_records=True, active=1, active_admin=1)
        cert = str(Path(tempfile.gettempdir()).resolve() / 'c.pem')
        ok = dict(explicit_port=False, vault_backend='dpapi', audit_ok=True)
        self.assertEqual(role.check_serve_preconditions(Settings(), bootstrap_key=None, keystore=none, **ok), [])
        with self.assertRaisesRegex(ValueError, 'Set TITANOS_API_KEY before binding to a network interface'):
            role.check_serve_preconditions(Settings(host='0.0.0.0'), bootstrap_key=None, keystore=none, **ok)
        self.assertEqual(role.check_serve_preconditions(Settings(host='0.0.0.0'), bootstrap_key=None, keystore=keyed, **ok),
                         [])
        expired_only = types.KeyStoreSummary(has_records=True, active=0, active_admin=0)
        with self.assertRaises(ValueError):
            role.check_serve_preconditions(Settings(host='0.0.0.0'), bootstrap_key=None, keystore=expired_only, **ok)
        lan_proxy = Settings(host='0.0.0.0')
        object.__setattr__(lan_proxy, 'behind_tls_proxy', True)  # bypass Settings' own check to reach this guard
        with self.assertRaisesRegex(ValueError, 'TITANOS_BEHIND_TLS_PROXY'):
            role.check_serve_preconditions(lan_proxy, bootstrap_key='bootstrap-fixture', keystore=none, **ok)
        broker = Settings(role='broker')
        with self.assertRaisesRegex(ValueError, 'TITANOS_ROLE=broker requires TITANOS_API_KEY'):
            role.check_serve_preconditions(broker, bootstrap_key=None, keystore=keyed, **ok)
        self.assertEqual(role.check_serve_preconditions(broker, bootstrap_key='bootstrap-fixture', keystore=none, **ok),
                         [])  # the default limit_concurrency (64) carries no warning
        warnings = role.check_serve_preconditions(Settings(role='broker', limit_concurrency=32),
                                                  bootstrap_key='b-fixture', keystore=none, **ok)
        self.assertEqual(len(warnings), 1)  # limit_concurrency 32 < 64
        with self.assertRaisesRegex(ValueError, 'native TLS'):
            role.check_serve_preconditions(Settings(role='broker', host='0.0.0.0'), bootstrap_key='b-fixture',
                                           keystore=none, **ok)
        role.check_serve_preconditions(Settings(role='broker', host='0.0.0.0', tls_certfile=cert, tls_keyfile=cert),
                                       bootstrap_key='b-fixture', keystore=none, **ok)
        with self.assertRaisesRegex(ValueError, 'audit'):
            role.check_serve_preconditions(broker, bootstrap_key='b-fixture', keystore=none,
                                           **{**ok, 'audit_ok': False})
        with self.assertRaisesRegex(ValueError, 'vault posture'):
            role.check_serve_preconditions(broker, bootstrap_key='b-fixture', keystore=none,
                                           **{**ok, 'vault_backend': 'unavailable'})
        role.check_serve_preconditions(Settings(role='broker', broker_vault_backend='refs-only'), bootstrap_key='b-fixture',
                                       keystore=none, **{**ok, 'vault_backend': 'unavailable'})
        with self.assertRaisesRegex(ValueError, 'kbadkey001'):
            role.check_serve_preconditions(broker, bootstrap_key='b-fixture', keystore=types.KeyStoreSummary(
                has_records=True, active=1, cloud_keys_missing_budget=('kbadkey001',)), **ok)
        managed = Settings(role='managed', port=12500)
        with self.assertRaisesRegex(ValueError, 'explicit --port'):
            role.check_serve_preconditions(managed, bootstrap_key=None, keystore=none, **ok)
        role.check_serve_preconditions(managed, bootstrap_key=None, keystore=none, **{**ok, 'explicit_port': True})
        for port in sorted(role.RESERVED_PORTS):
            with self.assertRaisesRegex(ValueError, 'reserved port'):
                role.check_serve_preconditions(Settings(role='managed', port=port), bootstrap_key=None, keystore=none,
                                               **{**ok, 'explicit_port': True})
        self.assertEqual(role.RESERVED_PORTS, frozenset({11434, 11435, 1234, 1235, 8000, 8001}))


class Errors(unittest.TestCase):
    def test_table(self):
        self.assertEqual({code for codes in STATUSES.values() for code in codes}, set(errors.ERROR_CODES))
        for status, codes in STATUSES.items():
            for code in codes:
                self.assertEqual(errors.spec(code).status, status, code)
        self.assertEqual({c for c, row in errors.ERROR_CODES.items() if row.no_retry}, NO_RETRY)
        for code, row in errors.ERROR_CODES.items():
            if row.status >= 500 and code not in errors.RETRYABLE_5XX:
                self.assertTrue(row.no_retry, code)
        self.assertLessEqual(errors.RETRYABLE_5XX, {c for c, r in errors.ERROR_CODES.items() if r.status >= 500})
        anthropic = {400: 'invalid_request_error', 401: 'authentication_error', 402: 'permission_error',
                     403: 'permission_error', 404: 'not_found_error', 413: 'request_too_large', 429: 'rate_limit_error',
                     503: 'overloaded_error', 529: 'overloaded_error', 500: 'api_error', 502: 'api_error', 504: 'api_error'}
        for status, kind in anthropic.items():
            self.assertEqual(errors.anthropic_type(status), kind)
        gemini = {400: 'INVALID_ARGUMENT', 401: 'UNAUTHENTICATED', 403: 'PERMISSION_DENIED', 404: 'NOT_FOUND',
                  429: 'RESOURCE_EXHAUSTED', 503: 'UNAVAILABLE', 504: 'DEADLINE_EXCEEDED', 500: 'INTERNAL', 502: 'INTERNAL'}
        for status, value in gemini.items():
            self.assertEqual(errors.gemini_status(status), value)
        with self.assertRaises(KeyError):
            errors.spec('made_up')

    def test_broker_error(self):
        exc = errors.BrokerError('Budget exhausted', 'budget_exceeded', retry_after=1.2, origin='policy')
        self.assertEqual((exc.status, exc.code, str(exc), exc.no_retry), (402, 'budget_exceeded', 'Budget exhausted', True))
        self.assertEqual(exc.response_headers(), {'X-Titanos-Error-Origin': 'policy', 'Retry-After': '2',
                                                  'x-should-retry': 'false'})
        from titanos.engine import EngineError
        self.assertIsInstance(exc, EngineError)
        self.assertEqual(errors.BrokerError('x', 'model_not_found', status=404).status, 404)
        with self.assertRaises(ValueError):
            errors.BrokerError('x', 'not_a_code')
        with self.assertRaises(ValueError):
            errors.BrokerError('x', 'not_found', origin='elsewhere')
        refused = errors.EgressRefused()
        self.assertEqual((refused.code, refused.status, refused.origin), ('egress_refused', 502, 'policy'))
        self.assertTrue(issubclass(errors.BrokerConfigSecretError, ValueError))
        self.assertEqual(errors.BrokerError('x', 'upstream_unavailable').response_headers(),
                         {'X-Titanos-Error-Origin': 'titanos'})


class Envelopes(unittest.TestCase):
    def test_protocol_for(self):
        cases = {'/v1/messages': 'anthropic', '/v1/messages/count_tokens': 'anthropic', '/v1/messagesx': 'openai',
                 '/v1beta/models/m:generateContent': 'gemini', '/v1/text-to-speech/v': 'elevenlabs',
                 '/v1/speech-to-text': 'elevenlabs', '/v1/voices/x': 'elevenlabs', '/api/chat': 'ollama',
                 '/api/embeddings': 'ollama', '/api/embed': 'ollama', '/api/tags': 'ollama', '/api/jobs': 'openai',
                 '/v1/chat/completions': 'openai', '/v1/audio/speech': 'openai', '/api/chatty': 'openai'}
        for path, protocol in cases.items():
            self.assertEqual(envelopes.protocol_for(path), protocol, path)

    def test_render_shapes(self):
        expected = {
            'openai': {'error': {'message': 'm', 'type': 'model_not_found', 'param': None, 'code': 'model_not_found'}},
            'anthropic': {'type': 'error', 'error': {'type': 'not_found_error', 'message': 'm'}},
            'gemini': {'error': {'code': 404, 'message': 'm', 'status': 'NOT_FOUND'}},
            'elevenlabs': {'detail': {'status': 'model_not_found', 'message': 'm'}},
            'ollama': {'error': 'm'},
        }
        for protocol, body in expected.items():
            with self.subTest(protocol=protocol):
                response = envelopes.render(protocol, 404, 'model_not_found', 'm')
                self.assertEqual(response.status_code, 404)
                self.assertEqual(json.loads(response.body), body)
                self.assertEqual(response.headers['x-titanos-error-origin'], 'titanos')
                self.assertNotIn('x-should-retry', response.headers)
                self.assertEqual(response.headers['content-type'], 'application/json')
        response = envelopes.render('anthropic', 503, 'breaker_open', 'open', retry_after=3.5, origin='upstream',
                                    extra_headers={'X-Request-ID': 'r1'})
        self.assertEqual(json.loads(response.body)['error']['type'], 'overloaded_error')
        self.assertEqual((response.headers['retry-after'], response.headers['x-titanos-error-origin'],
                          response.headers['x-request-id']), ('4', 'upstream', 'r1'))
        self.assertEqual(envelopes.render('openai', 403, 'cloud_disabled', 'off').headers['x-should-retry'], 'false')
        self.assertEqual(json.loads(envelopes.render_error('gemini', errors.BrokerError('slow', 'upstream_timeout')).body),
                         {'error': {'code': 504, 'message': 'slow', 'status': 'DEADLINE_EXCEEDED'}})
        with self.assertRaises(ValueError):
            envelopes.render('soap', 400, 'invalid_request', 'x')

    def test_stream_terminals(self):
        openai = envelopes.stream_terminal('openai', 'upstream_unavailable', 'gone')
        first, done = openai.decode().split('\n\n')[:2]
        self.assertEqual(json.loads(first[len('data: '):]),
                         {'error': {'message': 'gone', 'type': 'stream_interrupted', 'code': 'upstream_unavailable'}})
        self.assertEqual(done, 'data: [DONE]')
        anthropic = envelopes.stream_terminal('anthropic', 'upstream_unavailable', 'gone').decode()
        self.assertTrue(anthropic.startswith('event: error\ndata: '))
        self.assertEqual(json.loads(anthropic.split('data: ', 1)[1]),
                         {'type': 'error', 'error': {'type': 'overloaded_error', 'message': 'gone'}})
        self.assertIn('"api_error"', envelopes.stream_terminal('anthropic', 'upstream_timeout', 'x').decode())
        gemini = envelopes.stream_terminal('gemini', 'upstream_timeout', 'late').decode()
        self.assertEqual(json.loads(gemini[len('data: '):]),
                         {'error': {'code': 504, 'message': 'late', 'status': 'DEADLINE_EXCEEDED'}})
        self.assertEqual(json.loads(envelopes.stream_terminal('ollama', 'upstream_unavailable', 'x')), {'error': 'x'})
        self.assertTrue(envelopes.stream_terminal('ollama', 'upstream_unavailable', 'x').endswith(b'\n'))
        self.assertEqual(envelopes.stream_terminal('elevenlabs', 'upstream_unavailable', 'x'), b'')
        responses = envelopes.stream_terminal('responses', 'upstream_unavailable', 'x', sequence_number=7,
                                              response={'id': 'resp_1', 'status': 'in_progress'}).decode()
        self.assertTrue(responses.startswith('event: response.failed\n'))
        event = json.loads(responses.split('data: ', 1)[1])
        self.assertEqual((event['type'], event['sequence_number'], event['response']['status'], event['response']['id']),
                         ('response.failed', 7, 'failed', 'resp_1'))

    def test_messages_are_scrubbed_at_the_envelope(self):
        leaked = 'sk-' + os.urandom(12).hex()
        message = 'upstream said: bad key ' + leaked
        for protocol in ('openai', 'anthropic', 'gemini', 'elevenlabs', 'ollama'):
            with self.subTest(protocol=protocol):
                self.assertNotIn(leaked, envelopes.render(protocol, 401, 'upstream_auth_failed', message).body.decode())
                self.assertNotIn(leaked, envelopes.stream_terminal(protocol, 'upstream_unavailable', message).decode())
        self.assertNotIn(leaked, envelopes.stream_terminal('responses', 'upstream_unavailable', message).decode())
        self.assertNotIn(leaked, envelopes.render_error('openai', errors.BrokerError(message, 'upstream_timeout')).body.decode())
        self.assertIn('bad key <redacted>', envelopes.render('openai', 502, 'upstream_unavailable', message).body.decode())

    def test_asgi_error(self):
        sent = []

        async def send(message):
            sent.append(message)
        asyncio.run(envelopes.asgi_error({'type': 'http'}, send, 'openai', 429, 'rate_limit_exceeded', 'slow down',
                                         retry_after=2))
        self.assertEqual(sent[0]['status'], 429)
        headers = dict(sent[0]['headers'])
        self.assertEqual(headers[b'retry-after'], b'2')
        self.assertEqual(json.loads(sent[1]['body'])['error']['code'], 'rate_limit_exceeded')


class Adapters(unittest.TestCase):
    def test_abc_requires_every_method(self):
        with self.assertRaises(TypeError):
            base.ProviderAdapter()
        methods = ('supports', 'passthrough', 'invoke', 'invoke_sync', 'list_models', 'probe', 'map_error',
                   'usage_from_body')
        self.assertEqual(base.ProviderAdapter.__abstractmethods__, frozenset(methods))
        partial = type('Partial', (base.ProviderAdapter,), {name: (lambda self, *a: None) for name in methods[:-1]})
        with self.assertRaises(TypeError):
            partial()
        complete = type('Complete', (base.ProviderAdapter,), {name: (lambda self, *a: None) for name in methods})
        self.assertIsInstance(complete(), base.ProviderAdapter)

    def test_header_filters(self):
        inbound = base.filter_inbound_headers([(b'Authorization', b'Bearer x'), (b'anthropic-beta', b'b1'),
                                               (b'x-api-key', b'k'), (b'Cookie', b'c'), (b'x-titanos-target', b't'),
                                               (b'Accept', b'text/event-stream'), (b'OpenAI-Beta', b'assistants=v2')])
        self.assertEqual(inbound, {'anthropic-beta': 'b1', 'accept': 'text/event-stream', 'openai-beta': 'assistants=v2'})
        base.RawRequest(method='POST', headers=inbound, json={'model': 'm'})
        for bad in ({'authorization': 'Bearer x'}, {'x-titanos-hop': 'h'}, {'cookie': 'c'}):
            with self.assertRaises(ValueError):
                base.RawRequest(method='POST', headers=bad)
        response = base.filter_response_headers({'Content-Type': 'application/json', 'Set-Cookie': 's', 'Location': '/x',
                                                 'x-ratelimit-remaining-tokens': '9', 'anthropic-ratelimit-requests-remaining':
                                                     '3', 'request-id': 'r', 'Connection': 'keep-alive',
                                                 'character-cost': '12', 'server': 'up'})
        self.assertEqual(set(response), {'content-type', 'x-ratelimit-remaining-tokens',
                                         'anthropic-ratelimit-requests-remaining', 'request-id', 'character-cost'})
        self.assertEqual(base.RESPONSE_HEADER_ALLOW, frozenset({'content-type', 'retry-after', 'request-id', 'x-request-id',
                                                                'openai-processing-ms', 'character-cost'}))
        self.assertTrue(dataclasses.is_dataclass(base.ProbeResult) and base.ProbeResult(True, 200, 3.0).code is None)


class FrozenRecords(unittest.TestCase):
    def test_ir_is_frozen(self):
        usage = types.Usage(input_tokens=1)
        samples = [ir.Text('t'), ir.Image('image/png', data=b'x'), ir.Audio('audio/wav', b'x'), ir.ToolUse('i', 'n', '{}'),
                   ir.ToolResult('i', [ir.Text('r')]), ir.Thinking('t', 's'), ir.IRMessage('user', [ir.Text('x')]),
                   ir.IRTool('n', 'd', {}), ir.IRToolChoice('tool', 'n'), ir.IRReasoning(effort='low'),
                   ir.IRRequest('chat', 'm', messages=[ir.IRMessage('user', [ir.Text('x')])]), ir.Start('i', 'm'),
                   ir.TextDelta(0, 'x'), ir.ThinkingDelta(0, 'x'), ir.ToolCallStart(0, 'i', 'n'), ir.ToolArgsDelta(0, '{'),
                   ir.BlockStop(0), ir.UsageEvent(usage), ir.Stop('end_turn'), ir.ErrorEvent('c', 'm'),
                   ir.IRResponse('i', 'm', [ir.Text('x')], 'end_turn', usage)]
        self.assertEqual(len({type(s) for s in samples}), len(samples))
        self.assertLessEqual(set(ir.PART_TYPES) | set(ir.EVENT_TYPES), {type(s) for s in samples})
        for sample in samples:
            with self.subTest(type=type(sample).__name__):
                self.assertTrue(type(sample).__dataclass_params__.frozen)
                name = dataclasses.fields(sample)[0].name
                with self.assertRaises(dataclasses.FrozenInstanceError):
                    setattr(sample, name, None)
        self.assertEqual(ir.STOP_REASONS, ('end_turn', 'max_tokens', 'tool_use', 'stop_sequence', 'content_filter'))

    def test_shared_types(self):
        for cls in (types.Principal, types.KeyConstraints, types.KeyLimits, types.KeyBudget, types.Target, types.Hints,
                    types.BodySummary, types.RouteContext, types.EffectivePolicy, types.Exclusion, types.RoutePlan,
                    types.EgressGrant, types.Usage, types.UpstreamError, types.AdvisorOutcome, types.KeyStoreSummary,
                    types.AuthFailure):
            self.assertTrue(cls.__dataclass_params__.frozen, cls.__name__)
        grant = types.EgressGrant(decision_id='d', request_id='r', purpose='probe', delegated=False, target=None,
                                  upstream_id='openai', origin='https://api.openai.com:443', credential=None, deadline=1.0)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            grant.origin = 'https://elsewhere.example.invalid:443'
        self.assertEqual([f.name for f in dataclasses.fields(types.Hints)][:8],
                         ['location', 'target_header', 'delegated', 'hop', 'deadline', 'attempt', 'trace_id', 'request_id'])
        decision = types.Decision(ts='2026-10-01T00:00:00Z', request_id='r', decision_id='d', role='standalone',
                                  surface='openai.chat', inbound_protocol='openai', requested_model='m')
        decision.attempts.append(types.Attempt(ref='m', outcome='success', status=200))
        record = decision.to_dict()
        self.assertEqual(list(record)[:9], ['ts', 'request_id', 'decision_id', 'trace_id', 'key_id', 'role', 'surface',
                                            'inbound_protocol', 'requested_model'])
        self.assertEqual(record['attempts'][0]['status'], 200)
        json.dumps(record)
        self.assertEqual(types.min_cloud_policy('fallback', 'explicit'), 'explicit')
        self.assertEqual(types.min_cloud_policy('off', 'fallback'), 'off')
        self.assertIsNone(principal({'admin'}, kind='bootstrap').key_id)
        self.assertEqual(principal({'inference'}).key_id, 'kfixture01')
        self.assertTrue(isinstance(type('R', (), {'resolve': lambda s, c: None, 'explain': lambda s, c: None})(),
                                   types.RouterProtocol))


class Vault(unittest.TestCase):
    def test_secret_wrapper(self):
        value = 'sk-test-' + os.urandom(12).hex()
        secret = vault.Secret(value)
        fp = vault.fingerprint(value)
        self.assertEqual(len(fp), 8)
        for text in (repr(secret), str(secret), f'{secret}', f'{secret!r:>40}', '%s' % (secret,)):
            self.assertNotIn(value, text)
            self.assertIn(fp, text)
        self.assertEqual(secret.reveal(), value)
        for call in (lambda: pickle.dumps(secret), lambda: copy.deepcopy(secret), lambda: copy.copy(secret)):
            with self.assertRaises(TypeError):
                call()
        with self.assertRaises(AttributeError):
            secret.fingerprint = 'x'
        with self.assertRaises(TypeError):
            hash(secret)
        self.assertEqual(secret, vault.Secret(value))
        from titanos.broker.redact import scrub
        self.assertNotIn(value, scrub('leaked ' + value))
        for bad in ('short', ' padded-value ', None):
            with self.assertRaises(ValueError):
                vault.Secret(bad)

    def test_credential_handle_binds_origin(self):
        secret = vault.Secret('nvapi-' + os.urandom(16).hex())
        handle = vault.CredentialHandle('https://integrate.example.invalid:443', 'bearer', secret, credential_id='nv')
        request = httpx.Request('POST', 'https://INTEGRATE.example.invalid/v1/chat/completions')
        handle.apply(request)
        self.assertEqual(request.headers['authorization'], 'Bearer ' + secret.reveal())
        self.assertNotIn(secret.reveal(), repr(handle))
        for url in ('https://integrate.example.invalid:8443/v1', 'http://integrate.example.invalid/v1',
                    'https://evil.example.invalid/v1', 'https://integrate.example.invalid.evil.invalid/v1'):
            with self.subTest(url=url):
                with self.assertRaises(errors.EgressRefused):
                    handle.apply(httpx.Request('GET', url))
        goog = vault.CredentialHandle('https://g.example.invalid:443', 'goog', vault.Secret('AIza' + 'k' * 35))
        request = httpx.Request('GET', 'https://g.example.invalid/v1beta/models')
        goog.apply(request)
        self.assertIn('x-goog-api-key', request.headers)
        anon = vault.CredentialHandle('http://127.0.0.1:6089', 'none', None)
        request = httpx.Request('GET', 'http://127.0.0.1:6089/v1/models')
        anon.apply(request)
        self.assertNotIn('authorization', request.headers)
        with self.assertRaises(ValueError):
            vault.CredentialHandle('https://x.example.invalid:443', 'bearer', None)
        with self.assertRaises(ValueError):
            vault.CredentialHandle('https://x.example.invalid:443', 'cookie', secret)
        self.assertEqual(vault.request_origin('https://[::1]:8443/x'), 'https://[::1]:8443')
        # the origin binding cannot be re-pointed after construction
        for name in ('origin', 'template', 'credential_id', '_secret'):
            with self.subTest(attribute=name):
                with self.assertRaises(AttributeError):
                    setattr(handle, name, 'https://evil.example.invalid:443')
                with self.assertRaises(AttributeError):
                    delattr(handle, name)
        self.assertEqual(handle.origin, 'https://integrate.example.invalid:443')
        with self.assertRaises(errors.EgressRefused):
            handle.apply(httpx.Request('GET', 'https://evil.example.invalid/v1'))

    def test_credential_record(self):
        record = vault.CredentialRecord(id='openai-main', source='env', env_var='OPENAI_API_KEY', upstream_id='openai',
                                        origin='https://api.openai.com:443')
        self.assertFalse(record.usable)
        self.assertEqual(vault.CredentialRecord.from_dict(record.to_dict()), record)
        self.assertTrue(dataclasses.replace(record, validation='ok').usable)
        self.assertFalse(dataclasses.replace(record, validation='failed:upstream_auth').usable)
        for bad in (dict(id='Bad'), dict(source='env', env_var=None), dict(source='vault'), dict(validation='maybe'),
                    dict(purpose='probe')):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    vault.CredentialRecord(**{**record.to_dict(), **bad})
        self.assertEqual(vault.CREDENTIAL_ID_RE, '^[a-z][a-z0-9_-]{0,63}$')
        self.assertTrue(hasattr(vault.VaultProtocol, 'put_secret') and hasattr(vault.VaultBackend, 'load'))


class Context(unittest.TestCase):
    def test_bind_and_thread_inheritance(self):
        who = principal({'inference'})
        hints = types.Hints(request_id='req-ctx')
        token = context.bind(who, hints)
        try:
            self.assertIs(context.current_principal(), who)
            self.assertEqual(context.current_request_id(), 'req-ctx')
            # asyncio.run starts a fresh context copy of the current one: to_thread inside it sees the binding
            seen = asyncio.run(self._to_thread())
            self.assertIs(seen, who)
            plain = []
            thread = threading.Thread(target=lambda: plain.append(context.current_principal()))
            thread.start()
            thread.join()
            self.assertEqual(plain, [None])  # job worker threads do not inherit the request context
        finally:
            context.reset(token)
        self.assertIsNone(context.current_principal())
        self.assertIsNone(context.current_hints())

    async def _to_thread(self):
        return await asyncio.to_thread(context.current_principal)


class ImportHygiene(unittest.TestCase):
    def test_import_does_no_io_network_or_logging_change(self):
        code = r'''
import builtins, io, logging, os, socket, sys
writes, dirs = [], []
real_open = builtins.open
def guarded_open(file, mode='r', *a, **k):
    if any(c in mode for c in 'wax+'):
        writes.append(str(file))
    return real_open(file, mode, *a, **k)
builtins.open = io.open = guarded_open
real_mkdir = os.mkdir
os.mkdir = lambda *a, **k: dirs.append(a[0])
def refuse(*a, **k):
    raise AssertionError('network during import')
socket.socket.connect = socket.socket.connect_ex = refuse
socket.create_connection = socket.getaddrinfo = refuse
factory = logging.getLogRecordFactory()
handlers = [list(h.filters) for h in logging.getLogger().handlers]
import titanos.broker
import titanos.broker.config, titanos.broker.naming, titanos.broker.profiles, titanos.broker.surfaces
import titanos.broker.scopes, titanos.broker.role, titanos.broker.context, titanos.broker.redact
import titanos.broker.envelopes, titanos.broker.adapters.base, titanos.broker.translate.ir, titanos.broker.vault
assert not writes, writes
assert not dirs, dirs
assert logging.getLogRecordFactory() is factory
assert [list(h.filters) for h in logging.getLogger().handlers] == handlers
assert 'titanos.broker.facade' not in sys.modules
assert not os.listdir(os.environ['TITANOS_HOME'])
print('import-clean')
'''
        with tempfile.TemporaryDirectory(prefix='titanos-broker-import-') as home:
            env = {k: v for k, v in os.environ.items() if not k.startswith('TITANOS_')}
            env.update(PYTHONPATH=str(ROOT / 'src'), PYTHONDONTWRITEBYTECODE='1', TITANOS_HOME=home)
            result = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, env=env, capture_output=True,
                                    text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('import-clean', result.stdout)


if __name__ == '__main__':
    unittest.main(verbosity=1)
