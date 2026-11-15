"""Broker reachability (2026-10-02 round 2): a connected upstream answers well-formed requests without any client having
listed models first, OpenAI videos work through the stock base_url /v1, vendor-shaped model lookups resolve bare ids
exactly like inference, and requests refused locally on the native lane never count against the shared breaker. Over
the integrated app (create_app: Guard, BrokerDispatch) against the WP4 loopback mocks; the netguard is installed and
every test ends with zero non-loopback connection attempts. Synthetic sentinel credentials only.

- live catalogs on demand: an upstream created by connect (catalog 'live', no static model list) in a fresh process
  (the live cache is in memory only, so this is every restart): a chat to model@upstream and to the provider's bare id
  answers 200 after exactly one catalog fetch; a later request fetches nothing; an unknown model is 404 model_not_found
  after one fetch and the next one fetches nothing (TTL); a failing catalog is 404 and is not fetched again on the next
  request (retry window); a caller who cannot see the upstream never makes it contact the provider; concurrent misses
  share one fetch (single-flight); GET /v1/models/{ref} answers in a fresh process; a request whose deadline expires
  during the fetch is answered by its deadline and the fetch still completes for the next request; a moderations
  request without a model reaches the provider default listed only by the live catalog;
- OpenAI videos through base_url /v1 (the request shapes the OpenAI SDK sends: JSON without input_reference, multipart
  with it): sora-2 (not in any model list: the profile's video family), sora-2@<upstream> and a published alias reach
  OpenAI with the provider's own id (multipart model field rewritten, the file part intact); the created id is bound to
  the caller, so retrieve, content and delete reach OpenAI too, for a key that holds native, cloud and cloud:store but
  not jobs; an unknown model, a local video model and an id that was not created through Titanos stay on Titanos' local
  video routes (zero provider hits), and a key without jobs is refused there 403 scope_denied; keyless loopback reaches
  OpenAI only with policy.anonymous_cloud; with two OpenAI upstreams a bare sora-2 stays local and sora-2@<id> picks one;
- vendor-shaped lookups: the OpenAI SDK's models.retrieve('gpt-4o-mini') and google-genai's
  models.get('gemini-2.5-flash') (also 'models/...') answer 200 under the requested id with titanos.ref, while two
  candidate upstreams keep 404 and a local model of that name wins;
- native lane hygiene: a {model} path parameter '..@<upstream>' is 400 before egress (re-checked after the model
  rewrite), an empty segment inside a {x:path} parameter is 400 before egress, and requests the egress client refuses
  locally (502 egress_refused) never open the shared upstream breaker, so typed chat on that upstream keeps serving;
- round 3: a key's own provider objects never open the shared breaker either (a declared size over the media cap, a
  storage redirect: neutral, as on the typed lane); on-demand catalogs never hold local work: API Plus and job POSTs
  naming a local model, a local_only body or header, a local-only job task or a hop contact no provider, a cloud stage
  fetches only the catalog its model names, hop catalog lookups fetch nothing, every on-demand wait is capped
  (facade.CATALOG_WAIT_SECONDS), GET /api/tags never waits for a catalog, and a failing catalog's retry window runs from
  the end of the attempt;
- security round 2, the shared credential: OpenAI's restricted-key 401 ('insufficient permissions ... Missing scopes')
  on a native row (bare /v1/videos and /native) or a typed surface (moderations) never marks the credential and is not
  even checked, so every other key's chat keeps answering 200; a 401 without the provider's key-rejection code is
  checked once against the probe endpoint (GET /v1/models; three 401s, one check) and the credential stays usable when
  that answers 200; a key the provider rejects there too goes offline (failed:upstream_auth) and the provider is not
  contacted again; a 401 with invalid_api_key marks at once without a check.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import broker_wp7_support as W  # noqa: E402  (installs the netguard)
from broker_mocks import GeminiMock, OpenAIMock, netguard  # noqa: E402
from broker_mocks.native_routes import VIDEO_MP4 as MP4, clear_native, install_openai_videos  # noqa: E402

BOOT = 'reachability-check-bootstrap-0001'
MOCKS: dict = {}
CHAT = [{'role': 'user', 'content': 'Say hello.'}]
NATIVE = ('native', 'cloud', 'cloud:store')
PNG = bytes.fromhex('89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de0000000c4944415478'
                    '9c63f80f00000101000518d84e0000000049454e44ae426082')


def setUpModule():
    MOCKS['openai'] = OpenAIMock(models=['gpt-4o-mini', 'omni-moderation-latest'])
    MOCKS['openai2'] = OpenAIMock(models=['gpt-4o-mini'])
    MOCKS['gemini'] = GeminiMock()
    for name in ('openai', 'openai2'):
        install_openai_videos(MOCKS[name])
    for item in MOCKS.values():
        item.start()


def tearDownModule():
    for item in MOCKS.values():
        item.stop()


def bearer(token: str) -> dict:
    return {'Authorization': 'Bearer ' + token}


def paths(mock, path: str) -> int:
    return sum(1 for hit in mock.hits if hit.path == path)


class Base(unittest.TestCase):
    keyed = True
    models: dict = {}
    settings_kw: dict = {}

    def document(self) -> dict:
        raise NotImplementedError

    def setUp(self):
        netguard.reset_attempts()
        for item in MOCKS.values():
            item.reset()
            clear_native(item)
        self.full = W.FullApp(self.document(), api_key=BOOT if self.keyed else None, models=dict(self.models),
                              settings_kw=dict(self.settings_kw))
        self.http = self.full.http

    def tearDown(self):
        self.full.close()
        self.assertEqual(netguard.attempts(), [], 'a non-loopback connect was attempted')

    def key(self, *scopes, **kw) -> str:
        return self.full.mint(*scopes, name='r' + str(len(scopes)) + str(len(kw)), **kw)[1]

    def chat(self, model: str, token: str, **headers):
        return self.http.post('/v1/chat/completions', json={'model': model, 'messages': CHAT},
                              headers={**bearer(token), **headers})


# -- 1. live catalogs on demand -----------------------------------------------------------------------------------------
class LiveCatalogOnDemand(Base):
    """broker.json as `titanos broker connect openai` writes it (catalog live, no model list), in a fresh process."""

    def document(self) -> dict:
        return {'policy': {'cloud_policy': 'explicit'},
                'upstreams': {'oai': W.S.upstream(MOCKS['openai'], 'openai', credential='oai-key', catalog='live')}}

    def test_qualified_ref_after_a_restart(self):
        token = self.key('inference', 'cloud')
        answer = self.chat('gpt-4o-mini@oai', token)
        self.assertEqual(answer.status_code, 200, answer.text)
        self.assertEqual([hit.path for hit in MOCKS['openai'].hits], ['/v1/models', '/v1/chat/completions'])
        again = self.chat('gpt-4o-mini@oai', token)
        self.assertEqual(again.status_code, 200, again.text)
        self.assertEqual(paths(MOCKS['openai'], '/v1/models'), 1)  # cached for the TTL

    def test_bare_provider_id_after_a_restart(self):
        token = self.key('inference', 'cloud')
        answer = self.chat('gpt-4o-mini', token)
        self.assertEqual(answer.status_code, 200, answer.text)
        self.assertEqual(answer.headers.get('x-titanos-served-by'), 'cloud:oai')
        self.assertEqual(paths(MOCKS['openai'], '/v1/models'), 1)

    def test_unknown_model_fetches_once(self):
        token = self.key('inference', 'cloud')
        for _ in range(2):
            answer = self.chat('no-such-model@oai', token)
            self.assertEqual((answer.status_code, answer.json()['error']['code']), (404, 'model_not_found'))
        self.assertEqual([hit.path for hit in MOCKS['openai'].hits], ['/v1/models'])

    def test_failing_catalog_is_not_fetched_on_every_request(self):
        token = self.key('inference', 'cloud')
        MOCKS['openai'].faults.push('status', status=500, surface='openai.models')
        for _ in range(2):
            answer = self.chat('gpt-4o-mini@oai', token)
            self.assertEqual((answer.status_code, answer.json()['error']['code']), (404, 'model_not_found'))
        self.assertEqual([hit.path for hit in MOCKS['openai'].hits], ['/v1/models'])  # retried at most once a minute
        self.assertEqual(self.full.broker.snapshot().catalog.status_of('oai')['source'], 'static')

    def test_invisible_upstream_is_never_contacted(self):
        for token in (self.key('inference'), self.key('inference', 'cloud:other')):
            answer = self.chat('gpt-4o-mini@oai', token)
            self.assertEqual(answer.status_code, 404, answer.text)
        self.assertEqual(MOCKS['openai'].hits, [])

    def test_concurrent_misses_share_one_fetch(self):
        broker = self.full.broker
        self.http.get('/healthz')
        from titanos.broker.types import Principal
        admin = Principal(id='bootstrap', kind='bootstrap', scopes=frozenset({'admin'}))

        async def both():
            return await asyncio.gather(broker.ensure_catalogs(admin, ['oai']), broker.ensure_catalogs(admin, ['oai']),
                                        broker.refresh_catalogs(admin))
        results = asyncio.run(both())
        self.assertEqual(results, [['oai'], ['oai'], ['oai']])
        self.assertEqual(paths(MOCKS['openai'], '/v1/models'), 1)
        self.assertTrue(broker.snapshot().catalog.model_listed('oai', 'gpt-4o-mini'))

    def test_model_lookup_after_a_restart(self):
        token = self.key('models:read', 'cloud')
        one = self.http.get('/v1/models/gpt-4o-mini@oai', headers=bearer(token))
        self.assertEqual((one.status_code, one.json().get('id')), (200, 'gpt-4o-mini@oai'), one.text)
        self.assertEqual(paths(MOCKS['openai'], '/v1/models'), 1)

    def test_deadline_bounds_the_wait_and_the_fetch_still_lands(self):
        token = self.key('inference', 'cloud')
        MOCKS['openai'].faults.push('slow_first_byte', delay=1.5, surface='openai.models')
        started = time.monotonic()
        answer = self.chat('gpt-4o-mini@oai', token, **{'X-Titanos-Deadline-Ms': '300'})
        self.assertLess(time.monotonic() - started, 1.4)
        self.assertIn(answer.status_code, (404, 504), answer.text)
        deadline = time.monotonic() + 10
        while not self.full.broker.snapshot().catalog.model_listed('oai', 'gpt-4o-mini') and time.monotonic() < deadline:
            self.http.get('/healthz')  # the fetch runs on the server's loop
            time.sleep(0.05)
        later = self.chat('gpt-4o-mini@oai', token)
        self.assertEqual(later.status_code, 200, later.text)
        self.assertEqual(paths(MOCKS['openai'], '/v1/models'), 1)

    def test_provider_default_listed_only_by_the_live_catalog(self):
        token = self.key('inference', 'cloud')
        answer = self.http.post('/v1/moderations', json={'input': 'hello'}, headers=bearer(token))
        self.assertEqual(answer.status_code, 200, answer.text)
        self.assertEqual(json.loads(MOCKS['openai'].hits_for('openai.moderations')[-1].body)['model'],
                         'omni-moderation-latest')


class LiveCatalogCliConnect(unittest.TestCase):
    """The documented CLI flow: `titanos broker connect openai --from-env VAR` (another process: its catalog fetch never
    reaches the server), then a fresh server on that TITANOS_HOME serves the provider's models by qualified ref and bare
    id."""

    def test_cli_connect_then_serve(self):
        import os
        import shutil
        import tempfile
        from unittest import mock
        from fastapi.testclient import TestClient
        from broker_mocks import SENTINELS
        netguard.reset_attempts()
        MOCKS['openai'].reset()
        root = Path(tempfile.mkdtemp(prefix='titanos-reach-'))
        env = {'TITANOS_HOME': str(root / 'home'), 'REACH_OPENAI_KEY': SENTINELS['openai'],
               'TITANOS_BROKER_EGRESS': 'loopback-only', 'TITANOS_BROKER_VAULT_BACKEND': 'refs-only'}
        try:
            (root / 'home').mkdir()
            with mock.patch.dict(os.environ, env):
                from titanos.cli import main as cli_main
                import contextlib
                import io
                out = io.StringIO()
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                    try:
                        code = cli_main(['broker', 'connect', 'openai', '--from-env', 'REACH_OPENAI_KEY', '--base-url',
                                         MOCKS['openai'].api_base])
                    except SystemExit as exc:
                        code = exc.code
                self.assertIn(code, (0, None), out.getvalue())
                config = json.loads((root / 'home' / 'broker.json').read_text(encoding='utf-8'))
                self.assertEqual(config['upstreams']['openai']['catalog'], 'live')
                self.assertEqual(config['models'], {})  # no aliases: resolution depends on the live catalog
                MOCKS['openai'].reset()
                from titanos.config import Config
                from titanos.server import create_app
                from titanos.settings import Settings
                app = create_app(Config(models={}), api_key=None, output_dir=root / 'outputs',
                                 settings=Settings(broker_egress='loopback-only', broker_vault_backend='refs-only'))
                with TestClient(app, base_url='http://127.0.0.1:8000', client=('127.0.0.1', 50000)) as http:
                    for model in ('gpt-4o-mini@openai', 'gpt-4o-mini'):
                        answer = http.post('/v1/chat/completions', json={'model': model, 'messages': CHAT},
                                           headers={'X-Titanos-Studio': '1'})
                        self.assertEqual(answer.status_code, 200, answer.text)
                self.assertEqual(paths(MOCKS['openai'], '/v1/models'), 1)
        finally:
            shutil.rmtree(root, ignore_errors=True)
        self.assertEqual(netguard.attempts(), [], 'a non-loopback connect was attempted')


# -- 2. OpenAI videos through base_url /v1 -------------------------------------------------------------------------------
class OpenAIVideos(Base):
    models = {'local-video': ('video', 'diffusers')}

    def document(self) -> dict:
        return {'policy': {'cloud_policy': 'explicit'},
                'upstreams': {'oai': W.S.upstream(MOCKS['openai'], 'openai', credential='oai-key',
                                                  models=['gpt-4o-mini'], data_policy={'allow_store': True})},
                'models': {'video': {'target': 'sora-2-pro@oai'}}}

    def create(self, token, body=None, **kw):
        return self.http.post('/v1/videos', json=body, headers=bearer(token), **kw)

    def test_sdk_json_create_and_follow_ups_reach_openai(self):
        token = self.key(*NATIVE)  # no 'jobs': the cloud video needs only the native scopes
        created = self.create(token, {'prompt': 'a fox in the snow', 'model': 'sora-2', 'seconds': '4',
                                      'size': '720x1280'})
        self.assertEqual(created.status_code, 200, created.text)
        video_id = created.json()['id']
        self.assertEqual((video_id, created.json()['model']), ('video_mock0001', 'sora-2'))
        self.assertEqual(created.headers.get('x-titanos-translation'), 'native')
        hits = MOCKS['openai'].hits_for('native:videos')
        self.assertEqual(hits[-1].credentials[0][1], 'Bearer <sentinel:openai>')
        retrieved = self.http.get(f'/v1/videos/{video_id}', headers=bearer(token))
        self.assertEqual((retrieved.status_code, retrieved.json()['id']), (200, video_id), retrieved.text)
        content = self.http.get(f'/v1/videos/{video_id}/content', headers=bearer(token))
        self.assertEqual((content.status_code, content.content), (200, MP4), content.text[:200])
        self.assertTrue(content.headers['content-type'].startswith('video/mp4'))
        deleted = self.http.delete(f'/v1/videos/{video_id}', headers=bearer(token))
        self.assertEqual((deleted.status_code, deleted.json().get('deleted')), (200, True), deleted.text)
        self.assertEqual(len(MOCKS['openai'].hits_for('native:videos')), 4)

    def test_qualified_alias_and_multipart_reach_openai_with_the_provider_id(self):
        token = self.key(*NATIVE)
        qualified = self.create(token, {'prompt': 'a fox', 'model': 'sora-2@oai'})
        self.assertEqual((qualified.status_code, qualified.json()['model']), (200, 'sora-2'), qualified.text)
        alias = self.create(token, {'prompt': 'a fox', 'model': 'video'})
        self.assertEqual((alias.status_code, alias.json()['model']), (200, 'sora-2-pro'), alias.text)
        multipart = self.http.post('/v1/videos', data={'prompt': 'a fox', 'model': 'sora-2@oai', 'seconds': '8'},
                                   files={'input_reference': ('ref.png', PNG, 'image/png')}, headers=bearer(token))
        self.assertEqual(multipart.status_code, 200, multipart.text)
        received = multipart.json()['received']
        self.assertEqual((received['model'], received['seconds'], received['reference_bytes']), ('sora-2', '8', len(PNG)))

    def test_local_models_unknown_names_and_other_ids_stay_local(self):
        token = self.key(*NATIVE, 'jobs')
        unknown = self.create(token, {'prompt': 'a fox', 'model': 'no-such-video-model'})
        self.assertEqual(unknown.status_code, 404, unknown.text)
        self.assertIn('Unknown model', unknown.text)
        local = self.create(token, {'prompt': 'a fox', 'model': 'local-video'})
        self.assertEqual((local.status_code, local.json()['object']), (202, 'video'), local.text)
        self.assertEqual(self.http.get('/v1/videos/' + local.json()['id'], headers=bearer(token)).status_code, 200)
        foreign = self.http.get('/v1/videos/video_not_created_here', headers=bearer(token))
        self.assertEqual(foreign.status_code, 404, foreign.text)
        self.assertEqual(MOCKS['openai'].hits_for('native:videos'), [])

    def test_local_route_still_needs_the_jobs_scope(self):
        token = self.key(*NATIVE)
        for answer in (self.create(token, {'prompt': 'a fox', 'model': 'local-video'}),
                       self.http.get('/v1/videos/abc123', headers=bearer(token)),
                       self.http.delete('/v1/videos/abc123', headers=bearer(token))):
            self.assertEqual((answer.status_code, answer.json()['error']['code']), (403, 'scope_denied'), answer.text)
            self.assertIn('(jobs)', answer.text)
        listing = self.http.get('/v1/videos', headers=bearer(token))
        self.assertEqual(listing.status_code, 403, listing.text)
        self.assertEqual(MOCKS['openai'].hits_for('native:videos'), [])

    def test_cloud_video_needs_the_native_scopes(self):
        token = self.key('jobs', 'cloud', 'cloud:store')  # no native: the bare row is not a candidate for this key
        answer = self.create(token, {'prompt': 'a fox', 'model': 'sora-2@oai'})
        self.assertEqual(answer.status_code, 403, answer.text)
        self.assertEqual(MOCKS['openai'].hits_for('native:videos'), [])


class OpenAIVideosKeyless(Base):
    keyed = False
    anonymous_cloud = True

    def document(self) -> dict:
        return {'policy': {'cloud_policy': 'explicit', 'anonymous_cloud': self.anonymous_cloud},
                'upstreams': {'oai': W.S.upstream(MOCKS['openai'], 'openai', credential='oai-key',
                                                  models=['gpt-4o-mini'], data_policy={'allow_store': True})}}

    def test_keyless_owner_reaches_openai(self):
        created = self.http.post('/v1/videos', json={'prompt': 'a fox', 'model': 'sora-2'})
        self.assertEqual(created.status_code, 200, created.text)
        got = self.http.get('/v1/videos/' + created.json()['id'])
        self.assertEqual((got.status_code, got.json()['model']), (200, 'sora-2'), got.text)


class OpenAIVideosKeylessCloudOff(OpenAIVideosKeyless):
    anonymous_cloud = False

    def test_keyless_owner_reaches_openai(self):
        answer = self.http.post('/v1/videos', json={'prompt': 'a fox', 'model': 'sora-2'})
        self.assertEqual(answer.status_code, 404, answer.text)  # the local route: Unknown model
        self.assertEqual(MOCKS['openai'].hits_for('native:videos'), [])


class OpenAIVideosTwoUpstreams(Base):
    def document(self) -> dict:
        upstreams = {uid: W.S.upstream(MOCKS[name], 'openai', credential=uid + '-key', models=['gpt-4o-mini'],
                                       data_policy={'allow_store': True})
                     for uid, name in (('oai', 'openai'), ('oai2', 'openai2'))}
        return {'policy': {'cloud_policy': 'explicit'}, 'upstreams': upstreams}

    def test_bare_family_id_is_ambiguous_and_a_ref_picks_one(self):
        token = self.key(*NATIVE, 'jobs')
        bare = self.http.post('/v1/videos', json={'prompt': 'a fox', 'model': 'sora-2'}, headers=bearer(token))
        self.assertEqual(bare.status_code, 404, bare.text)
        picked = self.http.post('/v1/videos', json={'prompt': 'a fox', 'model': 'sora-2@oai2'}, headers=bearer(token))
        self.assertEqual((picked.status_code, picked.json()['model']), (200, 'sora-2'), picked.text)
        self.assertEqual((len(MOCKS['openai'].hits_for('native:videos')), len(MOCKS['openai2'].hits_for('native:videos'))),
                         (0, 1))
        got = self.http.get('/v1/videos/' + picked.json()['id'], headers=bearer(token))
        self.assertEqual(got.status_code, 200, got.text)
        self.assertEqual(len(MOCKS['openai2'].hits_for('native:videos')), 2)


# -- 3. vendor-shaped model lookups ------------------------------------------------------------------------------------
class VendorModelLookups(Base):
    models = {'local-chat': 'chat'}

    def document(self) -> dict:
        gemini = MOCKS['gemini']
        return {'policy': {'cloud_policy': 'explicit'},
                'upstreams': {'oai': W.S.upstream(MOCKS['openai'], 'openai', credential='oai-key',
                                                  models=['gpt-4o-mini']),
                              'gem': {'profile': 'gemini', 'base_url': gemini.api_base, 'credential': 'gem-key',
                                      'enabled': True, 'models': ['gemini-2.5-flash']}}}

    def test_openai_models_retrieve_with_a_bare_id(self):
        token = self.key('inference', 'cloud', 'models:read')
        one = self.http.get('/v1/models/gpt-4o-mini', headers=bearer(token))
        self.assertEqual(one.status_code, 200, one.text)
        self.assertEqual((one.json()['id'], one.json()['owned_by'], one.json()['titanos']['ref']),
                         ('gpt-4o-mini', 'oai', 'gpt-4o-mini@oai'))
        self.assertEqual(self.chat('gpt-4o-mini', token).status_code, 200)  # inference resolves it the same way
        self.assertEqual(self.http.get('/v1/models/gpt-4.1-nano', headers=bearer(token)).status_code, 404)
        self.assertEqual(self.http.get('/v1/models/gpt-4o-mini', headers=bearer(self.key('models:read'))).status_code,
                         404)  # a caller who cannot use the upstream sees nothing
        local = self.http.get('/v1/models/local-chat', headers=bearer(token))
        self.assertEqual(local.status_code, 200, local.text)
        self.assertNotIn('titanos', local.json())

    def test_gemini_models_get_with_a_bare_id(self):
        token = self.key('inference', 'cloud', 'models:read')
        for spelled in ('gemini-2.5-flash', 'models/gemini-2.5-flash'):
            one = self.http.get('/v1beta/models/' + spelled, headers={'x-goog-api-key': token})
            self.assertEqual(one.status_code, 200, one.text)
            self.assertEqual((one.json()['name'], one.json()['baseModelId'], one.json()['titanos']['ref']),
                             ('models/gemini-2.5-flash', 'gemini-2.5-flash', 'gemini-2.5-flash@gem'))
        generated = self.http.post('/v1beta/models/gemini-2.5-flash:generateContent',
                                   json={'contents': [{'role': 'user', 'parts': [{'text': 'hi'}]}]},
                                   headers={'x-goog-api-key': token})
        self.assertEqual(generated.status_code, 200, generated.text)
        self.assertEqual(self.http.get('/v1beta/models/gemini-0-none', headers={'x-goog-api-key': token}).status_code,
                         404)


class VendorModelLookupsTwoUpstreams(Base):
    models = {'gpt-4o-mini': 'chat'}

    def document(self) -> dict:
        upstreams = {uid: W.S.upstream(MOCKS[name], 'openai', credential=uid + '-key', models=['gpt-4o-mini',
                                                                                               'gpt-4.1-mini'])
                     for uid, name in (('oai', 'openai'), ('oai2', 'openai2'))}
        return {'policy': {'cloud_policy': 'explicit'}, 'upstreams': upstreams}

    def test_ambiguous_bare_id_stays_404_and_local_wins(self):
        token = self.key('inference', 'cloud', 'models:read')
        self.assertEqual(self.http.get('/v1/models/gpt-4.1-mini', headers=bearer(token)).status_code, 404)
        local = self.http.get('/v1/models/gpt-4o-mini', headers=bearer(token))
        self.assertEqual(local.status_code, 200, local.text)
        self.assertNotIn('titanos', local.json())  # the local registry row, not a cloud entry


# -- 4. native lane: local refusals never trip the shared breaker ------------------------------------------------------
class NativeBreakerHygiene(Base):
    def document(self) -> dict:
        gemini = MOCKS['gemini']
        return {'policy': {'cloud_policy': 'explicit'},
                'upstreams': {'oai': W.S.upstream(MOCKS['openai'], 'openai', credential='oai-key',
                                                  models=['gpt-4o-mini'], data_policy={'allow_store': True}),
                              'gem': {'profile': 'gemini', 'base_url': gemini.api_base, 'credential': 'gem-key',
                                      'enabled': True, 'models': ['gemini-2.5-flash'],
                                      'data_policy': {'allow_store': True}}}}

    def assert_typed_chat_serves(self):
        token = self.key('inference', 'cloud')
        answer = self.chat('gpt-4o-mini@oai', token)
        self.assertEqual(answer.status_code, 200, answer.text)
        gem = self.http.post('/v1/chat/completions', json={'model': 'gemini-2.5-flash@gem', 'messages': CHAT},
                             headers=bearer(token))
        self.assertEqual(gem.status_code, 200, gem.text)

    def test_dot_dot_model_ref_is_refused_before_egress(self):
        for _ in range(5):
            deleted = self.http.delete('/native/oai/models/..@oai', headers=bearer(BOOT))
            self.assertEqual((deleted.status_code, deleted.json()['error']['code']), (400, 'invalid_request'),
                             deleted.text)
            token = self.key('native', 'cloud')
            predict = self.http.post('/native/gem/v1beta/models/..@gem:predict', json={'instances': []},
                                     headers=bearer(token))
            self.assertEqual(predict.status_code, 400, predict.text)
        self.assertEqual((MOCKS['openai'].hits, MOCKS['gemini'].hits), ([], []))
        self.assert_typed_chat_serves()

    def test_empty_segment_in_a_path_parameter_is_refused_before_egress(self):
        for path in ('/native/gem/v1beta/environments/env1/files/a//b', '/native/oai/fine_tuning/jobs//x'):
            for _ in range(4):
                answer = self.http.get(path, headers=bearer(BOOT))
                self.assertEqual(answer.status_code, 400, answer.text)
                self.assertIn('Empty path segments', answer.text)
        self.assertEqual((MOCKS['openai'].hits, MOCKS['gemini'].hits), ([], []))
        self.assert_typed_chat_serves()

    def test_egress_refusals_never_open_the_shared_breaker(self):
        from titanos.broker.errors import EgressRefused
        broker = self.full.broker
        self.http.get('/healthz')
        egress = broker.egress
        original = egress.send
        calls = []

        async def refusing(grant, upstream, profile, method, path, **kw):
            if grant.purpose == 'inference' and upstream is not None and kw.get('native') is not None:
                calls.append(path)
                raise EgressRefused('Upstream path is not a plain path')
            return await original(grant, upstream, profile, method, path, **kw)

        egress.send = refusing
        try:
            token = self.key(*NATIVE)
            for _ in range(8):
                answer = self.http.post('/native/oai/responses/compact', json={'model': 'gpt-4o-mini', 'input': 'x'},
                                        headers=bearer(token))
                self.assertEqual((answer.status_code, answer.json()['error']['code']), (502, 'egress_refused'),
                                 answer.text)
        finally:
            egress.send = original
        self.assertEqual(len(calls), 8)
        upstream = broker.snapshot().config.upstreams['oai']
        self.assertTrue(broker.breakers.get('oai', upstream.credential).allow(consume=False))
        self.assert_typed_chat_serves()


class NativeBreakerOwnObjects(Base):
    """Round 3: the media cap lowered to 4 KiB stands in for TITANOS_BROKER_MAX_UPSTREAM_MEDIA_BYTES (64 MiB) and an
    8 KiB answer for a key's own larger object (video content, a batch output file, dubbing audio)."""
    settings_kw = {'broker_max_upstream_media_bytes': 4096}
    document = NativeBreakerHygiene.document
    assert_typed_chat_serves = NativeBreakerHygiene.assert_typed_chat_serves

    def own_video(self, token: str) -> str:
        created = self.http.post('/native/oai/videos', json={'model': 'sora-2', 'prompt': 'a lighthouse'},
                                 headers=bearer(token))
        self.assertEqual(created.status_code, 200, created.text)
        return created.json()['id']

    def assert_breaker_closed(self):
        broker = self.full.broker
        upstream = broker.snapshot().config.upstreams['oai']
        shared = broker.breakers.get('oai', upstream.credential)
        self.assertTrue(shared.allow(consume=False))
        self.assertEqual(shared.snapshot()['consecutive_failures'], 0)

    def test_oversized_own_objects_never_open_the_shared_breaker(self):
        token = self.key(*NATIVE)
        video = self.own_video(token)
        MOCKS['openai'].faults.push('oversized_body', size=8192, declared_length=True, path='/content$', times=8)
        for _ in range(8):
            answer = self.http.get(f'/native/oai/videos/{video}/content', headers=bearer(token))
            self.assertEqual((answer.status_code, answer.json()['error']['code']),
                             (502, 'upstream_response_too_large'), answer.text)
        self.assertEqual(paths(MOCKS['openai'], f'/v1/videos/{video}/content'), 8)  # each one reached the provider
        self.assert_breaker_closed()
        self.assert_typed_chat_serves()

    def test_redirected_own_objects_never_open_the_shared_breaker(self):
        token = self.key(*NATIVE)
        video = self.own_video(token)
        MOCKS['openai'].faults.push('redirect_foreign', path='/content$', times=6)
        sink = MOCKS['openai'].sink
        before = len(sink.hits)
        for _ in range(6):
            answer = self.http.get(f'/native/oai/videos/{video}/content', headers=bearer(token))
            self.assertEqual((answer.status_code, answer.json()['error']['code']),
                             (502, 'upstream_redirect_refused'), answer.text)
        self.assertEqual(len(sink.hits), before)  # never followed
        self.assert_breaker_closed()
        self.assert_typed_chat_serves()


# -- 5. on-demand catalogs never hold local work (round 3) ---------------------------------------------------------------
class StageCatalogs(Base):
    """Two live-catalog OpenAI upstreams (as `broker connect` writes them) in a fresh process, and a local chat model."""
    models = {'local-chat': 'chat'}

    def document(self) -> dict:
        return {'policy': {'cloud_policy': 'explicit'},
                'upstreams': {uid: W.S.upstream(MOCKS[name], 'openai', credential=uid + '-key', catalog='live')
                              for uid, name in (('oai', 'openai'), ('oai2', 'openai2'))}}

    def catalog_hits(self) -> tuple[int, int]:
        return paths(MOCKS['openai'], '/v1/models'), paths(MOCKS['openai2'], '/v1/models')

    def test_local_stage_posts_contact_no_provider(self):
        for name in ('openai', 'openai2'):
            MOCKS[name].faults.push('slow_first_byte', delay=2.0, surface='openai.models')
        token = self.key('inference', 'inference:plus', 'jobs', 'cloud')
        hop = {'X-Titanos-Hop': 'abc123'}
        cases = [
            ('/v1/translations', {'model': 'local-chat', 'text': 'hola', 'target_language': 'en'}, {}),
            ('/v1/translations', {'model': 'local-chat', 'text': 'hola', 'target_language': 'en', 'location': 'auto'},
             {}),
            # a cloud ref with the body's default location local_only, or the local_only header: placed locally
            ('/v1/translations', {'model': 'gpt-4o-mini@oai', 'text': 'hola', 'target_language': 'en'}, {}),
            ('/v1/translations', {'model': 'gpt-4o-mini@oai', 'text': 'hola', 'target_language': 'en',
                                  'location': 'auto'}, {'X-Titanos-Location': 'local_only'}),
            ('/v1/sentiment', {'model': 'local-chat', 'llm_model': 'gpt-4o-mini@oai', 'text': 'great',
                               'location': 'auto'}, hop),
            ('/api/jobs', {'kind': 'nothing-here'}, {}),
            ('/api/jobs', {'task': 'video', 'model': 'sora-2@oai', 'prompt': 'x'}, {}),
            ('/api/jobs', {'task': 'speech', 'model': 'gpt-4o-mini-tts@oai', 'input': 'x'}, hop),
        ]
        for path, body, headers in cases:
            started = time.monotonic()
            answer = self.http.post(path, json=body, headers={**bearer(token), **headers})
            self.assertLess(time.monotonic() - started, 1.5, (path, body, answer.status_code))
            self.assertEqual(self.catalog_hits(), (0, 0), (path, body, answer.text))

    def test_cloud_stage_fetches_only_the_catalog_its_model_names(self):
        token = self.key('inference', 'inference:plus', 'jobs', 'cloud')
        answer = self.http.post('/v1/translations', json={'model': 'gpt-4o-mini@oai', 'text': 'hola',
                                                          'target_language': 'en', 'location': 'auto'},
                                headers=bearer(token))
        self.assertEqual(answer.status_code, 200, answer.text)
        self.assertEqual(self.catalog_hits(), (1, 0))
        self.assertGreaterEqual(paths(MOCKS['openai'], '/v1/chat/completions'), 1)
        job = self.http.post('/api/jobs', json={'task': 'speech', 'model': 'gpt-4o-mini-tts@oai2', 'input': 'hi'},
                             headers=bearer(token))
        self.assertNotEqual(job.status_code, 500, job.text)
        self.assertEqual(self.catalog_hits(), (1, 1))

    def test_hop_catalog_lookups_contact_no_provider(self):
        token = self.key('inference', 'models:read', 'cloud')
        hop = {**bearer(token), 'X-Titanos-Hop': 'abc123'}
        for path in ('/v1/models/gpt-4o-mini@oai', '/v1/models/gpt-4o-mini', '/api/tags'):
            self.http.get(path, headers=hop)
        self.assertEqual(self.catalog_hits(), (0, 0))

    def test_ollama_tags_never_waits_for_a_catalog(self):
        MOCKS['openai'].faults.push('slow_first_byte', delay=2.0, surface='openai.models')
        token = self.key('inference', 'models:read', 'cloud')
        started = time.monotonic()
        tags = self.http.get('/api/tags', headers=bearer(token))
        self.assertLess(time.monotonic() - started, 1.5)
        self.assertEqual(tags.status_code, 200, tags.text)
        catalog = self.full.broker.snapshot().catalog
        deadline = time.monotonic() + 10
        while not catalog.model_listed('oai', 'gpt-4o-mini') and time.monotonic() < deadline:
            self.http.get('/healthz')  # the fetch runs on the server's loop
            time.sleep(0.05)
        later = self.http.get('/api/tags', headers=bearer(token))
        self.assertTrue(any('gpt-4o-mini' in str(item.get('name')) for item in later.json()['models']), later.text)

    def test_every_on_demand_wait_is_capped(self):
        from unittest import mock
        MOCKS['openai'].faults.push('slow_first_byte', delay=2.0, surface='openai.models')
        token = self.key('inference', 'cloud')
        with mock.patch('titanos.broker.facade.CATALOG_WAIT_SECONDS', 0.3):
            started = time.monotonic()
            answer = self.chat('gpt-4o-mini@oai', token)  # the default deadline is client_timeout (600 s)
            self.assertLess(time.monotonic() - started, 1.5)
        self.assertEqual(answer.status_code, 404, answer.text)
        catalog = self.full.broker.snapshot().catalog
        deadline = time.monotonic() + 10
        while not catalog.model_listed('oai', 'gpt-4o-mini') and time.monotonic() < deadline:
            self.http.get('/healthz')
            time.sleep(0.05)
        self.assertEqual(self.chat('gpt-4o-mini@oai', token).status_code, 200)

    def test_retry_window_runs_from_the_end_of_the_attempt(self):
        from titanos.broker.types import Principal
        broker = self.full.broker
        self.http.get('/healthz')
        catalog = broker.snapshot().catalog
        admin = Principal(id='bootstrap', kind='bootstrap', scopes=frozenset({'admin'}))
        now = [1000.0]
        calls = []

        async def hung(upstream_id):  # a fetch that used its whole connect + first-byte budget, then failed
            calls.append(upstream_id)
            now[0] += 130.0
            return {}

        async def scenario():
            await broker.ensure_catalogs(admin, ['oai'])
            now[0] += 30.0  # 160 s after the attempt started, 30 s after it ended
            await broker.ensure_catalogs(admin, ['oai'])
            now[0] += 31.0  # 61 s after it ended
            await broker.ensure_catalogs(admin, ['oai'])

        original_clock, original_refresh = broker._clock, catalog.refresh_upstream
        broker._clock, catalog.refresh_upstream = (lambda: now[0]), hung
        try:
            asyncio.run(scenario())
        finally:
            broker._clock, catalog.refresh_upstream = original_clock, original_refresh
        self.assertEqual(calls, ['oai', 'oai'])


# -- 6. a caller's 401 never takes the shared credential offline (security round 2) ---------------------------------------
SCOPES_401 = ("You have insufficient permissions for this operation. Missing scopes: api.videos.write. Check that you "
              "have the correct role in your organization (Reader, Writer, Owner) and project (Member, Owner), and if "
              "you're using a restricted API key, that it has the necessary scopes.")


class CredentialChecks(Base):
    """OpenAI answers a restricted or project-limited key with 401 'insufficient permissions ... Missing scopes' on every
    endpoint outside its permissions. Any key holding native, cloud and cloud:store (or inference and cloud) chooses
    which endpoint it calls, so such a 401, or any 401 without the provider's key-rejection code, must never take the
    shared credential offline for every other key; a key the provider really rejects still goes offline."""

    def document(self) -> dict:
        return {'policy': {'cloud_policy': 'explicit'},
                'upstreams': {'oai': W.S.upstream(MOCKS['openai'], 'openai', credential='oai-key',
                                                  models=['gpt-4o-mini', 'omni-moderation-latest'],
                                                  data_policy={'allow_store': True})}}

    def validation(self) -> str:
        return self.full.vault.record('oai-key').validation

    def assert_victim_serves(self):
        answer = self.chat('gpt-4o-mini@oai', self.key('inference', 'cloud'))
        self.assertEqual(answer.status_code, 200, answer.text)
        self.assertEqual(self.validation(), 'ok')

    def test_restricted_key_scope_401_keeps_the_credential(self):
        self.http.get('/healthz')
        native, typed = self.key(*NATIVE), self.key('inference', 'cloud')
        MOCKS['openai'].faults.push('status', status=401, message=SCOPES_401, path='/videos$', times=2)
        for path in ('/v1/videos', '/native/oai/videos'):
            answer = self.http.post(path, json={'model': 'sora-2', 'prompt': 'a lighthouse'}, headers=bearer(native))
            self.assertEqual((answer.status_code, answer.json()['error']['code']), (502, 'upstream_auth_failed'),
                             answer.text)
            self.assert_victim_serves()
        MOCKS['openai'].faults.push('status', status=401, message=SCOPES_401, path='/moderations$')
        answer = self.http.post('/v1/moderations', json={'model': 'omni-moderation-latest@oai', 'input': 'hi'},
                                headers=bearer(typed))
        self.assertEqual(answer.status_code, 502, answer.text)
        self.assert_victim_serves()
        self.assertEqual(paths(MOCKS['openai'], '/v1/videos'), 2)
        self.assertEqual(paths(MOCKS['openai'], '/v1/moderations'), 1)
        self.assertEqual(paths(MOCKS['openai'], '/v1/models'), 0)  # a scope refusal is not even worth a check
        self.assertEqual(self.full.broker.credential_checks, 0)

    def test_bare_401_is_checked_against_the_probe_endpoint(self):
        self.http.get('/healthz')
        native = self.key(*NATIVE)
        MOCKS['openai'].faults.push('status', status=401, path='/videos$', times=3)
        for _ in range(3):
            answer = self.http.post('/native/oai/videos', json={'model': 'sora-2', 'prompt': 'x'},
                                    headers=bearer(native))
            self.assertEqual((answer.status_code, answer.json()['error']['code']), (502, 'upstream_auth_failed'),
                             answer.text)
        self.assertTrue(wait_for(lambda: self.full.broker.credential_checks >= 1))
        self.assertEqual(paths(MOCKS['openai'], '/v1/models'), 1)  # one check for three 401s (single-flight, window)
        self.assert_victim_serves()
        self.assertEqual(self.full.broker.credential_checks, 1)

    def test_a_rejected_key_still_goes_offline_after_the_check(self):
        self.http.get('/healthz')
        token = self.key('inference', 'cloud')
        MOCKS['openai'].faults.push('status', status=401, path='/chat/completions$')
        MOCKS['openai'].faults.push('status', status=401, path='/models$')
        answer = self.chat('gpt-4o-mini@oai', token)
        self.assertEqual((answer.status_code, answer.json()['error']['code']), (502, 'upstream_auth_failed'),
                         answer.text)
        self.assertTrue(wait_for(lambda: self.full.broker.credential_checks >= 1))
        self.assertEqual(self.validation(), 'failed:upstream_auth')
        before = len(MOCKS['openai'].hits)
        again = self.chat('gpt-4o-mini@oai', token)
        self.assertNotEqual(again.status_code, 200, again.text)
        self.assertEqual(len(MOCKS['openai'].hits), before)  # the provider is not contacted with a rejected key
        self.assertEqual(paths(MOCKS['openai'], '/v1/models'), 1)

    def test_a_key_rejection_code_marks_at_once(self):
        self.http.get('/healthz')
        native = self.key(*NATIVE)
        MOCKS['openai'].faults.push('status', status=401, code='invalid_api_key', path='/videos$')
        answer = self.http.post('/native/oai/videos', json={'model': 'sora-2', 'prompt': 'x'}, headers=bearer(native))
        self.assertEqual(answer.status_code, 502, answer.text)
        self.assertEqual(self.validation(), 'failed:upstream_auth')
        self.assertEqual((paths(MOCKS['openai'], '/v1/models'), self.full.broker.credential_checks), (0, 0))


def wait_for(predicate, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


if __name__ == '__main__':
    unittest.main()
