"""WP7 asset jobs and artifacts with broker owner binding, over the integrated app and the WP4 loopback mocks (netguard
installed; zero non-loopback attempts).

- cloud speech (OpenAI and ElevenLabs mocks) and image (OpenAI) jobs run CloudInvoker on the job thread and produce .wav
  and .png /v1/files artifacts;
- a cloud job needs an Idempotency-Key (400 idempotency_key_required, before any hit); Idempotency-Key matches only
  within one owner, so the same key from two keys yields two jobs;
- owner scoping of /api/jobs (list, get, cancel), /api/video/jobs, the /v1/videos routes and /v1/files: a non-owner key
  gets 404, bootstrap sees everything; job artifacts are bound to the job owner;
- the budget is reserved at submit: a request over the key's budget is refused 402 at submit with zero hits;
- a key's model constraints apply to local jobs, video and the catalogs (404 model_not_found with zero engine calls,
  hidden from /v1/models and /v1/voices), and its local rate limits admit local jobs; the concurrency slot is released
  when a job settles and when an Idempotency-Key replays an existing job.
"""
from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import broker_wp7_support as W  # noqa: E402  (installs the netguard)
from broker_mocks import ElevenLabsMock, OpenAIMock, netguard  # noqa: E402

BOOT = 'wp7-jobs-bootstrap-0001'
MOCKS: dict = {}
LOCAL = {'voice': 'speech', 'vid-local': ('video', 'diffusers')}


def setUpModule():
    MOCKS['openai'] = OpenAIMock()
    MOCKS['elevenlabs'] = ElevenLabsMock()
    for item in MOCKS.values():
        item.start()


def tearDownModule():
    for item in MOCKS.values():
        item.stop()


def document() -> dict:
    return {'policy': {'cloud_policy': 'explicit'},
            'upstreams': {
                'openai': W.S.upstream(MOCKS['openai'], 'openai', credential='oa', models=['tts-1', 'dall-e-3']),
                'xi': W.S.upstream(MOCKS['elevenlabs'], 'elevenlabs', credential='xi',
                                   models=['eleven_multilingual_v2']),
            },
            'prices': [{'match': 'tts-1@openai', 'per_1k_chars': '1.00'},
                       {'match': 'dall-e-3@openai', 'per_image': '0.04'},
                       {'match': 'eleven_multilingual_v2@xi', 'per_1k_chars': '1.00'}]}


class Case(unittest.TestCase):
    def setUp(self):
        netguard.reset_attempts()
        for item in MOCKS.values():
            item.reset()
        self.full = W.FullApp(document(), models=LOCAL, api_key=BOOT)
        self.c = self.full.http
        self.a_id, self.a = self.full.mint('jobs', 'inference', 'cloud', name='a')
        self.b_id, self.b = self.full.mint('jobs', 'inference', 'cloud', name='b')

    def tearDown(self):
        self.full.close()
        self.assertEqual(netguard.attempts(), [], 'a non-loopback connect was attempted')

    def hits(self) -> int:
        return W.S.total_hits(*MOCKS.values())

    def submit(self, body, token, key='idem-0001'):
        headers = W.bearer(token)
        if key is not None:
            headers['Idempotency-Key'] = key
        return self.c.post('/api/jobs', json=body, headers=headers)

    def wait(self, job_id, token) -> dict:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            record = self.c.get(f'/api/jobs/{job_id}', headers=W.bearer(token)).json()
            if record['status'] in ('completed', 'failed', 'cancelled', 'interrupted'):
                return record
            time.sleep(0.05)
        self.fail('job did not finish')


class CloudJobs(Case):
    def test_cloud_speech_and_image_jobs_produce_owner_bound_artifacts(self):
        for model, task, suffix in (('tts-1@openai', 'speech', '.wav'), ('eleven_multilingual_v2@xi', 'speech', '.wav'),
                                    ('dall-e-3@openai', 'image', '.png')):
            with self.subTest(model=model):
                body = {'task': task, 'model': model, **({'input': 'Hello there.', 'voice': 'alloy'}
                                                         if task == 'speech' else {'prompt': 'a cat', 'size': '256x256'})}
                accepted = self.submit(body, self.a, key='idem-' + model)
                self.assertEqual(accepted.status_code, 202, accepted.text[:300])
                job = accepted.json()
                self.assertNotIn('owner_key_id', job)
                self.assertTrue(job.get('decision_id'))
                record = self.wait(job['id'], self.a)
                self.assertEqual(record['status'], 'completed', record)
                url = record['result']['url'] if task == 'speech' else record['result']['data'][0]['url']
                name = url.rsplit('/', 1)[-1]
                self.assertTrue(name.endswith(suffix))
                self.assertEqual(self.full.broker.ownership.owner('artifact', name), self.a_id)
                own = self.c.get('/v1/files/' + name, headers=W.bearer(self.a))
                self.assertEqual(own.status_code, 200)
                self.assertTrue(own.content.startswith(b'RIFF' if task == 'speech' else b'\x89PNG'))
                self.assertEqual(self.c.get('/v1/files/' + name, headers=W.bearer(self.b)).status_code, 404)
                self.assertEqual(self.c.get('/v1/files/' + name, headers=W.bearer(BOOT)).status_code, 200)
        self.assertTrue(MOCKS['openai'].hits and MOCKS['elevenlabs'].hits)

    def test_idempotency_key_required_and_per_owner(self):
        body = {'task': 'speech', 'model': 'tts-1@openai', 'input': 'Hi.', 'voice': 'alloy'}
        missing = self.submit(body, self.a, key=None)
        self.assertEqual((missing.status_code, missing.json()['error']['code']), (400, 'idempotency_key_required'))
        self.assertEqual(self.hits(), 0)
        first = self.submit(body, self.a, key='shared-key')
        again = self.submit(body, self.a, key='shared-key')
        other = self.submit(body, self.b, key='shared-key')
        self.assertEqual(first.json()['id'], again.json()['id'])
        self.assertNotEqual(first.json()['id'], other.json()['id'])
        for job, token in ((first.json(), self.a), (other.json(), self.b)):
            self.assertEqual(self.wait(job['id'], token)['status'], 'completed')
        local = {'task': 'speech', 'model': 'voice', 'input': 'Hi.'}
        la = self.submit(local, self.a, key='local-shared')
        lb = self.submit(local, self.b, key='local-shared')
        self.assertNotEqual(la.json()['id'], lb.json()['id'])

    def test_budget_is_reserved_at_submit(self):
        _, poor = self.full.mint('jobs', 'inference', 'cloud', name='poor', budget={'max_request_usd': '0.0001'},
                                 expires_at='2030-01-01T00:00:00Z')
        body = {'task': 'speech', 'model': 'tts-1@openai', 'input': 'A sentence long enough to cost money.',
                'voice': 'alloy'}
        refused = self.submit(body, poor, key='poor-1')
        self.assertEqual((refused.status_code, refused.json()['error']['code']), (402, 'budget_exceeded'))
        self.assertEqual(self.hits(), 0)
        self.assertEqual(self.c.get('/api/jobs', headers=W.bearer(poor)).json()['data'], [])


class OwnerScoping(Case):
    def test_jobs_videos_and_cancel_are_owner_scoped(self):
        local = self.submit({'task': 'speech', 'model': 'voice', 'input': 'Hi.'}, self.a, key=None)
        self.assertEqual(local.status_code, 202, local.text[:300])
        job = local.json()
        record = self.wait(job['id'], self.a)
        self.assertEqual(record['status'], 'completed')
        name = record['result']['url'].rsplit('/', 1)[-1]
        self.assertEqual(self.full.broker.ownership.owner('artifact', name), self.a_id)  # bound from the job owner
        self.assertEqual(self.c.get(f'/api/jobs/{job["id"]}', headers=W.bearer(self.b)).status_code, 404)
        self.assertEqual(self.c.delete(f'/api/jobs/{job["id"]}', headers=W.bearer(self.b)).status_code, 404)
        self.assertEqual(self.c.get(f'/api/jobs/{job["id"]}', headers=W.bearer(BOOT)).status_code, 200)
        ids_a = {row['id'] for row in self.c.get('/api/jobs', headers=W.bearer(self.a)).json()['data']}
        ids_b = {row['id'] for row in self.c.get('/api/jobs', headers=W.bearer(self.b)).json()['data']}
        ids_boot = {row['id'] for row in self.c.get('/api/jobs', headers=W.bearer(BOOT)).json()['data']}
        self.assertIn(job['id'], ids_a)
        self.assertNotIn(job['id'], ids_b)
        self.assertIn(job['id'], ids_boot)
        self.assertEqual(self.c.get('/v1/files/' + name, headers=W.bearer(self.b)).status_code, 404)
        video = self.c.post('/api/video/jobs', json={'model': 'vid-local', 'prompt': 'waves', 'size': '64x64',
                                                     'frames': 2}, headers=W.bearer(self.a))
        self.assertEqual(video.status_code, 202, video.text[:300])
        vid = video.json()['id']
        self.wait(vid, self.a)
        for path in (f'/v1/videos/{vid}', f'/api/video/jobs/{vid}'):
            self.assertEqual(self.c.get(path, headers=W.bearer(self.a)).status_code, 200, path)
            self.assertEqual(self.c.get(path, headers=W.bearer(self.b)).status_code, 404, path)
        self.assertEqual(self.c.get(f'/v1/videos/{vid}/content', headers=W.bearer(self.b)).status_code, 404)
        self.assertNotIn(vid, {r['id'] for r in self.c.get('/v1/videos', headers=W.bearer(self.b)).json()['data']})
        self.assertIn(vid, {r['id'] for r in self.c.get('/v1/videos', headers=W.bearer(self.a)).json()['data']})
        self.assertNotIn(vid, {r['id'] for r in self.c.get('/api/video/jobs', headers=W.bearer(self.b)).json()['data']})
        self.assertEqual(self.c.delete(f'/v1/videos/{vid}', headers=W.bearer(self.b)).status_code, 404)
        self.assertEqual(self.c.delete(f'/api/video/jobs/{vid}', headers=W.bearer(self.b)).status_code, 404)
        self.assertEqual(self.c.delete(f'/v1/videos/{vid}', headers=W.bearer(self.a)).status_code, 200)


class LocalJobConstraints(Case):
    def test_model_constraints_apply_to_local_jobs_video_and_catalogs(self):
        _, narrow = self.full.mint('jobs', 'inference', 'models:read', name='narrow',
                                   constraints={'models': ['voice']})
        before = len(self.full.engine.calls)
        refused = [
            self.c.post('/api/jobs', json={'task': 'video', 'model': 'vid-local', 'prompt': 'waves', 'size': '64x64',
                                           'frames': 2}, headers=W.bearer(narrow)),
            self.c.post('/api/video/jobs', json={'model': 'vid-local', 'prompt': 'waves', 'size': '64x64', 'frames': 2},
                        headers=W.bearer(narrow)),
            self.c.post('/v1/videos', json={'model': 'vid-local', 'prompt': 'waves', 'size': '64x64'},
                        headers=W.bearer(narrow)),
            self.c.post('/v1/videos/generations', json={'model': 'vid-local', 'prompt': 'waves', 'size': '64x64',
                                                        'frames': 2}, headers=W.bearer(narrow)),
            self.c.get('/v1/models/vid-local', headers=W.bearer(narrow)),
        ]
        for reply in refused:
            with self.subTest(path=str(reply.request.url)):
                self.assertEqual((reply.status_code, reply.json()['error']['code']), (404, 'model_not_found'),
                                 reply.text[:300])
        time.sleep(0.2)
        self.assertEqual(len(self.full.engine.calls), before)  # nothing ran
        allowed = self.submit({'task': 'speech', 'model': 'voice', 'input': 'Hi.'}, narrow, key=None)
        self.assertEqual(allowed.status_code, 202, allowed.text[:300])
        self.assertEqual(self.wait(allowed.json()['id'], narrow)['status'], 'completed')
        ids = {row['id'] for row in self.c.get('/v1/models', headers=W.bearer(narrow)).json()['data']}
        self.assertIn('voice', ids)
        self.assertNotIn('vid-local', ids)
        self.assertEqual(self.c.get('/v1/models/voice', headers=W.bearer(narrow)).status_code, 200)
        everything = {row['id'] for row in self.c.get('/v1/models', headers=W.bearer(BOOT)).json()['data']}
        self.assertTrue({'voice', 'vid-local'} <= everything)  # bootstrap sees every local model
        _, video_only = self.full.mint('jobs', 'inference', 'models:read', name='video-only',
                                       constraints={'models': ['vid-*']})
        self.assertEqual(self.c.get('/v1/voices', headers=W.bearer(video_only)).json()['voices'], [])
        self.assertTrue(self.c.get('/v1/voices', headers=W.bearer(narrow)).json()['voices'])
        speech = self.submit({'task': 'speech', 'model': 'voice', 'input': 'Hi.'}, video_only, key=None)
        self.assertEqual((speech.status_code, speech.json()['error']['code']), (404, 'model_not_found'))

    def test_local_rate_limits_admit_local_jobs(self):
        _, slow = self.full.mint('jobs', 'inference', name='slow', limits={'rpm': 1})
        first = self.submit({'task': 'speech', 'model': 'voice', 'input': 'Hi.'}, slow, key=None)
        self.assertEqual(first.status_code, 202, first.text[:300])
        second = self.submit({'task': 'speech', 'model': 'voice', 'input': 'Hi.'}, slow, key=None)
        self.assertEqual((second.status_code, second.json()['error']['code']), (429, 'rate_limit_exceeded'))
        self.assertIn('retry-after', second.headers)

    def test_concurrency_slot_is_released_when_a_job_settles_or_replays(self):
        _, single = self.full.mint('jobs', 'inference', name='single', limits={'concurrent': 1})
        body = {'task': 'speech', 'model': 'voice', 'input': 'Hi.'}
        first = self.submit(body, single, key='idem-local-1')
        self.assertEqual(first.status_code, 202, first.text[:300])
        self.wait(first.json()['id'], single)
        time.sleep(0.1)  # the slot is released when the job thread returns
        replay = self.submit(body, single, key='idem-local-1')
        self.assertEqual((replay.status_code, replay.json()['id']), (202, first.json()['id']))
        third = self.submit(body, single, key='idem-local-2')
        self.assertEqual(third.status_code, 202, third.text[:300])  # neither the job nor the replay kept the slot
        self.wait(third.json()['id'], single)


if __name__ == '__main__':
    unittest.main(verbosity=2)
