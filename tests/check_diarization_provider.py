"""Pinned provider lifecycle checks with byte fixtures, no models/network."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import httpx
from titanos.providers import AcquisitionRequest, GitHubSherpaDiarizationProvider, ProviderRegistry


class Checks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.catalog = copy.deepcopy(GitHubSherpaDiarizationProvider().catalog)
        self.profile = self.catalog['profiles'][0]
        self.payloads = {}
        for index, row in enumerate(self.profile['downloads']):
            raw = b's' * 300000 if index == 0 else ('fixture-' + str(index)).encode()
            self.payloads[row['url']] = raw
            row.update(size=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        for role, row in zip(['segmentation', 'embedding'], self.profile['downloads']):
            self.profile['components'][role].update(bytes=row['size'], sha256=row['sha256'])
        self.provider = GitHubSherpaDiarizationProvider(self.catalog)
        self.patches = [patch('titanos.providers.find_spec', return_value=object()),
                        patch('titanos.providers.package_metadata.version', return_value='1.13.8')]
        for p in self.patches: p.start()
        self.request = AcquisitionRequest(provider=self.provider.name, identifier=self.profile['id'],
            registry_name='fixture-diarization', task='diarization', backend='sherpa-onnx', device='auto')
        self.plan = self.provider.plan(self.request)
        self.requests = []

    def tearDown(self):
        for p in reversed(self.patches): p.stop()
        self.temp.cleanup()

    def transport(self, request):
        self.requests.append((str(request.url), request.headers.get('range')))
        body = self.payloads[str(request.url)]
        offset = int(request.headers.get('range', 'bytes=0-')[6:-1])
        if offset:
            return httpx.Response(206, content=body[offset:], headers={'Content-Range':f'bytes {offset}-{len(body)-1}/{len(body)}'})
        return httpx.Response(200, content=body)

    def client(self, handler=None):
        original = httpx.Client
        return patch('titanos.providers.httpx.Client', side_effect=lambda **kw:
                     original(transport=httpx.MockTransport(handler or self.transport), **kw))

    def test_plan_and_manifest_and_no_network_cache_reuse(self):
        self.assertIn(self.provider.name, ProviderRegistry().names())
        self.assertEqual(self.plan.device, 'cpu')
        self.assertEqual(self.plan.runtime['status'], 'available')
        self.assertEqual(self.provider.search('', 'speech'), [])
        with self.client(): result = self.provider.acquire(self.plan, self.root, lambda *a:None, threading.Event())
        manifest = json.loads((Path(result['path'])/'diarization.json').read_text())
        self.assertEqual(manifest['components'], self.profile['components'])
        self.assertEqual(result['license'], 'MIT AND Apache-2.0')
        self.assertEqual(len(result['license_evidence']), 4)
        count = len(self.requests)
        with self.client(): reused = self.provider.acquire(self.plan, self.root, lambda *a:None, threading.Event())
        self.assertTrue(reused['reused']);self.assertEqual(len(self.requests), count)

    def test_explicit_scope_and_revision_do_not_fall_back(self):
        for change in [{'device':'cuda:0'}, {'dtype':'float32'}, {'task':'speech'}, {'graph':'other.onnx'}]:
            with self.assertRaises(ValueError): AcquisitionRequest.model_validate({**self.request.model_dump(), **change})
        with self.assertRaises(ValueError): self.provider.plan(self.request.model_copy(update={'revision':'different'}))
        with self.assertRaisesRegex(ValueError, 'another provider'):
            self.provider.plan(self.request.model_copy(update={'provider':'controlled-other'}))

    def test_altered_agent_plan_rejected_before_download(self):
        changed = self.plan.model_copy(deep=True)
        changed.files[0].url = 'https://127.0.0.1/private'
        with self.client(), self.assertRaisesRegex(ValueError, 'trusted profile'):
            self.provider.acquire(changed, self.root, lambda *a:None, threading.Event())
        self.assertEqual(self.requests, [])

    def test_cancel_then_range_resume(self):
        cancel = threading.Event()
        def progress(stage, count, total):
            if stage == 'downloading' and count >= 256*1024: cancel.set()
        with self.client(), self.assertRaisesRegex(RuntimeError, 'cancelled'):
            self.provider.acquire(self.plan, self.root, progress, cancel)
        partial = self.root/'bundle/segmentation.onnx'
        self.assertEqual(partial.stat().st_size, 256*1024)
        with self.client(): result = self.provider.acquire(self.plan, self.root, lambda *a:None, threading.Event())
        self.assertTrue(any(header == 'bytes=262144-' for _, header in self.requests))
        self.assertEqual(partial.read_bytes(), self.payloads[self.plan.files[0].url])
        self.assertTrue((Path(result['path'])/'diarization.json').is_file())

    def test_bad_hash_cannot_publish_manifest(self):
        def wrong(request): return httpx.Response(200, content=b'x'*300000)
        with self.client(wrong), self.assertRaisesRegex(ValueError, 'SHA-256'):
            self.provider.acquire(self.plan, self.root, lambda *a:None, threading.Event())
        self.assertFalse((self.root/'bundle/diarization.json').exists())
        self.assertFalse((self.root/'bundle/segmentation.onnx').exists())

    def test_redirect_host_and_resume_offset_are_checked(self):
        calls = []
        def redirect(request):
            calls.append(str(request.url));return httpx.Response(302, headers={'Location':'https://127.0.0.1/private'})
        with self.client(redirect), self.assertRaisesRegex(ValueError, 'approved public HTTPS'):
            self.provider.acquire(self.plan, self.root, lambda *a:None, threading.Event())
        self.assertEqual(len(calls), 1)
        target = self.root/'bundle/segmentation.onnx';target.write_bytes(b's'*3)
        def wrong_range(request):return httpx.Response(206, content=b's', headers={'Content-Range':'bytes 0-0/300000'})
        with self.client(wrong_range), self.assertRaisesRegex(ValueError, 'resume range'):
            self.provider.acquire(self.plan, self.root, lambda *a:None, threading.Event())
        self.assertEqual(target.read_bytes(), b's'*3)


if __name__ == '__main__': unittest.main()
