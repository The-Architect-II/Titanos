"""Real client/HTTP boundaries with no models or remote workloads."""
from contextlib import ExitStack
import threading
import unittest

import httpx
from titanos.client import Client, TitanosError
from titanos.pool import Endpoint, Pool


class Tests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        self.calls = []
        def handle(request):
            self.calls.append((request.url.host, request.headers.get('authorization'), request.url.path))
            if request.url.path == '/fail':
                return httpx.Response(503, json={'error': {'message': 'resident worker failed'}})
            if request.url.path == '/redirect':
                return httpx.Response(307, headers={'Location': 'http://foreign.example/'})
            return httpx.Response(200, json={'ok': True})
        self.a = self.stack.enter_context(Client('http://a.example', api_key='test-a', transport=httpx.MockTransport(handle)))
        self.b = self.stack.enter_context(Client('http://b.example', api_key='test-b', transport=httpx.MockTransport(handle)))
        self.pool = Pool({'a': Endpoint(self.a, {'detect': 'yolos-a', 'image': 'sdxl'}),
                          'b': Endpoint(self.b, {'detect': 'yolos-b'})})

    def test_constructor_and_status_are_network_free_and_bindings_immutable(self):
        models = {'same': 'original'}; endpoint = Endpoint(self.a, models)
        models['same'] = 'changed'
        self.assertEqual(endpoint.models['same'], 'original')
        with self.assertRaises(TypeError): endpoint.models['same'] = 'changed'
        statuses = self.pool.status(); statuses[0]['models']['detect'] = 'changed'
        self.assertEqual(self.pool.status()[0]['models']['detect'], 'yolos-a')
        self.assertEqual(self.calls, [])

    def test_auto_balances_active_and_rotates_idle_ties(self):
        with self.pool.lease('detect') as first:
            with self.pool.lease('detect') as second:
                self.assertEqual((first.host, second.host), ('a', 'b'))
                with self.pool.lease('detect') as third:
                    self.assertEqual(third.host, 'a')
                self.assertEqual([r['active_leases'] for r in self.pool.status()], [1, 1])
        with self.pool.lease('detect') as next_one:
            self.assertEqual(next_one.host, 'b')

    def test_pin_and_missing_capability_do_not_fallback(self):
        with self.pool.lease('detect', host='b') as selected:
            self.assertIs(selected.client, self.b); self.assertEqual(selected.model, 'yolos-b')
        for args in [('image', 'b'), ('detect', 'missing')]:
            with self.assertRaises(ValueError), self.pool.lease(args[0], host=args[1]): pass
        self.assertEqual(self.calls, [])

    def test_error_and_redirect_are_one_send_without_retry_and_release(self):
        for path in ['/fail', '/redirect']:
            with self.assertRaises(TitanosError), self.pool.lease('detect', host='a') as selected:
                selected.client.request('POST', path, json={'model': selected.model})
        self.assertEqual(self.calls, [('a.example', 'Bearer test-a', '/fail'), ('a.example', 'Bearer test-a', '/redirect')])
        self.assertEqual([r['active_leases'] for r in self.pool.status()], [0, 0])
        with self.pool.lease('detect', host='b') as selected:
            selected.client.request('GET', '/ok')
        self.assertEqual(self.calls[-1], ('b.example', 'Bearer test-b', '/ok'))

    def test_disabled_host_keeps_active_lease_but_excludes_future_selection(self):
        with self.pool.lease('detect', host='a') as active:
            self.pool.set_enabled('a', False)
            with self.pool.lease('detect') as next_one: self.assertEqual(next_one.host, 'b')
            self.assertIs(active.client, self.a)
            with self.assertRaises(ValueError), self.pool.lease('detect', host='a'): pass
        self.pool.set_enabled('b', False)
        with self.assertRaises(ValueError), self.pool.lease('detect'): pass
        self.pool.set_enabled('a', True)
        with self.pool.lease('detect') as next_one: self.assertEqual(next_one.host, 'a')

    def test_concurrent_admission_is_atomic(self):
        barrier = threading.Barrier(9); release = threading.Event(); failures = []
        def call():
            try:
                with self.pool.lease('detect'):
                    barrier.wait(5); self.assertTrue(release.wait(5))
            except BaseException as exc: failures.append(exc)
        workers = [threading.Thread(target=call) for _ in range(8)]
        for worker in workers: worker.start()
        try:
            barrier.wait(5)
            self.assertEqual([r['active_leases'] for r in self.pool.status()], [4, 4])
        finally:
            release.set()
            for worker in workers: worker.join(6)
        self.assertFalse(failures); self.assertFalse(any(w.is_alive() for w in workers))
        self.assertEqual([r['active_leases'] for r in self.pool.status()], [0, 0])

    def test_origin_and_model_retained_for_followups(self):
        with self.pool.lease('detect', host='b') as saved: pass
        self.pool.set_enabled('b', False)
        self.assertIs(saved.client, self.b); self.assertEqual(saved.model, 'yolos-b')
        with self.assertRaises(ValueError):
            saved.client.save_asset({'url': 'http://a.example/v1/files/fake.png'}, 'never-written.png')
        self.assertEqual(self.calls, [])


if __name__ == '__main__': unittest.main()
