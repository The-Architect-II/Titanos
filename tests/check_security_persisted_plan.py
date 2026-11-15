"""A persisted acquisition plan that the file-name rules now refuse is a typed 422, not a 500.

Actual create_app/AcquisitionStore with records written straight into the store directory; the plan is a
fixture produced by HuggingFaceProvider.plan over a mocked HfApi and then tampered on disk. No network,
no download, no model, no listener, no request to any running service.
"""
from pathlib import Path
from types import SimpleNamespace as NS
import json
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from fastapi.testclient import TestClient  # noqa: E402
from titanos import server  # noqa: E402
from titanos.config import Config  # noqa: E402
from titanos.providers import AcquisitionRequest, HuggingFaceProvider  # noqa: E402

REPO = 'fixture-org/fixture-model'
SHA = 'a' * 40
BENIGN = [('config.json', 100), ('model.safetensors', 200), ('tokenizer/vocab.json', 50)]
IDENTIFIER = 'f' * 32


def fixture_plan():
    info = NS(id=REPO, pipeline_tag='feature-extraction', library_name='transformers', downloads=1, likes=0, gated=False,
              sha=SHA, card_data={'license': 'apache-2.0'}, siblings=[NS(rfilename=n, size=s) for n, s in BENIGN])
    with patch('titanos.providers.HfApi') as api, patch('titanos.providers.get_token', return_value=None):
        api.return_value.model_info.return_value = info
        request = AcquisitionRequest(provider='huggingface', identifier=REPO, revision='main', registry_name='fixture',
                                     task='embeddings', backend='transformers', device='cpu')
        return HuggingFaceProvider().plan(request).model_dump(mode='json')


class PersistedPlan(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='titanos-security-persisted-')
        self.addCleanup(self.tmp.cleanup)
        self.app = server.create_app(Config(), output_dir=Path(self.tmp.name) / 'outputs')
        self.client = TestClient(self.app, base_url='http://127.0.0.1:8000')
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.store = self.app.state.acquisitions
        self.store.root.mkdir(parents=True, exist_ok=True)
        self.plan = fixture_plan()
        self.assertEqual([row['path'] for row in self.plan['files']], ['config.json', 'model.safetensors', 'tokenizer/vocab.json'])

    def write_record(self, status, **extra):
        hostile = json.loads(json.dumps(self.plan))
        hostile['files'][0]['path'] = 'D:config.json'  # a name the validator now refuses (drive-relative)
        record = {'id': IDENTIFIER, 'object': 'model.acquisition', 'status': status, 'plan': hostile, 'error': None,
                  'created_at': time.time(), 'updated_at': time.time(), 'progress': None, 'fingerprint': 'b' * 64, **extra}
        (self.store.root / (IDENTIFIER + '.json')).write_text(json.dumps(record), encoding='utf-8')
        return record

    def stored_status(self):
        return json.loads((self.store.root / (IDENTIFIER + '.json')).read_text(encoding='utf-8'))['status']

    def test_retry_of_a_record_with_a_refused_name_is_a_typed_422(self):
        self.write_record('failed')
        with patch('titanos.providers.snapshot_download') as download:
            reply = self.client.post('/api/acquisitions/' + IDENTIFIER + '/retry')
            self.assertEqual(reply.status_code, 422, reply.text)
            body = reply.json()['error']
            self.assertEqual(body['code'], 'acquisition_plan_invalid')
            self.assertIn("':'", body['message'])
            self.assertNotIn('model.safetensors', body['message'], 'only the offending component is named')
            self.assertEqual(download.mock_calls, [])
        self.assertEqual(self.stored_status(), 'failed', 'the record is left as it was; nothing was queued')
        self.assertEqual(self.client.get('/api/acquisitions/' + IDENTIFIER).status_code, 200)

    def test_activate_of_a_prepared_record_with_a_refused_name_is_a_typed_422(self):
        prepared = self.store.model_root / IDENTIFIER
        prepared.mkdir(parents=True, exist_ok=True)
        (prepared / 'titanos-acquisition.json').write_text(json.dumps({'plan_fingerprint': 'b' * 64}), encoding='utf-8')
        self.write_record('prepared', prepared_path=str(prepared), license='apache-2.0')
        reply = self.client.post('/api/acquisitions/' + IDENTIFIER + '/activate', json={})
        self.assertEqual(reply.status_code, 422, reply.text)
        self.assertEqual(reply.json()['error']['code'], 'acquisition_plan_invalid')
        self.assertEqual(self.stored_status(), 'prepared')
        self.assertEqual(self.app.state.engine.target.__class__.__name__, 'Engine')  # no model was loaded or registered
        self.assertNotIn('fixture', self.client.get('/v1/models').text)

    def test_other_persisted_plan_errors_stay_as_before(self):
        # A missing record is still a 404 and a non-retryable status still a 409: only the validation path changed.
        self.assertEqual(self.client.post('/api/acquisitions/' + 'e' * 32 + '/retry').status_code, 404)
        self.write_record('prepared')
        self.assertEqual(self.client.post('/api/acquisitions/' + IDENTIFIER + '/retry').json()['error']['code'],
                         'acquisition_not_retryable')


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PersistedPlan))
    print(json.dumps({'passed': result.wasSuccessful(), 'tests_run': result.testsRun,
                      'scope': 'Persisted acquisition plans failing the file-name rules answer 422 acquisition_plan_invalid on retry/activate; '
                               'store records written as fixtures, mocked HfApi plan, no network/download/model'}))
    raise SystemExit(0 if result.wasSuccessful() else 1)
