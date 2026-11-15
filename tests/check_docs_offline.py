"""Offline API documentation: /docs renders from the packaged Swagger UI assets (no CDN, no remote favicon, validator badge
disabled), /docs/assets serves only the allowlisted files with pinned types, /redoc is not served, the documentation views are
public GET-only in both key modes, the vendored-asset catalog (Swagger UI and Studio's Chakra Petch fonts with their OFL) is
hash-bound to the repository files, retained licenses and wheel package data, and
build_binary's vendored-asset notice hook copies the licenses and records the SBOM component. Actual create_app and TestClient
over inert fixtures; no listener, model, device, network or browser."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tools'))

from fastapi.testclient import TestClient  # noqa: E402
from titanos import server  # noqa: E402
from titanos.config import Config  # noqa: E402

CATALOG = json.loads((ROOT / 'tools/licenses/vendored_assets.json').read_text(encoding='utf-8'))
EXTERNAL = re.compile(r'https?://', re.IGNORECASE)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class DocsOffline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='titanos-docs-')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def client(self, api_key=None):
        app = server.create_app(Config(), api_key=api_key, output_dir=Path(self.tmp) / ('out-' + ('keyed' if api_key else 'keyless')))
        return TestClient(app, base_url='http://127.0.0.1')

    def test_catalog_is_hash_bound_to_repository_files_and_retained_licenses(self):
        names, fonts = set(), set()
        for row in CATALOG:
            self.assertEqual(row['license_review'], 'not_performed')
            self.assertIn(row.get('kind'), ('vendored web asset', 'vendored font'), row['name'])
            for item in row['files']:
                path = ROOT / item['package_path']
                self.assertTrue(path.is_file(), item['package_path'])
                self.assertEqual(digest(path), item['sha256'], item['package_path'])
                self.assertEqual(path.stat().st_size, item['bytes'], item['package_path'])
                if row['kind'] == 'vendored font':
                    # Studio's display face: inlined into style.css, shipped beside it under web/fonts, never routed
                    self.assertEqual(item['package_path'], 'src/titanos/web/fonts/' + path.name)
                    self.assertEqual(item['packaged_path'], '_internal/titanos/web/fonts/' + path.name)
                    self.assertIsNone(item['route'])
                    fonts.add(path.name)
                    continue
                self.assertEqual(item['packaged_path'], '_internal/titanos/web/' + path.name)
                self.assertEqual(item['route'], '/docs/assets/' + path.name)
                names.add(path.name)
            for item in row['license_files']:
                self.assertEqual(digest(ROOT / 'tools/licenses' / item['file']), item['sha256'], item['file'])
                self.assertTrue(item['target'].startswith('THIRD_PARTY_LICENSES/' + row['name'] + '/'))
        self.assertEqual(names, set(server.DOCS_ASSETS), 'every served docs asset must be catalogued and vice versa')
        self.assertEqual(fonts, {p.name for p in (ROOT / 'src/titanos/web/fonts').iterdir() if p.is_file()},
                         'every file under web/fonts must be catalogued and vice versa')
        # the retained font licence is the shipped OFL, and the wheel's package data carries the fonts and their licence
        font_rows = [row for row in CATALOG if row['kind'] == 'vendored font']
        self.assertTrue(font_rows)
        for row in font_rows:
            self.assertIn(digest(ROOT / 'src/titanos/web/fonts/OFL.txt'), {item['sha256'] for item in row['license_files']})
        import tomllib
        package_data = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))['tool']['setuptools']['package-data']['titanos']
        for name in fonts:
            self.assertTrue(any(Path('web/fonts/' + name).match(pattern) for pattern in package_data), 'pyproject package-data omits web/fonts/' + name)


    def test_docs_page_references_only_packaged_assets(self):
        for api_key in (None, 'fixture-secret'):
            with self.client(api_key) as client:
                page = client.get('/docs')
                self.assertEqual(page.status_code, 200, api_key)
                self.assertTrue(page.headers['content-type'].startswith('text/html'))
                self.assertIsNone(EXTERNAL.search(page.text), 'the documentation shell must not reference any http(s) URL: ' + page.text[:400])
                for asset in server.DOCS_ASSETS:
                    self.assertIn('/docs/assets/' + asset, page.text)
                self.assertIn('/ui/titanos-honeycomb-t-v2.svg', page.text, 'favicon must be the packaged Studio mark')
                self.assertIn('"validatorUrl": null', page.text, 'the Swagger validator badge (external request) must be disabled')
                self.assertNotIn('oauth2RedirectUrl', page.text)
                # ReDoc is deliberately not served: 404 keyless; in keyed mode the bearer check answers 401 first, like any other non-public path
                self.assertEqual(client.get('/redoc').status_code, 401 if api_key else 404, 'ReDoc is deliberately not served')
                self.assertEqual(client.get('/docs/oauth2-redirect').status_code, 401 if api_key else 404)

    def test_assets_are_allowlisted_with_pinned_types_and_bytes(self):
        for api_key in (None, 'fixture-secret'):
            with self.client(api_key) as client:
                for name, media_type in server.DOCS_ASSETS.items():
                    reply = client.get('/docs/assets/' + name)
                    self.assertEqual(reply.status_code, 200, (api_key, name))
                    self.assertTrue(reply.headers['content-type'].startswith(media_type), (name, reply.headers['content-type']))
                    self.assertEqual(hashlib.sha256(reply.content).hexdigest(), digest(ROOT / 'src/titanos/web' / name), name)
                unlisted = client.get('/docs/assets/swagger-ui-standalone-preset.js')
                self.assertEqual(unlisted.status_code, 401 if api_key else 404, 'an unlisted asset name is not public; keyed mode meets the bearer first')
                self.assertEqual(client.get('/docs/assets/..%2Findex.html').status_code, 401 if api_key else 404)
                # names that DO reach the handler (no slash) are refused by the allowlist itself: backslashes, trailing dot, case variant
                for name in ('..%5C..%5Cserver.py', 'swagger-ui.css.', 'SWAGGER-UI.CSS', 'swagger-ui.css%00'):
                    reply = client.get('/docs/assets/' + name)
                    self.assertEqual(reply.status_code, 401 if api_key else 404, name)
                    if not api_key:
                        self.assertIn('not_found', reply.text, name)
                self.assertEqual(client.get('/docs/assets/swagger-ui.css').headers.get('cache-control'), 'no-cache')
                for path in ('/docs', '/'):
                    reply = client.get(path)
                    self.assertEqual(reply.headers.get('x-frame-options'), 'DENY', path)
                    self.assertEqual(reply.headers.get('content-security-policy'), "frame-ancestors 'none'", path)

    def test_documentation_views_are_public_get_only_in_both_modes(self):
        for api_key in (None, 'fixture-secret'):
            with self.client(api_key) as client:
                schema = client.get('/openapi.json')
                self.assertEqual(schema.status_code, 200, api_key)
                self.assertNotIn('/docs', schema.json()['paths'])
                self.assertNotIn('/docs/assets/{name}', schema.json()['paths'])
                for path in server.DOCS_ROUTES:
                    self.assertEqual(client.post(path, content=b'{}', headers={'content-type': 'application/json'}).status_code, 405, path)
                    self.assertEqual(client.request('GET', path, headers={'content-type': 'application/json'}, content=b'{}').status_code, 400, path)
                if api_key:
                    self.assertEqual(client.get('/api/info').status_code, 401, 'control routes stay behind the bearer')
                    self.assertEqual(client.get('/v1/models').status_code, 401)
        self.assertTrue(server.DOCS_ROUTES <= server.PUBLIC_ROUTES)

    def test_build_hook_copies_licenses_and_records_component(self):
        import build_binary
        bundle = Path(self.tmp) / 'bundle'
        (bundle / '_internal/titanos/web').mkdir(parents=True)
        for row in CATALOG:
            for item in row['files']:
                (bundle / item['packaged_path']).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / item['package_path'], bundle / item['packaged_path'])
        components, notes = [], []
        records = build_binary.apply_vendored_web_assets(bundle, components, notes)
        self.assertEqual([r['name'] for r in records], [row['name'] for row in CATALOG])
        for row, record, component in zip(CATALOG, records, components):
            self.assertEqual(component['purl'], row['purl'])
            self.assertEqual(component['licenses'], [{'expression': row['license_expression']}])
            for item in row['license_files']:
                target = bundle / item['target']
                self.assertTrue(target.is_file(), item['target'])
                self.assertEqual(digest(target), item['sha256'])
            self.assertEqual({f['packaged_path'] for f in record['files']}, {f['packaged_path'] for f in row['files']})
        self.assertTrue(notes and 'Swagger UI' in notes[0])
        self.assertTrue(any('Chakra Petch' in note and 'SIL Open Font License 1.1' in note for note in notes))
        # DEPENDENCIES.json records are re-hashed by the one-way payload validation
        import json as _json
        (bundle / 'DEPENDENCIES.json').write_text(_json.dumps({'scope': 'fixture', 'python': '3.12', 'components': [], 'vendored_web_assets': records, 'dependency_exceptions': {}}), encoding='utf-8')
        (bundle / 'sbom.cdx.json').write_text(_json.dumps({'bomFormat': 'CycloneDX', 'specVersion': '1.6', 'components': components}), encoding='utf-8')
        build_binary.validate_packaged_distributions(bundle)
        tampered_license = bundle / CATALOG[0]['license_files'][0]['target']
        original = tampered_license.read_bytes(); tampered_license.write_bytes(b'x')
        with self.assertRaises(ValueError):
            build_binary.validate_packaged_distributions(bundle)
        tampered_license.write_bytes(original)
        # an uncatalogued .js beside the Studio files refuses, with or without the catalog
        (bundle / '_internal/titanos/web/extra-vendor.js').write_bytes(b'// stray')
        with self.assertRaises(ValueError):
            build_binary.apply_vendored_web_assets(bundle, [], [])
        (bundle / '_internal/titanos/web/extra-vendor.js').unlink()
        # so does an uncatalogued file beside the vendored fonts
        (bundle / '_internal/titanos/web/fonts/Other-Regular.woff2').write_bytes(b'wOF2')
        with self.assertRaises(ValueError):
            build_binary.apply_vendored_web_assets(bundle, [], [])
        (bundle / '_internal/titanos/web/fonts/Other-Regular.woff2').unlink()
        # a missing vendored font refuses like any other catalogued asset
        font = next(item for row in CATALOG if row['kind'] == 'vendored font' for item in row['files'])
        held = (bundle / font['packaged_path']).read_bytes(); (bundle / font['packaged_path']).unlink()
        with self.assertRaises(ValueError):
            build_binary.apply_vendored_web_assets(bundle, [], [])
        (bundle / font['packaged_path']).write_bytes(held)
        with self.assertRaises(ValueError):
            build_binary.apply_vendored_web_assets(bundle, [], [], catalog=Path(self.tmp) / 'absent-catalog.json')
        # a shipped asset that differs from its catalog hash must refuse, never silently ship
        first = CATALOG[0]['files'][0]
        (bundle / first['packaged_path']).write_bytes(b'tampered')
        with self.assertRaises(ValueError):
            build_binary.apply_vendored_web_assets(bundle, [], [])
        # a missing shipped asset refuses too
        (bundle / first['packaged_path']).unlink()
        with self.assertRaises(ValueError):
            build_binary.apply_vendored_web_assets(bundle, [], [])


if __name__ == '__main__':
    result = unittest.main(exit=False, verbosity=1).result
    print(json.dumps({'passed': result.wasSuccessful(), 'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
                      'scope': 'create_app/TestClient over inert fixtures; no listener, model, device, network or browser'}))
    sys.exit(0 if result.wasSuccessful() else 1)
