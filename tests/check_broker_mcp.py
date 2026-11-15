"""WP8 MCP principal propagation and broker tools (mcp_server.create_mcp, broker/mcp_tools.py).

Over a real streamable-HTTP /mcp (uvicorn in a thread, the official MCP 2.2 client):
- a key holding mcp + inference that calls tools reaching admin and broker routes is refused by each inner route's
  scope (reload_registry, broker_status, test_upstream, list_upstreams), and a static resource (titanos://jobs) runs as
  the same key (no jobs scope: refused);
- the bootstrap key is never used for mounted internal dispatch: an instrumented Authenticator sees every in-process
  (titanos.internal) request carry the outer key's principal and no Authorization header;
- two keys on one MCP session id each get their own principal (whoami answers each key's id in turn);
- inference results carry _meta provenance (served_by, served_model, request_id, decision_id, fallback_count) when the
  server stamped provenance; a keyless, unconfigured server's results carry none (unchanged; the SDK's own serverInfo
  _meta entry is the only one).
In process:
- a mounted call whose outer request carries no authenticated principal is refused with ToolError;
- the tool list is exactly 72 names (60 existing + 12 broker tools), no tool's input schema has a secret-valued field,
  and no tool sets credentials, mints keys or changes policy;
- open_world_hint of the inference tools follows the startup rule: False with no upstream in broker.json, True with one,
  False in managed role without delegation; the instructions carry the cloud sentence.
Every test installs the netguard and asserts zero non-loopback connection attempts. Synthetic keys only.
"""
from __future__ import annotations

import asyncio
import re
import socket
import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import broker_wp7_support as W  # noqa: E402  (installs the netguard)
from broker_mocks import OpenAIMock, netguard  # noqa: E402

BOOT = 'wp8-mcp-bootstrap-0001'
CHAT = [{'role': 'user', 'content': 'Say hello.'}]
BROKER_TOOLS = {'broker_status', 'broker_catalog', 'broker_explain', 'broker_decisions', 'broker_usage', 'list_upstreams',
                'upstream_health', 'test_upstream', 'messages', 'count_tokens', 'moderate', 'whoami'}
EXISTING_TOOLS = {
    'server_info', 'list_models', 'model_capabilities', 'list_model_providers', 'discover_models', 'inspect_model_source',
    'import_model_page', 'plan_model_acquisition', 'submit_model_acquisition', 'list_model_acquisitions',
    'get_model_acquisition', 'cancel_model_acquisition', 'retry_model_acquisition', 'activate_model_acquisition',
    'model_details', 'load_model', 'unload_model', 'register_model', 'reload_registry', 'unregister_model', 'list_devices',
    'list_workers', 'chat', 'complete', 'responses', 'embed', 'rerank', 'retrieval_answer', 'classify_pairs', 'speak',
    'transcribe', 'diarize', 'analyze_audio', 'vad', 'translate', 'translate_audio', 'generate_image', 'edit_image',
    'generate_video', 'infer_tensor', 'sentiment', 'compose_sentiment', 'estimate_depth', 'segment_image',
    'extension_infer', 'submit_job', 'list_jobs', 'get_job', 'cancel_job', 'read_artifact', 'list_runtime_profiles',
    'list_runtime_setups', 'plan_runtime_setup', 'prepare_runtime', 'get_runtime_setup', 'cancel_runtime_setup',
    'retry_runtime_setup', 'list_assignment_devices', 'get_model_assignment', 'save_model_assignment'}
INFERENCE_TOOLS = {'chat', 'complete', 'responses', 'embed', 'speak', 'transcribe', 'generate_image', 'messages',
                   'count_tokens'}
SECRET_FIELD = re.compile(r'(^|_)(secret|password|passphrase|api_?key|apikey|access_token|bearer|credential_value)($|_)', re.I)


def free_port() -> int:
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    try:
        return sock.getsockname()[1]
    finally:
        sock.close()


def tools(url: str, calls, tokens):
    """Run calls [(name, args, token_index)] over ONE MCP session; the session's HTTP client switches its bearer
    between calls, so each POST message carries the key named for it."""
    import httpx2
    from mcp import Client
    from mcp.client.streamable_http import streamable_http_client

    async def run():
        out = []
        async with httpx2.AsyncClient(headers={'Authorization': 'Bearer ' + tokens[0]}, trust_env=False) as http:
            async with Client(streamable_http_client(url + '/mcp/', http_client=http), read_timeout_seconds=30) as client:
                for name, args, index in calls:
                    http.headers['Authorization'] = 'Bearer ' + tokens[index]
                    if name.startswith('titanos://'):
                        try:
                            out.append(await client.read_resource(name))
                        except Exception as exc:  # a refused resource read is an MCP error
                            out.append(exc)
                    else:
                        out.append(await client.call_tool(name, args))
        return out
    return asyncio.run(run())


class Served(unittest.TestCase):
    """uvicorn in a thread over a FullApp."""

    def serve(self, document=None, api_key=BOOT):
        import uvicorn
        full = W.FullApp(document, models={'chat-local': 'chat'}, api_key=api_key)
        self.addCleanup(full.close)
        port = free_port()
        server = uvicorn.Server(uvicorn.Config(full.app, host='127.0.0.1', port=port, log_level='warning', lifespan='on'))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()

        def stop():
            server.should_exit = True
            thread.join(15)
        self.addCleanup(stop)
        deadline = time.monotonic() + 30
        while not server.started:
            if time.monotonic() > deadline or not thread.is_alive():
                self.fail('uvicorn did not start')
            time.sleep(0.05)
        return full, f'http://127.0.0.1:{port}'

    def setUp(self):
        netguard.reset_attempts()

    def tearDown(self):
        self.assertEqual(netguard.attempts(), [], 'a non-loopback connect was attempted')

    def test_scoped_key_is_refused_on_admin_tools_and_never_elevated(self):
        full, url = self.serve()
        _, chatter = full.mint('mcp', 'inference', name='chatter')
        seen = []
        auth = full.broker.authenticator
        original = auth.authenticate

        def instrumented(scope, headers=None):
            if scope.get('titanos.internal'):
                raw = dict(scope.get('headers') or [])
                preset = scope.get('titanos.principal')
                seen.append((b'authorization' in raw, getattr(preset, 'id', None), getattr(preset, 'kind', None)))
            return original(scope, headers)
        with mock.patch.object(auth, 'authenticate', instrumented):
            out = tools(url, [('reload_registry', {}, 0), ('broker_status', {}, 0), ('list_upstreams', {}, 0),
                              ('test_upstream', {'id': 'none'}, 0), ('chat', {'payload': {'model': 'chat-local',
                                                                                         'messages': CHAT}}, 0),
                              ('titanos://jobs', {}, 0)], [chatter])
        for refused in out[:4]:
            self.assertTrue(refused.is_error, str(refused.content)[:300])
            self.assertIn('scope', str(refused.content).lower())
        self.assertFalse(out[4].is_error, str(out[4].content)[:300])
        self.assertIsInstance(out[5], Exception)  # titanos://jobs needs the jobs scope the key lacks
        self.assertTrue(seen)
        key_id = chatter.split('_')[1]
        self.assertEqual({entry for entry in seen}, {(False, key_id, 'key')}, seen)
        self.assertEqual(len(full.engine.calls), 1)

    def test_two_keys_on_one_session_keep_their_own_principal(self):
        full, url = self.serve()
        a_id, a_token = full.mint('mcp', name='a')
        b_id, b_token = full.mint('mcp', 'broker:read', name='b')
        out = tools(url, [('whoami', {}, 0), ('whoami', {}, 1), ('broker_status', {}, 0), ('broker_status', {}, 1),
                          ('whoami', {}, 0)], [a_token, b_token])
        self.assertEqual([out[0].structured_content['id'], out[1].structured_content['id'],
                          out[4].structured_content['id']], [a_id, b_id, a_id])
        self.assertTrue(out[2].is_error)
        self.assertFalse(out[3].is_error, str(out[3].content)[:300])

    def test_inference_results_carry_meta_provenance(self):
        openai = OpenAIMock()
        openai.start()
        self.addCleanup(openai.stop)
        document = {'policy': {'cloud_policy': 'explicit'},
                    'upstreams': {'openai': W.S.upstream(openai, 'openai', credential='oa', models=['gpt-4o-mini'])},
                    'models': {'mini': {'target': 'gpt-4o-mini@openai'}}}
        full, url = self.serve(document)
        _, token = full.mint('mcp', 'inference', 'cloud', name='cloud-user')
        cloud, local = tools(url, [('chat', {'payload': {'model': 'mini', 'messages': CHAT}}, 0),
                                   ('chat', {'payload': {'model': 'chat-local', 'messages': CHAT}}, 0)], [token])
        self.assertFalse(cloud.is_error, str(cloud.content)[:300])
        self.assertEqual(cloud.meta['served_by'], 'cloud:openai')
        self.assertEqual(cloud.meta['served_model'], 'gpt-4o-mini@openai')
        self.assertEqual(cloud.meta['fallback_count'], 0)
        self.assertTrue(cloud.meta['request_id'] and cloud.meta['decision_id'])
        self.assertEqual(cloud.structured_content['object'], 'chat.completion')
        self.assertEqual(local.meta['served_by'], 'local:engine')

    def test_keyless_unconfigured_results_are_unchanged(self):
        full, url = self.serve(api_key=None)
        (result,) = tools(url, [('chat', {'payload': {'model': 'chat-local', 'messages': CHAT}}, 0)], ['local'])
        self.assertFalse(result.is_error, str(result.content)[:300])
        self.assertFalse({'served_by', 'served_model', 'decision_id'} & set(result.meta or {}), result.meta)


class InProcess(unittest.TestCase):
    def setUp(self):
        netguard.reset_attempts()

    def tearDown(self):
        self.assertEqual(netguard.attempts(), [], 'a non-loopback connect was attempted')

    def app(self, document=None, **settings_kw):
        full = W.FullApp(document, models={'chat-local': 'chat'}, api_key=BOOT, settings_kw=settings_kw)
        self.addCleanup(full.close)
        return full

    def test_mounted_call_without_principal_is_refused(self):
        from mcp.server.mcpserver.context import Context
        from mcp.server.mcpserver.exceptions import ToolError
        full = self.app()
        outer = NS(request=NS(scope={'type': 'http', 'path': '/mcp/'}, url='http://127.0.0.1:8000/mcp/'))
        full.http  # lifespan
        with mock.patch.object(Context, 'request_context', new_callable=mock.PropertyMock, return_value=outer):
            with self.assertRaises(ToolError) as caught:
                asyncio.run(full.app.state.mcp.call_tool('server_info', {}))
        self.assertIn('no authenticated outer request', str(caught.exception))
        self.assertEqual(full.engine.calls, [])

    def test_tool_list_and_schemas(self):
        full = self.app()
        listed = asyncio.run(full.app.state.mcp.list_tools())
        names = {tool.name for tool in listed}
        self.assertEqual(len(listed), 72)
        self.assertEqual(names, EXISTING_TOOLS | BROKER_TOOLS)
        from titanos.broker.mcp_tools import TOOL_NAMES
        self.assertEqual(set(TOOL_NAMES), BROKER_TOOLS)

        def fields(schema, out):
            if isinstance(schema, dict):
                for name, value in (schema.get('properties') or {}).items():
                    out.add(name)
                    fields(value, out)
                for value in schema.values():
                    if isinstance(value, (dict, list)):
                        fields(value, out)
            elif isinstance(schema, list):
                for value in schema:
                    fields(value, out)
            return out
        for tool in listed:
            offending = {name for name in fields(tool.input_schema, set()) if SECRET_FIELD.search(name)}
            self.assertEqual(offending, set(), tool.name)
            self.assertNotIn('ctx', tool.input_schema.get('properties', {}), tool.name)
            self.assertFalse(re.search(r'credential|mint|key_create|rotate|revoke|policy|price|alias|virtual|advisor',
                                       tool.name) and tool.name not in EXISTING_TOOLS, tool.name)
        self.assertIn('broker_status or broker_explain', full.app.state.mcp.instructions)
        self.assertIn('_meta.served_by', full.app.state.mcp.instructions)

    def test_open_world_hint_follows_the_startup_rule(self):
        def hints(full):
            listed = asyncio.run(full.app.state.mcp.list_tools())
            return {t.name: t.annotations.open_world_hint for t in listed}
        bare = hints(self.app())
        self.assertEqual({bare[name] for name in INFERENCE_TOOLS}, {False})
        self.assertTrue(bare['test_upstream'] and bare['moderate'])
        self.assertFalse(bare['broker_status'] or bare['whoami'])
        document = {'upstreams': {'openai': {'profile': 'openai', 'base_url': 'http://127.0.0.1:9/v1'}}}
        configured = hints(self.app(document))
        self.assertEqual({configured[name] for name in INFERENCE_TOOLS}, {True})
        managed = hints(self.app(document, role='managed', port=12002))
        self.assertEqual({managed[name] for name in INFERENCE_TOOLS}, {False})
        delegated = hints(self.app(document, role='managed', port=12003, managed_delegation='parent'))
        self.assertEqual({delegated[name] for name in INFERENCE_TOOLS}, {True})


if __name__ == '__main__':
    unittest.main(verbosity=2)
