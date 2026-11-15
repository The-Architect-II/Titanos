"""Build docs/route_contract_matrix.json and docs/ROUTE_CONTRACT_MATRIX.md: one row per HTTP operation of the Titanos server,
bound to its typed contracts and to the semantics the source implements (INT-01 in docs/progress.json).

Everything is read in-process from this checkout: the FastAPI app is created (create_app(Config()) with the hardware probes of
the runtime catalog stubbed, so the default route surface is registered without touching a device), app.openapi() gives the
operations, app.routes gives the handler (file:line of the route decorator), request models, multipart fields, path/query
parameters, declared response models and status codes; create_mcp() gives the MCP tools, resources, templates and prompts, and
the route each tool or resource calls is read from that tool's own source (the string literals of its json_request/request/
infer call, with non-literal segments matched to a path parameter); the Python client's methods are read the same way from
src/titanos/client.py; the CLI subcommands are an explicit table below resolved through those client methods (the test checks
the table against the argparse help; `mcp` is a bridge joined to every operation an MCP tool calls). The adapter family of every
operation is an explicit table below; the deviation rows of docs/protocol_deviations.json are joined by route for cross-reference.
OPERATION_DOC_GAPS records documentation that under-declares an existing code path (a stale docstring) on the operation's
`doc_gaps` field, outside the semantics classes.

Semantics come from three sources, each row carrying its origin: the explicit SEMANTICS table below (origin `table`: one operation,
one declared control such as a request field, a mode, cancellation or a deadline, its class and one source citation `path:line`
plus an identifier that the cited line must contain; tests/check_route_contract_matrix.py verifies every citation);
DERIVED_FIELD_SEMANTICS (origin `derived`: a row emitted for every operation whose JSON request model declares the named field,
so a field inherited by many models such as `location` is classified on each of them); and SHARED_SEMANTICS (origin `shared`:
the rows of one operation copied onto another that runs the same handler, e.g. POST /api/jobs task=music runs the
POST /v1/music/generations rows, prefixed with the qualifier and pointing back through `shared_from`). Classes:
- implemented: the control reaches the backend or transport with the documented effect;
- ignored: accepted by the schema or transport but never read on that path (a client disconnect that no cancellation observes
  is recorded here);
- rejected: refused explicitly (400/422/409/499 or a schema Literal that admits one value) rather than approximated;
- rejected-in-job: accepted by the schema and answered 202 by a job route, then refused explicitly inside the job thread, so the
  refusal is only visible as job record status `failed` with the error string (no HTTP status or error code reaches the caller);
- declared-only: named by a document, capability record, docstring or field but with no code path that provides it.

Output is deterministic (sorted rows, sorted keys, no timestamps, LF newlines): rebuilding is a fixed point of the committed
JSON; the Markdown carries the source HEAD short hash in its Basis line and is otherwise a fixed point too. No server is
started, no model is loaded, no device is touched, nothing is written outside --out and the temporary TITANOS_HOME.
References are repository-relative; no lab path is written.
Usage: python tools/build_route_contract_matrix.py [--out DIR]"""
import argparse
import ast
import contextlib
import inspect
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import typing
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JSON_OUT = 'route_contract_matrix.json'
MD_OUT = 'ROUTE_CONTRACT_MATRIX.md'
STATUSES = ('implemented', 'ignored', 'rejected', 'rejected-in-job', 'declared-only')
STATUS_MEANING = {
    'implemented': 'the control reaches the backend or transport with the documented effect',
    'ignored': 'accepted by the schema or transport but never read on that path (including a client disconnect that no cancellation observes)',
    'rejected': 'refused explicitly (400/422/409/499 or a schema Literal admitting one value) instead of being approximated',
    'rejected-in-job': ('accepted by the schema and answered 202 by a job route, then refused explicitly inside the job thread: the refusal surfaces only as '
                        'job record status failed with the error string on GET /api/jobs/{identifier} (jobs.py:210, :177); no HTTP status or error code reaches the caller'),
    'declared-only': 'named by a document, capability record, docstring or field, but no code path provides it',
}
ORIGIN_MEANING = {
    'table': 'an explicit SEMANTICS row of the generator',
    'derived': 'emitted for every operation whose JSON request model declares the field (DERIVED_FIELD_SEMANTICS)',
    'shared': 'copied from the operation named in shared_from, whose handler this operation runs with the qualifier prefixed to the control (SHARED_SEMANTICS)',
}
AUTH_MEANING = {
    'public': ('exact path in the per-app public set (server.PUBLIC_ROUTES; GET /api/version joins it in managed role only): GET only (405 otherwise), '
               'a declared body is refused 400, no credential, no scope, no keyless cross-site check, not tracked by the registry gate'),
    'scoped': ('a principal is required and its scope is checked by the Guard before the body is read (src/titanos/broker/auth.py Authenticator with '
               'broker.scopes.required_scope; the row field `scope` is the required scope from broker.scopes.ROUTE_SCOPES, per action where a surface '
               'decides it; where a provider row of the native passthrough shares the shape and answers some requests (OpenAI /v1/videos: a cloud '
               'model in the body, or a video id created through Titanos by the caller), the Guard accepts `scope` or native, and BrokerDispatch '
               'still requires `scope` when the Titanos route answers). Keys enforced (a bootstrap TITANOS_API_KEY or any record in keys.json): no credential or a wrong one is 401, a principal '
               'without the scope is 403 scope_denied, and the credential headers are Authorization: Bearer plus the family vendor header named in the '
               'row field `credential_headers`. Keyless loopback mode: the anonymous loopback principal holds broker.scopes.ANONYMOUS_LOOPBACK_SCOPES '
               '(key and credential writes answer 403 keyless_mode_refused), cross-site browser requests are refused 403, and a non-loopback bind with '
               'nothing enforced answers 401. The registry gate answers 503 registry_reloading/registry_unavailable'),
}
# Scope vocabulary (broker/scopes.py module docstring): the umbrella rules the `scope` field is read with.
SCOPE_RULES = ('admin covers every scope; inference covers inference:*; cloud covers cloud:<upstream_id> but not cloud:store; broker:admin covers '
               'broker:read; models:write covers models:read; a value a|b is satisfied by either scope; authenticated is satisfied by any principal '
               '(never granted); a route missing from ROUTE_SCOPES needs admin (fail closed)')
FAMILY_MEANING = {
    'native': 'Titanos control, lifecycle, jobs, artifacts, health and typed native routes (including the pre-standard /v1/videos/generations, /v1/music/generations and /v1/files/{name})',
    'openai': 'OpenAI-compatible request/response shapes (docs/PROTOCOL_DEVIATIONS.md adapters openai and broker-openai)',
    'api_plus': 'HLI extended inference routes and the typed extensions interface (docs/API_PLUS.md)',
    'ollama': 'Ollama-compatible shapes (adapters ollama and broker-ollama)',
    'anthropic': 'Anthropic Messages shape (adapters anthropic and broker-anthropic)',
    'elevenlabs': 'ElevenLabs speech shapes: text-to-speech, streaming text-to-speech, speech-to-text and voices (adapters elevenlabs and broker-elevenlabs)',
    'gemini': 'Gemini generateContent family under /v1beta: models, generateContent, streamGenerateContent (alt=sse), embedContent, batchEmbedContents, countTokens (adapter broker-gemini)',
    'broker': 'cloud broker control, inbound keys and audit (src/titanos/broker/admin.py; docs/BROKER.md)',
}
FAMILY_ORDER = ['native', 'openai', 'api_plus', 'ollama', 'anthropic', 'elevenlabs', 'gemini', 'broker']
# Explicit family table (by OpenAPI path template); every other operation is native. tests/check_route_contract_matrix.py asserts
# that each listed path is a registered operation, so a removed route cannot linger here. The broker family is every path under
# BROKER_PREFIXES (asserted by the test against the app).
BROKER_PREFIXES = ('/api/broker', '/api/keys', '/api/audit')
FAMILY_PATHS = {
    'openai': {'/v1/models', '/v1/models/{name}', '/v1/chat/completions', '/v1/completions', '/v1/responses', '/v1/embeddings',
               '/v1/audio/speech', '/v1/audio/transcriptions', '/v1/audio/translations', '/v1/images/generations', '/v1/images/edits',
               '/v1/videos', '/v1/videos/{identifier}', '/v1/videos/{identifier}/content', '/v1/moderations',
               '/v1/responses/{response_id}', '/v1/responses/{response_id}/cancel', '/v1/responses/{response_id}/input_items'},
    'api_plus': {'/v1/extensions', '/v1/extensions/infer', '/v1/translations', '/v1/images/ocr', '/v1/images/depth', '/v1/images/segment',
                 '/v1/images/detect', '/v1/sentiment', '/v1/sentiment/compose', '/v1/rerank', '/v1/retrieval/answer',
                 '/v1/classifications/text-pairs', '/v1/audio/diarizations', '/v1/audio/analysis', '/v1/audio/vad'},
    'ollama': {'/api/chat', '/api/generate', '/api/tags', '/api/ps', '/api/embed', '/api/embeddings', '/api/version', '/api/show',
               '/api/pull', '/api/create', '/api/copy', '/api/delete'},
    'anthropic': {'/v1/messages', '/v1/messages/count_tokens'},
    'elevenlabs': {'/v1/text-to-speech/{voice_id}', '/v1/text-to-speech/{voice_id}/stream', '/v1/speech-to-text', '/v1/voices',
                   '/v1/voices/{voice_id}'},
    'gemini': {'/v1beta/models', '/v1beta/models/{model}', '/v1beta/models/{model_action}'},
    'broker': {'/api/audit', '/api/audit/verify', '/api/broker', '/api/broker/advisors', '/api/broker/advisors/{id}', '/api/broker/advisors/{id}/test',
               '/api/broker/catalog', '/api/broker/connect', '/api/broker/credentials', '/api/broker/credentials/{id}', '/api/broker/credentials/{id}/rotate',
               '/api/broker/credentials/{id}/validate', '/api/broker/decisions', '/api/broker/env-candidates', '/api/broker/explain', '/api/broker/metrics', '/api/broker/models',
               '/api/broker/models/{name}', '/api/broker/policy', '/api/broker/prices', '/api/broker/profiles', '/api/broker/reload',
               '/api/broker/upstreams', '/api/broker/upstreams/{id}', '/api/broker/upstreams/{id}/breaker/reset', '/api/broker/upstreams/{id}/health',
               '/api/broker/upstreams/{id}/test', '/api/broker/usage', '/api/broker/virtual-models', '/api/broker/virtual-models/{name}',
               '/api/keys', '/api/keys/self', '/api/keys/{id}', '/api/keys/{id}/revoke', '/api/keys/{id}/rotate'},
}
# Routes that are public only in another role: a scoped operation of this standalone build, recorded so the matrix states the role rule.
ROLE_ONLY_PUBLIC = {'/api/version': ('GET /api/version returns {name, version, role} in every role (Ollama clients read version: protocols.py outside managed role, '
                                     'broker/routes.py in managed role) and is public only in managed role (Guard per-app public set, broker.scopes.MANAGED_PUBLIC_ROUTES); '
                                     'this standalone build lists it as a scoped operation')}
# Response shapes of operations without a declared response_model whose 200 body is not a plain JSON object.
RESPONSE_SHAPES = {
    'POST /v1/chat/completions': 'JSON, or text/event-stream SSE when stream=true',
    'POST /v1/completions': 'JSON, or text/event-stream SSE when stream=true',
    'POST /v1/responses': 'JSON, or text/event-stream SSE when stream=true',
    'POST /v1/audio/speech': 'audio/wav or audio/pcm bytes (local), whole, or with stream_format audio sent as they are made (qwen3-tts); a cloud target answers its own audio type for the requested response_format, streamed through',
    'POST /v1/text-to-speech/{voice_id}': 'audio/wav bytes (local); a cloud target answers the upstream audio type of output_format',
    'POST /v1/text-to-speech/{voice_id}/stream': 'audio bytes, chunked: the local WAV handler of POST /v1/text-to-speech/{voice_id}, or the upstream audio type streamed from a cloud target',
    'POST /v1/speech-to-text': 'JSON in the ElevenLabs speech-to-text shape (multipart request read by BrokerDispatch; bridged to POST /v1/audio/transcriptions for a local model)',
    'POST /v1/messages/count_tokens': 'JSON {input_tokens}',
    'POST /v1/moderations': 'JSON moderation object from the cloud OpenAI target; a local model answers 503 classifier_unavailable',
    'GET /v1/responses/{response_id}': 'JSON passthrough from the upstream owning the handle; 404 response_not_found otherwise',
    'DELETE /v1/responses/{response_id}': 'JSON passthrough from the upstream owning the handle; 404 response_not_found otherwise',
    'POST /v1/responses/{response_id}/cancel': 'JSON passthrough from the upstream owning the handle; 404 response_not_found otherwise',
    'GET /v1/responses/{response_id}/input_items': 'JSON passthrough from the upstream owning the handle; 404 response_not_found otherwise',
    'GET /v1/voices': 'JSON {voices}',
    'GET /v1beta/models': 'JSON {models}',
    'POST /v1beta/models/{model_action}': 'JSON, or text/event-stream SSE for :streamGenerateContent (alt=sse; without it a Gemini upstream passes its JSON-array stream through and a translated cloud stream is refused 400)',
    'GET /v1/health/ready': 'JSON {status, version, role, models, checks} (200) or {status not_ready, reason, message, ...} (503)',
    'GET /v1/files/{name}': 'file (png, mp4 or wav artifact)',
    'GET /v1/videos/{identifier}/content': 'file (mp4; 409 video_not_ready before completion)',
    'POST /api/chat': 'JSON, or application/x-ndjson when stream=true (default true)',
    'POST /api/generate': 'JSON, or application/x-ndjson when stream=true (default true)',
    'POST /v1/messages': 'JSON, or text/event-stream SSE with Anthropic event names when stream=true',
    'POST /v1/audio/translations': 'JSON, or text/plain when response_format=text',
    'GET /readyz': 'JSON {ready, models}; 503 while no model is loaded',
}
# CLI subcommands (src/titanos/cli.py): either the Client methods they call (optionally restricted by 'routes' to the operations
# the subcommand's call actually reaches, e.g. Client.models() with loaded=False), raw routes for the client_request helper, a
# local reason when the subcommand sends nothing to a Titanos server, or 'bridge' when the subcommand forwards another interface's
# calls (mcp runs the stdio MCP bridge, so it is joined to every operation an MCP tool calls); 'note' records a client call that
# sends nothing (save_asset handed bytes writes them without an HTTP call). Keys are asserted against the argparse help by the test.
CLI_SUBCOMMANDS = {
    'native-install': {'local': 'installs a pinned llama.cpp runtime (native_install)'},
    'upgrade': {'local': 'installs a versioned runtime (hardware.install_main)'},
    'init': {'local': 'creates the local model registry file'},
    'settings': {'local': 'describes effective settings'},
    'add': {'local': 'writes the local registry'},
    'pull': {'local': 'Hugging Face download, then the local registry (no Titanos server route)'},
    'search': {'local': 'Hugging Face catalog search (no Titanos server route)'},
    'inspect': {'local': 'Hugging Face repository inspection (no Titanos server route)'},
    'login': {'local': 'Hugging Face token (no Titanos server route)'},
    'logout': {'local': 'Hugging Face token (no Titanos server route)'},
    'prepare': {'local': 'local export and registration'},
    'list': {'local': 'reads the local registry'},
    'devices': {'local': 'local device inventory (devices.inventory), not GET /api/devices'},
    'pin-devices': {'local': 'rewrites local registry assignments'},
    'capabilities': {'local': 'capabilities.describe over the local registry, not POST /api/models/capabilities'},
    'worker': {'local': 'local registry worker assignment'},
    'remove-worker': {'local': 'local registry worker assignment'},
    'remove': {'local': 'local registry'},
    'serve': {'local': 'starts the server that serves every operation in this matrix (uvicorn create_app)'},
    'doctor': {'local': 'hardware report (hardware.doctor)'},
    'mcp': {'bridge': 'stdio MCP bridge: cli.py:825 runs mcp_server.run_stdio(args.url) -> create_mcp(base_url).run(transport=stdio), so every MCP tool call an agent makes is sent to the server at --url; the operations are the union of the MCP tool operations'},
    'load': {'routes': ['POST /api/models/load']},
    'unload': {'routes': ['POST /api/models/unload']},
    'ps': {'routes': ['GET /api/models/loaded']},
    'details': {'routes': ['POST /api/models/details']},
    'workers': {'routes': ['GET /api/workers']},
    'run': {'client': ['chat'], 'note': 'save_asset (--history, --output) receives bytes and writes them without an HTTP call'},
    'models': {'routes': ['GET /v1/models'], 'client': ['models'], 'note': 'Client.models() is called with loaded=False, so only GET /v1/models is reached; GET /api/models/loaded is reached by ps'},
    'info': {'client': ['info']},
    'embed': {'client': ['embeddings']},
    'transcribe': {'client': ['transcribe']},
    'speak': {'client': ['speech'], 'note': 'save_asset receives the WAV bytes returned by POST /v1/audio/speech and writes them without an HTTP call'},
    'image': {'client': ['image', 'save_asset']},
    'edit': {'client': ['edit', 'save_asset']},
    'video': {'client': ['submit_video', 'wait_job', 'save_asset']},
    'music': {'client': ['submit_music', 'wait_job', 'save_asset']},
    'translate': {'client': ['translate']},
    'translate-audio': {'client': ['translate_audio']},
    'ocr': {'client': ['ocr']},
    'detect': {'client': ['detect']},
    'depth': {'client': ['depth']},
    'segment': {'client': ['segment']},
    'sentiment': {'client': ['sentiment']},
    'sentiment-compose': {'client': ['compose_sentiment']},
    'rerank': {'client': ['rerank']},
    'vad': {'client': ['vad']},
    'diarize': {'client': ['diarize']},
    'analyze-audio': {'client': ['analyze_audio']},
    'retrieval-answer': {'client': ['retrieval_answer']},
    'classify-pairs': {'client': ['classify_pairs']},
    'jobs': {'client': ['jobs', 'submit_job', 'job', 'cancel_job', 'wait_job', 'save_asset']},
    'providers': {'client': ['providers']},
    'discover': {'client': ['discover_models']},
    'source-inspect': {'client': ['inspect_model_source']},
    'import-link': {'client': ['import_model_link']},
    'acquire': {'client': ['plan_acquisition', 'submit_acquisition', 'wait_acquisition', 'activate_acquisition']},
    'acquisitions': {'client': ['acquisitions', 'acquisition', 'cancel_acquisition', 'retry_acquisition', 'wait_acquisition',
                                'activate_acquisition', 'delete_acquired_weights']},
    'runtime-prepare': {'client': ['plan_runtime_setup', 'submit_runtime_setup', 'wait_runtime_setup']},
    'runtime-setups': {'client': ['runtime_setups', 'runtime_setup', 'cancel_runtime_setup', 'retry_runtime_setup', 'wait_runtime_setup']},
    'runtime-profiles': {'client': ['runtime_profiles']},
    'assignment-devices': {'client': ['assignment_devices']},
    'assignment': {'client': ['assignment', 'save_assignment']},
    # broker/cli.py: --url mode calls these Client methods; without --url the same commands edit the TITANOS_HOME files
    # (broker.json, keys.json, the vault, the audit log) and send nothing (the note records it)
    'broker': {'client': ['broker_status', 'broker_profiles', 'broker_reload', 'upstreams', 'upstream', 'put_upstream',
                          'delete_upstream', 'test_upstream', 'upstream_health', 'reset_breaker', 'credentials',
                          'set_credential', 'rotate_credential', 'validate_credential', 'delete_credential', 'aliases',
                          'put_alias', 'delete_alias', 'virtual_models', 'put_virtual_model', 'delete_virtual_model', 'policy',
                          'set_policy', 'advisors', 'put_advisor', 'test_advisor', 'prices', 'put_prices', 'broker_catalog',
                          'broker_explain', 'broker_decisions', 'broker_usage', 'broker_env_candidates', 'broker_connect'],
               'note': 'with --url; without it the subcommand edits TITANOS_HOME files under the server locks and sends nothing (live-smoke runs tools/broker_live_smoke.py)'},
    'keys': {'client': ['create_key', 'keys', 'key', 'update_key', 'revoke_key', 'rotate_key', 'whoami'],
             'note': 'with --url; without it the subcommand edits keys.json under its lock (reset --keyless is offline only)'},
    'audit': {'client': ['audit', 'audit_verify'], 'note': 'with --url; without it the subcommand reads the local audit log'},
    'hivemind-manifest': {'local': 'writes a HiveMind engine-manager manifest naming routes; sends nothing'},
    'hivemind': {'routes': ['GET /v1/health/ready'],
                 'note': 'register-service probes the local readiness route, then POSTs /register-service (with a bearer from TITANOS_API_KEY or --key-env) to a HiveMind gateway, which is not a Titanos route'},
    'service': {'routes': ['GET /healthz'],
                'note': 'writes the host service unit (Scheduled Task, systemd, launchd) that runs serve, starts, stops and queries it; install (after checking the registry and the executable) and start poll GET /healthz of the configured server until it answers (service.wait_ready); on Windows stop and uninstall end the server by the id in its pid file'},
}
# Client methods whose route is not a literal in their body: save_asset streams the artifact URL a result carries (same origin only)
# when it is handed a result dict; handed bytes it sends nothing (the CLI table notes those callers).
CLIENT_OVERRIDES = {'save_asset': ['GET /v1/files/{name}']}
# MCP helper names inside create_mcp that stand for a fixed operation.
MCP_HELPERS = {'artifact_bytes': [('GET', '/v1/files/{name}')]}
# operation -> [(citation, identifier the cited line contains, note)]: documentation that under-declares a code path the operation
# has (the inverse of declared-only, so not a semantics class); the test verifies the citations like semantics rows.
OPERATION_DOC_GAPS = {
    'POST /api/jobs': [
        ('src/titanos/mcp_server.py:532', 'speech or video',
         'the MCP submit_job docstring names image, speech or video generation and omits music, although its AssetJob parameter and this route accept task=music (server.py:693 union member MusicJob; server.py:2239 task=music branch)'),
    ],
}
# (field, status, citation, identifier, note): one row per operation whose JSON request model (or any member of its discriminated
# union) declares the field; the test recomputes the set of operations from the live app.
DERIVED_FIELD_SEMANTICS = [
    ('location', 'ignored', 'src/titanos/api_plus.py:44', 'location',
     'declared Literal local_only|auto|prefer_local on every ExtensionPayload; propagated to composed child requests (api_plus.py:887, :890, :1045, :1056, :1103) but never acted upon: no code path branches on its value'),
]
# (target operation, qualifier, source operation, note): every semantics row of the source also applies to the target, whose handler
# is the same code path; the copies carry control '<qualifier>: <control>' and shared_from = source.
SHARED_SEMANTICS = [
    ('POST /api/jobs', 'task=music', 'POST /v1/music/generations',
     'POST /v1/music/generations wraps MusicRequest into MusicJob and calls the POST /api/jobs handler (server.py:2262); MusicJob is the task=music member of the AssetJob union with the same fields, so CLI music, Client.submit_music and MCP submit_job reach every music control through POST /api/jobs'),
]

# Appended to every POST /v1/responses row that describes the local adapter (responses.py chat_request and the vision check that
# follows it): the handler forwards a llama-cpp native model to native.proxy before either runs (server.py:1768-1770), so those
# rows hold for local backends only and carry the qualifier "(local backends)" in their control.
LOCAL_RESPONSES = ('; this row describes the local adapter (responses.py:20 chat_request): a llama-cpp native model\'s request is forwarded to '
                   'native.proxy before this check or transformation runs (server.py:1774), see the llama-cpp native rows')

# (operation, control, status, citation, identifier the cited line contains, note)
SEMANTICS = [
    # front door
    ('GET /healthz', 'unauthenticated GET', 'implemented', 'src/titanos/server.py:76', 'PUBLIC_ROUTES', 'exact-path public route: GET only, no body, no bearer, not registry-gate tracked'),
    ('GET /readyz', 'unauthenticated access (keyed mode)', 'rejected', 'src/titanos/server.py:282', 'authentication_error', 'not in PUBLIC_ROUTES (server.py:76): when create_app received an api_key the Guard answers 401 authentication_error to a request without the bearer; in keyless loopback mode no bearer exists and only the cross-site check runs (server.py:283-288: 403 cross_site_refused for a cross-site browser request), so header-less local clients reach it unauthenticated, as docs/CONFIGURATION.md front-door rules and tests/check_security_public_routes.py record; tracked by the registry gate'),
    ('POST /api/models/reload', 'registry_unavailable bypass', 'implemented', 'src/titanos/server.py:343', '/api/models/reload', 'the only operation answered while the registry gate is unavailable; still 503 while reloading'),
    # music (POST /v1/music/generations wraps MusicJob into the asset job runner; SHARED_SEMANTICS copies these rows onto POST /api/jobs task=music)
    ('POST /v1/music/generations', 'prompt', 'implemented', 'src/titanos/music_backend.py:117', 'prompt', 'bounded by max_prompt_chars at the server, the parent and the child; passed as the ACE caption'),
    ('POST /v1/music/generations', 'mode', 'implemented', 'src/titanos/music_backend.py:449', 'mode', 'refused before the child runs when the bundle manifest capability for the mode is false; instrumental forces [Instrumental] lyrics in the child'),
    ('POST /v1/music/generations', 'lyrics', 'implemented', 'src/titanos/music_backend.py:452', 'lyrics', 'instrumental+lyrics is 422 at the server and refused again in the child; song lyrics need the bundle lyrics capability'),
    ('POST /v1/music/generations', 'duration', 'implemented', 'src/titanos/music_backend.py:459', 'duration', 'server schema 10..600 s (server.py:666) and <= TITANOS_MAX_AUDIO_SECONDS (server.py:682); the parent validates the value against the bundle duration_min..duration_max intersected with TITANOS_MAX_AUDIO_SECONDS (music_backend.py:460) and refuses an out-of-range value with ValueError instead of adjusting it into range, which inside the job thread becomes status failed with that message (row duration outside the bundle range); the child re-checks the sample budget'),
    ('POST /v1/music/generations', 'duration outside the bundle range', 'rejected-in-job', 'src/titanos/music_backend.py:461', 'Music duration must be', 'schema-accepted on MusicRequest and MusicJob (10..600 s and <= TITANOS_MAX_AUDIO_SECONDS); the parent computes minimum = bundle duration_min (default 10) and maximum = min(bundle duration_max (default 600), TITANOS_MAX_AUDIO_SECONDS) (music_backend.py:460) and refuses a duration outside them as ValueError -> EngineError 400 (engine.py:252) raised inside the job thread after the 202, so the job record shows status failed with the message Music duration must be min..max seconds and no HTTP status or code (jobs.py:210, :177)'),
    ('POST /v1/music/generations', 'vocal_language', 'implemented', 'src/titanos/music_backend.py:454', 'vocal_language', 'refused unless in the bundle vocal_languages; forwarded as GenerationParams.vocal_language'),
    ('POST /v1/music/generations', 'bpm', 'implemented', 'src/titanos/music_backend.py:462', 'bpm', '30..300; forwarded to the pinned ACE GenerationParams'),
    ('POST /v1/music/generations', 'key_scale', 'implemented', 'src/titanos/music_backend.py:462', 'key_scale', 'forwarded as GenerationParams.keyscale'),
    ('POST /v1/music/generations', 'seed', 'implemented', 'tools/music_runtime/ace_adapter.py:1226', 'seed', 'child uses use_random_seed=False with the given seed; an omitted seed is drawn with secrets and echoed in the receipt'),
    ('POST /v1/music/generations', 'guidance_scale', 'implemented', 'tools/music_runtime/ace_adapter.py:917', 'guidance_scale', 'server 0..20; child default 7.0'),
    ('POST /v1/music/generations', 'steps', 'implemented', 'src/titanos/music_backend.py:448', 'steps', 'default settings.default_steps at the parent; bounded by TITANOS_MAX_STEPS; forwarded as inference_steps'),
    ('POST /v1/music/generations', 'thinking', 'implemented', 'tools/music_runtime/ace_adapter.py:50', 'thinking', 'optional bool, default true (an absent flag keeps the route of every earlier receipt; a registry default may set it): true has the 4B planner write audio codes that the DiT decodes into precomputed conditioning hints; false sends the DiT no planner audio codes, so it samples from the caption and metadata alone, and needs use_cot_metas (row planner flags without a planner route); forwarded as GenerationParams.thinking; the child requires hint decodes with thinking and none without it, and the receipt echoes the flag next to precomputed_lm_hints_calls'),
    ('POST /v1/music/generations', 'use_cot_metas', 'implemented', 'tools/music_runtime/ace_adapter.py:50', 'use_cot_metas', 'optional bool, default false: true runs the planner metadata pass, whose BPM, key and time signature fill the values the request leaves out before the DiT is conditioned; the receipt records the parsed plan as planner_plan (the planner parser keys, each value at most 4000 characters) and what the DiT received as dit_conditioning'),
    ('POST /v1/music/generations', 'use_cot_caption', 'implemented', 'tools/music_runtime/ace_adapter.py:50', 'use_cot_caption', 'optional bool, default false; needs use_cot_metas: true has the metadata pass write a caption from the prompt, and the DiT text encoder then reads that planner caption in place of the prompt; the receipt records it as dit_conditioning.caption (at most 4000 characters)'),
    ('POST /v1/music/generations', 'use_cot_language', 'implemented', 'tools/music_runtime/ace_adapter.py:50', 'use_cot_language', 'optional bool, default false; needs use_cot_metas: true lets the metadata pass write a language value (recorded in planner_plan); the pinned orchestrator reads a vocal_language key the planner parser never writes, so the DiT keeps the request vocal_language (dit_conditioning.vocal_language)'),
    ('POST /v1/music/generations', 'planner flags without a planner route', 'rejected-in-job', 'src/titanos/music_backend.py:577', 'use_cot_metas', 'schema-accepted on MusicRequest and MusicJob: use_cot_caption or use_cot_language without use_cot_metas (the pinned planner reads them only in its metadata pass), thinking=false without use_cot_metas (the planner would load and never run) and non-boolean flags are refused after the registry defaults are merged (music_backend.py:447, :462) as ValueError -> EngineError 400 (engine.py:252) raised inside the job thread after the 202, so the job record shows status failed with the message and no HTTP status or code (jobs.py:210, :177); the child re-checks the same rules'),
    ('POST /v1/music/generations', 'worker', 'implemented', 'src/titanos/engine.py:215', "data.get('worker')", 'routed to the assigned replica worker (replicas.py:625); a single-process server refuses a worker with 400 (engine.py:208), raised inside the job thread (status failed)'),
    ('POST /v1/music/generations', 'reference_audio_base64', 'rejected-in-job', 'src/titanos/music_backend.py:456', 'reference_audio', 'schema-accepted on MusicRequest and MusicJob; the parent refuses it when the bundle capability reference_audio is false (the packaged provider declares false, providers.py:700) as ValueError -> EngineError 400 (engine.py:252) raised inside the job thread after the 202, so the job record shows status failed with the message and no HTTP status or code (jobs.py:210, :177)'),
    ('POST /v1/music/generations', 'reference_audio_base64 (child runner)', 'rejected-in-job', 'tools/music_runtime/ace_adapter.py:376', 'reference_audio', 'the child refuses reference audio unconditionally (no runtime path exists); the parent maps that child exit to a generic EngineError 502 native_process_failed inside the job thread (music_backend.py:518), so the job record shows status failed naming the exit code, not the child message'),
    ('POST /v1/music/generations', 'timeout', 'rejected', 'src/titanos/server.py:381', 'extra', 'MusicRequest has no timeout field: an unknown field is 422 (Payload extra=forbid); music jobs run without a deadline (jobs.py:159)'),
    ('POST /v1/music/generations', 'stream', 'rejected', 'src/titanos/server.py:381', 'extra', 'no stream field on MusicRequest: 422 unknown field; music is job-only'),
    ('POST /v1/music/generations', 'cancellation (DELETE /api/jobs/{identifier})', 'implemented', 'src/titanos/music_backend.py:490', 'cancelled', 'the job event is checked by the owned command runner; the child process tree is terminated and partial output is discarded; record status cancelled (jobs.py:210)'),
    ('POST /v1/music/generations', 'progress.stage', 'implemented', 'src/titanos/music_backend.py:497', 'stage', 'the parent republishes native stage events written by the child (ace_adapter.py:1299)'),
    ('POST /v1/music/generations', 'progress.completed/total', 'declared-only', 'src/titanos/music_backend.py:501', 'completed', 'the parent accepts completed/total from the child progress file, but the child writes only stage and a constant native_cancelled (ace_adapter.py:1299); no step counts exist'),
    ('POST /v1/music/generations', 'progress.native_cancelled', 'declared-only', 'tools/music_runtime/ace_adapter.py:1299', 'native_cancelled', 'always False: cancellation is enforced by process termination, never observed natively'),
    ('POST /v1/music/generations', 'variant other than xl-sft', 'rejected-in-job', 'src/titanos/music_backend.py:27', 'ACE_SUPPORTED_VARIANTS', 'a bundle property of the registry spec, not a request field: AceStepModel.__init__ refuses xl-turbo and xl-base at load (music_backend.py:435); on the request path the load runs inside the job thread, so the refusal is EngineError 503 model_load_failed (engine.py:155) recorded as status failed; an explicit POST /api/models/load of the same bundle answers 503 model_load_failed; the child re-checks variant == xl-sft (ace_adapter.py:373)'),
    ('POST /v1/music/generations', 'placement (registry spec)', 'implemented', 'src/titanos/music_backend.py:307', 'placement_refused', 'auto/offload/resident is a spec contract sent to the child and verified from its receipt; a resident placement the child refuses before load is surfaced verbatim as EngineError 503 placement_refused inside the job thread, so the job record shows status failed with that message'),
    ('POST /v1/music/generations', 'model_sharding / tensor_parallel', 'declared-only', 'src/titanos/capabilities.py:80', 'model_sharding', 'the capability record declares both modes implemented False, qualified False; no code path'),
    ('POST /v1/music/generations', 'output (PCM16 WAV, 1-2 channels)', 'implemented', 'src/titanos/music_backend.py:528', 'channels', 'validated against max_audio_seconds, max_music_samples and max_music_output_bytes; channel-preserving'),
    ('POST /v1/music/generations', 'capability progress claim', 'implemented', 'src/titanos/music_backend.py:500', 'stage', "the capability record (capabilities.py:82) declares 'request-bound native stage events': the parent reads the child's progress file and republishes its stage events through the job observer bound to the job identity (jobs.py:187), which is what the source provides; the claim names stage events only, and no completion fraction exists (progress.completed/total)"),
    # anonymous diarization
    ('POST /v1/audio/diarizations', 'num_speakers', 'implemented', 'src/titanos/diarization.py:204', 'num_speakers', '0 = automatic (num_clusters -1); bounded by TITANOS_DIARIZATION_MAX_SPEAKERS'),
    ('POST /v1/audio/diarizations', 'cluster_threshold', 'implemented', 'src/titanos/diarization.py:205', 'threshold', 'FastClusteringConfig threshold; finite and >= 0'),
    ('POST /v1/audio/diarizations', 'min_duration_on / min_duration_off', 'implemented', 'src/titanos/diarization.py:206', 'min_duration_on', 'passed to OfflineSpeakerDiarizationConfig; bounded by TITANOS_MAX_AUDIO_SECONDS'),
    ('POST /v1/audio/diarizations', 'num_speakers / cluster_threshold on clips <= 10 s', 'rejected', 'src/titanos/diarization.py:67', 'num_speakers', 'Sherpa bypasses clustering for short clips: only num_speakers=0 with cluster_threshold=0.5 is accepted before load'),
    ('POST /v1/audio/diarizations', 'anonymous speaker labels (per request)', 'implemented', 'src/titanos/diarization.py:162', 'speaker_', 'labels speaker_N by first appearance, rebuilt on every call; the runtime is reconstructed when controls change, so no state crosses requests'),
    ('POST /v1/audio/diarizations', 'overlaps', 'implemented', 'src/titanos/diarization.py:254', 'overlap', 'exact half-open sample intersections of per-speaker intervals; no audio separation'),
    ('POST /v1/audio/diarizations', 'cancellation (disconnect, MCP cancel)', 'implemented', 'src/titanos/diarization.py:243', 'check_cancel', 'checked before decode, around configure, inside the embedding callback and after the native return'),
    ('POST /v1/audio/diarizations', 'cancellation inside native segmentation/clustering', 'declared-only', 'src/titanos/capabilities.py:58', 'no checks during native segmentation or clustering', 'the capability record declares the gap: a running Sherpa segmentation/clustering pass finishes before the cancel is observed'),
    ('POST /v1/audio/diarizations', 'timeout', 'implemented', 'src/titanos/api_plus.py:456', 'deadline_exceeded', 'millisecond deadline including the model wait; 504 deadline_exceeded'),
    ('POST /v1/audio/diarizations', 'worker', 'implemented', 'src/titanos/api_plus.py:1066', 'worker', 'checked against the diarization task assignment before dispatch'),
    ('POST /v1/audio/diarizations', 'unknown diarization controls', 'rejected', 'src/titanos/diarization.py:40', 'Unsupported diarization controls', 'ValueError -> 400; unknown top-level fields are 422 (api_plus.py:40 extra=forbid)'),
    ('POST /v1/audio/diarizations', 'model spec (task diarization, backend sherpa-onnx, cpu)', 'rejected', 'src/titanos/diarization.py:33', 'sherpa-onnx', 'other tasks, backends, devices, dtypes or loader overrides are refused at construction'),
    ('POST /v1/audio/diarizations', 'stream', 'rejected', 'src/titanos/api_plus.py:40', 'extra', 'no stream field on DiarizationRequest, so ExtensionPayload extra=forbid answers 422 unknown field; the capability record declares stream False (capabilities.py:52)'),
    ('POST /v1/audio/diarizations', 'speaker identity / calibrated confidence', 'declared-only', 'src/titanos/capabilities.py:60', 'identity', 'the qualification note declares anonymous interval estimates are not identity or calibrated confidence; no request field or code path'),
    ('POST /v1/audio/diarizations', 'result control echo', 'implemented', 'src/titanos/api_plus.py:1106', 'parsed.parameters', 'the parent verifies the typed result echoes the effective controls, input digest and size budget'),
    ('POST /v1/audio/analysis', 'diarization_options', 'implemented', 'src/titanos/api_plus.py:1144', 'diarization_options', 'nested diarization controls are forwarded to the diarization stage under the shared deadline'),
    ('POST /v1/audio/analysis', 'speaker attribution of ASR segments', 'implemented', 'src/titanos/api_plus.py:1192', 'AttributedASRSegment', 'anonymous speakers and overlap flags per transcript segment'),
    ('POST /v1/audio/analysis', 'shared deadline (VAD gate + ASR + diarization + sentiment)', 'implemented', 'src/titanos/api_plus.py:1113', 'audio_analysis', 'one timeout across the composed stages'),
    ('POST /v1/audio/analysis', 'diarization_model / diarization_worker', 'implemented', 'src/titanos/api_plus.py:1142', 'diarization_model', 'the diarization stage runs on the named model under the shared deadline; the worker is checked against the diarization assignment like the direct route (audio_worker)'),
    ('POST /v1/audio/analysis', 'text_analysis_model / text_analysis_worker / text_analysis_options', 'implemented', 'src/titanos/api_plus.py:1199', 'text_analysis_model', 'sentiment stage over the transcript (input_modality asr_transcript) after a static-controls preflight (api_plus.py:1152); text_analysis_status not_run_empty_transcript when the transcript is blank'),
    ('POST /v1/audio/analysis', 'vad_enabled / vad_model / vad_options (ASR gate)', 'implemented', 'src/titanos/api_plus.py:1127', 'resolve_gate', 'inherited ASR gate fields resolved as on POST /v1/audio/transcriptions; the gate controls are validated before any stage runs'),
    ('POST /v1/audio/analysis', 'language / prompt / max_tokens / timestamp_granularities (ASR stage)', 'implemented', 'src/titanos/api_plus.py:1116', 'ASRRequest.model_validate', 'the ASRRequest fields are copied to the transcription stage; response_format is Literal verbose_json'),
    ('POST /v1/audio/analysis', 'model / worker (ASR stage)', 'implemented', 'src/titanos/api_plus.py:1117', 'audio_worker', 'the top-level model and worker are copied into the ASRRequest (api_plus.py:1116) and checked by audio_worker before any stage runs: the model must be a transcription-task registration (404 model_not_found, 400 unsupported_capability) and the worker must be assigned that model (404 worker_not_found); the diarization and text-analysis stages carry their own model/worker fields'),
    ('POST /v1/audio/analysis', 'response_format other than verbose_json', 'rejected', 'src/titanos/api_plus.py:227', 'verbose_json', 'AudioAnalysisRequest narrows the inherited ASRRequest Literal to verbose_json: json is a 422 schema error, so the transcript stage always carries segments for speaker attribution'),
    # stateful Responses
    ('POST /v1/responses', 'store=true', 'rejected', 'src/titanos/request_compat.py:80', 'store', '400 unsupported_capability before the schema (request_compat.split; store stays Literal[False] at server.py:633); the response echoes store false (responses.py:116)'),
    ('POST /v1/responses', 'include / reasoning / truncation / user / safety_identifier / prompt_cache_key / prompt_cache_retention / service_tier / top_logprobs / text.verbosity / background=false', 'ignored', 'src/titanos/request_compat.py:34', 'RESPONSES', 'type-checked, removed before the strict schema and never read; named in the X-Titanos-Ignored response header (request_compat.attach); agent harnesses send them (docs/PROTOCOL_COMPATIBILITY.md)'),
    ('POST /v1/responses', 'background=true', 'rejected', 'src/titanos/request_compat.py:86', 'background', '400 unsupported_capability: local Responses run synchronously'),
    ('POST /v1/chat/completions', 'metadata / user / store=false / prompt_cache_key / prompt_cache_retention / safety_identifier / service_tier / top_logprobs / reasoning_effort / verbosity', 'ignored', 'src/titanos/request_compat.py:32', 'CHAT', 'type-checked, removed before the strict schema and never forwarded to the engine or llama.cpp; named in the X-Titanos-Ignored response header (request_compat.attach); agent harnesses send them (docs/PROTOCOL_COMPATIBILITY.md)'),
    ('POST /v1/chat/completions', 'store=true', 'rejected', 'src/titanos/request_compat.py:80', 'store', '400 unsupported_capability: no local completion store'),
    ('POST /v1/chat/completions', 'options.num_ctx (the Ollama options object HiveMind sends with every OpenAI chat request to a non-cloud backend)', 'implemented', 'src/titanos/request_compat.py:133', 'context_option', 'a foreign extension here, never forwarded to the runtime: on a llama-cpp model it sizes the leased runtime, which starts, or restarts once its requests have drained, with a per-slot context of the next power of two at or above num_ctx up to the largest per-slot context (native.max_slot_context, TITANOS_NATIVE_MAX_SLOT_CONTEXT or the GGUF trained context_length; native_context.py, passed on by server.generate), and nothing is named; above that maximum, and on the other backends (no context option), the value is not honoured and options.num_ctx is named in X-Titanos-Ignored (server.py:1761), never a 400, where POST /api/chat answers 400 context_limit (protocols.py:200) because num_ctx is a native field there; the 2026-10-05 HiveMind smoke body answered 422 before the options object was accepted'),
    ('POST /v1/chat/completions', 'options.<key> other than num_ctx (num_gpu, temperature, ...)', 'ignored', 'src/titanos/request_compat.py:107', 'options', 'options is type-checked (a non-object is a typed 400, a non-positive-integer num_ctx too), removed before the strict schema and each remaining key is named as options.<key> in X-Titanos-Ignored; never applied over the sampling fields the request itself carries and never a 422'),
    ('POST /v1/completions', 'options.num_ctx', 'implemented', 'src/titanos/request_compat.py:133', 'context_option', 'as POST /v1/chat/completions (server.py:1766)'),
    ('POST /v1/completions', 'options.<key> other than num_ctx', 'ignored', 'src/titanos/request_compat.py:107', 'options', 'as POST /v1/chat/completions'),
    ('POST /v1/responses', 'previous_response_id', 'rejected', 'src/titanos/request_compat.py:82', 'previous_response_id', '400 unsupported_capability: the adapter is stateless (responses.py:1) and no response store exists'),
    ('POST /v1/responses', 'conversation', 'rejected', 'src/titanos/server.py:381', 'extra', '422 unknown field; no conversation state'),
    ('POST /v1/responses', 'tools (local backends)', 'rejected', 'src/titanos/responses.py:28', 'responses_capabilities', '400 unsupported_capability whose message names the model, its backend (transformers, openvino-genai, ...) and the missing capability (agent_capability.responses_capabilities, agent_capability.py:49; agent_capability.refusal, responses.py:30); an empty list is a no-op' + LOCAL_RESPONSES),
    ('POST /v1/responses', 'tools / tool_choice / text (llama-cpp native)', 'implemented', 'src/titanos/server.py:1774', 'native', 'forwarded to the llama.cpp server (native.proxy); a streamed answer is completed by responses.native_stream (the stream (llama-cpp native) row); semantics belong to that binary'),
    ('POST /v1/responses', 'input items / roles / images / instructions / sampling / usage (llama-cpp native)', 'implemented', 'src/titanos/server.py:1774', 'native',
     'a model of the llama-cpp native set is forwarded to the llama.cpp server (native.proxy) before chat_request and the vision check run, after model_defaults has filled omitted sampling and token fields from the model spec (server.py:1775, request_defaults.py:5), so the local-backend refusals and transformations of this table (the rows qualified local backends) do not apply: input item types and roles, image parts, instructions, stream, metadata and usage carry the semantics of that binary; max_output_tokens, temperature and top_p keep the schema bounds and process defaults of server.py:625-627 (422 out of range for every backend; an omitted value reaches the binary filled with the process default or the model-spec default because the branch dumps with exclude_none, server.py:1775-1776); the Responses schema types input as a string or a list of dicts without validating the items (server.py:623); tests/check_responses.py:293, :318, :321 exercises the passthrough (tools, tool_choice, parallel_tool_calls, max_output_tokens, temperature, top_p and stream on a string input; the forwarded body carries every request field unchanged; an upstream 503 is returned with its headers); item roles, types and images on this branch follow from the source order, not from a test'),
    ('POST /v1/responses', 'tool_choice (local backends)', 'rejected', 'src/titanos/responses.py:28', 'responses_capabilities', 'only None, auto or none are accepted and echoed; anything else is a 400 naming the model, its backend and tool_choice (agent_capability.py:51, agent_capability.refusal)' + LOCAL_RESPONSES),
    ('POST /v1/responses', 'parallel_tool_calls=true (local backends)', 'rejected', 'src/titanos/responses.py:28', 'responses_capabilities', 'true is a 400 naming the model, its backend and parallel_tool_calls (agent_capability.py:53); false/None are echoed as false' + LOCAL_RESPONSES),
    ('POST /v1/responses', 'text.format other than text (local backends)', 'rejected', 'src/titanos/responses.py:28', 'responses_capabilities', 'only None, {} or {format: {type: text}} are accepted; anything else is a 400 naming the model, its backend and structured output (text.format) (agent_capability.py:55)' + LOCAL_RESPONSES),
    ('POST /v1/responses', 'input items function_call / function_call_output (local backends)', 'rejected', 'src/titanos/responses.py:28', 'responses_capabilities', '400 naming the model, its backend and the tool-call history capability before the item loop (agent_capability.py:58); a llama-cpp native model serves the tool loop (tests/check_agent_loops.py, manual tier)' + LOCAL_RESPONSES),
    ('POST /v1/responses', 'instructions (local backends)', 'implemented', 'src/titanos/responses.py:37', 'instructions', 'mapped to a system message; bounded by max_text_chars' + LOCAL_RESPONSES),
    ('POST /v1/responses', 'max_output_tokens (local backends)', 'implemented', 'src/titanos/responses.py:101', 'max_output_tokens', 'mapped to Chat.max_tokens; model default aliases apply when omitted' + LOCAL_RESPONSES),
    ('POST /v1/responses', 'temperature / top_p / worker (local backends)', 'implemented', 'src/titanos/responses.py:98', 'temperature', 'forwarded only when explicitly supplied so model and process defaults apply' + LOCAL_RESPONSES),
    ('POST /v1/responses', 'stream (local backends)', 'implemented', 'src/titanos/responses.py:103', 'stream', 'SSE created/in_progress/output_item/content_part/output_text.delta/done/completed events with sequence numbers; the MCP responses tool forces stream false' + LOCAL_RESPONSES),
    ('POST /v1/responses', 'stream (llama-cpp native)', 'implemented', 'src/titanos/server.py:1776', 'native_stream',
     "the llama.cpp server's SSE is forwarded with sequence_number, output_index / content_index and a response object completed from NATIVE_RESPONSE_SKELETON (created_at, the requested model, output []) on response.created and response.in_progress (responses.native_stream, responses.py:248-296): the fields the stock OpenAI SDK's responses.stream() helper indexes, which b10809 omits (it sends {id, object, status} only); whatever the binary sent is kept (tests/check_responses.py native passthrough; the frames were captured from the CPU binary)"),
    ('POST /v1/responses', 'metadata (local backends)', 'implemented', 'src/titanos/responses.py:118', 'metadata', 'echoed (16 pairs, 64/512 character limits at the schema, server.py:638); never stored' + LOCAL_RESPONSES),
    ('POST /v1/responses', 'input role developer (local backends)', 'rejected', 'src/titanos/responses.py:47', "'system', 'user', 'assistant'", 'only system, user and assistant items are accepted (400)' + LOCAL_RESPONSES),
    ('POST /v1/responses', 'input item type other than message (local backends)', 'rejected', 'src/titanos/responses.py:40', 'message', '400 for non-message items' + LOCAL_RESPONSES),
    ('POST /v1/responses', 'input_image file IDs / non-inline URLs (local backends)', 'rejected', 'src/titanos/responses.py:73', 'input_image', 'inline data URLs with detail auto only' + LOCAL_RESPONSES),
    ('POST /v1/responses', 'image input on a non-vision model (local backends)', 'rejected', 'src/titanos/server.py:1783', 'unsupported_capability', '400 unsupported_capability; the check runs on the local path after chat_request' + LOCAL_RESPONSES),
    ('POST /v1/responses', 'usage (local backends)', 'implemented', 'src/titanos/responses.py:121', 'usage', 'from the backend usage; cached/cache_write/reasoning counts fixed at 0; inconsistent totals are 502' + LOCAL_RESPONSES),
    ('POST /v1/responses', 'cancellation (stream, local backends)', 'implemented', 'src/titanos/responses.py:230', 'disconnect', 'a disconnect after created propagates the chat stream cancellation' + LOCAL_RESPONSES),
    ('POST /v1/responses', 'cancellation (nonstream, local backends)', 'ignored', 'src/titanos/server.py:1708', 'engine.infer', 'the handler delegates to generate() (server.py:1785), whose nonstream local path calls engine.infer without a cancel event, so a disconnect is not observed (the require_loaded=false path of POST /v1/chat/completions; Responses declares no require_loaded field, so the strict-residency disconnect observer of that route is unreachable here); docs/RESPONSES.md:53-61 documents stream cancellation only'),
    # chat and completions
    ('POST /v1/chat/completions', 'tools / tool_choice / parallel_tool_calls / response_format (local backends)', 'rejected', 'src/titanos/server.py:1548', 'tools', '400 unsupported_capability whose message names the model, its backend (transformers, openvino-genai, ...), the missing capability and the route the client used (agent_capability.refusal, server.py:1549; the Ollama and Anthropic routes reach the same line and are named as such); tests/check_agent_tools.py'),
    ('POST /v1/chat/completions', 'roles tool / developer and assistant tool_calls (local backends)', 'rejected', 'src/titanos/server.py:1548', 'developer', '400 unsupported_capability naming the model, its backend and the tool-call history or developer role (agent_capability.chat_capabilities); the Message model validates the tool shape first (server.py:441)'),
    ('POST /v1/chat/completions', 'tools / json_schema (llama-cpp native)', 'implemented', 'src/titanos/native.py:686', 'proxy', 'forwarded untouched to the llama.cpp server; the capability record lists tools and json_schema (capabilities.py:172)'),
    ('POST /v1/chat/completions', 'stream', 'implemented', 'src/titanos/server.py:1757', 'text/event-stream', 'SSE with [DONE]; require_loaded and native paths stream too'),
    ('POST /v1/chat/completions', 'stream_options.include_usage', 'implemented', 'src/titanos/server.py:496', 'include_usage', 'the only stream_options key; others are 422'),
    ('POST /v1/chat/completions', 'n (llama-cpp native)', 'implemented', 'src/titanos/server.py:488', 'n: int', 'forwarded to the llama.cpp server, which returns n choices (b10809 CPU binary, 2026-10-05); 1..TITANOS_MAX_CHOICES (server.py:407), above is 422'),
    ('POST /v1/chat/completions', 'n other than 1 (local backends)', 'rejected', 'src/titanos/openai_params.py:52', 'n', '400 unsupported_capability naming n and the backend (openai_params.local_generation); n=1 is the OpenAI default and is served'),
    ('POST /v1/chat/completions', 'logprobs / top_logprobs / logit_bias / presence_penalty / frequency_penalty (llama-cpp native)', 'implemented', 'src/titanos/server.py:483', 'presence_penalty', 'Generation/Chat schema fields (server.py:483-485, :509-510) forwarded untouched to the llama.cpp server, which honours each (logprobs in the body and SSE chunks; b10809 CPU binary, 2026-10-05)'),
    ('POST /v1/chat/completions', 'logprobs / top_logprobs / presence_penalty / frequency_penalty (local backends)', 'rejected', 'src/titanos/openai_params.py:25', 'NATIVE_ONLY', '400 unsupported_capability naming the field and the backend (openai_params.local_generation); false and 0 are inert and served'),
    ('POST /v1/chat/completions', 'logit_bias (transformers)', 'implemented', 'src/titanos/openai_params.py:79', 'transformers_bias', 'integer token-id keys become generate() sequence_bias (backends.py:380); a non-integer key is a ValueError -> 400 naming logit_bias; {} is served; openvino-genai refuses the field by its own check (genai.py:18)'),
    ('POST /v1/chat/completions', 'temperature / top_p / max_tokens / stop', 'implemented', 'src/titanos/backends.py:386', 'top_p', 'transformers generation kwargs; context overflow is a 400 (backends.py:339)'),
    ('POST /v1/chat/completions', 'top_p when temperature is 0', 'ignored', 'src/titanos/backends.py:385', 'temperature > 0', 'greedy decoding drops top_p on transformers (and openvino-genai, genai.py:44)'),
    ('POST /v1/chat/completions', 'seed', 'implemented', 'src/titanos/backends.py:391', 'manual_seed', 'torch.manual_seed on transformers; rng_seed on openvino-genai'),
    ('POST /v1/chat/completions', 'require_loaded', 'implemented', 'src/titanos/engine.py:213', 'require_loaded', 'strict residency: 409 model_not_loaded, admission futures and a disconnect observer (server.py:1643)'),
    ('POST /v1/chat/completions', 'max_completion_tokens', 'implemented', 'src/titanos/server.py:504', 'max_completion_tokens', 'alias of max_tokens'),
    ('POST /v1/chat/completions', 'cancellation (stream)', 'implemented', 'src/titanos/server.py:1752', 'cancel.set()', 'the generation thread observes the cancel event when the SSE consumer stops'),
    ('POST /v1/chat/completions', 'cancellation (nonstream, local backends, require_loaded=false)', 'ignored', 'src/titanos/server.py:1708', 'engine.infer', 'the default path calls engine.infer through asyncio.to_thread without a cancel event, so a disconnect is not observed and the generation runs to completion; the native nonstream path returns 499 on disconnect (native.py)'),
    ('POST /v1/chat/completions', 'cancellation (nonstream, local backends, require_loaded=true)', 'implemented', 'src/titanos/server.py:1647', 'http.disconnect', 'the strict-residency path (server.py:1565) creates a cancel Event passed to engine.infer (server.py:1601); observe_disconnect sets it on http.disconnect and resolves the admission and outcome futures with 499 cancelled, so the nonstream request is cancelled when the client leaves'),
    ('POST /v1/chat/completions', 'timeout', 'rejected', 'src/titanos/server.py:381', 'extra', 'no timeout field: 422 unknown field; deadlines exist only for internal compositions'),
    ('POST /v1/completions', 'prompt', 'implemented', 'src/titanos/server.py:524', 'prompt', 'raw prompt completion; Completion has no tools fields (422 if sent)'),
    ('POST /v1/completions', 'logprobs (llama-cpp native): legacy shape', 'implemented', 'src/titanos/openai_params.py:109', 'legacy_logprobs',
     "the chat-shaped logprobs.content[] the llama.cpp server returns is rewritten to the OpenAI legacy completions shape (tokens, token_logprobs, top_logprobs maps, text_offset counted across chunks) in the body and in every SSE chunk (server.py:1546); chat keeps llama-server's chat shape; until 2026-10-06 the chat shape was forwarded as is and the stock SDK's typed Logprobs fields stayed null (tests/check_openai_params.py)"),
    # provider adapters
    ('POST /api/chat', 'options (num_predict, temperature, top_p, top_k, min_p, repeat_penalty, seed, stop)', 'implemented', 'src/titanos/protocols.py:188', 'num_predict', 'num_predict maps to max_tokens; others share the inference controls'),
    ('POST /api/chat', 'options num_gpu / mirostat / repeat_last_n and unknown top-level fields (think, raw, template, ...)', 'rejected', 'src/titanos/protocols.py:54', 'extra', '422 unknown field on Options and Ollama (protocols.py:70); keep_alive and options.num_ctx are accepted (rows below)'),
    ('POST /api/chat', 'keep_alive', 'implemented', 'src/titanos/protocols.py:34', 'keep_alive_policy', '0 unloads after the answer (release: the POST /api/models/unload operation, protocols.py:217), a negative value keeps the model resident (the Titanos default), a positive duration has no Titanos timer and is named in X-Titanos-Dropped; wrong types and unparseable durations are 400'),
    ('POST /api/chat', 'options.num_ctx', 'implemented', 'src/titanos/protocols.py:200', 'context_window', 'llama-cpp: honoured up to the largest per-slot context (the leased runtime starts at, or once idle restarts at, the next power of two at or above num_ctx; native_context.py), 400 context_limit above it; other backends have no context option and the field is named in X-Titanos-Dropped'),
    ('POST /api/chat', 'messages[] role=tool with tool_name (Ollama tool results)', 'implemented', 'src/titanos/protocols.py:253', 'ollama_messages', 'the tool loop in Ollama\'s shape on llama-cpp models: assistant tool_calls get synthesized ids, a tool result with tool_name and no tool_call_id is bound to the pending call of that name (the next unanswered call without one) and tool_name travels as name; a result with no call to answer is 400 naming the message (tests/check_ollama_compat.py, tests/check_ollama_sdk.py)'),
    ('POST /api/generate', 'keep_alive', 'implemented', 'src/titanos/protocols.py:34', 'keep_alive_policy', 'as POST /api/chat (one ollama_run)'),
    ('POST /api/generate', 'options.num_ctx', 'implemented', 'src/titanos/protocols.py:200', 'context_window', 'as POST /api/chat (one ollama_run)'),
    ('POST /api/embed', 'keep_alive', 'implemented', 'src/titanos/protocols.py:450', 'keep_alive', 'same policy as POST /api/chat; options stays a 400 on this route'),
    ('POST /api/embeddings', 'keep_alive', 'implemented', 'src/titanos/protocols.py:463', 'keep_alive', 'same policy as POST /api/chat'),
    ('GET /api/version', 'version', 'implemented', 'src/titanos/protocols.py:346', 'ollama_version', 'the managed role body {name, version, role} (broker/routes.py registers the managed, public copy); Ollama clients read version'),
    ('POST /api/show', 'model / name', 'implemented', 'src/titanos/protocols.py:388', 'ollama_show', 'registry entry plus the passive model inspection (capabilities.describe); never loads; 404 model_not_found; verbose is accepted and named in X-Titanos-Ignored; unknown fields 400'),
    ('POST /api/pull', 'model', 'rejected', 'src/titanos/protocols.py:408', 'ollama_pull', '400 unsupported_capability (param model) naming the Titanos acquisition path (protocols.ACQUISITION_PATH)'),
    ('POST /api/create', 'model / from / files', 'rejected', 'src/titanos/protocols.py:413', 'ollama_create', '400 unsupported_capability (param model): no Modelfile; defaults belong to the registration'),
    ('POST /api/copy', 'source / destination', 'rejected', 'src/titanos/protocols.py:419', 'ollama_copy', '400 unsupported_capability (param destination): register the directory under another name'),
    ('DELETE /api/delete', 'model', 'rejected', 'src/titanos/protocols.py:425', 'ollama_delete', '400 unsupported_capability (param model): DELETE /api/models/registrations/{name} and DELETE /api/acquisitions/{identifier}/weights'),
    ('POST /api/chat', 'format', 'implemented', 'src/titanos/protocols.py:280', 'format', 'json|schema maps to response_format, then 400 on local backends or native passthrough'),
    ('POST /api/chat', 'per-message images', 'implemented', 'src/titanos/protocols.py:242', 'images', 'mapped to inline image parts (labelled image/png); top-level images are refused'),
    ('POST /api/chat', 'stream (NDJSON, default true)', 'implemented', 'src/titanos/protocols.py:334', 'x-ndjson', 'the OpenAI tool-call argument fragments llama-server streams are gathered by index into whole tool_calls with object arguments and emitted in one frame ahead of the final frame (protocols.ToolCallBuffer, protocols.py:564; ollama_run.stream, protocols.py:323); the complete argument text is parsed at protocols.py:309 on both paths; text deltas pass through as they arrive (tests/check_ollama_compat.py, tests/check_ollama_sdk.py streaming tool loop)'),
    ('POST /api/generate', 'messages / images / format / tools', 'rejected', 'src/titanos/protocols.py:284', 'messages', 'raw-mode completion refuses chat-only fields (400)'),
    ('POST /api/embed', 'user / dimensions=null', 'ignored', 'src/titanos/protocols.py:166', 'run_embed', 'named in X-Titanos-Ignored as on /v1/embeddings (request_compat.EMBEDDINGS): run_embed hands the shared embeddings handler a carrier response and carry_ignored (protocols.py:553) copies the header across the native Response -> dict step that used to lose it (a silent drop until 2026-10-06); dimensions with a value is 400 (request_compat.py:73); tests/check_ollama_compat.py'),
    ('POST /api/embeddings', 'user / dimensions=null', 'ignored', 'src/titanos/protocols.py:166', 'run_embed', 'as POST /api/embed (one run_embed)'),
    ('POST /api/embeddings', 'input other than one prompt string', 'rejected', 'src/titanos/protocols.py:465', 'Legacy embeddings', 'legacy route accepts exactly one prompt (400)'),
    ('POST /v1/messages', 'tools / tool_use / tool_result (nonstream)', 'implemented', 'src/titanos/protocols.py:513', "values['tools']", 'mapped to the OpenAI tool shape; native models only (400 elsewhere)'),
    ('POST /v1/messages', 'tools with stream=true', 'implemented', 'src/titanos/agent_capability.py:75', 'anthropic_stream', 'tool_use content blocks (content_block_start with id, name and empty input) with input_json_delta fragments built from the llama.cpp tool_calls deltas, parallel calls as successive blocks, stop_reason tool_use; native models only (400 naming the backend elsewhere); tool_choice.disable_parallel_tool_use maps to parallel_tool_calls false; tests/check_agent_tools.py, stock Anthropic SDK loop in tests/check_agent_loops.py (manual tier)'),
    ('POST /v1/messages', 'image source other than base64', 'rejected', 'src/titanos/protocols.py:505', 'base64', 'URL and file sources are 400'),
    ('POST /v1/messages', 'stop_sequences / top_k', 'implemented', 'src/titanos/protocols.py:489', 'stop_sequences', 'mapped to stop and top_k (protocols.py:487)'),
    ('POST /v1/messages', 'unknown content block types', 'rejected', 'src/titanos/protocols.py:509', 'Unsupported content block', 'document/thinking/server_tool_use blocks are 400'),
    ('POST /v1/messages', 'betas / mcp_servers / output_format', 'rejected', 'src/titanos/protocols.py:94', 'extra', '422 unknown field (Anthropic model extra=forbid)'),
    ('POST /v1/messages', 'metadata / thinking / service_tier / inference_geo / cache_control / context_management / output_config without format', 'ignored', 'src/titanos/request_compat.py:35', 'ANTHROPIC', 'type-checked, removed before the strict schema and never read; named in the X-Titanos-Ignored response header (request_compat.attach); agent harnesses send them (docs/PROTOCOL_COMPATIBILITY.md)'),
    ('POST /v1/messages', 'container / output_config.format', 'rejected', 'src/titanos/request_compat.py:99', 'container', '400 unsupported_capability (output_config.format: src/titanos/request_compat.py:102)'),
    ('POST /v1/messages', 'stream (text deltas)', 'implemented', 'src/titanos/agent_capability.py:116', 'text_delta', 'Anthropic event names; a text block is opened on the first text delta and stopped before a tool_use block starts; an empty stream ends with one empty text block'),
    ('POST /v1/text-to-speech/{voice_id}', 'output_format other than wav', 'rejected', 'src/titanos/protocols.py:539', 'output_format', '400 unsupported_capability'),
    ('POST /v1/text-to-speech/{voice_id}', 'body fields other than text and model_id (voice_settings, ...)', 'rejected', 'src/titanos/protocols.py:543', 'Supported fields', '400'),
    ('POST /v1/text-to-speech/{voice_id}', 'voice_id', 'implemented', 'src/titanos/protocols.py:547', 'voice', 'the path segment is the Titanos voice alias passed to the speech handler'),
    # speech
    ('POST /v1/audio/speech', 'speed (sherpa-onnx)', 'implemented', 'src/titanos/sherpa_tts.py:95', 'speed', '0.25..4.0 at the schema (server.py:563) and passed to the sherpa generator; 1.0 is the default'),
    ('POST /v1/audio/speech', 'speed other than 1.0 (transformers, xtts, qwen3-tts)', 'rejected', 'src/titanos/openai_params.py:101', 'speech_speed', 'ValueError -> 400 naming speed from speak() (backends.py:438); qwen3-tts re-checks (qwen_tts_backend.py:273)'),
    ('POST /v1/audio/speech', 'voice_map (client voice names)', 'implemented', 'src/titanos/openai_params.py:91', 'speech_voice', 'ModelSpec.voice_map (config.py:83, speech only) maps alloy/nova/... onto the model voice id inside Engine.infer (engine.py:222), so every speech caller shares it; names outside the map keep the backend check'),
    ('POST /v1/audio/speech', 'response_format other than wav or pcm', 'rejected', 'src/titanos/server.py:560', 'response_format', 'Literal["wav", "pcm"]: 422'),
    ('POST /v1/audio/speech', 'voice', 'implemented', 'src/titanos/backends.py:438', 'voice', 'xtts named reference, sherpa default or numeric speaker id (sherpa_tts.py:83), qwen3-tts configured alias (qwen_tts_backend.py:275)'),
    ('POST /v1/audio/speech', 'language (xtts, qwen3-tts)', 'implemented', 'src/titanos/backends.py:443', 'language', 'xtts language; qwen3-tts language map (qwen_tts_backend.py:280)'),
    ('POST /v1/audio/speech', 'language (transformers speech)', 'rejected', 'src/titanos/backends.py:449', 'language', '400 for transformers speech models'),
    ('POST /v1/audio/speech', 'language (sherpa-onnx)', 'ignored', 'src/titanos/sherpa_tts.py:75', 'def infer', 'the sherpa backend never reads language'),
    ('POST /v1/audio/speech', 'seed (qwen3-tts)', 'implemented', 'src/titanos/qwen_tts_backend.py:300', 'seed', 'the only speech backend that reads seed'),
    ('POST /v1/audio/speech', 'seed (transformers, xtts, sherpa-onnx)', 'ignored', 'src/titanos/backends.py:436', 'speak', 'declared on Speech (server.py:557) but never read by speak() or the sherpa backend'),
    ('POST /v1/audio/speech', 'stream', 'rejected', 'src/titanos/server.py:556', 'Speech', 'no stream field on Speech, so Payload extra=forbid (server.py:381) answers 422 unknown field; streaming is stream_format (server.py:561)'),
    ('POST /v1/audio/speech', 'stream_format audio (qwen3-tts)', 'implemented', 'src/titanos/qwen_tts_backend.py:351', 'emit', 'the format record, then each decoded chunk as PCM16 (qwen_tts_backend.py:354), reach the client as they are made (speech_stream.py:66): with response_format wav the body opens with the gim_api TTS Super streaming header (RIFF and data sizes 0x7FFFFFFF, speech_stream.py:28), with pcm it is bare PCM16; an error before audio is a JSON error, one after audio aborts the transfer (speech_stream.py:132); tests/check_speech_stream.py, tests/check_qwen_tts_streaming.py'),
    ('POST /v1/audio/speech', 'stream_format audio (transformers, xtts, sherpa-onnx)', 'ignored', 'src/titanos/speech_stream.py:109', 'complete(', 'these backends emit no audio while they synthesize, so the route answers with the complete body; a disconnect before it still sets the cancel event (speech_stream.py:101), which only sherpa-onnx reads (sherpa_tts.py:94)'),
    ('POST /v1/audio/speech', 'stream_format sse (local models)', 'rejected', 'src/titanos/server.py:1824', 'stream_format sse', '400 unsupported_capability naming stream_format audio; the broker relays sse for cloud speech targets'),
    ('POST /v1/audio/speech', 'response_format pcm', 'implemented', 'src/titanos/speech_stream.py:58', 'audio/pcm', 'bare little-endian PCM16 at the model rate, named in X-Sample-Rate, X-Bits-Per-Sample and X-Channels (speech_stream.py:48), whole or streamed'),
    ('POST /v1/audio/speech', 'temperature / top_k / top_p / subtalker_temperature / subtalker_top_k / subtalker_top_p (qwen3-tts)', 'implemented', 'src/titanos/qwen_tts_backend.py:309', 'sampling_value', 'override the registration sampling defaults for this request; the gim_api TTS Super ranges (qwen_tts_backend.py:30), a value outside them -> 422 before inference (server.py:575)'),
    ('POST /v1/audio/speech', 'frame_budget (qwen3-tts)', 'implemented', 'src/titanos/qwen_tts_backend.py:279', 'frame_budget', 'the codec-frame cap of this request (12.5 frames a second) instead of the estimate from the text length; below 16 -> 422 (server.py:570), above the model max_frames a ValueError -> 400 (qwen_tts_backend.py:131)'),
    ('POST /v1/audio/speech', 'Qwen sampling fields or frame_budget (transformers, xtts, sherpa-onnx)', 'rejected', 'src/titanos/server.py:1802', 'unsupported_capability', '400 naming the fields before inference'),
    ('POST /v1/audio/speech', 'cancellation (disconnect), stream_format audio', 'implemented', 'src/titanos/speech_stream.py:135', 'cancel.set()', 'a disconnect before audio (is_disconnected polled every 0.25 s) or mid-body (the body generator is cancelled) sets the cancel event engine.infer holds; tests/check_speech_stream.py shows both on uvicorn'),
    ('POST /v1/audio/speech', 'cancellation (disconnect), whole body', 'ignored', 'src/titanos/server.py:1819', '/v1/audio/speech', 'without stream_format the route runs engine.infer to the end with no cancel event; POST /api/jobs task=speech cancels (server.py:2252, sherpa_tts.py:94)'),
    ('POST /v1/audio/speech', 'timeout', 'rejected', 'src/titanos/server.py:381', 'extra', 'no timeout field: 422'),
    # images and video
    ('POST /v1/images/generations', 'steps / n / seed / negative_prompt / guidance_scale', 'implemented', 'src/titanos/backends.py:489', 'steps', 'diffusers kwargs; seed seeds a torch.Generator'),
    ('POST /v1/images/generations', 'cancellation (disconnect)', 'implemented', 'src/titanos/backends.py:491', 'cancel', 'step callback under dispatch_cancellable with the default client_timeout deadline (api_plus.py:432)'),
    ('POST /v1/images/generations', 'timeout', 'rejected', 'src/titanos/server.py:381', 'extra', 'no timeout field: 422; the deadline is the process client_timeout'),
    ('POST /v1/images/edits', 'undeclared multipart fields', 'rejected', 'src/titanos/server.py:1964', 'unsupported_capability', '400 naming the unknown and supported fields'),
    ('POST /v1/images/edits', 'mask / mask_base64 / preserve_unmasked', 'implemented', 'src/titanos/backends.py:471', 'mask_image', 'diffusers and openvino-genai SD/SDXL inpainting only; preserve_unmasked must be a literal true/false with a mask (server.py:1972)'),
    ('POST /v1/images/edits', 'image_guidance_scale', 'implemented', 'src/titanos/backends.py:464', 'InstructPix2Pix', 'InstructPix2Pix only; 400 for other pipelines'),
    ('POST /v1/images/edits', 'strength', 'implemented', 'src/titanos/backends.py:519', 'strength', 'img2img strength for SD/SDXL edits'),
    ('POST /v1/videos/generations', 'cancellation (disconnect)', 'ignored', 'src/titanos/server.py:2114', '/v1/videos/generations', 'synchronous make_video with no cancel event; the job routes cancel (server.py:2127)'),
    ('POST /v1/videos', 'seconds', 'implemented', 'src/titanos/server.py:2300', 'seconds', 'converted to frames with fps on the OpenAI-shaped route'),
    ('POST /v1/videos', 'input_reference', 'implemented', 'src/titanos/backends.py:465', 'input_reference', 'image conditioning for compatible diffusers pipelines'),
    ('POST /v1/videos', 'pipeline_options without input_reference', 'rejected-in-job', 'src/titanos/backends.py:468', 'pipeline_options', 'ValueError -> 400 raised by diffuse() inside the job thread after the 202 (POST /v1/videos submits a video job, server.py:2127), so the job record shows status failed with the message (jobs.py:210, :177)'),
    ('POST /v1/videos', 'frames / fps', 'implemented', 'src/titanos/backends.py:512', 'num_frames', 'a pipeline producing fewer frames than requested fails explicitly'),
    ('POST /v1/videos', 'cancellation (DELETE /v1/videos/{identifier}, DELETE /api/jobs/{identifier})', 'implemented', 'src/titanos/server.py:2127', 'cancel', 'job event passed into make_video'),
    # embeddings, tensors, jobs
    ('POST /v1/embeddings', 'max_tokens / truncate / encoding_format', 'implemented', 'src/titanos/backends.py:415', 'truncate', 'truncation policy from request, model defaults or process setting; base64 output (embedding_output.py:8)'),
    ('POST /v1/embeddings', 'cancellation (disconnect)', 'ignored', 'src/titanos/server.py:1787', '/v1/embeddings', 'the local path runs engine.infer in a thread without a cancel event; the llama-cpp path returns 499 on disconnect (native.proxy)'),
    ('POST /v1/embeddings', 'llama-cpp GGUF embedding model', 'implemented', 'src/titanos/server.py:1791', 'native.proxy', 'a task=embeddings llama-cpp model is proxied to its owned llama-server started with --embeddings and the registered native.pooling (native.py:863 embedding_arguments; b10809 CPU binary, 2026-10-05); encoding_format reaches the server, user is named in X-Titanos-Ignored, dimensions is 400 (request_compat.py:73), max_tokens/truncate are 400 naming the field (openai_params.py:68); Ollama /api/embed and /api/embeddings share the handler and keep the X-Titanos-Ignored header through protocols.carry_ignored (their own rows)'),
    ('POST /v1/embeddings', 'dimensions', 'rejected', 'src/titanos/request_compat.py:72', 'dimensions', '400 unsupported_capability naming dimensions on every backend (no Matryoshka truncation locally; llama.cpp b10809 ignores the field); an explicit null is named in X-Titanos-Ignored'),
    ('POST /v1/embeddings', 'user', 'implemented', 'src/titanos/request_compat.py:37', 'EMBEDDINGS', 'accepted, stripped before the engine or llama-server and named in X-Titanos-Ignored (Embeddings._compat_kind, server.py:535)'),
    ('POST /api/infer', 'cancellation (disconnect)', 'ignored', 'src/titanos/server.py:1529', '/api/infer', 'synchronous; no cancel event or deadline'),
    ('POST /api/jobs', 'task discriminator (image, speech, video, music)', 'implemented', 'src/titanos/server.py:693', 'discriminator', 'AssetJob union; the OpenAPI requestBody is an inline anyOf'),
    ('POST /api/jobs', 'Idempotency-Key', 'implemented', 'src/titanos/server.py:2133', 'Idempotency-Key', '1..128 characters; a repeated key with the same fingerprint returns the existing job; the fingerprint hashes the task and the whole payload.model_dump() (server.py:2134), so a field the worker ignores still distinguishes two requests under one key: 409 idempotency_conflict (jobs.py:171-172)'),
    ('POST /api/jobs', 'cancellation (DELETE /api/jobs/{identifier})', 'implemented', 'src/titanos/jobs.py:281', 'cancel', 'sets the job event; status cancelled (jobs.py:210)'),
    ('POST /api/jobs', 'deadline / timeout', 'rejected', 'src/titanos/server.py:381', 'extra', 'no AssetJob member declares a deadline or timeout field: an unknown field is 422 Extra inputs are not permitted (Payload extra=forbid); jobs run without a deadline (jobs.py:159) and cancellation is the only bound'),
    ('POST /api/jobs', 'native progress evidence', 'implemented', 'src/titanos/jobs.py:187', 'observe_native', 'stage events bound to the job identity'),
    ('POST /api/jobs', 'model (all tasks)', 'implemented', 'src/titanos/server.py:2231', 'engine.validate', 'validated against the registry and the task before the 202: 404 model_not_found for an unknown model, 400 when the registered task differs (engine.py:36)'),
    ('POST /api/jobs', 'worker (image, speech, video)', 'implemented', 'src/titanos/engine.py:215', "data.get('worker')", 'the Payload field reaches engine.infer inside the job thread (image jobs through the api_plus run helper, api_plus.py:475): routed to the assigned replica worker (replicas.py:625); a single-process server refuses it with 400 (engine.py:208) recorded as status failed; task=music: worker is the shared music row'),
    ('POST /api/jobs', 'task=image / task=video: prompt / size / steps / seed / negative_prompt / guidance_scale', 'implemented', 'src/titanos/backends.py:489', 'steps', 'ImageJob is rebuilt as ImageRequest and runs image_result (server.py:2238), VideoJob as VideoRequest through make_video (server.py:2236): the same diffusers call as the sync routes, width/height from size, num_inference_steps from steps, negative_prompt and guidance_scale when given, seed as a torch.Generator (backends.py:506-510)'),
    ('POST /api/jobs', 'task=image: n', 'implemented', 'src/titanos/backends.py:517', 'num_images_per_prompt', 'num_images_per_prompt; one PNG artifact per image in the job result'),
    ('POST /api/jobs', 'task=image: response_format', 'ignored', 'src/titanos/server.py:2238', "'response_format':'url'", 'b64_json is schema-accepted on ImageJob and overwritten with url before image_result runs: the job result is always an artifact URL under GET /v1/files/{name}; the value still participates in the Idempotency-Key fingerprint (server.py:2134), so ignored means unread by the worker, not without observable effect'),
    ('POST /api/jobs', 'task=speech: input / voice', 'implemented', 'src/titanos/backends.py:438', 'voice', 'SpeechJob data runs the same engine.infer speech call as POST /v1/audio/speech (server.py:2252): xtts named reference, sherpa default or numeric speaker id (sherpa_tts.py:83), qwen3-tts configured alias (qwen_tts_backend.py:275); an unknown voice is a ValueError -> 400 raised inside the job thread (status failed)'),
    ('POST /api/jobs', 'task=speech: language (xtts, qwen3-tts)', 'implemented', 'src/titanos/backends.py:443', 'language', 'as on POST /v1/audio/speech: xtts language; qwen3-tts language map (qwen_tts_backend.py:280)'),
    ('POST /api/jobs', 'task=speech: language (transformers speech)', 'rejected-in-job', 'src/titanos/backends.py:449', 'language', 'the 400 of the sync route is raised inside the job thread after the 202, so the job record shows status failed with the message (jobs.py:210, :177)'),
    ('POST /api/jobs', 'task=speech: language (sherpa-onnx)', 'ignored', 'src/titanos/sherpa_tts.py:75', 'def infer', 'the sherpa backend never reads language; the value still participates in the Idempotency-Key fingerprint (server.py:2134), so ignored means unread by the worker, not without observable effect'),
    ('POST /api/jobs', 'task=speech: seed (qwen3-tts)', 'implemented', 'src/titanos/qwen_tts_backend.py:300', 'seed', 'the only speech backend that reads seed'),
    ('POST /api/jobs', 'task=speech: seed (transformers, xtts, sherpa-onnx)', 'ignored', 'src/titanos/backends.py:436', 'speak', 'declared on Speech (server.py:557), never read by speak() or the sherpa backend; the value still participates in the Idempotency-Key fingerprint (server.py:2134), so ignored means unread by the worker, not without observable effect'),
    ('POST /api/jobs', 'task=speech: speed', 'implemented', 'src/titanos/server.py:563', 'speed', '0.25..4.0 on SpeechJob through Speech; honoured by sherpa-onnx (sherpa_tts.py:95), a value other than 1.0 on transformers/xtts/qwen3-tts is a ValueError -> 400 inside the job (openai_params.py:100)'),
    ('POST /api/jobs', 'task=speech: response_format other than wav', 'rejected', 'src/titanos/server.py:653', 'response_format', 'Speech takes wav or pcm; the SpeechJob validator refuses pcm: 422 before the 202; the artifact is always WAV'),
    ('POST /api/jobs', 'task=speech: stream_format', 'rejected', 'src/titanos/server.py:653', 'stream_format', 'the SpeechJob validator refuses it: 422 "A speech job writes a WAV file" before the 202; streaming is a POST /v1/audio/speech option'),
    ('POST /api/jobs', 'task=speech: temperature / top_k / top_p / subtalker_temperature / subtalker_top_k / subtalker_top_p / frame_budget (qwen3-tts)', 'implemented', 'src/titanos/server.py:2232', 'speech_data', 'the job data carries them to engine.infer as POST /v1/audio/speech does: sampling (qwen_tts_backend.py:309) and the codec-frame cap (qwen_tts_backend.py:279); out-of-range sampling or a frame_budget below 16 -> 422 before the 202 (server.py:575, :570)'),
    ('POST /api/jobs', 'task=speech: frame_budget above the model max_frames', 'rejected-in-job', 'src/titanos/qwen_tts_backend.py:131', 'max_frames', 'the ValueError is raised inside the job thread after the 202, so the job record shows status failed with the message (jobs.py:210, :177)'),
    ('POST /api/jobs', 'task=speech: Qwen sampling fields or frame_budget (other backends, cloud stage)', 'rejected', 'src/titanos/server.py:1802', 'unsupported_capability', '400 before the 202 for a local model of another backend; a cloud stage builds its own upstream body and would drop them, so it refuses them too (server.py:2229)'),
    ('POST /api/jobs', 'task=video: frames / fps', 'implemented', 'src/titanos/backends.py:512', 'num_frames', 'num_frames from frames (a pipeline producing fewer frames than requested is a ValueError -> 400 inside the job thread, status failed); fps is the write_video frame rate (server.py:2080)'),
    ('POST /api/jobs', 'task=video: input_reference', 'implemented', 'src/titanos/backends.py:465', 'input_reference', 'image conditioning for compatible diffusers pipelines, as on POST /v1/videos'),
    ('POST /api/jobs', 'task=video: pipeline_options without input_reference', 'rejected-in-job', 'src/titanos/backends.py:468', 'pipeline_options', 'ValueError -> 400 raised by diffuse() inside the job thread after the 202, so the job record shows status failed with the message (jobs.py:210, :177)'),
    ('POST /api/jobs', 'task=video: n other than 1 / response_format other than url', 'rejected', 'src/titanos/server.py:616', 'Literal[1]', 'VideoRequest Literals n=1 and response_format=url (server.py:619) on VideoJob: 422 before the 202'),
    # extensions
    ('POST /v1/extensions/infer', 'timeout', 'implemented', 'src/titanos/api_plus.py:921', 'deadline', 'shared across composition stages; 504 deadline_exceeded'),
    ('POST /v1/extensions/infer', 'cancellation (disconnect, MCP cancel)', 'implemented', 'src/titanos/api_plus.py:917', 'dispatch_cancellable', 'one threading.Event across stages; shielded until the worker settles'),
    ('POST /v1/extensions/infer', 'unknown operation', 'rejected', 'src/titanos/api_plus.py:336', 'ExtensionRequest', 'discriminated union on operation: 422'),
    ('POST /v1/audio/transcriptions', 'undeclared multipart fields (temperature, include[], stream, chunking_strategy)', 'rejected', 'src/titanos/api_plus.py:1394', 'TRANSCRIPTION_FORM_FIELDS', '400 unsupported_capability naming the fields'),
    ('POST /v1/audio/transcriptions', 'response_format=text', 'implemented', 'src/titanos/api_plus.py:1436', 'PlainTextResponse', 'text/plain body'),
    ('POST /v1/audio/transcriptions', 'prompt', 'implemented', 'src/titanos/asr.py:44', 'prompt', 'Whisper prompt conditioning; supplied once and non-empty'),
    ('POST /v1/audio/transcriptions', 'timestamp_granularities[]', 'implemented', 'src/titanos/asr.py:26', 'timestamp_granularities', 'one granularity; requires response_format=verbose_json (asr.py:30)'),
    ('POST /v1/audio/transcriptions', 'language', 'implemented', 'src/titanos/asr.py:58', 'language', 'English-only checkpoints accept en hints only'),
    ('POST /v1/audio/transcriptions', 'vad_enabled / vad_model / vad_options', 'implemented', 'src/titanos/audio_gating.py:31', 'vad_enabled', 'omitted inherits the model default; false disables; enabled requires a registered vad_model (audio_gating.py:22)'),
    ('POST /v1/audio/transcriptions', 'cancellation (disconnect)', 'implemented', 'src/titanos/asr.py:82', 'StoppingCriteria', 'cooperative stopping criterion for seq2seq generation'),
    ('POST /v1/audio/translations', 'undeclared multipart fields', 'rejected', 'src/titanos/api_plus.py:1396', 'TRANSLATION_FORM_FIELDS', '400 unsupported_capability'),
    ('POST /v1/audio/translations', 'include_provenance', 'implemented', 'src/titanos/api_plus.py:1466', 'include_provenance', 'typed AudioTranslationResult; incompatible with response_format=text'),
    ('POST /v1/translations', 'include_provenance', 'implemented', 'src/titanos/api_plus.py:506', 'include_provenance', 'typed TextTranslationResult with loaded NMT identity'),
    ('POST /v1/translations', 'source_language / target_language on fixed-pair checkpoints', 'rejected', 'src/titanos/extra_backends.py:151', 'target_language', 'a fixed-pair model refuses other directions'),
    ('POST /v1/translations', 'max_tokens / temperature / top_p', 'implemented', 'src/titanos/extra_backends.py:87', 'max_tokens', 'request overrides the allowed generation keys within TITANOS_MAX_TOKENS'),
    ('POST /v1/images/ocr', 'prompt (TrOCR)', 'rejected', 'src/titanos/extra_backends.py:194', 'prompt', 'prompt is refused for TrOCR checkpoints; tesseract controls require the tesseract backend (api_plus.py:682)'),
    ('POST /v1/images/detect', 'threshold / nms / nms_iou', 'implemented', 'src/titanos/extra_backends.py:261', 'nms', 'post-processing threshold and optional non-maximum suppression'),
    ('POST /v1/sentiment', 'decoding=label_choices outside transformers label-v2/v3', 'rejected', 'src/titanos/api_plus.py:779', 'label_choices', '400'),
    ('POST /v1/sentiment', 'decoding=native_json_schema outside llama-cpp', 'rejected', 'src/titanos/api_plus.py:783', 'native_json_schema', '400'),
    ('POST /v1/retrieval/answer', 'stream', 'rejected', 'src/titanos/retrieval_workflow_contract.py:43', 'stream', 'Literal[False]: 422'),
    ('POST /v1/rerank', 'top_n', 'implemented', 'src/titanos/api_plus.py:612', 'top_n', 'results truncated after scoring with original indices preserved'),
    ('POST /v1/audio/vad', 'unknown VAD controls', 'rejected', 'src/titanos/api_plus.py:40', 'extra="forbid"', 'VADRequest is an ExtensionPayload (api_plus.py:242) with extra=forbid: an unknown request control is a 422 schema error before the handler runs; the handler forwards only the six declared controls (api_plus.py:1027) plus the audio bytes and the worker, so the Unsupported VAD controls ValueError -> 400 guard (vad.py:100) is an internal guard unreachable from this route'),
    ('POST /v1/audio/vad', 'sample_rate / threshold / min_speech_ms / min_silence_ms / speech_pad_ms', 'implemented', 'src/titanos/vad.py:20', 'threshold', 'defaults from settings; echoed as parameters'),
    ('POST /v1/audio/vad', 'return_frames', 'implemented', 'src/titanos/vad.py:127', 'return_frames', 'per-frame scores on request'),
    ('POST /api/models/capabilities', 'modes.model_sharding / modes.tensor_parallel available', 'declared-only', 'src/titanos/capabilities.py:194', 'tensor_parallel', 'declared with available None and reason runtime_unverified (capabilities.py:15): no passive runtime inspection exists; the tensor-parallel load path itself exists (engine.py:115)'),
    ('POST /api/models/capabilities', 'music.controls', 'implemented', 'src/titanos/capabilities.py:81', 'controls', 'the control list is the bundle manifest capabilities plus the constants seed/guidance_scale/steps (capabilities.py:76); every named control has a code path under POST /v1/music/generations (mode, lyrics, vocal_languages, duration_min/max, bpm, key_scale, seed, guidance_scale, steps); reference_audio is declared False by the packaged provider (providers.py:700) and refused in the job; configuration_error when metadata is unavailable'),
]

PARAM = '{}'


@contextlib.contextmanager
def hermetic_environment(state_dir):
    """Make the in-process app build hermetic while it runs: no TITANOS_* environment leaks into the schema, TITANOS_HOME is a
    fresh scratch folder (so a populated broker.json, keys.json or vault index of the caller's home cannot change the output),
    the role is standalone, and the runtime catalog's hardware probes are stubbed so the default route surface registers
    without any device call. Everything is restored on exit, so other tests importing this module in the same interpreter see
    the original environment and module attributes."""
    saved = {key: value for key, value in os.environ.items() if key.startswith('TITANOS_')}
    for key in saved:
        del os.environ[key]
    os.environ['TITANOS_HOME'] = str(Path(state_dir) / 'home')
    os.environ['TITANOS_ROLE'] = 'standalone'
    if str(ROOT / 'src') not in sys.path:
        sys.path.insert(0, str(ROOT / 'src'))
    import titanos.runtime_catalog as runtime_catalog
    import titanos.devices as devices
    originals = (runtime_catalog.compatible_profiles, devices.visibility_constraints)
    runtime_catalog.compatible_profiles = lambda catalog: []
    devices.visibility_constraints = lambda environ=None: {}
    try:
        yield
    finally:
        runtime_catalog.compatible_profiles, devices.visibility_constraints = originals
        os.environ.pop('TITANOS_HOME', None)
        os.environ.pop('TITANOS_ROLE', None)
        os.environ.update(saved)


def build_app(state_dir):
    with hermetic_environment(state_dir):
        from titanos.config import Config
        from titanos.server import create_app
        outputs = Path(state_dir) / 'state' / 'outputs'
        outputs.mkdir(parents=True, exist_ok=True)
        return create_app(Config(), output_dir=outputs)


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def unwrap(function):
    """The plain Python function behind pydantic validate_call / functools wrappers (MCP resources and prompts are wrapped)."""
    seen = set()
    while id(function) not in seen:
        seen.add(id(function))
        inner = getattr(function, 'raw_function', None) or getattr(function, '__wrapped__', None)
        if inner is None or not callable(inner):
            break
        function = inner
    return function


def cite(function):
    function = unwrap(function)
    return f'{relative(inspect.getsourcefile(function))}:{function.__code__.co_firstlineno}'


def type_name(annotation, aliases):
    for name, value in aliases.items():
        # FastAPI 0.163 hands the body parameter the Annotated alias's bare union (the discriminator moves into
        # field_info), older releases the alias itself; both name the alias.
        if value is annotation or (typing.get_origin(value) is typing.Annotated and typing.get_args(value)[0] == annotation):
            return name
    if inspect.isclass(annotation):
        module = annotation.__module__.split('.')[-1] if annotation.__module__.startswith('titanos.') else annotation.__module__
        return f'{module}.{annotation.__qualname__}'
    origin = typing.get_origin(annotation)
    if origin is typing.Annotated:
        return type_name(typing.get_args(annotation)[0], aliases)
    if origin is typing.Union or str(origin) == "<class 'types.UnionType'>":
        return ' | '.join(type_name(a, aliases) for a in typing.get_args(annotation))
    if origin is typing.Literal:
        return 'Literal[' + ', '.join(repr(a) for a in typing.get_args(annotation)) + ']'
    if origin is not None:
        return f'{getattr(origin, "__name__", str(origin))}[' + ', '.join(type_name(a, aliases) for a in typing.get_args(annotation)) + ']'
    return str(annotation).replace('typing.', '')


def model_classes(annotation):
    """The pydantic model classes a body annotation names, through Annotated and Union (a discriminated union yields its members)."""
    from pydantic import BaseModel
    if inspect.isclass(annotation):
        return [annotation] if issubclass(annotation, BaseModel) else []
    origin = typing.get_origin(annotation)
    if origin is typing.Annotated:
        return model_classes(typing.get_args(annotation)[0])
    if origin is typing.Union or str(origin) == "<class 'types.UnionType'>":
        return [c for a in typing.get_args(annotation) for c in model_classes(a)]
    return []


def forms(node):
    """Every concrete spelling of a route expression with each non-literal segment replaced by PARAM (a path parameter)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return [a + b for a in forms(node.left) for b in forms(node.right)]
    if isinstance(node, ast.IfExp):
        return forms(node.body) + forms(node.orelse)
    if isinstance(node, ast.JoinedStr):
        out = ['']
        for part in node.values:
            piece = [part.value] if isinstance(part, ast.Constant) else [PARAM]
            out = [a + b for a in out for b in piece]
        return out
    return [PARAM]


def normalize(template):
    return re.sub(r'\{[^}]*\}', PARAM, template)


def call_name(node):
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def parse_source(function):
    return ast.parse(textwrap.dedent(inspect.getsource(unwrap(function))))


def match_operations(method, node, ops_by_form):
    matched = []
    for form in forms(node):
        for key in ops_by_form.get((method, form), ()):
            matched.append(key)
    return matched


def route_scope(method, route_path):
    """(scope, scope_by_action, broker_surfaces) of one route: the broker.scopes.ROUTE_SCOPES entry (fail closed to admin when
    unlisted), the per-action scopes of the surfaces a ':path' route dispatches on its last ':' (the Gemini actions, which
    broker.scopes.required_scope resolves through broker.surfaces before the table), and the names of the broker surfaces
    (broker/surfaces.py) the route serves. A direct surface whose scope differs from ROUTE_SCOPES stops the build."""
    from titanos.broker import scopes as scopes_module
    from titanos.broker.surfaces import SURFACES
    template = scopes_module.normalize_template(route_path)
    listed = scopes_module.scope_for_template(method, route_path)
    scope = listed if listed is not None else f'{scopes_module.ADMIN} (not in ROUTE_SCOPES: fail closed)'
    names, by_action = [], {}
    action_prefix = None
    tail = re.search(r'\{([A-Za-z_][A-Za-z0-9_]*):path\}$', route_path)
    if tail:
        action_prefix = route_path[:tail.start()]
    for surface in SURFACES.values():
        if surface.method != method:
            continue
        if scopes_module.normalize_template(surface.path) == template:
            names.append(surface.name)
            if listed is not None and surface.scope != listed:
                raise SystemExit(f'surface {surface.name} scope {surface.scope} differs from ROUTE_SCOPES {method} {route_path} {listed}')
            continue
        if action_prefix is not None:
            found = re.fullmatch(re.escape(action_prefix) + r'\{[A-Za-z_][A-Za-z0-9_]*:path\}:([A-Za-z]+)', surface.path)
            if found:
                names.append(surface.name)
                by_action[found.group(1)] = surface.scope
    return scope, dict(sorted(by_action.items())), sorted(names)


def mount_scope(path):
    """The scope broker.scopes.required_scope demands on a mount (the ('*', path) rows of ROUTE_SCOPES)."""
    from titanos.broker import scopes as scopes_module
    return scopes_module.required_scope('POST', path)


def credential_headers(path_format):
    """The request headers the Authenticator reads as a credential on this path (broker/auth.py VENDOR_HEADERS)."""
    from titanos.broker.auth import VENDOR_HEADERS
    names = ['Authorization: Bearer']
    for header, prefixes in sorted(VENDOR_HEADERS.items()):
        if any(path_format == p or path_format.startswith(p + '/') for p in prefixes):
            names.append(header.decode('ascii'))
    return names


def enumerate_operations(app, public_routes):
    """One row per app.openapi() operation joined to its APIRoute; plus the routes the schema hides."""
    from fastapi.routing import APIRoute
    from starlette.routing import Mount, Route
    import titanos.server as server_module
    import titanos.api_plus as api_plus_module
    aliases = {'server.AssetJob': server_module.AssetJob, 'api_plus.ExtensionRequest': api_plus_module.ExtensionRequest}
    schema = app.openapi()
    declared = {(method.upper(), path) for path, item in schema['paths'].items() for method in item}
    rows, unlisted, seen = [], [], set()
    for route in app.routes:
        if isinstance(route, APIRoute):
            endpoint = route.endpoint
            source = inspect.getsource(endpoint)
            for method in sorted(route.methods):
                key = (method, route.path_format)
                if not route.include_in_schema:
                    prefix = route.path_format.split('{')[0]
                    hidden_scope = route_scope(method, route.path)[0]
                    if route.path_format in public_routes:
                        auth = 'public'
                    elif '{' in route.path_format and any(p.startswith(prefix) for p in public_routes):
                        auth = f'public for the exact PUBLIC_ROUTES paths under this template, otherwise scoped ({hidden_scope})'
                    else:
                        auth = f'scoped ({hidden_scope})'
                    unlisted.append({'method': method, 'path': route.path_format, 'handler': cite(endpoint), 'handler_name': endpoint.__name__,
                                     'auth': auth, 'reason': 'include_in_schema=False'})
                    continue
                if key not in declared:
                    raise SystemExit(f'route {method} {route.path_format} is not in app.openapi()')
                seen.add(key)
                body_params = list(route.dependant.body_params)
                multipart = [getattr(p, 'alias', None) or p.name for p in body_params if type(p.field_info).__name__ in ('Form', 'File')]
                json_params = [p for p in body_params if type(p.field_info).__name__ not in ('Form', 'File')]
                json_models = [type_name(p.field_info.annotation, aliases) for p in json_params]
                manual = [m for m in ('request.json()', 'request.form()') if m in source]
                if multipart:
                    request = {'kind': 'multipart', 'model': None, 'fields': multipart}
                elif json_models:
                    # declared field names of the model (the union of the members for a discriminated union); DERIVED_FIELD_SEMANTICS keys on them
                    fields = sorted({name for p in json_params for c in model_classes(p.field_info.annotation) for name in c.model_fields})
                    request = {'kind': 'json', 'model': ', '.join(json_models), 'fields': fields}
                elif manual:
                    request = {'kind': 'manual', 'model': None, 'fields': None, 'read_by': manual, 'note': 'body read by the handler; no OpenAPI requestBody'}
                else:
                    request = {'kind': 'none', 'model': None, 'fields': None}
                response_model = route.response_model
                op_key = f'{method} {route.path_format}'
                response = f'typed: {type_name(response_model, aliases)}' if response_model is not None else RESPONSE_SHAPES.get(op_key, 'untyped JSON (dict)')
                public = route.path_format in public_routes
                scope, scope_by_action, surfaces = route_scope(method, route.path)
                rows.append({'method': method, 'path': route.path_format, 'route_path': route.path, 'handler': cite(endpoint), 'handler_name': endpoint.__name__,
                             'status_code': route.status_code or 200, 'auth': 'public' if public else 'scoped',
                             'scope': None if public else scope, 'scope_by_action': {} if public else scope_by_action,
                             'credential_headers': [] if public else credential_headers(route.path_format), 'broker_surfaces': surfaces,
                             'registry_gate': 'not_tracked' if route.path_format in public_routes else ('bypasses_unavailable' if op_key == 'POST /api/models/reload' else 'tracked'),
                             'request': request, 'path_params': [p.name for p in route.dependant.path_params], 'query_params': [p.name for p in route.dependant.query_params],
                             'response': response, 'family': next((f for f, paths in FAMILY_PATHS.items() if route.path_format in paths), 'native')})
        elif isinstance(route, Route):
            unlisted.append({'method': ','.join(sorted(route.methods or [])), 'path': route.path, 'handler': 'fastapi.applications.FastAPI.setup (framework route)', 'handler_name': 'openapi',
                             'auth': 'public' if route.path in public_routes else 'gated', 'reason': 'starlette Route registered by FastAPI, not an APIRoute (HEAD is answered 405 by the Guard: PUBLIC_METHODS is GET only)'})
        elif isinstance(route, Mount):
            unlisted.append({'method': 'ANY', 'path': route.path, 'handler': 'src/titanos/server.py (app.mount of create_mcp(...).streamable_http_app; src/titanos/mcp_server.py create_mcp)', 'handler_name': 'mcp',
                             'auth': (f"scoped ({mount_scope(route.path)} on the transport; each tool call then carries the principal of its own HTTP message "
                                      'and is checked against the scope of the operation it calls; keyless cross-site check on the transport; the registry gate tracks '
                                      'each inner tool operation, not the transport)'),
                             'reason': 'MCP streamable HTTP mount; JSON-RPC, not an OpenAPI operation'})
    missing = declared - seen
    if missing:
        raise SystemExit(f'app.openapi() operations without an APIRoute: {sorted(missing)}')
    rows.sort(key=lambda r: (r['path'], r['method']))
    unlisted.sort(key=lambda r: (r['path'], r['method']))
    return schema, rows, unlisted


def operation_index(rows):
    ops_by_form = {}
    for row in rows:
        ops_by_form.setdefault((row['method'], normalize(row['path'])), []).append(f"{row['method']} {row['path']}")
    return ops_by_form


def enumerate_mcp(mcp, ops_by_form, all_ops, unlisted):
    """Tools, resources, templates and prompts with the operations each one calls, read from its own source. A call to a route
    outside the OpenAPI schema (titanos://schema reads /openapi.json) is reported under unlisted_routes, never as an operation."""
    tools = {name: tool for name, tool in mcp._tool_manager._tools.items()}
    index = dict(ops_by_form)
    for route in unlisted:
        for method in route['method'].split(','):
            index.setdefault((method, normalize(route['path'])), []).append(f"{method} {route['path']}")

    def calls_of(function, depth=0):
        found = []
        for node in ast.walk(parse_source(function)):
            if not isinstance(node, ast.Call):
                continue
            name = call_name(node)
            if name in ('json_request', 'request') and len(node.args) >= 2 and isinstance(node.args[0], ast.Constant):
                found += match_operations(node.args[0].value.upper(), node.args[1], index)
            elif name in ('infer', 'inference') and node.args:
                found += match_operations('POST', node.args[0], index)
            elif name in MCP_HELPERS:
                found += [f'{m} {p}' for m, p in MCP_HELPERS[name]]
            elif name in tools and depth < 3 and tools[name].fn is not function:
                found += calls_of(tools[name].fn, depth + 1)
        return sorted(set(found))

    def split(function):
        found = calls_of(function)
        return {'operations': [k for k in found if k in all_ops], 'unlisted_routes': [k for k in found if k not in all_ops]}

    tool_rows = []
    for name in sorted(tools):
        tool = tools[name]
        annotations = tool.annotations.model_dump(exclude_none=True) if tool.annotations is not None else {}
        tool_rows.append({'name': name, 'source': cite(tool.fn), **split(tool.fn), 'annotations': {k: v for k, v in sorted(annotations.items())},
                          'required_arguments': sorted(tool.parameters.get('required', [])), 'has_output_schema': tool.output_schema is not None})
    resources = [{'uri': r.uri if isinstance(r.uri, str) else str(r.uri), 'name': r.name, 'mime_type': r.mime_type, 'source': cite(r.fn), **split(r.fn)}
                 for r in mcp._resource_manager._resources.values()]
    templates = [{'uri_template': t.uri_template, 'name': t.name, 'mime_type': t.mime_type, 'source': cite(t.fn), **split(t.fn)}
                 for t in mcp._resource_manager._templates.values()]
    prompts = [{'name': p.name, 'source': cite(p.fn), 'arguments': [{'name': a.name, 'required': bool(a.required)} for a in (p.arguments or [])]}
               for p in mcp._prompt_manager._prompts.values()]
    return (tool_rows, sorted(resources, key=lambda r: r['uri']), sorted(templates, key=lambda t: t['uri_template']), sorted(prompts, key=lambda p: p['name']))


def enumerate_client(ops_by_form):
    """Public methods of titanos.client.Client with the operations their bodies call (one level of self.method() indirection)."""
    from titanos import client as client_module
    tree = parse_source(client_module.Client)
    class_node = tree.body[0]
    methods = {node.name: node for node in class_node.body if isinstance(node, ast.FunctionDef)}
    direct = {}
    for name, node in methods.items():
        found, calls = [], []
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Call):
                continue
            attr = call_name(sub)
            is_self = isinstance(sub.func, ast.Attribute) and isinstance(sub.func.value, ast.Name) and sub.func.value.id == 'self'
            if not is_self:
                continue
            if attr in ('_json', 'request') and len(sub.args) >= 2 and isinstance(sub.args[0], ast.Constant):
                found += match_operations(sub.args[0].value.upper(), sub.args[1], ops_by_form)
            elif attr in ('_stream', '_audio_upload', '_sse_events', '_byte_stream') and sub.args:
                found += match_operations('POST', sub.args[0], ops_by_form)
            elif attr in methods and attr != name:
                calls.append(attr)
        direct[name] = (found, calls)
    resolved = {}
    for name in methods:
        found, calls = direct[name]
        ops = set(found) | set(CLIENT_OVERRIDES.get(name, []))
        for other in calls:
            ops |= set(direct[other][0]) | set(CLIENT_OVERRIDES.get(other, []))
        resolved[name] = sorted(ops)
    first_line = tree.body[0].lineno
    base = inspect.getsourcelines(client_module.Client)[1]
    rows = [{'name': name, 'source': f"{relative(client_module.__file__)}:{base + methods[name].lineno - first_line}", 'operations': resolved[name]}
            for name in sorted(methods) if not name.startswith('_')]
    return rows


def resolve_cli(client_rows, all_ops, mcp_tools):
    by_name = {row['name']: row['operations'] for row in client_rows}
    bridged = sorted({op for tool in mcp_tools for op in tool['operations']})
    rows = []
    for name in sorted(CLI_SUBCOMMANDS):
        entry = CLI_SUBCOMMANDS[name]
        kinds = [k for k in ('local', 'bridge') if k in entry] + (['client/routes'] if entry.get('client') or entry.get('routes') else [])
        if len(kinds) != 1:
            raise SystemExit(f'CLI table entry {name} must be exactly one of local, bridge or client/routes: {kinds}')
        reachable = set()
        for method in entry.get('client', []):
            if method not in by_name:
                raise SystemExit(f'CLI table names unknown Client method {method}')
            reachable |= set(by_name[method])
        if 'bridge' in entry:
            # the stdio MCP bridge forwards every MCP tool call to the server: joined to the union of the MCP tool operations
            ops = set(bridged)
        elif 'routes' in entry:
            # explicit routes: the client_request helper, or the subset a Client method call actually reaches (Client.models(loaded=False))
            ops = set(entry['routes'])
            if entry.get('client') and not ops <= reachable:
                raise SystemExit(f'CLI table restricts {name} to routes its Client methods do not reach: {sorted(ops - reachable)}')
        else:
            ops = reachable
        unknown = ops - all_ops
        if unknown:
            raise SystemExit(f'CLI table names unknown operations {sorted(unknown)} for {name}')
        rows.append({'name': name, 'operations': sorted(ops), 'client_methods': sorted(entry.get('client', [])), 'local': entry.get('local'),
                     'bridge': entry.get('bridge'), 'note': entry.get('note')})
    return rows


def deviation_index():
    data = json.loads((ROOT / 'docs' / 'protocol_deviations.json').read_text(encoding='utf-8'))
    index = {}
    for row in data['rows']:
        if row['adapter'] == 'option-parity':
            continue
        index.setdefault(normalize(row['route']), []).append(row['adapter'])
    return index


def assemble(state_dir):
    app = build_app(state_dir)
    import titanos
    from titanos import server as server_module
    public_routes = sorted(server_module.PUBLIC_ROUTES)
    schema, rows, unlisted = enumerate_operations(app, set(public_routes))
    ops_by_form = operation_index(rows)
    all_ops = {f"{r['method']} {r['path']}" for r in rows}
    for family, paths in FAMILY_PATHS.items():
        stray = {p for p in paths if not any(r['path'] == p for r in rows)}
        if stray:
            raise SystemExit(f'FAMILY_PATHS[{family}] names paths that are not operations: {sorted(stray)}')
    broker_paths = {r['path'] for r in rows if r['path'].startswith(BROKER_PREFIXES)}
    if broker_paths != FAMILY_PATHS['broker']:
        raise SystemExit(f"FAMILY_PATHS[broker] differs from the operations under {BROKER_PREFIXES}: missing {sorted(broker_paths - FAMILY_PATHS['broker'])}")
    for path in ROLE_ONLY_PUBLIC:
        if path in public_routes or not any(r['path'] == path and r['auth'] == 'scoped' for r in rows):
            raise SystemExit(f'ROLE_ONLY_PUBLIC {path} must be a scoped (not public) operation of this standalone build')
    for op in RESPONSE_SHAPES:
        if op not in all_ops:
            raise SystemExit(f'RESPONSE_SHAPES names an unknown operation {op}')
    tools, resources, templates, prompts = enumerate_mcp(app.state.mcp, ops_by_form, all_ops, unlisted)
    client_rows = enumerate_client(ops_by_form)
    cli_rows = resolve_cli(client_rows, all_ops, tools)
    deviations = deviation_index()
    for operation in OPERATION_DOC_GAPS:
        if operation not in all_ops:
            raise SystemExit(f'OPERATION_DOC_GAPS names an unknown operation {operation}')
    semantics_by_op = {}
    seen_controls = set()

    def add_row(operation, control, status, citation, identifier, note, origin, shared_from=None):
        if operation not in all_ops:
            raise SystemExit(f'semantics ({origin}) name an unknown operation {operation}')
        if status not in STATUSES:
            raise SystemExit(f'semantics status {status!r} is not one of {STATUSES}')
        if (operation, control) in seen_controls:
            raise SystemExit(f'semantics ({origin}) repeat {operation} / {control}')
        seen_controls.add((operation, control))
        semantics_by_op.setdefault(operation, []).append({'control': control, 'status': status, 'citation': citation, 'identifier': identifier,
                                                          'note': note, 'origin': origin, 'shared_from': shared_from})

    for operation, control, status, citation, identifier, note in SEMANTICS:
        add_row(operation, control, status, citation, identifier, note, 'table')
    declared_fields = {f"{r['method']} {r['path']}": set(r['request']['fields'] or []) for r in rows if r['request']['kind'] == 'json'}
    for field, status, citation, identifier, note in DERIVED_FIELD_SEMANTICS:
        targets = sorted(op for op, names in declared_fields.items() if field in names)
        if not targets:
            raise SystemExit(f'DERIVED_FIELD_SEMANTICS field {field} is declared by no JSON request model')
        for op in targets:
            add_row(op, field, status, citation, identifier, note, 'derived')
    for target, qualifier, source, note in SHARED_SEMANTICS:
        if source not in all_ops or target not in all_ops:
            raise SystemExit(f'SHARED_SEMANTICS names an unknown operation: {target} <- {source}')
        if not semantics_by_op.get(source):
            raise SystemExit(f'SHARED_SEMANTICS source {source} has no semantics rows')
        for s in list(semantics_by_op[source]):
            add_row(target, f'{qualifier}: {s["control"]}', s['status'], s['citation'], s['identifier'], s['note'], 'shared', source)
    joins = {op: {'mcp_tools': [], 'mcp_resources': [], 'cli': [], 'client': []} for op in all_ops}
    for tool in tools:
        for op in tool['operations']:
            joins[op]['mcp_tools'].append(tool['name'])
    for resource in resources + templates:
        for op in resource['operations']:
            joins[op]['mcp_resources'].append(resource.get('uri') or resource.get('uri_template'))
    for row in cli_rows:
        for op in row['operations']:
            joins[op]['cli'].append(row['name'])
    for row in client_rows:
        for op in row['operations']:
            joins[op]['client'].append(row['name'])
    for row in rows:
        op = f"{row['method']} {row['path']}"
        row.update({k: sorted(set(v)) for k, v in joins[op].items()})
        adapters = deviations.get(normalize(row['path']), [])
        row['deviation_adapters'] = sorted(set(adapters))
        row['deviation_rows'] = len(adapters)
        row['semantics'] = sorted(semantics_by_op.get(op, []), key=lambda s: s['control'])
        row['doc_gaps'] = [{'citation': citation, 'identifier': identifier, 'note': note} for citation, identifier, note in OPERATION_DOC_GAPS.get(op, [])]
        row['shared_semantics_from'] = [{'operation': source, 'qualifier': qualifier, 'note': note} for target, qualifier, source, note in SHARED_SEMANTICS if target == op]
        row['shared_semantics_to'] = [{'operation': target, 'qualifier': qualifier} for target, qualifier, source, note in SHARED_SEMANTICS if source == op]
    status_counts = Counter(s['status'] for r in rows for s in r['semantics'])
    origin_counts = Counter(s['origin'] for r in rows for s in r['semantics'])
    counts = {'operations': len(rows), 'paths': len(schema['paths']), 'methods': dict(sorted(Counter(r['method'] for r in rows).items())),
              'families': dict(sorted(Counter(r['family'] for r in rows).items())), 'unlisted_routes': len(unlisted),
              'auth': dict(sorted(Counter(r['auth'] for r in rows).items())), 'scopes': dict(sorted(Counter(r['scope'] for r in rows if r['scope']).items())),
              'public_routes': len(public_routes), 'broker_surface_operations': sum(1 for r in rows if r['broker_surfaces']),
              'mcp_tools': len(tools), 'mcp_resources': len(resources), 'mcp_resource_templates': len(templates), 'mcp_prompts': len(prompts),
              'cli_subcommands': len(cli_rows), 'client_methods': len(client_rows), 'semantics': sum(status_counts.values()),
              'semantics_by_origin': {o: origin_counts.get(o, 0) for o in ORIGIN_MEANING},
              'semantics_by_status': {s: status_counts.get(s, 0) for s in STATUSES},
              'operations_with_semantics': sum(1 for r in rows if r['semantics']), 'doc_gaps': sum(len(r['doc_gaps']) for r in rows),
              'operations_without_mcp_tool': sorted(op for op in all_ops if not joins[op]['mcp_tools']),
              'operations_without_cli': sorted(op for op in all_ops if not joins[op]['cli']),
              'operations_without_client_method': sorted(op for op in all_ops if not joins[op]['client'])}
    return {'schema_version': 1,
            'basis': ('Generated by tools/build_route_contract_matrix.py from app.openapi() and app.routes of create_app(Config()) (hardware probes of the '
                      'runtime catalog stubbed), the MCP registry of create_mcp(), the source of each MCP tool/resource and Client method, and the explicit '
                      'family, CLI and semantics tables in the generator (semantics rows of origin table, derived from the declared request model fields, '
                      'or shared from the operation whose handler another operation runs); deviation adapters joined from docs/protocol_deviations.json; the '
                      'required scope of every operation from broker.scopes.ROUTE_SCOPES and the broker surfaces from broker/surfaces.py. The app is built in '
                      'standalone role under a fresh temporary TITANOS_HOME, so a populated broker.json, keys.json or vault index of the caller cannot change it. '
                      'Semantics rows describe the local handlers: a request BrokerDispatch routes to a cloud or router target (operations with broker_surfaces) is '
                      'answered before FastAPI parses it, with the semantics recorded in the broker-* sections of docs/PROTOCOL_DEVIATIONS.md and the cells of '
                      'docs/BROKER_MATRIX.md. Deterministic: sorted rows and keys, no timestamps; tests/check_route_contract_matrix.py asserts the rebuild is a '
                      'fixed point and that every semantics citation resolves. References are repository-relative.'),
            'source_version': titanos.__version__, 'openapi_version': schema['openapi'],
            'vocabulary': {'semantics': STATUS_MEANING, 'semantics_origin': ORIGIN_MEANING, 'auth': AUTH_MEANING, 'scope_rules': SCOPE_RULES, 'families': FAMILY_MEANING,
                           'registry_gate': {'tracked': 'answers 503 registry_reloading/registry_unavailable while the registry reloads or is unavailable',
                                             'not_tracked': 'public route: never answered by the registry gate', 'bypasses_unavailable': 'answered while unavailable (still 503 while reloading)'}},
            'counts': counts, 'public_routes': public_routes, 'role_only_public_routes': ROLE_ONLY_PUBLIC, 'operations': rows, 'unlisted_routes': unlisted,
            'mcp_tools': tools, 'mcp_resources': resources, 'mcp_resource_templates': templates, 'mcp_prompts': prompts,
            'cli_subcommands': cli_rows, 'client_methods': client_rows}


def source_head():
    try:
        return subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True, timeout=30).stdout.strip() or 'unknown'
    except (OSError, subprocess.SubprocessError):
        return 'unknown'


def cell(text):
    return str(text if text is not None else '').replace('|', '\\|').replace('\n', ' ').strip() or '-'


def auth_cell(r):
    if r['auth'] != 'scoped':
        return r['auth']
    by_action = '; '.join(f'{action}: `{scope}`' for action, scope in r['scope_by_action'].items())
    return f"scoped: `{r['scope']}`" + (f' ({by_action})' if by_action else '')


def render_md(data, head):
    c = data['counts']
    L = ['# Route-to-typed-contract matrix', '',
         f"Basis: generated by `tools/build_route_contract_matrix.py` from the in-process OpenAPI schema (`app.openapi()`), the MCP registry (`create_mcp`) and the source under `src/titanos` at source HEAD `{head}` (Titanos {data['source_version']}, OpenAPI {data['openapi_version']}); the machine-readable form is `docs/route_contract_matrix.json`. Rebuild with `python tools/build_route_contract_matrix.py`; do not edit by hand. `tests/check_route_contract_matrix.py` asserts the rebuild is a fixed point and that every semantics citation resolves. This records what the source declares and implements; it is not execution evidence (see `docs/CAPABILITY_MATRIX.md`).", '',
         f"Counts: {c['operations']} operations over {c['paths']} paths ({', '.join(f'{k} {v}' for k, v in c['methods'].items())}); families: {', '.join(f'{k} {v}' for k, v in c['families'].items())}; {c['unlisted_routes']} routes outside the schema; MCP {c['mcp_tools']} tools, {c['mcp_resources']} resources, {c['mcp_resource_templates']} resource templates, {c['mcp_prompts']} prompts; {c['cli_subcommands']} CLI subcommands; {c['client_methods']} Python client methods; {c['semantics']} semantics rows over {c['operations_with_semantics']} operations: " + ', '.join(f"{v} {k}" for k, v in c['semantics_by_status'].items()) + '.', '',
         '## Vocabulary', '', 'Semantics classes:', '']
    L += [f"- `{k}`: {v}" for k, v in data['vocabulary']['semantics'].items()]
    L += ['', 'Semantics origins (the `origin` of every row in the JSON):', ''] + [f"- `{k}`: {v}" for k, v in data['vocabulary']['semantics_origin'].items()]
    L += ['', 'Auth classes (src/titanos/server.py Guard with src/titanos/broker/auth.py):', ''] + [f"- `{k}`: {v}" for k, v in data['vocabulary']['auth'].items()]
    L += ['', f"Scope rules: {data['vocabulary']['scope_rules']}. The Auth column names the required scope of each scoped operation; operations by scope: "
          + ', '.join(f'`{k}` {v}' for k, v in c['scopes'].items()) + '.']
    L += ['', 'Families:', ''] + [f"- `{k}`: {v}" for k, v in data['vocabulary']['families'].items()]
    L += ['', f"Public routes ({c['public_routes']} exact paths, `server.PUBLIC_ROUTES`): " + ', '.join(f'`{p}`' for p in data['public_routes']), '']
    L += [f"Public in another role only (a scoped operation of this standalone build): `{p}`: {note}" for p, note in data['role_only_public_routes'].items()]
    L += ['',
          'Handler citations are `file:line` of the route decorator (the `def` follows). Request `multipart:` lists the declared form fields; `manual` means the handler reads the body itself and the schema shows no requestBody; the declared field names of JSON request models are in the JSON (`request.fields`). Semantics cells count rows per class as implemented/ignored/rejected/rejected-in-job/declared-only, shared rows included; the rows themselves are listed under "Semantics per operation". An MCP/CLI/Client cell reading `via ...` means no interface calls this operation directly; the named interfaces reach its semantics through the operation listed under "Shared semantics".']
    head_row = '| Method | Path | Handler | Request | Response | Auth | MCP tools | CLI | Client | Semantics (impl/ign/rej/rej-in-job/decl) |'
    sep = '|---|---|---|---|---|---|---|---|---|---|'
    by_op = {f"{r['method']} {r['path']}": r for r in data['operations']}

    def join_cell(r, key):
        if r[key]:
            return cell(', '.join(r[key]))
        via = [f"via `{t['operation']}` ({t['qualifier']}): {', '.join(by_op[t['operation']][key])}" for t in r['shared_semantics_to'] if by_op[t['operation']][key]]
        return cell('; '.join(via))

    for family in FAMILY_ORDER:
        rows = [r for r in data['operations'] if r['family'] == family]
        if not rows:
            continue
        L += ['', f'## Family: {family}', '', head_row, sep]
        for r in rows:
            req = r['request']
            request = {'json': req['model'], 'multipart': 'multipart: ' + ', '.join(req['fields'] or []), 'manual': 'manual (' + ', '.join(req.get('read_by', [])) + ')', 'none': '-'}[req['kind']]
            extras = []
            if r['path_params']:
                extras.append('path: ' + ', '.join(r['path_params']))
            if r['query_params']:
                extras.append('query: ' + ', '.join(r['query_params']))
            if extras:
                request = (request + '; ' if request != '-' else '') + '; '.join(extras)
            counts = Counter(s['status'] for s in r['semantics'])
            sem = '/'.join(str(counts.get(s, 0)) for s in STATUSES) if r['semantics'] else '-'
            status = f" [{r['status_code']}]" if r['status_code'] != 200 else ''
            L.append(f"| {r['method']} | `{r['path']}`{status} | `{r['handler']}` | {cell(request)} | {cell(r['response'])} | {auth_cell(r)} | {join_cell(r, 'mcp_tools')} | {join_cell(r, 'cli')} | {join_cell(r, 'client')} | {sem} |")
    L += ['', '## Broker surfaces and credential headers', '',
          'Operations BrokerDispatch serves (broker/surfaces.py: intercept answers before FastAPI parses, catalog merges inside the handler, stage resolves in-process) '
          'and the operations where a vendor credential header is read besides Authorization: Bearer.', '',
          '| Method | Path | Broker surfaces | Credential headers |', '|---|---|---|---|']
    for r in data['operations']:
        if r['broker_surfaces'] or len(r['credential_headers']) > 1:
            L.append(f"| {r['method']} | `{r['path']}` | {cell(', '.join(r['broker_surfaces']))} | {cell(', '.join(r['credential_headers']))} |")
    L += ['', '## Routes outside the OpenAPI schema', '', '| Method | Path | Handler | Auth | Reason |', '|---|---|---|---|---|']
    for u in data['unlisted_routes']:
        L.append(f"| {u['method']} | `{u['path']}` | `{u['handler']}` | {cell(u['auth'])} | {cell(u['reason'])} |")
    L += ['', '## Shared semantics', '', 'Operations whose handler runs another operation\'s code path carry that operation\'s semantics rows too, prefixed with the qualifier (`origin` shared, `shared_from` names the source).', '']
    for r in data['operations']:
        for s in r['shared_semantics_from']:
            copied = sum(1 for x in r['semantics'] if x['shared_from'] == s['operation'])
            L.append(f"- **{r['method']} `{r['path']}`** with `{s['qualifier']}` runs the {copied} semantics rows of `{s['operation']}` (listed below as `{s['qualifier']}: <control>`): {s['note']}")
    L += ['', '## Declared but not implemented', '', 'Every semantics row whose class is not `implemented` (ignored, rejected, rejected-in-job or declared-only), grouped by operation. A `rejected` row is an explicit refusal, not a silent gap; a `rejected-in-job` row is an explicit refusal the caller only sees as job status failed; `ignored` and `declared-only` rows are the gaps.', '']
    for r in data['operations']:
        gaps = [s for s in r['semantics'] if s['status'] != 'implemented']
        if not gaps:
            continue
        L.append(f"- **{r['method']} `{r['path']}`**")
        for s in gaps:
            shared = f" [shared from {s['shared_from']}]" if s['shared_from'] else ''
            L.append(f"  - {s['control']}: `{s['status']}` ({s['citation']}){shared} {s['note']}")
    L += ['', '## Semantics per operation', '', '| Operation | Control | Class | Origin | Citation | Note |', '|---|---|---|---|---|---|']
    for r in data['operations']:
        for s in r['semantics']:
            origin = s['origin'] + (f" from `{s['shared_from']}`" if s['shared_from'] else '')
            L.append(f"| {r['method']} `{r['path']}` | {cell(s['control'])} | {s['status']} | {origin} | `{s['citation']}` | {cell(s['note'])} |")
    L += ['', '## Documentation gaps', '', 'Documentation that under-declares a code path the operation has (a stale docstring or document): the inverse of `declared-only`, so recorded on the operation (`doc_gaps` in the JSON) rather than as a semantics class.', '']
    for r in data['operations']:
        for g in r['doc_gaps']:
            L.append(f"- **{r['method']} `{r['path']}`** ({g['citation']}): {g['note']}")
    L += ['', '## MCP tools', '', '| Tool | Source | Operations | Annotations | Required arguments |', '|---|---|---|---|---|']
    def calls(entry):
        return ', '.join(entry['operations'] + [f'{u} (outside the schema)' for u in entry.get('unlisted_routes', [])])

    for t in data['mcp_tools']:
        L.append(f"| `{t['name']}` | `{t['source']}` | {cell(calls(t))} | {cell(', '.join(f'{k}={v}' for k, v in t['annotations'].items()))} | {cell(', '.join(t['required_arguments']))} |")
    L += ['', '## MCP resources, templates and prompts', '', '| Kind | URI / name | MIME | Source | Operations |', '|---|---|---|---|---|']
    for res in data['mcp_resources']:
        L.append(f"| resource | `{res['uri']}` | {res['mime_type']} | `{res['source']}` | {cell(calls(res))} |")
    for t in data['mcp_resource_templates']:
        L.append(f"| template | `{t['uri_template']}` | {t['mime_type']} | `{t['source']}` | {cell(calls(t))} |")
    for p in data['mcp_prompts']:
        L.append(f"| prompt | `{p['name']}` | - | `{p['source']}` | {cell(', '.join(a['name'] + ('' if a['required'] else '?') for a in p['arguments']))} (arguments; no HTTP call) |")
    L += ['', '## CLI subcommands', '', 'Kinds: `client` (calls the listed Client methods or routes), `local` (sends nothing to a Titanos server; the reason is given), `bridge` (forwards another interface, joined to every operation that interface reaches).', '',
          '| Subcommand | Kind | Operations | Client methods | Local-only reason / bridge | Note |', '|---|---|---|---|---|---|']
    for row in data['cli_subcommands']:
        kind = 'local' if row['local'] else 'bridge' if row['bridge'] else 'client'
        L.append(f"| `{row['name']}` | {kind} | {cell(', '.join(row['operations']))} | {cell(', '.join(row['client_methods']))} | {cell(row['local'] or row['bridge'])} | {cell(row['note'])} |")
    L += ['', '## Python client methods', '', '| Method | Source | Operations |', '|---|---|---|']
    for row in data['client_methods']:
        L.append(f"| `Client.{row['name']}` | `{row['source']}` | {cell(', '.join(row['operations']))} |")
    L += ['', '## Coverage gaps', '',
          f"Operations with no MCP tool ({len(c['operations_without_mcp_tool'])}): " + ', '.join(f'`{o}`' for o in c['operations_without_mcp_tool']), '',
          f"Operations with no CLI subcommand ({len(c['operations_without_cli'])}): " + ', '.join(f'`{o}`' for o in c['operations_without_cli']), '',
          f"Operations with no Python client method ({len(c['operations_without_client_method'])}): " + ', '.join(f'`{o}`' for o in c['operations_without_client_method']), '']
    return '\n'.join(L)


def render_json(data):
    return json.dumps(data, indent=1, ensure_ascii=False, sort_keys=True) + '\n'


def build(state_dir):
    """The matrix dict and its Markdown rendering; the HEAD hash is read here so the JSON itself carries none."""
    data = assemble(state_dir)
    return data, render_md(data, source_head())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', type=Path, default=ROOT / 'docs', help='output directory (default docs/)')
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix='titanos-route-matrix-') as state:
        data, markdown = build(state)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / JSON_OUT).write_bytes(render_json(data).encode('utf-8'))
    (args.out / MD_OUT).write_bytes((markdown + '\n').encode('utf-8'))
    print(json.dumps({k: v for k, v in data['counts'].items() if not k.startswith('operations_without')}))


if __name__ == '__main__':
    main()
