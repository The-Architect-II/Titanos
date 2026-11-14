"""Build docs/broker_matrix.json and docs/BROKER_MATRIX.md: the cloud broker's support matrix, every cell bound to evidence.

Cells (each with a stable id; tests/check_broker_docs_claims.py maps documentation support claims to these ids):
- provider x surface x stream: every inbound broker surface (broker/surfaces.py, kind intercept) against every provider
  profile (broker/profiles.py), grouped by the client protocol the surface speaks. The mode is read from the real adapter
  classes: P (passthrough: the profile serves the surface natively), T (translated by the adapter) or na (the adapter
  neither serves nor translates it: 400 broker_capability_unavailable before egress; the pending_contract slot has no wire
  contract). Surfaces whose stream is selected by a field have an off and an on cell; surfaces that always stream
  (streamGenerateContent, text-to-speech/stream) have an on cell only. id: '<surface>|<profile>|<off|on>'.
- local: the same surfaces against a local model (L today's route, L* bridged through the inner OpenAI route, na).
  id: '<surface>|local|<off|on>'.
- catalog: catalog surfaces merged in their handlers. id: '<surface>|catalog'.
- stage: API Plus stages and asset jobs resolved in-process (CloudInvoker, GatewayRouter). id: '<surface>|cloud[:<task>]'.
- error: upstream error classes per target adapter (classification and credential marking) and client-facing renderings
  per inbound protocol envelope. ids: 'error:<class>|<adapter>' and 'envelope:<rendering>|<protocol>'.
- role: behaviour per role (standalone, broker, managed). id: 'role:<role>|<behaviour>'.
- feature: inbound auth, keys, limits, vault, egress, redaction, routing, breakers, advisors, observability, audit and the
  operator surfaces. id: 'feature:<area>.<name>'.

States: pass (every cited test id exists in the test source and passed in this run's results file), live-pass (a live
evidence id from a live evidence file names the cell), na (with the reason) or untried (with the reason: no check cites
the cell, or the cited test was skipped on this host). Every broker check runs loopback-only under the netguard against
protocol-faithful mocks, fakes or an httpx MockTransport, so every pass without live evidence is labelled 'pass (mock)';
a provider cell proven only through its adapter column (the column's reference profile row on the same adapter code) is
labelled 'pass (mock, adapter column)'. Nothing claims live without a live evidence id.

Live observations: a live evidence file may also carry records with outcome 'observation' and a required note (a live
call that did not pass, for example a provider refusing the account). They are embedded under live_observations and
listed on the cell they name (live_notes), but they are never evidence: the cell keeps the state its tests and live
evidence give it.

Evidence comes from a TITANOS_BROKER_RESULTS JSONL results file: one line per check {check, returncode, passed, failed,
skipped, cells}. `--run` produces it from a fresh run of every tests/check_broker_*.py (except the matrix consumers
check_broker_matrix.py and check_broker_docs_claims.py) under a recording shim that records every unittest outcome, plus the
cell maps the WP5 checks report themselves (tests/broker_adapter_support.main), plus the case outcomes the lanes that are
not unittest modules append themselves: the SDK lanes (tests/check_broker_sdk.py, tests/check_broker_sdk_vendors.py; an
SDK that is not importable records its cases as skipped) and the JavaScript Studio checks run with node
(tests/check_broker_ui.js, tests/check_studio_broker_browser.js; a host without node or Playwright records them as
skipped). The run record and the live evidence are embedded in the JSON, so a rebuild without --results/--live reproduces
the committed files byte for byte (fixed point).

Usage:
  python -B tools/build_broker_matrix.py --run [--results PATH] [--jobs N]   fresh run of every broker check, then build
  python -B tools/build_broker_matrix.py --results PATH                      build from an existing results file
  python -B tools/build_broker_matrix.py [--live PATH ...] [--check]         rebuild from the embedded run record
--live may be given several times: the files are merged (an id may appear once across them) and together replace the
live evidence and observations embedded in docs/broker_matrix.json.
--check compares the rebuild with docs/ and exits 1 on drift (writes nothing). Output is deterministic: sorted keys, no
timestamps, LF newlines, repository-relative references only."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JSON_OUT = 'broker_matrix.json'
MD_OUT = 'BROKER_MATRIX.md'
EXCLUDED_CHECKS = ('check_broker_docs_claims.py', 'check_broker_matrix.py')  # they read this matrix
CHECK_TIMEOUT_SECONDS = 1800
STATES = ('pass', 'live-pass', 'na', 'untried')
STATE_MEANING = {
    'pass': ('every cited test id exists in the test source and passed in the embedded run record (results file); all broker checks run '
             'loopback-only against mocks, fakes or an httpx MockTransport, so a pass without live evidence is labelled pass (mock)'),
    'live-pass': 'a live evidence id from the live evidence file names this cell (the mock evidence stays cited beside it)',
    'na': 'not applicable, with the reason (a refusal the code makes on purpose, or a combination the design excludes)',
    'untried': 'no check cites the cell, or the cited test was skipped on this host; the reason says which',
}
MODE_MEANING = {
    'P': 'passthrough: the profile serves the surface natively (model rewritten, credential swapped, bytes streamed through)',
    'T': 'translated by the adapter through the canonical IR (broker/translate), incremental for streams',
    'L': 'local: the unchanged local route answers',
    'L*': 'local bridge: the request is re-encoded as an OpenAI body and answered by the local OpenAI route',
    'na': 'not served: 400 broker_capability_unavailable before egress (or the refusal the reason names)',
}
PROOF_MEANING = {
    'profile': 'a cited test drove this profile row itself (the reference row of its adapter column, or a profile-row test)',
    'column': ('proven for the adapter column through its reference profile row on the same adapter code; this profile row differs only in '
               'data (base URL, auth template, ParamMap, surfaces) and is covered by the profile table consistency test'),
}
# adapter column codes (the codes the WP5 checks use in their cell keys) and the reference profile each column's mock drives
COLUMNS = {'openai_compat': 'OAI', 'anthropic': 'ANT', 'gemini': 'GEM', 'elevenlabs': 'XI', 'ollama_native': 'OLC'}
REFERENCE_PROFILE = {'OAI': 'openai', 'AZ': 'azure_openai', 'ANT': 'anthropic', 'GEM': 'gemini', 'XI': 'elevenlabs', 'OLC': 'ollama_cloud',
                     'SY': 'switchyard'}
COLUMN_MEANING = {'OAI': 'OpenAI-compatible adapter (cloud and router rows)', 'AZ': 'OpenAI-compatible adapter, azure_openai row (URL template, api-key)',
                  'ANT': 'Anthropic Messages adapter', 'GEM': 'Gemini adapter', 'XI': 'ElevenLabs adapter', 'OLC': 'Ollama native adapter (ollama_cloud)',
                  'SY': ('split-protocol router (switchyard row): OpenAI-compatible adapter with the Anthropic surfaces passed through natively; '
                         'translated cells use the OAI column evidence')}
# the inbound name the streaming check uses for a translated stream pair ('stream:<alias><-<column>')
STREAM_ALIAS = {'openai.chat': 'openai.chat', 'openai.completions': 'openai.completions', 'openai.responses': 'responses',
                'anthropic.messages': 'anthropic', 'gemini.streamGenerateContent': 'gemini', 'ollama.chat': 'ollama',
                'elevenlabs.tts_stream': 'elevenlabs.tts_stream'}
# surfaces the OpenAI-compatible adapter translates onto its chat surface (the streaming check names that column OAI-chat)
VIA_CHAT = ('openai.completions', 'openai.responses')

ADAPTERS_FILE = 'check_broker_adapters.py'
ROUTING = 'check_broker_routing.py'
HTTP = 'check_broker_http.py'
DISPATCH = 'check_broker_dispatch.py'
COMPAT = 'check_broker_compat_family.py'


def ev(check, *tests):
    return [f'{check}::{test}' for test in tests]


# Profile-row evidence beyond the column reference rows: (profile, surface, stream variant, evidence ids).
PROFILE_ROW_EVIDENCE = [
    ('mistral', 'openai.chat', 'off', ev(ADAPTERS_FILE, 'ProfileRows.test_mistral_param_map_and_tool_id_round_trip')),
    ('groq', 'openai.chat', 'off', ev(ADAPTERS_FILE, 'ProfileRows.test_groq_refusals_happen_before_egress')),
    ('nvidia_api_catalog', 'openai.chat', 'off', ev(ADAPTERS_FILE, 'LiveRowFidelity.test_openai_nvidia_and_hf_router_requests')),
    ('nvidia_api_catalog', 'openai.embeddings', 'off', ev(ADAPTERS_FILE, 'ProfileRows.test_nvidia_input_type_never_defaulted',
                                                          'LiveRowFidelity.test_openai_nvidia_and_hf_router_requests')),
    ('hf_router', 'openai.chat', 'off', ev(ADAPTERS_FILE, 'LiveRowFidelity.test_openai_nvidia_and_hf_router_requests')),
    ('hf_router', 'openai.chat', 'on', ev(ADAPTERS_FILE, 'LiveRowFidelity.test_openai_nvidia_and_hf_router_requests')),
    ('hivemind_gateway', 'openai.chat', 'off', ev(ROUTING, 'Hops.test_hop_restricts_to_local_and_own_hop_is_508')),
    ('xai', 'openai.images.generations', 'off', ev(DISPATCH, 'Images.test_png_url_rehosting_and_jpeg_refusal')
     + ev(COMPAT, 'Images.test_url_native_providers_answer_their_own_urls')),
    # the OpenAI-compatible family rows against their own provider mocks (tests/broker_mocks/compat_mock.py)
    ('together', 'openai.images.generations', 'off', ev(COMPAT, 'Images.test_url_native_providers_answer_their_own_urls')),
    ('groq', 'openai.audio.speech', 'off', ev(COMPAT, 'TypedSurfaces.test_groq_speech_and_responses_are_native')),
    ('groq', 'openai.responses', 'off', ev(COMPAT, 'TypedSurfaces.test_groq_speech_and_responses_are_native')),
    ('groq', 'openai.chat', 'off', ev(COMPAT, 'Arguments.test_local_engine_knobs_are_mapped_or_dropped',
                                      'Errors.test_groq_498_is_a_rate_limit')),
    ('together', 'openai.audio.speech', 'off', ev(COMPAT, 'TypedSurfaces.test_together_speech_transcription_and_rerank')),
    ('together', 'openai.audio.transcriptions', 'off',
     ev(COMPAT, 'TypedSurfaces.test_together_speech_transcription_and_rerank')),
    ('together', 'openai.rerank', 'off', ev(COMPAT, 'TypedSurfaces.test_together_speech_transcription_and_rerank')),
    ('together', 'openai.chat', 'off', ev(COMPAT, 'Arguments.test_local_engine_knobs_are_mapped_or_dropped')),
    ('fireworks', 'openai.rerank', 'off', ev(COMPAT, 'TypedSurfaces.test_fireworks_rerank_and_responses_store_rule')),
    ('fireworks', 'openai.responses', 'off', ev(COMPAT, 'TypedSurfaces.test_fireworks_rerank_and_responses_store_rule')),
    ('nvidia_ai_endpoints', 'openai.rerank', 'off',
     ev(COMPAT, 'TypedSurfaces.test_nvidia_rerank_translates_the_nim_ranking_shape')),
    ('xai', 'openai.responses', 'off', ev(COMPAT, 'TypedSurfaces.test_xai_responses_store_and_handle_retrieval')),
    ('xai', 'openai.responses.retrieve', 'off', ev(COMPAT, 'TypedSurfaces.test_xai_responses_store_and_handle_retrieval')),
    ('xai', 'openai.chat', 'off', ev(COMPAT, 'Arguments.test_xai_reasoning_models_never_get_penalties_or_stop')),
    ('mistral', 'openai.audio.transcriptions', 'off',
     ev(COMPAT, 'TypedSurfaces.test_mistral_voxtral_transcription_and_openrouter_embeddings')),
    ('openrouter', 'openai.embeddings', 'off',
     ev(COMPAT, 'TypedSurfaces.test_mistral_voxtral_transcription_and_openrouter_embeddings')),
    ('deepseek', 'anthropic.messages', 'off', ev(COMPAT, 'Arguments.test_deepseek_tool_loop_sends_reasoning_content_back')),
    ('perplexity', 'openai.chat', 'off', ev(COMPAT, 'Arguments.test_perplexity_extensions_pass_through_and_native_search')),
    ('cerebras', 'openai.responses', 'off',
     ev(COMPAT, 'Arguments.test_translated_reasoning_effort_only_reaches_models_that_take_it')),
    ('azure_openai', 'openai.images.generations', 'off',
     ev(ADAPTERS_FILE, 'AzurePassthrough.test_azure_gpt_image_deployment_never_gets_response_format')),
    ('pair_facade', 'openai.chat', 'off', ev(ROUTING, 'PairFacade.test_pair_facade_serves_as_a_router_upstream')),
    ('pair_facade', 'openai.chat', 'on', ev(ROUTING, 'PairFacade.test_pair_facade_serves_as_a_router_upstream')),
    ('pair_facade', 'openai.embeddings', 'off', ev(ROUTING, 'PairFacade.test_pair_facade_serves_as_a_router_upstream')),
    # the SDK lanes and the real-serve e2e drive the reference rows through a real titanos serve
    ('openai', 'openai.chat', 'on', ev('check_broker_sdk.py', 'openai.chat.stream_tools')),
    ('openai', 'openai.responses', 'on', ev('check_broker_sdk.py', 'openai.responses.stream', 'openai.responses.stream_tools')
     + ev('check_broker_e2e.py', 'KeyedStandalone.test_openai_responses_embeddings_audio_images_moderations_models')),
    ('openai', 'openai.embeddings', 'off', ev('check_broker_sdk.py', 'openai.embeddings')),
    ('openai', 'openai.images.generations', 'off', ev('check_broker_sdk.py', 'openai.images')),
    ('openai', 'openai.audio.speech', 'off', ev('check_broker_sdk.py', 'openai.audio.speech')),
    ('openai', 'openai.audio.transcriptions', 'off', ev('check_broker_sdk.py', 'openai.audio.transcriptions')),
    ('openai', 'openai.moderations', 'off', ev('check_broker_sdk.py', 'openai.moderations')),
    ('anthropic', 'anthropic.messages', 'on', ev('check_broker_sdk_vendors.py', 'anthropic.messages.stream_tools')),
    ('anthropic', 'anthropic.count_tokens', 'off', ev('check_broker_sdk_vendors.py', 'anthropic.count_tokens')),
    ('gemini', 'gemini.streamGenerateContent', 'on', ev('check_broker_sdk_vendors.py', 'gemini.streamGenerateContent.tools')),
    ('gemini', 'gemini.countTokens', 'off', ev('check_broker_sdk_vendors.py', 'gemini.countTokens')),
    ('elevenlabs', 'elevenlabs.tts', 'off', ev('check_broker_sdk_vendors.py', 'elevenlabs.tts')),
    ('elevenlabs', 'elevenlabs.tts_stream', 'on', ev('check_broker_sdk_vendors.py', 'elevenlabs.tts_stream')),
    ('elevenlabs', 'elevenlabs.stt', 'off', ev('check_broker_sdk_vendors.py', 'elevenlabs.stt')),
    ('ollama_cloud', 'ollama.chat', 'on', ev('check_broker_e2e.py', 'KeyedStandalone.test_ollama_chat_passthrough_stream_and_translated')),
    # NVIDIA NeMo Switchyard: translated inbound protocols driven through the switchyard row itself
    ('switchyard', 'ollama.chat', 'off', ev('check_broker_switchyard.py', 'AnthropicSurfaces.test_other_protocols_translate_onto_chat')),
    ('switchyard', 'gemini.generateContent', 'off',
     ev('check_broker_switchyard.py', 'AnthropicSurfaces.test_other_protocols_translate_onto_chat')),
]
EVERY_ROW = {'openai_compat': ev(ADAPTERS_FILE, 'EveryProfileRow.test_openai_compatible_rows'),
             'native': ev(ADAPTERS_FILE, 'EveryProfileRow.test_native_adapter_rows')}


def every_row_evidence():
    """(profile, surface, variant, ids) for every native (P) surface of every profile row: the row's own request (URL, auth
    form, static and hop headers, model, include_usage) captured from its adapter code by EveryProfileRow."""
    from titanos.broker.profiles import PROFILES
    from titanos.broker.surfaces import SURFACES
    rows = []
    for profile in PROFILES.values():
        if profile.adapter == 'none' or profile.extras.get('surface_adapters'):
            continue  # split-protocol rows (switchyard) are driven by tests/check_broker_switchyard.py instead
        ids = EVERY_ROW['openai_compat' if profile.adapter == 'openai_compat' else 'native']
        for surface_name in sorted(profile.surfaces):
            for variant in _variants(SURFACES[surface_name]):
                rows.append((profile.id, surface_name, variant, ids))
    return rows

# Local target per intercept surface: (mode, note). Modes from the translation matrix of the spec and the WP7 code.
LOCAL_MODES = {
    'openai.chat': ('L', 'today\'s route; local strictness unchanged'),
    'openai.completions': ('L', 'today\'s route'),
    'openai.responses': ('L', 'today\'s stateless Responses adapter'),
    'openai.responses.retrieve': ('na', '404 response_not_found: no local response store; only the upstream owning a handle answers'),
    'openai.responses.delete': ('na', '404 response_not_found: no local response store; only the upstream owning a handle answers'),
    'openai.responses.cancel': ('na', '404 response_not_found: no local response store; only the upstream owning a handle answers'),
    'openai.responses.input_items': ('na', '404 response_not_found: no local response store; only the upstream owning a handle answers'),
    'openai.embeddings': ('L', 'today\'s route'),
    'openai.audio.speech': ('L', 'today\'s route (WAV)'),
    'openai.audio.transcriptions': ('L', 'today\'s route'),
    'openai.audio.translations': ('L', 'today\'s route'),
    'openai.images.generations': ('L', 'today\'s route'),
    'openai.images.edits': ('L', 'today\'s route'),
    'openai.moderations': ('na', '503 classifier_unavailable: no local classifier; fails closed, never flagged:false'),
    'openai.rerank': ('L', 'today\'s API Plus rerank route (local cross-encoder models)'),
    'anthropic.messages': ('L', 'today\'s route (tool calls stream as tool_use blocks on llama-cpp models; in-process backends answer 400 unsupported_capability)'),
    'anthropic.count_tokens': ('L', 'llama-cpp native models only (apply-template plus tokenize); other local backends 400 broker_capability_unavailable'),
    'gemini.generateContent': ('L*', 'bridged to POST /v1/chat/completions'),
    'gemini.streamGenerateContent': ('L*', 'bridged to POST /v1/chat/completions with stream; re-encoded as Gemini SSE'),
    'gemini.embedContent': ('L*', 'bridged to POST /v1/embeddings'),
    'gemini.batchEmbedContents': ('L*', 'bridged to POST /v1/embeddings'),
    'gemini.countTokens': ('L', 'llama-cpp native models only; other local backends 400 broker_capability_unavailable'),
    'ollama.chat': ('L', 'today\'s route'),
    'ollama.generate': ('L', 'today\'s route'),
    'ollama.embed': ('L', 'today\'s route'),
    'ollama.embeddings': ('L', 'today\'s route'),
    'elevenlabs.tts': ('L', 'today\'s WAV-only route; a request without model_id still records a Decision (R16)'),
    'elevenlabs.tts_stream': ('L', 'the local WAV handler of POST /v1/text-to-speech/{voice_id}, chunked'),
    'elevenlabs.stt': ('L*', 'bridged to POST /v1/audio/transcriptions (word timestamps only)'),
}
DISABLED = 'check_broker_disabled_regression.py'
COV = 'check_broker_coverage.py'
DISABLED_A = ev(DISABLED, 'DisabledLane.test_a_no_broker_json_is_byte_identical')
LOCAL_EVIDENCE = {
    ('openai.chat', 'off'): DISABLED_A + ev(HTTP, 'Provenance.test_provenance_headers_on_cloud_and_local_answers'),
    ('openai.chat', 'on'): DISABLED_A,
    ('openai.completions', 'off'): DISABLED_A,
    ('openai.responses', 'off'): DISABLED_A,
    ('openai.responses.retrieve', 'off'): ev(HTTP, 'Responses.test_handle_affinity_and_owner_only'),
    ('openai.embeddings', 'off'): DISABLED_A,
    ('openai.audio.speech', 'off'): DISABLED_A,
    ('openai.moderations', 'off'): ev(HTTP, 'Moderation.test_cloud_passthrough_and_local_fail_closed'),
    ('anthropic.messages', 'off'): DISABLED_A,
    ('anthropic.messages', 'on'): DISABLED_A,
    ('anthropic.count_tokens', 'off'): ev(HTTP, 'CountTokens.test_anthropic_passthrough_and_llama_cpp_local_count',
                                          'CountTokens.test_local_count_while_the_broker_is_inactive'),
    ('gemini.generateContent', 'off'): ev(HTTP, 'Gemini.test_generate_content_reaches_a_local_model')
    + ev(DISPATCH, 'Rewrites.test_gemini_and_elevenlabs_stt_reach_local_models_through_the_openai_route'),
    ('gemini.streamGenerateContent', 'on'): ev('check_broker_streaming.py', 'Incrementality.test_gemini_from_local_openai_stream')
    + ev(DISPATCH, 'Rewrites.test_gemini_and_elevenlabs_stt_reach_local_models_through_the_openai_route'),
    ('ollama.chat', 'off'): DISABLED_A,
    ('ollama.chat', 'on'): DISABLED_A,
    ('elevenlabs.tts', 'off'): DISABLED_A + ev(HTTP, 'TextToSpeech.test_no_model_id_reaches_the_local_route_with_a_decision',
                                               'Unconfigured.test_disabled_lane_tts_is_unchanged'),
    ('elevenlabs.stt', 'off'): ev(HTTP, 'SpeechToTextBridge.test_elevenlabs_stt_reaches_a_local_transcription_model')
    + ev(DISPATCH, 'Rewrites.test_gemini_and_elevenlabs_stt_reach_local_models_through_the_openai_route'),
    # local targets through an active BrokerDispatch over the integrated app (tests/check_broker_coverage.py)
    ('openai.images.generations', 'off'): ev(COV, 'LocalThroughActiveBroker.test_images_generations_and_edits_reach_the_local_route'),
    ('openai.images.edits', 'off'): ev(COV, 'LocalThroughActiveBroker.test_images_generations_and_edits_reach_the_local_route'),
    ('openai.audio.transcriptions', 'off'): ev(COV, 'LocalThroughActiveBroker.test_audio_transcriptions_and_translations_rewrite_the_multipart_alias')
    + ev(DISPATCH, 'Rewrites.test_alias_rewrite_on_json_and_multipart'),
    ('openai.audio.translations', 'off'): ev(COV, 'LocalThroughActiveBroker.test_audio_transcriptions_and_translations_rewrite_the_multipart_alias'),
    ('ollama.generate', 'off'): ev(COV, 'LocalThroughActiveBroker.test_ollama_generate_embed_and_legacy_embeddings'),
    ('ollama.generate', 'on'): ev(COV, 'LocalThroughActiveBroker.test_ollama_generate_embed_and_legacy_embeddings'),
    ('ollama.embed', 'off'): ev(COV, 'LocalThroughActiveBroker.test_ollama_generate_embed_and_legacy_embeddings'),
    ('ollama.embeddings', 'off'): ev(COV, 'LocalThroughActiveBroker.test_ollama_generate_embed_and_legacy_embeddings'),
    ('gemini.embedContent', 'off'): ev(COV, 'LocalThroughActiveBroker.test_gemini_embeddings_bridge_and_native_count_tokens'),
    ('gemini.batchEmbedContents', 'off'): ev(COV, 'LocalThroughActiveBroker.test_gemini_embeddings_bridge_and_native_count_tokens'),
    ('gemini.countTokens', 'off'): ev(COV, 'LocalThroughActiveBroker.test_gemini_embeddings_bridge_and_native_count_tokens'),
    ('openai.completions', 'on'): ev(COV, 'LocalThroughActiveBroker.test_completions_and_responses_streams'),
    ('openai.responses', 'on'): ev(COV, 'LocalThroughActiveBroker.test_completions_and_responses_streams'),
    ('elevenlabs.tts_stream', 'on'): ev(COV, 'LocalThroughActiveBroker.test_elevenlabs_stream_route_answers_locally'),
    ('openai.rerank', 'off'): ev('check_broker_compat_family.py',
                                 'TypedSurfaces.test_local_rerank_model_still_reaches_the_local_route'),
}

# Explicit cells: (id, title, evidence ids, na reason or None). An empty evidence list without a reason is untried.
CATALOG_CELLS = [
    ('openai.models|catalog', 'GET /v1/models lists local models and the broker catalog the principal may see; a stale live catalog refreshes only for principals that see the upstream (R21)',
     ev(HTTP, 'Catalogs.test_live_catalog_refresh_only_for_principals_that_see_the_upstream') + ev(ROUTING, 'Facade.test_catalog_entries',
                                                                                                 'Facade.test_live_catalog_ttl_status_static_fallback_and_switch'), None),
    ('openai.model|catalog', 'GET /v1/models/{name} resolves qualified refs before local validation',
     ev(COV, 'Catalogs.test_qualified_model_entry_tags_and_v1beta_merge_the_broker_catalog'), None),
    ('ollama.tags|catalog', 'GET /api/tags answers exactly as before without broker.json', DISABLED_A, None),
    ('ollama.tags|catalog:cloud', 'GET /api/tags merges the broker entries the principal may see',
     ev(COV, 'Catalogs.test_qualified_model_entry_tags_and_v1beta_merge_the_broker_catalog'), None),
    ('ollama.ps|catalog', 'GET /api/ps lists the loaded native models; cloud targets are never loaded, so no broker entry appears',
     ev(COV, 'Catalogs.test_ps_lists_loaded_native_models_only'), None),
    ('gemini.models|catalog', 'GET /v1beta/models lists the local chat and embedding models with their supportedGenerationMethods',
     ev(HTTP, 'Gemini.test_action_allowlist_and_listings'), None),
    ('gemini.models|catalog:cloud', 'GET /v1beta/models lists the broker catalog entries the principal may see',
     ev(COV, 'Catalogs.test_qualified_model_entry_tags_and_v1beta_merge_the_broker_catalog'), None),
    ('gemini.model|catalog', 'GET /v1beta/models/{model} for a local model; unknown names 404', ev(HTTP, 'Gemini.test_action_allowlist_and_listings'), None),
    ('elevenlabs.voices|catalog', 'GET /v1/voices lists the local speech-model voices the principal may see',
     ev(COV, 'Catalogs.test_voices_merge_upstream_voices_for_keys_that_see_the_upstream') + ev('check_broker_jobs.py', 'LocalJobConstraints.test_model_constraints_apply_to_local_jobs_video_and_catalogs'), None),
    ('elevenlabs.voices|catalog:cloud', 'GET /v1/voices merges the upstream ElevenLabs voices after the local ones for a key that sees the upstream (cloud:<id>); '
     'fetched with a catalog grant and cached for the catalog TTL; a key without the scope causes no upstream request',
     ev(COV, 'Catalogs.test_voices_merge_upstream_voices_for_keys_that_see_the_upstream') + ev(ADAPTERS_FILE, 'ElevenLabsAdapterRow.test_tts_stream_stt_voices'), None),
    ('elevenlabs.voice|catalog', 'GET /v1/voices/{voice_id} for a local or an upstream voice; unknown voices 404',
     ev(COV, 'Catalogs.test_voices_merge_upstream_voices_for_keys_that_see_the_upstream'), None),
    ('broker.catalog|catalog', 'GET /api/broker/catalog with per-upstream live or static source and refresh',
     ev(ROUTING, 'Facade.test_live_catalog_ttl_status_static_fallback_and_switch'), None),
]
PLUS = 'check_broker_plus.py'
STAGES = 'check_broker_stages.py'
JOBS = 'check_broker_jobs.py'
STAGE_CELLS = [
    ('plus.translations|cloud', 'text translation through a cloud chat target (prompt broker-translation-v1); include_provenance refused for cloud; location semantics',
     ev(PLUS, 'Locations.test_location_semantics_per_operation') + ev(STAGES, 'Invoker.test_chat_translation_and_embeddings'), None),
    ('plus.audio_analysis|cloud', 'cloud ASR stage after the local VAD gate (no speech means zero egress)',
     ev(PLUS, 'CloudASR.test_cloud_asr_after_the_local_vad_gate') + ev(STAGES, 'Invoker.test_transcription_returns_an_asr_shaped_dict'), None),
    ('plus.sentiment|cloud', 'sentiment LLM stage through GatewayRouter on a cloud target',
     ev(PLUS, 'CloudLLM.test_sentiment_and_compose_reach_the_cloud_llm_through_gateway_router') + ev(STAGES, 'Gateway.test_candidates_and_complete_for_cloud_and_local'), None),
    ('plus.sentiment_compose|cloud', 'sentiment composition LLM stage through GatewayRouter on a cloud target',
     ev(PLUS, 'CloudLLM.test_sentiment_and_compose_reach_the_cloud_llm_through_gateway_router'), None),
    ('plus.retrieval_answer|cloud', 'retrieval answers refuse cloud LLMs with a typed 400 and zero upstream calls (R24: native context accounting and checkpoint provenance)',
     ev(PLUS, 'CloudLLM.test_retrieval_answer_needs_a_local_native_llm'), 'refused by design (R24): exact context accounting needs a local native model'),
    ('plus.local_only_stages|cloud', 'ocr, depth, detection, segmentation, vad, diarization, pair classification and rerank stay local: cloud_only gives 400 broker_capability_unavailable',
     ev(PLUS, 'Locations.test_location_semantics_per_operation'), 'local-only stages by design: cloud_only answers 400 broker_capability_unavailable'),
    ('plus.extensions_infer|cloud', 'POST /v1/extensions/infer compositions with cloud stages (cloud ASR after the local VAD gate)',
     ev(PLUS, 'CloudASR.test_cloud_asr_after_the_local_vad_gate'), None),
    ('plus.extensions|runtime', 'GET /v1/extensions reports the runtime (standalone-local or local+broker) and cloud_stages',
     ev(PLUS, 'Extensions.test_runtime_reporting'), None),
    ('jobs.create|cloud:speech', 'POST /api/jobs task=speech on a cloud target: WAV artifact bound to its owner, Idempotency-Key required, budget reserved at submit',
     ev(JOBS, 'CloudJobs.test_cloud_speech_and_image_jobs_produce_owner_bound_artifacts', 'CloudJobs.test_idempotency_key_required_and_per_owner',
        'CloudJobs.test_budget_is_reserved_at_submit') + ev(STAGES, 'Invoker.test_speech_returns_wav'), None),
    ('jobs.create|cloud:image', 'POST /api/jobs task=image on a cloud target: PNG artifact (non-PNG refused), owner-bound',
     ev(JOBS, 'CloudJobs.test_cloud_speech_and_image_jobs_produce_owner_bound_artifacts') + ev(STAGES, 'Invoker.test_image_returns_png_and_refuses_jpeg'), None),
    ('jobs.create|cloud:video', 'video jobs on a cloud target', [], 'cloud asset jobs cover speech and image only; video stays local (R24)'),
    ('jobs.create|cloud:music', 'music jobs on a cloud target', [], 'cloud asset jobs cover speech and image only; music stays local (R24)'),
    ('jobs.owner_scope|local', 'jobs, videos, cancel and artifacts are owner-scoped for keys; model constraints apply to local jobs',
     ev(JOBS, 'OwnerScoping.test_jobs_videos_and_cancel_are_owner_scoped', 'LocalJobConstraints.test_model_constraints_apply_to_local_jobs_video_and_catalogs'), None),
    ('invoker.cancel_deadline|cloud', 'CloudInvoker honours cancel and the deadline on job and stage threads and never takes the engine lock',
     ev(STAGES, 'Invoker.test_cancel_and_deadline_are_honoured', 'Invoker.test_the_engine_lock_is_never_acquired',
        'Invoker.test_infer_sync_rechecks_the_principal_and_the_current_policy'), None),
]
UE = 'check_broker_upstream_errors.py'
ERROR_ADAPTERS = ('openai_compat', 'anthropic', 'gemini', 'elevenlabs', 'ollama_native')
# error class -> (title, {adapter: evidence ids} or a list for every adapter, na reasons by adapter)
ERROR_CLASSES = {
    'client_4xx_scrubbed': ('400/404/413/422 pass through in the inbound envelope with the body scrubbed, never retried',
                            {'*': ev(UE, 'Table.test_passthrough_bodies_are_scrubbed', 'Table.test_status_to_code_table'),
                             'openai_compat': ev(UE, 'MockErrors.test_client_errors_pass_through_scrubbed') + ev(DISPATCH, 'CloudFallback.test_upstream_400_is_passed_through_and_never_retried')}, {}),
    'upstream_auth': ('401 and auth-coded 403 become 502 upstream_auth_failed and mark the credential failed; a plain 403 never takes it offline (R10)',
                      {'openai_compat': ev(UE, 'MockErrors.test_r10_openai_compatible_shapes', 'Table.test_credential_failure_rules'),
                       'anthropic': ev(UE, 'MockErrors.test_r10_anthropic_shapes', 'Table.test_credential_failure_rules'),
                       'gemini': ev(UE, 'MockErrors.test_r10_gemini_shapes', 'Table.test_credential_failure_rules'),
                       'elevenlabs': ev(UE, 'MockErrors.test_r10_elevenlabs_shapes', 'Table.test_credential_failure_rules'),
                       'ollama_native': ev(UE, 'MockErrors.test_r10_ollama_401_marks')}, {}),
    'rate_limited': ('429 becomes upstream_rate_limited with Retry-After preserved (header, retry-after-ms or Gemini RetryInfo)',
                     {'openai_compat': ev(UE, 'MockErrors.test_rate_limit_and_overload_shapes', 'Table.test_retry_after_is_preserved'),
                      'gemini': ev(UE, 'Table.test_status_to_code_table', 'Table.test_retry_after_is_preserved'),
                      'anthropic': ev(UE, 'Table.test_stream_errors'),
                      'elevenlabs': ev(COV, 'Envelopes.test_upstream_rate_limits_from_elevenlabs_and_ollama'),
                      'ollama_native': ev(COV, 'Envelopes.test_upstream_rate_limits_from_elevenlabs_and_ollama')}, {}),
    'server_error': ('5xx and 529 become 503 upstream_unavailable (fallback-eligible before commit)',
                     {'*': ev(UE, 'Table.test_status_to_code_table'),
                      'anthropic': ev(UE, 'MockErrors.test_rate_limit_and_overload_shapes'), 'gemini': ev(UE, 'MockErrors.test_rate_limit_and_overload_shapes'),
                      'ollama_native': ev(UE, 'MockErrors.test_rate_limit_and_overload_shapes')}, {}),
    'network_and_timeout': ('connect refusal, connect/read timeouts and 408 become upstream_unavailable or upstream_timeout (adapter-independent, EgressClient)',
                            {'*': ev(UE, 'Table.test_network_errors') + ev('check_broker_egress.py', 'Responses.test_first_byte_timeout', 'Responses.test_idle_timeout_between_stream_chunks')}, {}),
    'redirect_refused': ('3xx becomes 502 upstream_redirect_refused and a foreign redirect is never followed (adapter-independent)',
                         {'*': ev(UE, 'Table.test_status_to_code_table') + ev('check_broker_egress.py', 'Refusals.test_redirect_to_a_foreign_host_is_refused_and_never_followed')}, {}),
    'error_first_stream': ('an error as the first stream frame is a pre-commit failure (fallback-eligible), never a committed 200',
                           {'openai_compat': ev(UE, 'MockErrors.test_error_first_stream_is_a_pre_commit_failure'),
                            'anthropic': ev(UE, 'MockErrors.test_error_first_stream_is_a_pre_commit_failure') + ev(DISPATCH, 'CloudFallback.test_anthropic_error_first_stream_falls_back_before_commit'),
                            'ollama_native': ev(UE, 'MockErrors.test_error_first_stream_is_a_pre_commit_failure'),
                            'gemini': ev(UE, 'Table.test_stream_errors')},
                           {'elevenlabs': 'binary audio streams carry no error frame; a failed TTS stream is a non-2xx answer before commit'}),
    'secret_bearing_codes': ('provider error codes that look like secrets are dropped from every rendering (adapter-independent)',
                             {'*': ev(UE, 'Table.test_provider_codes_carrying_secrets_are_dropped')}, {}),
    'no_retry_headers': ('x-should-retry:false on upstream auth, redirect, budget and loop answers; retryable codes carry none (adapter-independent)',
                         {'*': ev(UE, 'Table.test_no_retry_headers')}, {}),
}
CONTRACTS = 'check_broker_contracts.py'
ENVELOPE_PROTOCOLS = ('openai', 'responses', 'anthropic', 'gemini', 'elevenlabs', 'ollama')
ENVELOPES = {
    'upstream_4xx_body': ('an upstream 4xx body re-rendered in the inbound envelope',
                          {'*': ev(UE, 'Table.test_per_protocol_envelopes', 'Table.test_passthrough_bodies_are_scrubbed')}, {}),
    'broker_error': ('a Titanos/broker refusal in the inbound envelope with X-Titanos-Error-Origin',
                     {'*': ev(CONTRACTS, 'Envelopes.test_render_shapes', 'Envelopes.test_messages_are_scrubbed_at_the_envelope'),
                      'openai': ev(HTTP, 'Envelopes.test_error_envelope_per_route_family'), 'anthropic': ev(HTTP, 'Envelopes.test_error_envelope_per_route_family'),
                      'gemini': ev(HTTP, 'Envelopes.test_error_envelope_per_route_family'), 'elevenlabs': ev(HTTP, 'Envelopes.test_error_envelope_per_route_family'),
                      'ollama': ev(HTTP, 'Envelopes.test_error_envelope_per_route_family')},
                     {'responses': None}),
    'mid_stream_terminal': ('a failure after commit ends the stream with the protocol terminal frame (OpenAI error chunk then [DONE]; Responses response.failed; '
                            'Anthropic event error; Gemini final error chunk; Ollama error line; binary audio aborted without a terminating chunk)',
                            {'*': ev(CONTRACTS, 'Envelopes.test_stream_terminals'),
                             'openai': ev(DISPATCH, 'Streaming.test_mid_stream_disconnect_gives_the_terminal_frame_and_no_second_hit')}, {}),
    'deadline_terminal': ('R15: a committed stream outliving the request deadline ends with the terminal frame naming the expired deadline '
                          '(OpenAI code deadline_exceeded; Anthropic error type timeout_error; Responses response.failed; Gemini 504 '
                          'DEADLINE_EXCEEDED chunk; Ollama error line; ElevenLabs binary audio ends without a terminating chunk)',
                          {'openai': ev(HTTP, 'StreamDeadline.test_committed_stream_ends_with_the_deadline_terminal_frame'),
                           'anthropic': ev(HTTP, 'StreamDeadline.test_anthropic_deadline_frame_is_timeout_error'),
                           'responses': ev(COV, 'Envelopes.test_deadline_terminal_frames'),
                           'gemini': ev(COV, 'Envelopes.test_deadline_terminal_frames'),
                           'ollama': ev(COV, 'Envelopes.test_deadline_terminal_frames'),
                           'elevenlabs': ev(COV, 'Envelopes.test_elevenlabs_binary_stream_ends_without_a_terminating_chunk')}, {}),
    'auth_refusal': ('an inbound authentication refusal in the inbound envelope (Gemini carries a google.rpc.ErrorInfo detail with the Titanos code)',
                     {'gemini': ev(HTTP, 'QueryCredential.test_gemini_query_key_refusal_is_readable_and_typed'),
                      'openai': ev('check_broker_auth_http.py', 'Keyed.test_wrong_key_from_loopback_is_401'),
                      'anthropic': ev('check_broker_auth_http.py', 'Keyed.test_wrong_key_from_loopback_is_401')
                      + ev('check_broker_auth.py', 'Keyed.test_auth_denial_is_an_auth_failure_with_headers'),
                      'elevenlabs': ev(COV, 'Envelopes.test_auth_refusals_in_the_elevenlabs_and_ollama_envelopes'),
                      'ollama': ev(COV, 'Envelopes.test_auth_refusals_in_the_elevenlabs_and_ollama_envelopes')}, {'responses': None}),
}
ROLES_CHECK = 'check_broker_roles.py'
ROLE_CELLS = [
    ('role:standalone|no_broker_json_no_egress', 'standalone without broker.json makes no egress at all',
     ev(ROLES_CHECK, 'Standalone.test_no_broker_json_means_no_egress_at_all') + ev(DISPATCH, 'FastPath.test_install_broker_creates_no_files_and_opens_no_sockets'), None),
    ('role:standalone|cloud_policy', 'cloud policy is min(ceiling, broker.json) ordered off < explicit < fallback; off by default',
     ev(ROUTING, 'PolicyMatrix.test_cloud_policy_times_ceiling', 'PolicyMatrix.test_location_matrix'), None),
    ('role:standalone|anonymous_cloud', 'keyless loopback gets cloud only when policy.anonymous_cloud is true, in standalone role only',
     ev('check_broker_auth.py', 'Keyless.test_anonymous_cloud_only_in_standalone'), None),
    ('role:broker|serve_preconditions', 'broker role refuses to serve without its preconditions (bootstrap key, native TLS for a LAN bind, audit chain, egress-key budgets)',
     ev(ROLES_CHECK, 'Preconditions.test_broker_role_refusals_name_the_switch') + ev(CONTRACTS, 'Roles.test_serve_preconditions') + ev('check_broker_cli.py', 'Serve.test_preconditions'), None),
    ('role:broker|egress_key_rules', 'broker role: keys with egress scopes (admin included, R3) need a budget and an expiry',
     ev('check_broker_keys.py', 'Rules.test_broker_role_requires_budget_and_expiry_for_egress_scopes'), None),
    ('role:managed|egress_refused', 'managed role refuses every egress purpose without parent delegation (typed errors, zero upstream hits)',
     ev(ROLES_CHECK, 'Managed.test_cloud_and_fallback_give_typed_errors_with_zero_hits') + ev('check_broker_egress.py', 'ManagedRole.test_every_purpose_is_refused_before_any_socket')
     + ev(ROUTING, 'Grants.test_managed_role_refuses_every_purpose_without_delegation'), None),
    ('role:managed|delegation', 'delegated cloud use needs all five conditions and an explicit target, and never raises the configured cloud policy',
     ev(ROLES_CHECK, 'Delegation.test_delegation_needs_all_five_conditions') + ev(ROUTING, 'Managed.test_delegation_needs_all_conditions_and_an_explicit_target',
                                                                                'Managed.test_delegated_target_must_be_a_qualified_ref_or_one_x_titanos_target',
                                                                                'Managed.test_delegation_never_raises_the_configured_cloud_policy')
     + ev('check_broker_egress.py', 'ManagedRole.test_delegated_inference_reaches_the_network_layer_only_with_parent_delegation'), None),
    ('role:managed|sealed_no_probes', 'managed role: vault sealed, no probes, no live catalogs, no advisors',
     ev(ROLES_CHECK, 'Managed.test_vault_sealed_no_probes_no_live_catalogs') + ev('check_broker_advisor.py', 'Slot.test_managed_role_never_consults_advisors')
     + ev(ROUTING, 'Facade.test_startup_info_and_managed_seal'), None),
    ('role:managed|parent_headers', 'managed role: deadline, trace and placement headers from the parent (local_only fail-closed)',
     ev(ROLES_CHECK, 'Managed.test_parent_headers') + ev(CONTRACTS, 'Roles.test_parent_headers'), None),
    ('role:managed|public_routes', 'managed role: GET /api/version public; /api/info reports the role',
     ev(ROLES_CHECK, 'Managed.test_info_and_public_routes'), None),
    ('role:managed|reload_refusal', 'a reload that would enable egress in managed role without delegation is refused',
     ev(ROUTING, 'Managed.test_reload_refuses_enabling_egress_in_managed_role_without_delegation'), None),
    ('role:managed|ports', 'managed role needs an explicit port and refuses the reserved ports',
     ev(ROLES_CHECK, 'Preconditions.test_managed_port_refusal_set') + ev('check_broker_manifest.py', 'PairManifest.test_port_refusals_kept'), None),
    ('role:any|hops', 'an inbound hop header restricts to local targets; this instance\'s own hop answers 508 loop_detected',
     ev(ROUTING, 'Hops.test_hop_restricts_to_local_and_own_hop_is_508'), None),
    ('role:any|switch_table', 'the role switch table and the per-purpose egress rule',
     ev(CONTRACTS, 'Roles.test_switch_table', 'Roles.test_purpose_allowed'), None),
    ('role:managed|pair_manifest', 'pair-manifest args gain --role managed with the identity block unchanged (re-qualification against PAIR pending, R28)',
     ev('check_broker_manifest.py', 'PairManifest.test_identity_unchanged_and_role_in_args'), None),
    ('role:managed|hivemind_manifest', 'hivemind-manifest placeholder rules and gateway args',
     ev('check_broker_manifest.py', 'HiveMindManifest.test_placeholder_rules_and_shape', 'HiveMindManifest.test_the_port_of_the_rules_rejects_host_and_unknown_tokens',
        'HiveMindManifest.test_gateway_args_and_refusals'), None),
]
AUTH = 'check_broker_auth.py'
AUTH_HTTP = 'check_broker_auth_http.py'
E2E = 'check_broker_e2e.py'
VAULT = 'check_broker_vault.py'
EGRESS = 'check_broker_egress.py'
FEATURE_CELLS = [
    # inbound authentication and scopes
    ('feature:inbound_auth.bearer', 'Authorization: Bearer (scheme case-insensitive, trimmed); a wrong or missing key is 401 even from loopback',
     ev(AUTH, 'Keyed.test_bearer_scheme_is_case_insensitive_and_trimmed', 'Keyed.test_wrong_or_missing_key_from_loopback_is_401') + ev(AUTH_HTTP, 'Keyed.test_wrong_key_from_loopback_is_401'), None),
    ('feature:inbound_auth.vendor_headers', 'x-api-key on /v1/messages*, xi-api-key on ElevenLabs routes and x-goog-api-key on /v1beta/* are accepted credentials in their family only',
     ev(AUTH, 'Keyed.test_vendor_headers_only_in_their_family'), None),
    ('feature:inbound_auth.query_key_refused', '?key= on /v1beta is refused 401 query_credential_refused in every mode (R4)',
     ev(AUTH, 'Keyed.test_query_credential_refused_on_v1beta', 'Keyless.test_query_key_refused_even_keyless') + ev(HTTP, 'QueryCredential.test_gemini_query_key_refusal_is_readable_and_typed'), None),
    ('feature:inbound_auth.ambiguous', 'two different presented credentials are 400 ambiguous_credentials', ev(AUTH, 'Keyed.test_ambiguous_credentials'), None),
    ('feature:inbound_auth.keyless', 'keyless loopback ignores placeholder keys; a non-loopback bind with nothing enforced fails closed',
     ev(AUTH, 'Keyless.test_placeholder_credentials_are_ignored', 'Keyless.test_non_loopback_bind_with_nothing_enforced_fails_closed')
     + ev(AUTH_HTTP, 'Keyless.test_placeholder_bearer_is_ignored_in_keyless_mode'), None),
    ('feature:inbound_auth.sticky_enforcement', 'any key record keeps enforcement (expired and revoked included); only the offline keyless reset reopens keyless mode',
     ev(AUTH, 'Keyed.test_expired_and_revoked_keys_keep_enforcement') + ev(AUTH_HTTP, 'Keyless.test_host_guard_until_enforcement_and_sticky_after_revocation')
     + ev('check_broker_keys.py', 'Reload.test_has_records_stays_true_when_every_key_expired', 'Reload.test_reset_keyless_is_offline_only_and_audited', 'Reload.test_enforcement_survives_a_deleted_table'), None),
    ('feature:inbound_auth.scopes', 'route scopes are enforced before the body is read (403 scope_denied); every operation and mount is listed or public; unlisted needs admin',
     ev(AUTH, 'Keyed.test_scope_denied_before_the_route') + ev(AUTH_HTTP, 'Keyed.test_scope_matrix_samples')
     + ev('check_broker_scope_coverage.py', 'Coverage.test_every_operation_and_mount_is_listed_or_public', 'Coverage.test_an_unlisted_route_needs_admin')
     + ev(CONTRACTS, 'Scopes.test_required_scope', 'Scopes.test_every_app_route_is_listed_or_public', 'Scopes.test_allows_and_umbrellas'), None),
    ('feature:inbound_auth.keyless_writes_refused', 'keyless mode refuses key and credential writes with 403 keyless_mode_refused and guards anonymous_cloud and allow_http_lan',
     ev(AUTH, 'Keyless.test_key_and_credential_writes_are_keyless_mode_refused') + ev(AUTH_HTTP, 'Keyless.test_keyless_key_and_credential_writes_and_guarded_policy_fields'), None),
    ('feature:inbound_auth.transport_secure', 'key and credential writes need TLS or a loopback peer; a forged X-Forwarded-Proto never satisfies it',
     ev(AUTH, 'HostGuardAndTransport.test_transport_secure') + ev(AUTH_HTTP, 'Keyed.test_transport_secure_for_key_and_credential_writes', 'UvicornPinnedProxy.test_forged_forwarded_proto_never_satisfies_transport_secure'), None),
    ('feature:inbound_auth.throttle', 'failed credential claims are throttled per address (429 auth_throttled) and audited; absent credentials do not count (R5)',
     ev(AUTH, 'Keyed.test_throttling_and_audit', 'Keyed.test_absent_credentials_do_not_feed_the_throttle'), None),
    ('feature:inbound_auth.settings_block', 'the /api/info settings block needs settings:read', ev(AUTH_HTTP, 'Keyed.test_settings_block_needs_settings_read'), None),
    ('feature:inbound_auth.host_guard', 'KeylessHostGuard reproduces TrustedHostMiddleware byte for byte while keyless and steps aside once enforced',
     ev(AUTH, 'HostGuardAndTransport.test_keyless_host_guard_matches_trusted_host_bytes_and_toggles'), None),
    # keys
    ('feature:keys.lifecycle', 'mint (secret shown once, digest stored), verify, expire, revoke and rotate without a restart; owner-only keys file',
     ev('check_broker_keys.py', 'Lifecycle.test_mint_returns_the_secret_once_and_stores_only_the_digest', 'Lifecycle.test_expiry_and_revocation_take_effect_without_restart',
        'Lifecycle.test_rotate_returns_a_new_secret_once_and_retires_the_old', 'Lifecycle.test_keys_file_is_owner_only'), None),
    ('feature:keys.offline_mint_seen', 'an offline CLI mint is seen by a running server through the mtime check',
     ev('check_broker_keys.py', 'Reload.test_second_instance_mint_is_seen_after_the_mtime_check') + ev('check_broker_cli.py', 'RunningServer.test_in_process_server_sees_an_offline_minted_key'), None),
    ('feature:keys.last_admin_guard', 'the last admin key cannot be revoked on a network bind', ev('check_broker_keys.py', 'Rules.test_last_admin_key_guard_on_network_bind'), None),
    # limits and budgets
    ('feature:limits.rate', 'per-key RPM, TPM (settled against actual usage) and concurrency limits answer 429',
     ev('check_broker_limits.py', 'RateLimits.test_rpm_bucket', 'RateLimits.test_tpm_bucket_and_settlement_reconciles', 'RateLimits.test_concurrency'), None),
    ('feature:limits.budgets', 'USD and token budgets answer 402 budget_exceeded before any hit; an unpriced target with a budget fails closed; spend survives a restart',
     ev('check_broker_limits.py', 'Budgets.test_usd_per_day_and_x_should_retry', 'Budgets.test_unpriced_target_with_a_budget_fails_closed',
        'Budgets.test_spend_survives_restart_through_the_usage_ledger') + ev(DISPATCH, 'Budgets.test_budget_is_admitted_before_any_hit'), None),
    ('feature:limits.audio_budgets', 'transcription and translation reserve the upload duration (exact for WAV, the size at 8 kbit/s otherwise); '
     'an answer without usage (text, srt, vtt) settles at that reservation in the limiter, the decision and the usage ledger',
     ev('check_broker_limits.py', 'AudioBudgets.test_upload_duration_estimate', 'AudioBudgets.test_router_estimate_and_pricing',
        'AudioBudgets.test_unknown_usage_settles_at_the_reservation')
     + ev(DISPATCH, 'Budgets.test_budgeted_transcription_without_usage_spends_its_reservation'), None),
    ('feature:limits.upstream_concurrency', 'an upstream\'s concurrency slots answer 429 concurrency_exceeded when exhausted (R9)',
     ev(EGRESS, 'Responses.test_per_upstream_concurrency_slot'), None),
    ('feature:accounting.pricing', 'operator Decimal prices, specificity and cache-write rates; usage rollups by key, upstream, model, surface and day',
     ev('check_broker_accounting.py', 'Pricing.test_specificity_and_ties', 'Pricing.test_decimal_cost_and_estimate', 'Ledger.test_rollups'), None),
    # vault
    ('feature:vault.dpapi', 'Windows DPAPI vault: real round trip, owner-only files, per-install entropy',
     ev(VAULT, 'Dpapi.test_real_round_trip_with_owner_only_files', 'Dpapi.test_another_install_entropy_cannot_open_the_blob'), None),
    ('feature:vault.keychain', 'macOS Keychain vault: real round trip in a temporary keychain', ev(VAULT, 'HelperTools.test_real_keychain_round_trip_in_a_temporary_keychain'), None),
    ('feature:vault.secret_service', 'Linux secret-service vault: real secret-tool round trip', ev(VAULT, 'HelperTools.test_real_secret_tool_round_trip'), None),
    ('feature:vault.helper_argv', 'helper backends never put a secret in argv and children never inherit env-reference variables',
     ev(VAULT, 'HelperTools.test_fake_security_records_argv_and_stdin', 'HelperTools.test_fake_secret_tool_records_argv_and_stdin',
        'HelperTools.test_helper_children_never_inherit_env_reference_variables'), None),
    ('feature:vault.sealed_file', 'sealed-file vault: scrypt parameters with maxmem, known-answer vectors, tamper and wrong-passphrase refusal, passphrase-file ACL; never auto-selected',
     ev(VAULT, 'SealedFile.test_scrypt_parameters_need_maxmem', 'SealedFile.test_known_answer_vectors', 'SealedFile.test_tamper_wrong_passphrase_and_id_binding',
        'SealedFile.test_vault_on_sealed_file_and_the_passphrase_file_acl', 'SealedFile.test_sealed_file_is_never_auto_selected'), None),
    ('feature:vault.references', 'env references are never persisted; file references re-check permissions on every use; ambient provider variables are ignored without a record',
     ev(VAULT, 'References.test_env_values_are_never_written', 'References.test_file_references_recheck_permissions_on_every_use',
        'References.test_ambient_provider_variables_are_ignored_without_a_record'), None),
    ('feature:vault.origin_binding', 'a credential handle needs validation for the exact origin, purpose and upstream; a base_url change invalidates it; rotation is atomic',
     ev(VAULT, 'Records.test_handle_needs_validation_and_the_exact_origin', 'Records.test_handle_enforces_purpose_and_upstream_binding',
        'Records.test_base_url_change_invalidates_and_validation_rebinds', 'Records.test_rotation_is_atomic_and_deletes_the_old_value'), None),
    ('feature:vault.host_sources', 'env: and file: credential sources name host values: over HTTP they need the admin scope, re-binding a stored credential to a changed origin needs admin, and TITANOS_* variables '
     '(the bootstrap key included) are never an upstream credential',
     ev(AUTH_HTTP, 'HostCredentialSources.test_reserved_names_and_host_sources_need_admin',
        'HostCredentialSources.test_moving_a_stored_credential_to_a_new_origin_needs_admin') + ev(VAULT, 'References.test_env_refusals'), None),
    ('feature:vault.write_only', 'no API returns a credential value; the index is metadata only', ev(VAULT, 'Records.test_no_api_returns_a_value', 'Records.test_index_is_metadata_only_and_values_are_registered_for_scrubbing'), None),
    ('feature:vault.lazy', 'opening the vault creates nothing until the first write; managed role seals it', ev(VAULT, 'Lazy.test_opening_the_vault_creates_nothing_until_the_first_write', 'Lazy.test_seal_refuses_every_resolution'), None),
    # egress
    ('feature:egress.credential_isolation', 'only the bound credential is presented; inbound secrets and the client Authorization are never forwarded',
     ev(EGRESS, 'MockRoundTrip.test_bound_credential_only_and_inbound_secrets_never_forwarded', 'MockRoundTrip.test_raw_json_content_and_form_fields_are_stripped_too',
        'RequestHygiene.test_returned_request_carries_no_auth_header'), None),
    ('feature:egress.url_refusals', 'userinfo, port, scheme, look-alike, path-trick and query-credential URLs are refused; ambient base URLs, keys and proxies are ignored',
     ev(EGRESS, 'Refusals.test_userinfo_port_scheme_and_look_alike_urls', 'Refusals.test_path_tricks_and_query_credentials', 'Refusals.test_proxy_environment_is_never_used',
        'Refusals.test_ambient_base_urls_and_keys_are_ignored'), None),
    ('feature:egress.resolved_address', 'every resolved address is checked before connect (metadata and private ranges refused; RFC 1918 needs allow_http_lan)',
     ev(EGRESS, 'Resolver.test_blocked_resolved_addresses_are_refused_before_connect', 'Resolver.test_every_resolved_address_is_checked',
        'Resolver.test_rfc1918_needs_allow_http_lan', 'Resolver.test_egress_modes'), None),
    ('feature:egress.azure_host', 'Azure: only the exact resource host; look-alikes and other resources are refused',
     ev(EGRESS, 'AzureAndOrigins.test_exact_azure_resource_host_is_allowed', 'AzureAndOrigins.test_azure_look_alikes_and_other_resources_are_refused'), None),
    ('feature:egress.response_caps', 'response size caps, compression bombs, set-cookie and location dropped', ev(EGRESS, 'Responses.test_response_size_caps',
     'CompressedAndCapped.test_gzip_and_deflate_bombs_are_refused_while_decoding', 'Responses.test_set_cookie_and_location_are_dropped_and_cookies_never_replayed'), None),
    ('feature:egress.tls_name_binding', 'TLS SNI and hostname checks use the origin host over the checked IP (needs the cryptography package; untried when absent, R14)',
     ev(EGRESS, 'TlsNameBinding.test_sni_and_hostname_check_use_the_origin_host_over_the_checked_ip', 'TlsNameBinding.test_a_certificate_for_another_name_is_refused'), None),
    ('feature:egress.source_scan', 'auth headers are set only in egress; reveal only in egress and the vault; grants only from the router; no subprocess or raw network imports in the broker',
     ev('check_broker_source_scan.py', 'SourceScan.test_auth_headers_are_set_only_in_egress', 'SourceScan.test_reveal_only_in_egress_and_the_vault',
        'SourceScan.test_egress_grant_constructed_only_by_the_router', 'SourceScan.test_network_and_subprocess_imports', 'SourceScan.test_provider_hostnames_only_in_profiles'), None),
    # redaction
    ('feature:redaction.logging', 'secrets are scrubbed in log records (child loggers, records after dictConfig, exc_text, uvicorn access lines with ?key=)',
     ev('check_broker_redact.py', 'Logging.test_child_logger_records_are_scrubbed_at_root_handler', 'Logging.test_factory_is_idempotent_and_handler_filters_reapply_after_dictconfig',
        'Logging.test_uvicorn_access_record_with_query_key', 'Logging.test_exc_text_scrubbed_by_factory'), None),
    ('feature:redaction.records', 'decisions, usage ledger and audit records carry no secret',
     ev('check_broker_accounting.py', 'Ledger.test_records_are_scrubbed', 'Decisions.test_decisions_are_scrubbed') + ev('check_broker_audit.py', 'Content.test_no_secret_in_records'), None),
    ('feature:redaction.response_headers', 'an allowlisted upstream response header that reflects a registered secret is redacted; others never pass',
     ev('check_broker_redact.py', 'ResponseHeaders.test_allowed_upstream_headers_never_reflect_a_registered_secret'), None),
    ('feature:redaction.child_env', 'child processes drop provider variables and base URLs', ev('check_broker_redact.py', 'ChildEnv.test_provider_names_and_base_urls_dropped'), None),
    # routing
    ('feature:routing.leak_guard', 'unknown, typo\'d and blank names, disabled, unvalidated and slot upstreams make zero upstream hits',
     ev(ROUTING, 'LeakGuard.test_unknown_typo_and_blank_names_make_zero_hits', 'LeakGuard.test_disabled_unvalidated_and_slot_upstreams_make_zero_hits'), None),
    ('feature:routing.visibility', 'an invisible model is 404 (no 403 oracle); key constraint globs; X-Titanos-Target needs route:explicit',
     ev(ROUTING, 'Visibility.test_missing_cloud_scope_is_404_even_with_policy_off', 'Visibility.test_cloud_other_upstream_is_denied', 'Visibility.test_constraint_globs',
        'Visibility.test_target_header_needs_route_explicit'), None),
    ('feature:routing.virtual_models', 'virtual-model strategies (ordered, weighted, cheapest, lowest latency) and max_attempts defaults',
     ev(ROUTING, 'Strategies.test_ordered_weighted_cheapest_lowest_latency', 'Strategies.test_max_attempts_defaults'), None),
    ('feature:routing.aliases', 'published aliases, collisions and compat suffixes (longest first)', ev(ROUTING, 'Names.test_alias_collisions', 'Names.test_compat_suffix_longest_first')
     + ev('check_broker_naming.py', 'LastAtSplit.test_qualified_refs', 'CompatSuffix.test_longest_suffix_first'), None),
    ('feature:routing.router_upstreams', 'router-as-upstream: hivemind_gateway and pair_facade upstreams are served by router:<id> with this '
     'instance\'s hop header; a request carrying our own hop back is 508',
     ev(ROUTING, 'Hops.test_hop_restricts_to_local_and_own_hop_is_508', 'PairFacade.test_pair_facade_serves_as_a_router_upstream'), None),
    ('feature:router.switchyard', 'NVIDIA NeMo Switchyard as a router upstream: origin base_url, auth none (a credential needs auth bearer), this '
     'instance\'s hop header and the request id, the harness session headers on its row only, the selected backend returned and recorded as a '
     'decision note, Anthropic Messages and count_tokens passed through natively, store forced false unless allowed, inbound hops kept local',
     ev('check_broker_switchyard.py', 'ProfileRow.test_row_and_config_rules', 'OpenAISurfaces.test_chat_passthrough_headers_and_provenance',
        'OpenAISurfaces.test_responses_store_forced_false_and_stream_events', 'AnthropicSurfaces.test_messages_pass_through_natively',
        'NarrowedSurfaces.test_anthropic_is_translated_when_the_operator_narrows_the_row', 'Hops.test_inbound_hop_stays_local_and_own_hop_is_a_loop',
        'BearerUpstream.test_auth_bearer_sends_the_bound_credential_only_there',
        'Connect.test_connect_creates_the_router_without_a_credential'), None),
    ('feature:router.catalog_capabilities', 'a router\'s per-route tool_calling, vision and context window from its live catalog: shown to clients and '
     'Studio, and tools or images to a route that declares them false are refused before egress',
     ev('check_broker_switchyard.py', 'ProfileRow.test_catalog_metadata_parser',
        'Catalog.test_route_capabilities_reach_the_catalog_and_refuse_before_egress'), None),
    ('feature:breakers.per_route', 'a switchyard row keeps one breaker per routed model: a failing route never takes the other routes offline',
     ev('check_broker_switchyard.py', 'OpenAISurfaces.test_errors_keep_their_meaning'), None),
    ('feature:native.switchyard', 'Switchyard\'s auxiliary endpoints (responses compact and input_tokens, decision, health) through the native lane; '
     'its operator surfaces (stats, stats reset, metrics) need broker:admin',
     ev('check_broker_switchyard.py', 'NativeLane.test_auxiliary_and_operator_endpoints'), None),
    ('feature:local.compat_fields', 'the local OpenAI and Anthropic routes accept the standard identity, cache, storage and reasoning fields and name '
     'them in X-Titanos-Ignored; store=true, previous_response_id, background, container and output_config.format answer a typed 400',
     ev('check_broker_switchyard.py', 'LocalTarget.test_chat_accepts_the_standard_fields_and_names_them', 'LocalTarget.test_refusals_stay_typed',
        'LocalTarget.test_responses_and_messages_accept_the_codex_and_claude_code_fields'), None),
    # provider completeness (owner direction 2026-10-02): native passthrough, one-step connect and per-provider rules
    ('feature:native.lane', 'native passthrough /native/<upstream>/<path>: body, repeated query and allowlisted headers forwarded untouched, ids bound to the '
     'creating key, refusals before egress (native scope, allowlist, store, query credentials, policy off, managed role, keyless without anonymous_cloud), '
     'provider errors scrubbed, a provider 401 is 502 upstream_auth_failed and marks the credential',
     ev('check_broker_native.py', 'PrefixLane.test_upload_reaches_the_provider_untouched_and_binds_the_owner', 'PrefixLane.test_every_argument_passes_untouched',
        'PrefixLane.test_refusals_before_egress', 'PrefixLane.test_list_with_broker_admin_and_bootstrap', 'PrefixLane.test_provider_errors',
        'PrefixLane.test_provider_401_is_upstream_auth_failed_and_marks_the_credential', 'NoStore.test_upstream_without_allow_store',
        'PolicyOff.test_cloud_disabled', 'Managed.test_managed_role_refuses', 'KeylessDefault.test_keyless_without_anonymous_cloud',
        'KeylessCloud.test_keyless_with_anonymous_cloud', 'Unconfigured.test_native_prefix_without_broker'), None),
    ('feature:native.bare_paths', 'a provider\'s own public path (OpenAI files, Anthropic batches with x-api-key, ElevenLabs /v2/voices and sound effects, Gemini '
     'resumable upload and :predict) reaches the single visible upstream that offers it; two candidates need X-Titanos-Upstream',
     ev('check_broker_native.py', 'BarePaths.test_bare_openai_path_with_one_candidate', 'BarePaths.test_anthropic_batches_with_x_api_key',
        'BarePaths.test_elevenlabs_v2_voices_and_sound_effects', 'BarePaths.test_gemini_resumable_upload_and_predict',
        'TwoCandidates.test_ambiguous_bare_path_needs_the_header'), None),
    ('feature:native.websocket', 'WebSocket relay (OpenAI Realtime, ElevenLabs stream-input) over the egress RFC 6455 client: a key inside the first client frame '
     'is removed, refusals happen before any upstream connection, a provider handshake 401 is typed, keyless cross-site origins are refused',
     ev('check_broker_native_ws.py', 'Relay.test_openai_realtime_relay', 'Relay.test_elevenlabs_stream_input_strips_a_key_in_the_first_frame',
        'Refusals.test_denials_before_any_upstream_connection', 'Refusals.test_provider_handshake_401', 'Refusals.test_http_request_to_a_websocket_row',
        'Keyless.test_cross_site_origin_refused_same_origin_relayed'), None),
    ('feature:observability.dropped_header', 'translated requests name the fields they dropped in X-Titanos-Dropped',
     ev('check_broker_native.py', 'DroppedHeader.test_translated_request_names_dropped_fields')
     + ev('check_broker_openai_completeness.py', 'Arguments.test_local_sampling_knobs_are_dropped_with_a_header'), None),
    ('feature:connect.keyed', 'one-step connect: POST /api/broker/connect creates, validates and enables the upstream and raises cloud_policy to explicit; '
     'GET /api/broker/env-candidates and `titanos broker quickstart` show variable names only; `titanos broker connect --from-env` offline',
     ev('check_broker_connect.py', 'Keyed.test_env_candidates_names_only', 'Keyed.test_connect_end_to_end_with_aliases', 'Keyed.test_connect_refusals',
        'Cli.test_quickstart_lists_names_only', 'Cli.test_offline_connect_from_env'), None),
    ('feature:connect.keyless_studio', 'keyless Studio provider setup (R29): with X-Titanos-Studio: 1 and JSON on a loopback standalone server, connect, credential '
     'and policy writes are accepted and audited as keyless-studio, connect turns anonymous_cloud on for the keyless owner (bare provider ids then work), '
     'host sources are limited to the profile well-known variable at its default address (no file:, no fingerprint), keyless cloud off is a typed 403; '
     'the header never elevates in managed role',
     ev('check_broker_connect.py', 'KeylessStudio.test_studio_connect_turns_keyless_cloud_on_and_bare_ids_work',
        'KeylessStudio.test_studio_host_sources_are_narrow', 'KeylessStudio.test_keyless_cloud_off_is_a_typed_403',
        'KeylessStudio.test_studio_credential_and_policy_writes',
        'ManagedKeyless.test_header_never_elevates_in_managed_role', 'ManagedKeyed.test_managed_role_refuses_connect'), None),
    ('feature:openai.media_streams', 'OpenAI image generation, image edit, transcription and speech event streams relayed with usage, a resumed stored Responses '
     'stream, and OpenAI\'s error frame after a failure past the first event',
     ev('check_broker_openai_completeness.py', 'MediaStreams.test_image_generation_stream_is_relayed_and_metered', 'MediaStreams.test_image_stream_failures',
        'MediaStreams.test_image_edit_stream_is_relayed', 'MediaStreams.test_a_partial_image_over_the_sse_line_limit_still_commits',
        'MediaStreams.test_transcription_stream_and_whisper_json', 'MediaStreams.test_speech_sse_is_relayed_with_token_usage',
        'MediaStreams.test_resumed_response_stream', 'StreamUnits.test_usage_observer_media_family', 'StreamUnits.test_responses_web_search_calls_are_counted',
        'StreamUnits.test_chat_audio_usage'), None),
    ('feature:openai.errors', 'OpenAI error meaning kept: a plain 403 passes through and never marks the credential, insufficient_quota is a billing stop '
     '(503 without Retry-After), 400 keeps code, type and param',
     ev('check_broker_openai_completeness.py', 'ErrorEnvelopes.test_plain_403_passes_through_with_the_provider_message',
        'ErrorEnvelopes.test_insufficient_quota_is_a_billing_stop_not_a_rate_limit', 'ErrorEnvelopes.test_400_keeps_code_type_and_param',
        'ErrorClassification.test_table', 'ErrorClassification.test_param_and_type_are_plain_identifiers_only'), None),
    ('feature:openai.arguments', 'OpenAI and Azure argument rules: local sampling and VAD fields dropped and named, reasoning-model sampling rules, unhostable '
     'url image requests refused before egress, the moderation default model, image JSON under the media cap',
     ev('check_broker_openai_completeness.py', 'Arguments.test_local_sampling_knobs_are_dropped_with_a_header', 'Arguments.test_reasoning_sampling_rules',
        'Arguments.test_vad_fields_are_dropped_on_cloud_transcription', 'Arguments.test_unhostable_url_requests_are_refused_before_egress',
        'Arguments.test_moderation_without_model_uses_the_provider_default',
        'ArgumentsTwoUpstreams.test_moderation_without_model_stays_model_required_when_ambiguous',
        'RewriteUnits.test_azure_deployments_follow_the_openai_rules', 'RewriteUnits.test_rows', 'MediaCap.test_images_json_uses_the_media_cap'), None),
    ('feature:native.openai', 'OpenAI native rows (Realtime mints, evals and ChatKit need broker:admin; custom voices and consents are stored and owner-bound; '
     'retired endpoints stay out) and the Azure OpenAI classic native lane with its api-version',
     ev('check_broker_openai_completeness.py', 'NativeRows.test_admin_rows_need_broker_admin', 'NativeRows.test_voice_consents_are_stored_rows_and_variations_stay_out',
        'AzureLane.test_native_files_reach_the_resource_with_api_version', 'AzureLane.test_azure_error_shapes'), None),
    ('feature:catalog.model_capabilities', 'catalog entries carry per-model modalities: titanos.model_capabilities on OpenAI and Azure models, per-model '
     'capabilities and published metadata on Gemini models',
     ev('check_broker_openai_completeness.py', 'Catalog.test_entries_carry_model_capabilities')
     + ev('check_broker_gemini_completeness.py', 'Codec.test_model_capabilities_by_family_and_published_methods',
          'LiveCatalog.test_capabilities_and_metadata_per_model'), None),
    ('feature:pricing.modality_tokens', 'audio and image token prices, 1-hour cache writes, web search requests and fast mode priced at their own rates',
     ev('check_broker_openai_completeness.py', 'PricingUnits.test_modality_tokens_at_their_own_prices')
     + ev('check_broker_anthropic_models.py', 'EstimateUsageAndPrices.test_usage_details_and_prices', 'EstimateUsageAndPrices.test_estimate'), None),
    ('feature:anthropic.model_rules', 'translations into Anthropic follow the target Claude model (thinking and effort, sampling, forced tool_choice, structured '
     'output, max_tokens floor, omitted thinking replay); every model family accepted by the mock\'s independent rules',
     ev('check_broker_anthropic_models.py', 'ModelTable.test_every_family_resolves', 'ModelTable.test_effort_mapping',
        'TranslatedBodies.test_every_family_gets_a_body_the_model_accepts', 'TranslatedBodies.test_current_models_lose_what_they_refuse_with_notes',
        'TranslatedBodies.test_older_models_keep_their_forms', 'TranslatedBodies.test_structured_outputs_and_default_cap',
        'TranslatedBodies.test_omitted_thinking_round_trip', 'DecodeAnthropic.test_output_config_adaptive_thinking_and_documents',
        'AdvisorInjection.test_shaped_per_model', 'EstimateUsageAndPrices.test_model_meta')
     + ev('check_broker_anthropic_platform.py', 'Translated.test_openai_client_to_opus_5_5'), None),
    ('feature:anthropic.platform', 'Anthropic Files and Message Batches through the native lane (results_url rewritten to this server), provider-side state in '
     'Messages bodies gated on cloud:store and owner-bound, Anthropic-shaped /v1/models, bare model ids from the Anthropic SDK, Managed Agents and the '
     'Admin API for broker:admin only',
     ev('check_broker_anthropic_platform.py', 'Files.test_upload_metadata_ownership_content_and_delete',
        'ProviderState.test_file_sources_and_containers_are_store_gated_and_owner_bound', 'Batches.test_results_url_is_rewritten_and_followed',
        'Batches.test_batch_naming_another_keys_file_is_refused', 'Catalog.test_anthropic_shaped_models', 'Translated.test_bare_model_id_from_the_anthropic_sdk',
        'AdminRows.test_managed_agents_and_admin_api')
     + ev('check_broker_anthropic_models.py', 'ProviderState.test_references'), None),
    ('feature:elevenlabs.typed', 'ElevenLabs typed surfaces: bare model ids on one visible upstream, text-to-speech without model_id to the provider default, every '
     'text-to-speech and speech-to-text argument forwarded, raw voice ids on OpenAI speech; 401 codes decide the credential mark (quota 503, permissions 502)',
     ev('check_broker_elevenlabs.py', 'ErrorMapping.test_401_codes_decide_the_credential_mark', 'ErrorMapping.test_401_quota_is_unavailable_and_permissions_is_auth',
        'ErrorMapping.test_current_envelope_code_is_read_first', 'Probe.test_restricted_or_exhausted_keys_validate',
        'Translation.test_raw_voice_id_passes_and_openai_names_still_need_a_mapping', 'Translation.test_speech_to_text_usage',
        'TypedSurfaces.test_bare_model_id_and_every_tts_argument', 'TypedSurfaces.test_tts_without_model_uses_the_provider_default',
        'TypedSurfaces.test_speech_to_text_forwards_every_field', 'TypedSurfaces.test_quota_and_permission_401s_keep_the_credential',
        'TypedSurfaces.test_openai_speech_with_a_raw_elevenlabs_voice', 'TwoUpstreams.test_bare_ids_are_ambiguous',
        'LocalSpeechKeepsPrecedence.test_model_absent_goes_to_the_local_route'), None),
    ('feature:native.elevenlabs', 'ElevenLabs native rows on the SDK\'s own paths: generation endpoints with character-cost usage, voices and owner-bound '
     'objects, the provider model list for xi-api-key callers, account rows for broker:admin, unlisted paths refused before egress',
     ev('check_broker_elevenlabs.py', 'Allowlist.test_sdk_paths_are_allowlisted_with_the_right_gates', 'NativeLane.test_models_listing_follows_the_sdk_header',
        'NativeLane.test_generation_endpoints', 'NativeLane.test_voices_and_owned_objects', 'NativeLane.test_unlisted_paths_are_refused_before_egress',
        'NativeLane.test_account_endpoints_need_broker_admin'), None),
    ('feature:gemini.codec', 'Gemini request and answer rules: both field spellings, thinking per model family, lossless JSON schemas, inline audio, generated '
     'images and code execution kept, media classified for the data policy, multi-MiB first frames and media-capped answers',
     ev('check_broker_gemini_completeness.py', 'Codec.test_snake_case_request_decodes_like_camel_case',
        'Codec.test_output_cap_and_thinking_follow_the_body_spelling_and_model', 'Codec.test_reasoning_maps_per_model_family',
        'Codec.test_lossless_json_schema_on_new_models', 'Codec.test_audio_input_round_trips_and_answers_keep_media',
        'Codec.test_stream_decode_notes_media_and_streams_code', 'Codec.test_data_policy_classifies_gemini_media', 'Codec.test_media_codecs',
        'Bridges.test_chat_answer_carries_generated_images', 'Bridges.test_large_first_image_frame_streams_through',
        'Bridges.test_budgeted_snake_case_body_gets_one_cap', 'Bridges.test_refusals_before_egress',
        'MediaCap.test_generate_content_answers_use_the_media_cap', 'Jobs.test_speech_and_image_jobs_on_gemini_models',
        'Codec.test_stage_transcription_through_gemini'), None),
    ('feature:native.gemini', 'Gemini native rows: the media download, Interactions and File Search stores; account-wide rows need broker:admin and storage '
     'redirects are refused',
     ev('check_broker_gemini_completeness.py', 'NativeRows.test_download_interactions_and_file_search',
        'NativeRows.test_account_wide_rows_need_admin_and_redirects_are_refused'), None),
    ('feature:compat.rules', 'OpenAI-compatible family rules: provider image URLs passed through, xAI reasoning-model field drops, local-engine knobs mapped or '
     'dropped, reasoning effort only to models that take it, DeepSeek reasoning_content round trip, Groq 498 and NVCF 202 mapping, asset jobs',
     ev(COMPAT, 'Images.test_url_native_providers_answer_their_own_urls', 'Arguments.test_xai_reasoning_models_never_get_penalties_or_stop',
        'Arguments.test_local_engine_knobs_are_mapped_or_dropped', 'Arguments.test_translated_reasoning_effort_only_reaches_models_that_take_it',
        'Arguments.test_deepseek_tool_loop_sends_reasoning_content_back', 'AssetJobs.test_together_image_and_groq_speech_jobs',
        'Errors.test_groq_498_is_a_rate_limit', 'Errors.test_nvcf_accepted_is_a_timeout_not_an_empty_success',
        'ProfileData.test_rows_declare_their_surfaces_extras_and_env_names', 'ProfileData.test_rewrite_body_rules', 'ProfileData.test_reasoning_effort_globs',
        'ProfileData.test_nim_rerank_codec', 'ProfileData.test_reasoning_content_round_trip_is_per_profile'), None),
    ('feature:native.compat', 'OpenAI-compatible family native rows (Perplexity Search, Groq files and the endpoints no typed surface covers) and provider '
     'extensions passed through untouched',
     ev(COMPAT, 'Arguments.test_perplexity_extensions_pass_through_and_native_search', 'Arguments.test_groq_files_through_the_native_lane',
        'ProfileData.test_native_rows_cover_the_endpoints_without_a_typed_surface'), None),
    ('feature:profiles.every_row', 'every profile row through its own adapter code: URL from the row default (or operator) base URL, auth header '
     'form, static and hop headers, model rewrite, include_usage, catalog parse and usage extraction (httpx MockTransport captures)',
     EVERY_ROW['openai_compat'] + EVERY_ROW['native'], None),
    ('feature:routing.data_policy', 'an upstream data policy that excludes a modality answers 403 data_policy_denied', ev(ROUTING, 'ExplainAndDataPolicy.test_data_policy_denial'), None),
    ('feature:routing.store_forced_false', 'store is forced false for Responses and chat unless cloud:store and allow_store', ev(ROUTING, 'ExplainAndDataPolicy.test_store_forced_false_for_responses_and_chat')
     + ev(ADAPTERS_FILE, 'OpenAIPassthrough.test_responses_store_forced_false_and_handle_bound_routes'), None),
    ('feature:routing.handles', 'Responses handles: owner-only, previous_response_id of another key is 404 with zero hits; TTL, cap and persistence',
     ev(ROUTING, 'Handles.test_previous_response_id_of_another_key_is_404_with_zero_hits', 'HandleTable.test_ttl_cap_persistence_and_ownership')
     + ev(HTTP, 'Responses.test_handle_affinity_and_owner_only') + ev(DISPATCH, 'HandleRoutes.test_handle_bound_routes_serve_the_owner_only'), None),
    ('feature:fallback.before_commit', 'typed fallback only before commit; never on 4xx, after an upload, for store=true or previous_response_id',
     ev(DISPATCH, 'LocalFallback.test_local_triggers_advance_only_before_commit', 'LocalFallback.test_local_4xx_and_success_never_fall_back',
        'CloudFallback.test_no_fallback_for_edits_store_or_previous_response_id', 'CloudFallback.test_anthropic_error_first_stream_falls_back_before_commit'), None),
    ('feature:streaming.keepalive_and_disconnect', 'keepalives only after commit; a client disconnect closes the upstream; mid-stream failure gives the terminal frame with no second hit',
     ev(DISPATCH, 'Streaming.test_no_keepalive_before_commit_and_keepalives_after_it', 'Streaming.test_client_disconnect_closes_the_upstream',
        'Streaming.test_mid_stream_disconnect_gives_the_terminal_frame_and_no_second_hit'), None),
    ('feature:streaming.whole_tool_calls', 'Gemini and Ollama tool calls are released whole at BlockStop', ev('check_broker_streaming.py', 'Incrementality.test_tool_call_streams_are_whole_for_gemini_and_ollama'), None),
    ('feature:breakers', 'per-upstream breakers: exponential durations with jitter, single half-open probe, 429 and 4xx never counted; open breakers skipped before egress',
     ev('check_broker_breakers.py', 'Breaker.test_opens_after_n_counted_failures', 'Breaker.test_exponential_durations_within_jitter_bounds', 'Breaker.test_single_half_open_probe',
        'Breaker.test_client_error_auth_and_429_are_never_counted', 'RouterSkipsOpenBreakers.test_open_breakers_and_cooldowns_are_skipped_before_egress'), None),
    ('feature:advisor', 'advisors reorder the chain, time out to the deterministic order and see no prompt text unless share_excerpt; the pending_contract slot is never called',
     ev('check_broker_advisor.py', 'Mapping.test_generic_v1_reorders_the_chain', 'Robustness.test_timeout_keeps_the_deterministic_order', 'Robustness.test_no_prompt_text_unless_share_excerpt',
        'Slot.test_pending_contract_is_never_called_and_a_url_is_409'), None),
    ('feature:reasoning_injection', 'the advisor reasoning mode maps to the target\'s reasoning field (Anthropic thinking budget, Gemini thinkingBudget) on passthrough and translated requests (R19)',
     ev(HTTP, 'Reasoning.test_passthrough_injection', 'Reasoning.test_translated_requests_carry_the_reasoning'), None),
    ('feature:images.rehosting', 'cloud image url answers are re-hosted as absolute owner-bound /v1/files URLs (PNG only; JPEG-only profiles refused before egress)',
     ev(HTTP, 'ImageRehosting.test_url_rehosting_is_owner_bound_and_jpeg_is_refused') + ev(DISPATCH, 'Images.test_png_url_rehosting_and_jpeg_refusal', 'Rehost.test_rehost_error_paths_leave_no_files'), None),
    # observability and audit
    ('feature:observability.provenance', 'provenance headers on every inference answer (served-by, requested and served model, fallback count, translation, decision id, request id)',
     ev(HTTP, 'Provenance.test_provenance_headers_on_cloud_and_local_answers') + ev(DISPATCH, 'Provenance.test_provenance_headers_on_every_path'), None),
    ('feature:observability.explain', 'POST /api/broker/explain returns the Decision with zero egress', ev(ROUTING, 'ExplainAndDataPolicy.test_explain_makes_zero_egress'), None),
    ('feature:observability.decisions', 'decisions: own key for broker:read, all for broker:admin; bounded ring', ev('check_broker_accounting.py', 'Decisions.test_visibility_own_versus_admin',
     'Decisions.test_ring_is_bounded_and_jsonl_optional'), None),
    ('feature:audit.chain', 'the audit log is a verifiable hash chain across days and writers; tampering is located; audit-first admin writes change nothing when the audit fails',
     ev('check_broker_audit.py', 'Chain.test_chain_verifies_and_records_are_canonical', 'Chain.test_tampered_line_is_detected_at_the_right_record', 'Chain.test_two_writer_processes_interleave_under_the_lock')
     + ev(AUTH_HTTP, 'AuditFirst.test_config_writes_with_an_unavailable_audit_change_nothing', 'AuditFirst.test_vault_writes_with_an_unavailable_audit_change_nothing'), None),
    # configuration
    ('feature:config.schema', 'broker.json v1 schema (extra forbidden), secret-shaped values refused anywhere, origin forms, locked atomic writes',
     ev('check_broker_config.py', 'Schema.test_round_trip', 'Schema.test_extra_fields_and_bad_values_refused', 'Secrets.test_secret_shape_anywhere_is_refused',
        'Origins.test_refused_forms', 'Files.test_update_holds_the_lock_and_retries_replace'), None),
    ('feature:config.settings', 'every TITANOS_BROKER_* and role setting parses and validates; describe() is secret-free',
     ev('check_broker_settings.py', 'BrokerSettings.test_every_environment_name_parses', 'BrokerSettings.test_bad_values_raise', 'BrokerSettings.test_tls_pair_and_proxy_rules',
        'BrokerSettings.test_describe_is_secret_free'), None),
    ('feature:disabled_lane', 'with the broker installed but unused, local inference answers are byte-identical; with policy off only provenance headers are added',
     ev(DISABLED, 'DisabledLane.test_a_no_broker_json_is_byte_identical', 'DisabledLane.test_b_cloud_upstream_with_policy_off_adds_only_provenance_headers')
     + ev(DISPATCH, 'FastPath.test_inactive_fast_path_touches_neither_scope_nor_body'), None),
    # operator surfaces
    ('feature:mcp.principal_propagation', 'MCP tools run as the outer principal of each message, never the bootstrap key; two keys on one session keep their own principal',
     ev('check_broker_mcp.py', 'Served.test_scoped_key_is_refused_on_admin_tools_and_never_elevated', 'Served.test_two_keys_on_one_session_keep_their_own_principal',
        'InProcess.test_mounted_call_without_principal_is_refused') + ev(AUTH_HTTP, 'MountedMcp.test_tools_run_as_the_outer_principal_never_the_bootstrap_key'), None),
    ('feature:mcp.tools', 'MCP broker tools and schemas; inference results carry provenance meta; open_world_hint fixed at startup',
     ev('check_broker_mcp.py', 'InProcess.test_tool_list_and_schemas', 'Served.test_inference_results_carry_meta_provenance', 'InProcess.test_open_world_hint_follows_the_startup_rule'), None),
    ('feature:cli.no_secret_argv', 'no CLI option takes a secret value; parser messages never echo a typed value',
     ev('check_broker_cli.py', 'ParserIntrospection.test_no_option_takes_a_secret_value', 'ParserIntrospection.test_parser_messages_never_echo_a_typed_value'), None),
    ('feature:cli.offline_and_url', 'broker, keys and audit commands offline (TITANOS_HOME files) and in --url mode',
     ev('check_broker_cli.py', 'Offline.test_offline_broker_configuration_and_secret_handling', 'Offline.test_offline_keys', 'UrlMode.test_url_mode_against_a_served_app'), None),
    ('feature:cli.serve_pinning', 'serve pins uvicorn proxy headers to behind_tls_proxy and forwarded_allow_ips to loopback', ev('check_broker_cli.py', 'Serve.test_uvicorn_kwargs_are_pinned'), None),
    ('feature:client.methods', 'Python client protocol, admin, key and audit methods; hint headers and provenance helper; credential writes refuse insecure transport',
     ev('check_broker_client.py', 'ProtocolMethods.test_anthropic_messages_and_count_tokens', 'ProtocolMethods.test_gemini_methods', 'ProtocolMethods.test_elevenlabs_methods',
        'AdminMethods.test_broker_reads', 'AdminMethods.test_keys_and_audit', 'HeaderContract.test_default_and_per_call_hints', 'HeaderContract.test_credential_writes_refuse_insecure_transport'), None),
    # end to end: a real `titanos serve` subprocess under the netguard sitecustomize, driven by raw httpx and the MCP client
    ('feature:e2e.openai_chat', 'real serve: OpenAI chat with tools, stream and non-stream', ev(E2E, 'KeyedStandalone.test_openai_chat_with_tools_stream_and_non_stream'), None),
    ('feature:e2e.openai_surfaces', 'real serve: Responses, embeddings, audio, images, moderations and models',
     ev(E2E, 'KeyedStandalone.test_openai_responses_embeddings_audio_images_moderations_models'), None),
    ('feature:e2e.anthropic_x_api_key', 'real serve: /v1/messages with x-api-key, passthrough and translated, streaming tool calls',
     ev(E2E, 'KeyedStandalone.test_anthropic_messages_passthrough_and_translated_with_streaming_tools'), None),
    ('feature:e2e.gemini_goog_header', 'real serve: generateContent and streamGenerateContent with x-goog-api-key',
     ev(E2E, 'KeyedStandalone.test_gemini_generate_and_stream_with_the_native_header'), None),
    ('feature:e2e.elevenlabs_xi_api_key', 'real serve: text-to-speech, /stream and speech-to-text with xi-api-key',
     ev(E2E, 'KeyedStandalone.test_elevenlabs_tts_stream_and_stt_with_xi_api_key'), None),
    ('feature:e2e.ollama', 'real serve: /api/chat passthrough, stream and translated', ev(E2E, 'KeyedStandalone.test_ollama_chat_passthrough_stream_and_translated'), None),
    ('feature:e2e.mcp_scoped_key', 'real serve: the official MCP client over /mcp with a scoped key (chat, broker_status, broker_explain)',
     ev(E2E, 'KeyedStandalone.test_mcp_chat_status_and_explain_with_a_scoped_key'), None),
    ('feature:e2e.error_envelopes', 'real serve: error envelopes per family with zero upstream hits', ev(E2E, 'KeyedStandalone.test_error_envelopes_per_family_with_zero_hits'), None),
    ('feature:e2e.limits_and_breakers', 'real serve: budget, visibility and scope refusals before any hit; breaker opens and resets; fallback only before the first byte',
     ev(E2E, 'KeyedStandalone.test_budget_visibility_and_scope_refusals_happen_before_any_hit', 'KeyedStandalone.test_breaker_opens_answers_without_a_hit_and_resets',
        'KeyedStandalone.test_virtual_model_falls_back_before_the_first_byte_only'), None),
    ('feature:e2e.keyless', 'real serve in keyless mode: placeholder and absent keys reach the cloud alias; key and credential writes refused',
     ev(E2E, 'KeylessStandalone.test_placeholder_and_absent_keys_reach_the_cloud_alias', 'KeylessStandalone.test_keyless_writes_are_refused'), None),
    ('feature:e2e.managed_role', 'real serve in managed role: cloud refused with zero hits, sealed vault, delegation conditions, parent deadline 504',
     ev(E2E, 'ManagedRole.test_cloud_targets_are_refused_with_zero_hits_and_the_parent_trace_echoed', 'ManagedRole.test_sealed_vault_public_routes_and_refused_writes',
        'ManagedDelegation.test_delegated_explicit_target_reaches_the_mock', 'ManagedDelegation.test_missing_conditions_are_refused_without_a_hit',
        'ManagedDelegation.test_parent_deadline_gives_504'), None),
    ('feature:e2e.broker_role', 'real serve in broker role: cloud keys need a budget and an expiry; the pre-dispatch budget stops a minted key',
     ev(E2E, 'BrokerRole.test_cloud_keys_need_a_budget_and_an_expiry', 'BrokerRole.test_the_pre_dispatch_budget_stops_a_minted_key'), None),
    ('feature:e2e.netguard', 'the serve subprocess loads the netguard (self-test) and every served case ends with a clean shutdown and zero non-loopback attempts',
     ev(E2E, 'GuardSelfTest.test_the_subprocess_guard_refuses_and_reports', 'KeyedStandalone.test_zz_clean_shutdown_and_zero_non_loopback_attempts',
        'KeylessStandalone.test_zz_clean_shutdown_and_zero_non_loopback_attempts', 'ManagedRole.test_zz_clean_shutdown_and_zero_non_loopback_attempts',
        'BrokerRole.test_zz_clean_shutdown_and_zero_non_loopback_attempts'), None),
    # the official SDKs against a real serve plus the mocks (tests/requirements-sdk.txt; untried where an SDK is missing)
    ('feature:sdk.openai', 'official OpenAI SDK: models, chat with tools (stream and not), Responses event family, embeddings, images, speech, '
     'transcription, moderations, translated chat and audio, typed errors; 402 and 508 sent once; upstream 400 scrubbed, cut stream, 429',
     ev('check_broker_sdk.py', 'openai.models', 'openai.chat.tools', 'openai.chat.stream_tools', 'openai.responses.stream',
        'openai.responses.stream_tools', 'openai.embeddings', 'openai.images', 'openai.audio.speech', 'openai.audio.transcriptions',
        'openai.moderations', 'openai.translated.chat_tools', 'openai.translated.audio', 'errors.not_found', 'errors.authentication',
        'errors.budget_402_not_retried', 'errors.loop_508_not_retried', 'upstream.400_scrubbed', 'upstream.stream_cut', 'upstream.429'), None),
    ('feature:sdk.mcp', 'official MCP client over streamable HTTP with a scoped key', ev('check_broker_sdk.py', 'mcp.streamable_http'), None),
    ('feature:sdk.anthropic', 'official Anthropic SDK with x-api-key: messages with tools (stream and not), count_tokens, translated to OpenAI, '
     'typed errors, 402 and 508 refusals',
     ev('check_broker_sdk_vendors.py', 'anthropic.auth_header', 'anthropic.messages.tools', 'anthropic.messages.stream_tools',
        'anthropic.count_tokens', 'anthropic.translated_openai.stream_tools', 'anthropic.errors', 'refusals.anthropic'), None),
    ('feature:sdk.gemini', 'official google-genai SDK with x-goog-api-key: generateContent with tools, streamGenerateContent (alt=sse), '
     'embeddings, countTokens, models, translated to OpenAI, typed errors, refusals',
     ev('check_broker_sdk_vendors.py', 'gemini.auth_header', 'gemini.generateContent.tools', 'gemini.streamGenerateContent',
        'gemini.streamGenerateContent.tools', 'gemini.embedContent', 'gemini.countTokens', 'gemini.models', 'gemini.translated_openai',
        'gemini.errors', 'refusals.gemini'), None),
    ('feature:sdk.elevenlabs', 'official ElevenLabs SDK with xi-api-key: text-to-speech, streaming, speech-to-text, voices, translated to '
     'OpenAI, typed errors, refusals',
     ev('check_broker_sdk_vendors.py', 'elevenlabs.auth_header', 'elevenlabs.tts', 'elevenlabs.tts_stream', 'elevenlabs.stt',
        'elevenlabs.voices', 'elevenlabs.translated_openai', 'elevenlabs.errors', 'refusals.elevenlabs'), None),
    ('feature:sdk.local_models', 'the official SDKs against local fixture models through the same serve (needs TITANOS_SDK_FIXTURE_MODELS)',
     ev('check_broker_sdk.py', 'local.openai.chat', 'local.openai.embeddings', 'local.openai.audio.speech', 'local.openai.audio.transcriptions')
     + ev('check_broker_sdk_vendors.py', 'local.anthropic.messages', 'local.anthropic.count_tokens', 'local.gemini.generateContent',
          'local.gemini.embedContent', 'local.elevenlabs.tts', 'local.elevenlabs.stt'), None),
    ('feature:sdk.lane_hygiene', 'SDK lanes: netguard active in server and client, zero non-loopback attempts, each mock saw only its own '
     'credential, no secret in any output, clean shutdown',
     ev('check_broker_sdk.py', 'netguard.active', 'netguard.zero_non_loopback', 'mocks.credentials_exact', 'outputs.no_secret_values',
        'serve.clean_shutdown'), None),
    ('feature:studio.providers_panel', 'Studio Cloud and routing panel: write-only credential form, env-sourced credential, cloud alias chat with '
     'the provenance line, design spot checks and the keyless CLI note (Playwright against a real serve; and a DOM-double contract check)',
     ev('check_studio_broker_browser.js', 'Studio.providers_panel_against_a_real_serve') + ev('check_broker_ui.js', 'Panel.dom_double_contract'), None),
]
UNTRIED_STUDIO = 'no Studio check reported (tests/check_broker_ui.js and tests/check_studio_broker_browser.js run with node)'


# ---------------------------------------------------------------------------------------------------------------------
# running the checks under the recording shim

JS_CHECKS = ('check_broker_ui.js', 'check_studio_broker_browser.js')  # node; they append their own results lines
CHECK_NAME = re.compile(r'check_(?:broker|studio_broker)_[a-z0-9_]+\.(?:py|js)')


def discover_checks():
    python = [path for path in sorted((ROOT / 'tests').glob('check_broker_*.py')) if path.name not in EXCLUDED_CHECKS]
    return python + [ROOT / 'tests' / name for name in JS_CHECKS if (ROOT / 'tests' / name).is_file()]


def _test_id(test):
    method = getattr(test, '_testMethodName', None)
    if method:
        return f'{type(test).__name__}.{method}'
    description = getattr(test, 'description', None) or str(test)  # _ErrorHolder (setUpClass/module failures)
    return 'error:' + re.sub(r'\s+', ' ', str(description))[:160]


_PATHISH = re.compile(r'(?:[A-Za-z]:[\\/]|/(?:home|Users|tmp|var|private)/|\\\\)[^\s\'",;)]*')


def _clean_reason(text):
    return _PATHISH.sub('<path>', re.sub(r'\s+', ' ', str(text))).strip()[:200]


def exec_check(path):
    """Run one check module as __main__ with every unittest outcome recorded, then append one merged line to
    TITANOS_BROKER_RESULTS: {check, returncode, passed, failed, skipped, cells}. A check that reports its own matrix cells
    (tests/broker_adapter_support.main) writes them to a private file, which is read back and merged."""
    import runpy
    import unittest
    path = Path(path).resolve()
    target = os.environ.get('TITANOS_BROKER_RESULTS')
    if not target:
        raise SystemExit('--exec-check needs TITANOS_BROKER_RESULTS')
    outcome = {'passed': set(), 'failed': set(), 'skipped': {}}
    base = unittest.TestResult

    def wrap(name, record):
        original = getattr(base, name)

        def method(self, test, *args):
            record(test, *args)
            return original(self, test, *args)
        setattr(base, name, method)
    wrap('addSuccess', lambda test: outcome['passed'].add(_test_id(test)))
    wrap('addFailure', lambda test, err: outcome['failed'].add(_test_id(test)))
    wrap('addError', lambda test, err: outcome['failed'].add(_test_id(test)))
    wrap('addSkip', lambda test, reason: outcome['skipped'].__setitem__(_test_id(test), _clean_reason(reason)))
    wrap('addExpectedFailure', lambda test, err: outcome['passed'].add(_test_id(test)))
    wrap('addUnexpectedSuccess', lambda test: outcome['failed'].add(_test_id(test)))
    wrap('addSubTest', lambda test, subtest, err: err is not None and outcome['failed'].add(_test_id(test)))
    own = Path(tempfile.mkdtemp(prefix='titanos-broker-cells-')) / 'cells.jsonl'
    os.environ['TITANOS_BROKER_RESULTS'] = str(own)
    sys.argv = [str(path)]
    sys.path[0] = str(path.parent)
    code = 0
    try:
        runpy.run_path(str(path), run_name='__main__')
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    finally:
        os.environ['TITANOS_BROKER_RESULTS'] = target
    cells = {}
    if own.is_file():
        for line in own.read_text(encoding='utf-8').splitlines():
            if line.strip():
                record = json.loads(line)
                cells.update(record.get('cells') or {})
                outcome['passed'].update(record.get('passed') or [])
                outcome['failed'].update(record.get('failed') or [])
                skipped = record.get('skipped') or {}
                if isinstance(skipped, list):
                    skipped = {name: 'skipped' for name in skipped}
                outcome['skipped'].update({name: _clean_reason(reason) for name, reason in skipped.items()})
        own.unlink()
    own.parent.rmdir()
    passed = outcome['passed'] - outcome['failed'] - set(outcome['skipped'])
    line = {'check': path.name, 'returncode': code, 'passed': sorted(passed), 'failed': sorted(outcome['failed']),
            'skipped': dict(sorted(outcome['skipped'].items())), 'cells': dict(sorted(cells.items()))}
    with open(target, 'a', encoding='utf-8') as handle:
        handle.write(json.dumps(line, sort_keys=True) + '\n')
    return code


def _child_env(results):
    env = dict(os.environ)
    env['TITANOS_BROKER_RESULTS'] = str(results)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'
    src = str(ROOT / 'src')
    env['PYTHONPATH'] = src + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    return env


def run_js(path, results, env, python):
    """Run one JavaScript check with node (it appends its own results line): (returncode, output). Without node the check
    is recorded as skipped."""
    import shutil
    node = shutil.which('node')
    if node is None:
        with open(results, 'a', encoding='utf-8') as handle:
            handle.write(json.dumps({'check': path.name, 'passed': [], 'failed': [], 'skipped': {'run': 'node is not on PATH'}}) + '\n')
        return 0, 'node is not on PATH'
    child = dict(env, TITANOS_PYTHON=env.get('TITANOS_PYTHON') or python)
    try:
        done = subprocess.run([node, str(path)], cwd=ROOT, env=child, capture_output=True, text=True, encoding='utf-8',
                              errors='replace', timeout=CHECK_TIMEOUT_SECONDS)
        return done.returncode, (done.stdout or '') + (done.stderr or '')
    except subprocess.TimeoutExpired:
        return 124, f'timed out after {CHECK_TIMEOUT_SECONDS} s'


def run_checks(results, *, jobs=1, python=None, log=None, checks=None):
    """Run every broker check (or `checks`) under the shim, appending to `results`. Returns {check: returncode}."""
    python = python or sys.executable
    checks = list(checks or discover_checks())
    env = _child_env(results)

    def one(path):
        if path.suffix == '.js':
            code, output = run_js(path, results, env, python)
            if log is not None:
                log(path.name, code, output)
            return path.name, code
        command = [python, '-B', str(Path(__file__).resolve()), '--exec-check', str(path)]
        try:
            done = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, encoding='utf-8', errors='replace',
                                  timeout=CHECK_TIMEOUT_SECONDS)
            code, output = done.returncode, (done.stdout or '') + (done.stderr or '')
        except subprocess.TimeoutExpired:
            code, output = 124, f'timed out after {CHECK_TIMEOUT_SECONDS} s'
        if log is not None:
            log(path.name, code, output)
        return path.name, code
    if jobs <= 1:
        return dict(one(path) for path in checks)
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        return dict(pool.map(one, checks))


# ---------------------------------------------------------------------------------------------------------------------
# results and live evidence

def read_results(path):
    """The run record from a results JSONL file: {check: {returncode, passed, failed, skipped, cells}}. Several lines for
    one check (a direct S.main line and a shim line) are merged; a failure anywhere wins over a pass."""
    checks = {}
    for number, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        name = record.get('check')
        if not isinstance(name, str) or not CHECK_NAME.fullmatch(name):
            raise SystemExit(f'{path}:{number}: not a broker check results line')
        entry = checks.setdefault(name, {'returncode': 0, 'passed': set(), 'failed': set(), 'skipped': {}, 'cells': {}})
        entry['passed'].update(record.get('passed') or [])
        entry['failed'].update(record.get('failed') or [])
        entry['skipped'].update({k: _clean_reason(v) for k, v in (record.get('skipped') or {}).items()})
        entry['cells'].update(record.get('cells') or {})
        if record.get('returncode'):
            entry['returncode'] = record['returncode']
        elif 'returncode' not in record and record.get('failed'):
            entry['returncode'] = 1
    return {name: {'returncode': e['returncode'], 'passed': sorted(e['passed'] - e['failed'] - set(e['skipped'])),
                   'failed': sorted(e['failed']), 'skipped': dict(sorted(e['skipped'].items())), 'cells': dict(sorted(e['cells'].items()))}
            for name, e in sorted(checks.items())}


LIVE_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9._:-]{2,95}')


def read_live(path):
    """Live evidence records (JSON list or JSONL): {evidence_id, outcome:'pass', cell} or {evidence_id, outcome, profile, surface,
    stream:'off'|'on'}, optional observed_on (YYYY-MM-DD) and note. Values must be public-safe: no secret shape, no path.
    Observation records (outcome 'observation') are validated too but are not evidence: read_live_files returns them."""
    return read_live_files([path])[0]


def read_live_files(paths):
    """(evidence records, observation records) of one or more live evidence files (read_live). An observation record is
    {evidence_id, outcome:'observation', cell or profile/surface/stream, observed_on, note} with a required note: a live
    call that did not pass (it is listed on its cell and never changes the cell's state). An id may appear once across
    all files; any other outcome (a failed live call recorded as 'fail') is refused."""
    _use_src()
    from titanos.broker.redact import scrub
    records = []
    for path in paths:
        text = Path(path).read_text(encoding='utf-8').strip()
        records += json.loads(text) if text.startswith('[') else [json.loads(line) for line in text.splitlines() if line.strip()]
    out, observations, seen = [], [], set()
    for record in records:
        if not isinstance(record, dict) or set(record) - {'evidence_id', 'outcome', 'cell', 'profile', 'surface', 'stream', 'observed_on', 'note'}:
            raise SystemExit(f'live evidence record has unknown fields: {record!r}'[:200])
        evidence_id = record.get('evidence_id')
        if not isinstance(evidence_id, str) or not LIVE_ID.fullmatch(evidence_id) or evidence_id in seen:
            raise SystemExit(f'live evidence id {evidence_id!r} is missing, malformed or repeated')
        seen.add(evidence_id)
        outcome = record.get('outcome')
        if outcome not in ('pass', 'observation'):
            raise SystemExit(f'live evidence {evidence_id}: only outcome pass is recorded (a failed live call is not evidence; '
                             'record it as outcome observation with a note)')
        if outcome == 'observation' and not (isinstance(record.get('note'), str) and record['note'].strip()):
            raise SystemExit(f'live observation {evidence_id}: a note is required')
        if 'cell' in record:
            cell = record['cell']
        else:
            cell = f"{record.get('surface')}|{record.get('profile')}|{record.get('stream', 'off')}"
        for value in record.values():
            if isinstance(value, str) and (scrub(value) != value or _PATHISH.search(value)):
                raise SystemExit(f'live evidence {evidence_id}: a value looks like a secret or a path')
        if record.get('observed_on') is not None and not re.fullmatch(r'\d{4}-\d{2}-\d{2}', str(record['observed_on'])):
            raise SystemExit(f'live evidence {evidence_id}: observed_on must be YYYY-MM-DD')
        entry = {'evidence_id': evidence_id, 'cell': cell, 'observed_on': record.get('observed_on'), 'note': record.get('note')}
        (out if outcome == 'pass' else observations).append(entry)
    return sorted(out, key=lambda r: r['evidence_id']), sorted(observations, key=lambda r: r['evidence_id'])


# ---------------------------------------------------------------------------------------------------------------------
# cells

class Evidence:
    """Resolves evidence ids and reported cell keys against the run record."""

    def __init__(self, run):
        self.run = run
        self.reported = {}
        for check, entry in run.items():
            for key, test in entry['cells'].items():
                self.reported.setdefault(key, []).append(f'{check}::{test}')
        self.failed_cited = []

    def status(self, evidence_id):
        check, _, test = evidence_id.partition('::')
        entry = self.run.get(check)
        if entry is None:
            return 'missing', f'{check} is not in the results file'
        if test in entry['passed']:
            return 'passed', None
        if test in entry['skipped']:
            return 'skipped', f'{evidence_id} was skipped on this host: {entry["skipped"][test]}'
        if test in entry['failed']:
            self.failed_cited.append(evidence_id)
            return 'failed', f'{evidence_id} failed in this run'
        return 'missing', f'{evidence_id} did not run (no such test in the results file)'

    def resolve(self, ids):
        """(state 'pass'|'untried', reason): pass only when every id passed."""
        if not ids:
            return 'untried', 'no check cites this cell'
        for evidence_id in ids:
            state, reason = self.status(evidence_id)
            if state != 'passed':
                return 'untried', reason
        return 'pass', None


def _variants(surface):
    if surface.stream_from == 'json:stream':
        return ('off', 'on')
    if surface.stream_from.startswith('path:'):
        return ('on',)
    return ('off',)


def _column(profile):
    if profile.id == 'azure_openai':
        return 'AZ'
    if profile.extras.get('surface_adapters'):
        return 'SY'
    return COLUMNS.get(profile.adapter)


def _adapter_classes():
    from titanos.broker.adapters.anthropic import AnthropicAdapter
    from titanos.broker.adapters.elevenlabs import ElevenLabsAdapter
    from titanos.broker.adapters.gemini import GeminiAdapter
    from titanos.broker.adapters.ollama_native import OllamaNativeAdapter
    from titanos.broker.adapters.openai_compat import OpenAICompatAdapter
    return {'openai_compat': OpenAICompatAdapter, 'anthropic': AnthropicAdapter, 'gemini': GeminiAdapter,
            'elevenlabs': ElevenLabsAdapter, 'ollama_native': OllamaNativeAdapter}


def mode_for(profile, surface, classes):
    """('P'|'T'|'na', reason) from the real adapter: native surfaces pass through, supports() means translated."""
    if profile.adapter == 'none':
        return 'na', 'pending_contract slot: no wire contract is assumed (503 upstream_unconfigured)'
    adapter = classes[profile.adapter](profile, None)
    if surface.name in profile.surfaces:
        return 'P', None
    if adapter.supports(profile, surface):
        if profile.adapter == 'ollama_native' and surface.name == 'ollama.embeddings':
            return 'P', None  # the legacy Ollama embeddings shape is served by the native /api/embed route
        return 'T', None
    return 'na', f'the {profile.adapter} adapter neither serves nor translates {surface.name} for this profile (400 broker_capability_unavailable before egress)'


def provider_keys(surface, column, mode, variant):
    """The cell keys the WP5 checks report for one provider cell."""
    name = surface.name
    if mode == 'P':
        if variant == 'on' and surface.stream_from == 'json:stream':
            return [f'{name}|{column}|P|stream']
        return [f'{name}|{column}|P']
    keys = []
    if variant == 'off' or surface.stream_from.startswith('path:'):
        keys.append(f'{name}|{column}|T')
    if variant == 'on':
        alias = STREAM_ALIAS.get(name)
        source = 'OAI-chat' if column == 'OAI' and name in VIA_CHAT else column
        keys.append(f'stream:{alias or name}<-{source}')
    if name == 'openai.audio.speech' and column == 'XI':
        keys.append('stream:openai.speech<-XI')
    return keys


def provider_cells(evidence, classes):
    from titanos.broker.profiles import PROFILES
    from titanos.broker.surfaces import SURFACES
    row_evidence = {}
    for profile_id, surface_name, variant, ids in PROFILE_ROW_EVIDENCE + every_row_evidence():
        if profile_id not in PROFILES or surface_name not in SURFACES:
            raise SystemExit(f'PROFILE_ROW_EVIDENCE names an unknown profile or surface: {profile_id} {surface_name}')
        row_evidence.setdefault((profile_id, surface_name, variant), []).extend(ids)
    cells = {}
    for surface in SURFACES.values():
        if surface.kind != 'intercept':
            continue
        for profile in PROFILES.values():
            column = _column(profile)
            mode, reason = mode_for(profile, surface, classes)
            for variant in _variants(surface):
                cell_id = f'{surface.name}|{profile.id}|{variant}'
                cell = {'section': 'provider', 'surface': surface.name, 'protocol': surface.protocol, 'profile': profile.id,
                        'adapter': profile.adapter, 'column': column, 'profile_kind': profile.kind, 'stream': variant, 'mode': mode,
                        'cites': [], 'evidence': [], 'proof': None, 'reason': None, 'live': []}
                if mode == 'na':
                    if (profile.id, surface.name, variant) in row_evidence:
                        raise SystemExit(f'PROFILE_ROW_EVIDENCE cites an na cell: {cell_id}')
                    cell.update(state='na', reason=reason)
                    cells[cell_id] = cell
                    continue
                # a split-protocol row translates onto chat with the OpenAI-compatible adapter's own code (OAI evidence)
                key_column = 'OAI' if column == 'SY' and mode == 'T' else column
                keys = provider_keys(surface, key_column, mode, variant)
                cell['cites'] = keys
                missing = [key for key in keys if key not in evidence.reported]
                ids = sorted({i for key in keys for i in evidence.reported.get(key, [])})
                state, why = evidence.resolve(ids) if not missing else ('untried', f'no check reports cell {missing[0]}')
                own = row_evidence.get((profile.id, surface.name, variant), [])
                own_state = evidence.resolve(own)[0] if own else 'untried'
                if missing and own_state == 'pass':
                    # no column proof exists (the column's reference row does not serve the surface, e.g. rerank on
                    # the OpenAI-compatible column, image edits on Azure): the row's own passing tests prove the cell
                    state, why = 'pass', None
                cell['evidence'] = sorted(set(ids) | (set(own) if own_state == 'pass' else set()))
                cell['state'], cell['reason'] = state, why
                if state == 'pass':
                    cell['proof'] = 'profile' if (REFERENCE_PROFILE.get(key_column) == profile.id or own_state == 'pass') else 'column'
                cells[cell_id] = cell
    return cells


def local_cells(evidence):
    from titanos.broker.surfaces import SURFACES
    intercept = {s.name for s in SURFACES.values() if s.kind == 'intercept'}
    if set(LOCAL_MODES) != intercept:
        raise SystemExit(f'LOCAL_MODES must name every intercept surface: missing {sorted(intercept - set(LOCAL_MODES))}, '
                         f'unknown {sorted(set(LOCAL_MODES) - intercept)}')
    for surface_name, variant in LOCAL_EVIDENCE:
        if surface_name not in LOCAL_MODES or variant not in _variants(SURFACES[surface_name]):
            raise SystemExit(f'LOCAL_EVIDENCE names an unknown cell {surface_name} {variant}')
    cells = {}
    for name, (mode, note) in sorted(LOCAL_MODES.items()):
        surface = SURFACES[name]
        for variant in _variants(surface):
            ids = sorted(LOCAL_EVIDENCE.get((name, variant), []))
            cell = {'section': 'local', 'surface': name, 'protocol': surface.protocol, 'profile': 'local', 'stream': variant, 'mode': mode,
                    'note': note, 'evidence': [], 'reason': None, 'live': []}
            if mode == 'na':
                verified = ids and evidence.resolve(ids)[0] == 'pass'
                cell.update(state='na', reason=note, evidence=ids if verified else [])
            else:
                state, why = evidence.resolve(ids)
                if state == 'untried' and not ids:
                    why = 'no broker check exercises this local path (the existing route suites cover the local handler outside this matrix)'
                cell.update(state=state, reason=why, evidence=ids if state == 'pass' else [])
            cells[f'{name}|local|{variant}'] = cell
    return cells


def explicit_cells(section, rows, evidence, *, untried_reason=None):
    cells = {}
    for cell_id, title, ids, na_reason in rows:
        if cell_id in cells:
            raise SystemExit(f'duplicate {section} cell {cell_id}')
        ids = sorted(ids)
        cell = {'section': section, 'title': title, 'evidence': [], 'reason': None, 'live': []}
        if na_reason:
            verified = ids and evidence.resolve(ids)[0] == 'pass'
            cell.update(state='na', reason=na_reason, evidence=ids if verified else [])
        else:
            state, why = evidence.resolve(ids)
            if not ids and untried_reason:
                why = untried_reason
            cell.update(state=state, reason=why, evidence=ids if state == 'pass' else [])
        cells[cell_id] = cell
    return cells


def grid_cells(section, prefix, table, columns, evidence):
    rows = []
    for name, (title, by_column, na_by_column) in table.items():
        for column in columns:
            if column in na_by_column:
                reason = na_by_column[column]
                if reason is None:
                    continue  # the rendering does not exist for this protocol
                rows.append((f'{prefix}:{name}|{column}', title, [], reason))
                continue
            ids = list(by_column.get('*', [])) + list(by_column.get(column, []))
            if not by_column.get(column) and '*' not in by_column:
                ids = []
            rows.append((f'{prefix}:{name}|{column}', title, ids, None))
    return explicit_cells(section, rows, evidence)


def apply_live(cells, live):
    for record in live:
        cell = cells.get(record['cell'])
        if cell is None:
            raise SystemExit(f"live evidence {record['evidence_id']} names an unknown cell {record['cell']}")
        if cell['state'] == 'na':
            raise SystemExit(f"live evidence {record['evidence_id']} names an na cell {record['cell']}")
        cell['live'].append(record['evidence_id'])
        cell['state'] = 'live-pass'
        cell['reason'] = None


def apply_observations(cells, observations):
    """List each live observation on the cell it names (live_notes); the cell's state never changes."""
    for record in observations:
        cell = cells.get(record['cell'])
        if cell is None:
            raise SystemExit(f"live observation {record['evidence_id']} names an unknown cell {record['cell']}")
        if cell['state'] == 'na':
            raise SystemExit(f"live observation {record['evidence_id']} names an na cell {record['cell']}")
        cell.setdefault('live_notes', []).append(record['evidence_id'])


def label(cell):
    if cell['state'] == 'pass':
        return 'pass (mock, adapter column)' if cell.get('proof') == 'column' else 'pass (mock)'
    return cell['state']


def _use_src():
    if str(ROOT / 'src') not in sys.path:
        sys.path.insert(0, str(ROOT / 'src'))


def assemble(run, live, observations=()):
    _use_src()
    import titanos
    from titanos.broker.profiles import DEFERRED, PROFILES
    from titanos.broker.surfaces import SURFACES
    evidence = Evidence(run)
    classes = _adapter_classes()
    cells = {}
    for part in (provider_cells(evidence, classes), local_cells(evidence),
                 explicit_cells('catalog', CATALOG_CELLS, evidence), explicit_cells('stage', STAGE_CELLS, evidence),
                 grid_cells('error', 'error', ERROR_CLASSES, ERROR_ADAPTERS, evidence),
                 grid_cells('envelope', 'envelope', ENVELOPES, ENVELOPE_PROTOCOLS, evidence),
                 explicit_cells('role', ROLE_CELLS, evidence),
                 explicit_cells('feature', FEATURE_CELLS[:-1], evidence),
                 explicit_cells('feature', FEATURE_CELLS[-1:], evidence, untried_reason=UNTRIED_STUDIO)):
        overlap = set(part) & set(cells)
        if overlap:
            raise SystemExit(f'cell ids repeat across sections: {sorted(overlap)[:5]}')
        cells.update(part)
    if evidence.failed_cited:
        raise SystemExit('cited tests failed in this run (fix them or rerun): ' + ', '.join(sorted(set(evidence.failed_cited))[:20]))
    failed_checks = sorted(name for name, entry in run.items() if entry['returncode'] or entry['failed'])
    if failed_checks:
        raise SystemExit('broker checks failed in this run: ' + ', '.join(failed_checks))
    apply_live(cells, live)
    observations = list(observations)
    apply_observations(cells, observations)
    for cell in cells.values():
        cell['label'] = label(cell)
    sections = Counter(c['section'] for c in cells.values())
    counts = {'cells': len(cells), 'by_state': {s: sum(1 for c in cells.values() if c['state'] == s) for s in STATES},
              'by_section': {section: {s: sum(1 for c in cells.values() if c['section'] == section and c['state'] == s) for s in STATES}
                             for section in sorted(sections)},
              'pass_mock_adapter_column': sum(1 for c in cells.values() if c['state'] == 'pass' and c.get('proof') == 'column'),
              'checks': len(run), 'tests_passed': sum(len(e['passed']) for e in run.values()),
              'tests_skipped': sum(len(e['skipped']) for e in run.values()), 'live_evidence': len(live),
              'live_observations': len(observations)}
    profiles = [{'id': p.id, 'adapter': p.adapter, 'column': _column(p), 'kind': p.kind, 'surfaces': sorted(p.surfaces),
                 'operator_surfaces': p.operator_surfaces} for p in PROFILES.values()]
    surfaces = [{'name': s.name, 'method': s.method, 'path': s.path, 'protocol': s.protocol, 'capability': s.capability, 'kind': s.kind,
                 'stream_variants': list(_variants(s))} for s in SURFACES.values()]
    run_record = {name: {'returncode': e['returncode'], 'passed': e['passed'], 'failed': e['failed'], 'skipped': e['skipped'],
                         'cells': e['cells']} for name, e in sorted(run.items())}
    return {'schema_version': 1,
            'basis': ('Generated by tools/build_broker_matrix.py from broker/surfaces.py, broker/profiles.py and the adapter classes (modes), the explicit '
                      'evidence tables in the generator, the run record of a fresh run of every tests/check_broker_*.py and the JavaScript Studio checks with TITANOS_BROKER_RESULTS set '
                      '(embedded under run) and the live evidence file (embedded under live_evidence). Every broker check is loopback-only under the netguard, '
                      'so every pass without a live evidence id is mock evidence and is labelled pass (mock). Deterministic: sorted keys, no timestamps; '
                      'tests/check_broker_matrix.py asserts the rebuild is a fixed point and that every cited test exists and passed.'),
            'source_version': titanos.__version__,
            'vocabulary': {'states': STATE_MEANING, 'modes': MODE_MEANING, 'proof': PROOF_MEANING, 'columns': COLUMN_MEANING,
                           'stream': {'off': 'the request does not stream (or the surface has no stream switch)',
                                      'on': 'the request streams (a stream field, or a surface that always streams)'},
                           'cell_ids': {'provider': '<surface>|<profile>|<off|on>', 'local': '<surface>|local|<off|on>', 'catalog': '<surface>|catalog',
                                        'stage': '<surface>|cloud[:<task>] or <area>|<scope>', 'error': 'error:<class>|<adapter>',
                                        'envelope': 'envelope:<rendering>|<protocol>', 'role': 'role:<role>|<behaviour>', 'feature': 'feature:<area>.<name>'},
                           'deferred_profiles': list(DEFERRED)},
            'run': {'platform': platform.system(), 'excluded_checks': list(EXCLUDED_CHECKS), 'checks': run_record},
            'live_evidence': live, 'live_observations': observations, 'profiles': profiles, 'surfaces': surfaces, 'counts': counts, 'cells': dict(sorted(cells.items()))}


# ---------------------------------------------------------------------------------------------------------------------
# rendering

def _cell_text(text):
    return str(text if text is not None else '').replace('|', '\\|').replace('\n', ' ').strip() or '-'


SHORT = {'pass': 'pass', 'live-pass': 'live', 'na': 'na', 'untried': 'untried'}


def _short(cell):
    if cell['state'] == 'pass' and cell.get('proof') == 'column':
        return 'pass~'
    return SHORT[cell['state']]


def _notes_text(cell):
    notes = cell.get('live_notes') or []
    return ('; live observation (not evidence): ' + ', '.join(f'`{n}`' for n in notes)) if notes else ''


def render_md(data):
    cells, c = data['cells'], data['counts']
    run = data['run']
    L = ['# Broker support matrix', '',
         'Basis: generated by `tools/build_broker_matrix.py` from the broker surface and profile tables, the real adapter classes, the evidence tables in the generator, '
         'a fresh run of every `tests/check_broker_*.py` and the JavaScript Studio checks with `TITANOS_BROKER_RESULTS` set and the live evidence file; the machine-readable form is `docs/broker_matrix.json`. '
         'Rebuild with `python -B tools/build_broker_matrix.py --run`; do not edit by hand. `tests/check_broker_matrix.py` asserts the rebuild is a fixed point and that every cited test exists and passed.', '',
         f"Evidence run: {c['checks']} broker checks on {run['platform']} ({c['tests_passed']} tests passed, {c['tests_skipped']} skipped, none failed); "
         f"excluded as consumers of this matrix: {', '.join('`' + x + '`' for x in run['excluded_checks'])}. "
         f"Live evidence records: {c['live_evidence']}" + ('. No cell claims a live call: every pass is mock evidence and reads **pass (mock)**.' if not c['live_evidence'] else '.')
         + (f" Live observations (not evidence; listed on their cells, which keep their state): {c['live_observations']}." if c.get('live_observations') else ''), '',
         f"Cells: {c['cells']}: " + ', '.join(f"{v} {k}" for k, v in c['by_state'].items()) + f"; {c['pass_mock_adapter_column']} of the passes are proven through the adapter column only.", '',
         '## Legend', '', 'States:', '']
    L += [f'- `{k}`: {v}' for k, v in data['vocabulary']['states'].items()]
    L += ['', 'Labels: `pass (mock)` (this row was driven by a cited test), `pass (mock, adapter column)` (shown as `pass~` in the grids; proven through the column reference row on the same adapter code), `live-pass`, `na`, `untried`.', '',
          'Modes:', ''] + [f'- `{k}`: {v}' for k, v in data['vocabulary']['modes'].items()]
    L += ['', 'Proof:', ''] + [f'- `{k}`: {v}' for k, v in data['vocabulary']['proof'].items()]
    L += ['', 'Adapter columns:', ''] + [f'- `{k}`: {v}' for k, v in data['vocabulary']['columns'].items()]
    L += ['', 'Cell ids:', ''] + [f'- {k}: `{v}`' for k, v in data['vocabulary']['cell_ids'].items()]
    L += ['', 'Profiles documented as not registered (absent, not stubbed): ' + ', '.join(f'`{p}`' for p in data['vocabulary']['deferred_profiles']) + '.', '',
          '## Summary by section', '', '| Section | ' + ' | '.join(STATES) + ' |', '|---|' + '---|' * len(STATES)]
    for section, by_state in c['by_section'].items():
        L.append(f'| {section} | ' + ' | '.join(str(by_state[s]) for s in STATES) + ' |')
    provider = {k: v for k, v in cells.items() if v['section'] == 'provider'}
    local = {k: v for k, v in cells.items() if v['section'] == 'local'}
    profiles = data['profiles']
    for protocol in ('openai', 'anthropic', 'gemini', 'ollama', 'elevenlabs'):
        names = [s['name'] for s in data['surfaces'] if s['kind'] == 'intercept' and s['protocol'] == protocol]
        L += ['', f'## Client protocol: {protocol}', '',
              'Rows are targets (profile, adapter column, kind); columns are the inbound surfaces. A cell reads `<mode> <state>`, with `off / on` states where the request may stream.', '',
              '| Target | ' + ' | '.join(f'`{n}`' for n in names) + ' |', '|---|' + '---|' * len(names)]
        targets = [('local', 'local', '-')] + [(p['id'], p['column'] or '-', p['kind']) for p in profiles]
        for target, column, kind in targets:
            row = []
            for name in names:
                variants = [v for v in ('off', 'on') if f'{name}|{target}|{v}' in cells]
                items = [cells[f'{name}|{target}|{v}'] for v in variants]
                mode = items[0]['mode']
                row.append(mode if mode == 'na' else f"{mode} " + ' / '.join(_short(i) for i in items))
            L.append(f'| `{target}` ({column}, {kind}) | ' + ' | '.join(row) + ' |')
        detail = [(k, v) for k, v in sorted(provider.items()) if v['protocol'] == protocol and v['state'] != 'na']
        detail += [(k, v) for k, v in sorted(local.items()) if v['protocol'] == protocol]
        L += ['', f'Cells served on {protocol} (na cells are summarised below):', '', '| Cell | Mode | State | Evidence or reason |', '|---|---|---|---|']
        for key, cell in detail:
            why = ', '.join(f'`{e}`' for e in cell['evidence'] + cell['live']) if cell['state'] in ('pass', 'live-pass') else _cell_text(cell['reason'])
            if cell['state'] == 'na':
                why = _cell_text(cell['reason'])
            L.append(f"| `{key}` | {cell['mode']} | {cell['label']} | {why}{_notes_text(cell)} |")
    reasons = Counter((v['reason']) for v in provider.values() if v['state'] == 'na')
    L += ['', '## Not applicable provider cells', '', '| Reason | Cells |', '|---|---|']
    for reason, count in sorted(reasons.items(), key=lambda item: (-item[1], item[0])):
        L.append(f'| {_cell_text(reason)} | {count} |')
    for section, title in (('catalog', 'Catalogs'), ('stage', 'API Plus stages and asset jobs'), ('role', 'Roles'), ('feature', 'Features')):
        L += ['', f'## {title}', '', '| Cell | State | Behaviour | Evidence or reason |', '|---|---|---|---|']
        for key, cell in sorted(cells.items()):
            if cell['section'] != section:
                continue
            why = ', '.join(f'`{e}`' for e in cell['evidence'] + cell['live'])
            if cell['state'] in ('na', 'untried'):
                why = _cell_text(cell['reason']) + (('; verified by ' + ', '.join(f'`{e}`' for e in cell['evidence'])) if cell['evidence'] else '')
            L.append(f"| `{key}` | {cell['label']} | {_cell_text(cell['title'])} | {why}{_notes_text(cell)} |")
    for section, title, columns, table in (('error', 'Upstream error classes by target adapter', ERROR_ADAPTERS, ERROR_CLASSES),
                                            ('envelope', 'Client-facing error renderings by inbound protocol', ENVELOPE_PROTOCOLS, ENVELOPES)):
        L += ['', f'## {title}', '', '| Class | ' + ' | '.join(f'`{col}`' for col in columns) + ' |', '|---|' + '---|' * len(columns)]
        for name, (meaning, _, _) in table.items():
            row = []
            for col in columns:
                cell = cells.get(f'{section}:{name}|{col}')
                row.append('-' if cell is None else _short(cell))
            L.append(f'| `{name}`: {_cell_text(meaning)} | ' + ' | '.join(row) + ' |')
        L += ['', '| Cell | State | Evidence or reason |', '|---|---|---|']
        for key, cell in sorted(cells.items()):
            if cell['section'] == section:
                why = ', '.join(f'`{e}`' for e in cell['evidence'] + cell['live']) if cell['state'] in ('pass', 'live-pass') else _cell_text(cell['reason'])
                L.append(f"| `{key}` | {cell['label']} | {why}{_notes_text(cell)} |")
    L += ['', '## Live evidence', '']
    if data['live_evidence']:
        L += ['| Evidence id | Cell | Observed on | Note |', '|---|---|---|---|']
        for record in data['live_evidence']:
            L.append(f"| `{record['evidence_id']}` | `{record['cell']}` | {record['observed_on'] or '-'} | {_cell_text(record['note'])} |")
    else:
        L.append('None yet. The owner-authorized live lane (`tools/broker_live_smoke.py`, manual tier, never in CI) adds records; until then no cell is live-pass.')
    if data.get('live_observations'):
        L += ['', '### Live observations (not evidence)', '',
              'Live calls that did not pass. Each is listed on the cell it names; the cell keeps the state its tests and live evidence give it.', '',
              '| Observation id | Cell | Cell state | Observed on | Note |', '|---|---|---|---|---|']
        for record in data['live_observations']:
            L.append(f"| `{record['evidence_id']}` | `{record['cell']}` | {cells[record['cell']]['label']} | {record['observed_on'] or '-'} | {_cell_text(record['note'])} |")
    L += ['', '## Evidence run', '', '| Check | Passed | Skipped | Failed | Reported cells |', '|---|---|---|---|---|']
    for name, entry in run['checks'].items():
        L.append(f"| `{name}` | {len(entry['passed'])} | {len(entry['skipped'])} | {len(entry['failed'])} | {len(entry['cells'])} |")
    skipped = [(name, test, reason) for name, entry in run['checks'].items() for test, reason in entry['skipped'].items()]
    if skipped:
        L += ['', 'Skipped on this host (cells citing them are untried):', '']
        L += [f'- `{name}::{test}`: {_cell_text(reason)}' for name, test, reason in skipped]
    return '\n'.join(L) + '\n'


def render_json(data):
    return json.dumps(data, indent=1, ensure_ascii=False, sort_keys=True) + '\n'


def committed():
    path = ROOT / 'docs' / JSON_OUT
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None


def run_from_committed(data):
    return {name: {'returncode': e['returncode'], 'passed': list(e['passed']), 'failed': list(e['failed']), 'skipped': dict(e['skipped']),
                   'cells': dict(e['cells'])} for name, e in data['run']['checks'].items()}


def build(*, results=None, live=None, base=None):
    """(data, markdown). results: a path; live: a path or a list of paths (merged; they replace the embedded live evidence
    and observations); base: a previous matrix dict whose run record or live evidence is reused."""
    base = base if base is not None else committed()
    if results is not None:
        run = read_results(results)
    elif base is not None:
        run = run_from_committed(base)
    else:
        raise SystemExit('no results file and no committed docs/broker_matrix.json: run with --run or --results PATH')
    if live is not None:
        records, observations = read_live_files([live] if isinstance(live, (str, Path)) else list(live))
    else:
        records = list(base['live_evidence']) if base is not None else []
        observations = list(base.get('live_observations') or []) if base is not None else []
    data = assemble(run, records, observations)
    if results is None and base is not None and data['run']['platform'] != base['run']['platform']:
        data['run']['platform'] = base['run']['platform']  # a rebuild from the embedded record keeps the host it ran on
    return data, render_md(data)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--run', action='store_true', help='run every tests/check_broker_*.py under the recording shim (and the JavaScript Studio checks with node) first')
    parser.add_argument('--results', type=Path, help='TITANOS_BROKER_RESULTS JSONL file to build from (with --run: where to write it)')
    parser.add_argument('--live', type=Path, action='append',
                        help='live evidence file (JSON list or JSONL); repeat to merge several files; default: the evidence embedded in docs/broker_matrix.json')
    parser.add_argument('--jobs', type=int, default=1, help='checks run in parallel with --run (default 1)')
    parser.add_argument('--check', action='store_true', help='compare the rebuild with docs/ instead of writing; exit 1 on drift')
    parser.add_argument('--out', type=Path, default=ROOT / 'docs', help='output directory (default docs/)')
    parser.add_argument('--exec-check', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.exec_check:
        sys.exit(exec_check(args.exec_check))
    results = args.results
    scratch = None
    if args.run:
        if results is None:
            scratch = tempfile.TemporaryDirectory(prefix='titanos-broker-results-')
            results = Path(scratch.name) / 'results.jsonl'
        if results.exists():
            raise SystemExit(f'{results} exists; --run writes a fresh results file')

        def log(name, code, output):
            print(f'{name}: exit {code}', flush=True)
            if code:
                print(output[-4000:], flush=True)
        codes = run_checks(results, jobs=max(1, args.jobs), log=log)
        bad = sorted(name for name, code in codes.items() if code)
        if bad:
            raise SystemExit('broker checks failed: ' + ', '.join(bad))
    try:
        data, markdown = build(results=results, live=args.live)
    finally:
        if scratch is not None:
            scratch.cleanup()
    rendered = render_json(data)
    if args.check:
        drift = []
        for name, text in ((JSON_OUT, rendered), (MD_OUT, markdown)):
            path = args.out / name
            current = path.read_bytes().replace(b'\r\n', b'\n').decode('utf-8') if path.is_file() else ''
            if current != text:
                drift.append(name)
        if drift:
            raise SystemExit(f"{', '.join(drift)} differ from the rebuild; run python -B tools/build_broker_matrix.py --run")
        print(json.dumps({'fixed_point': True, **{k: v for k, v in data['counts'].items() if k != 'by_section'}}))
        return
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / JSON_OUT).write_bytes(rendered.encode('utf-8'))
    (args.out / MD_OUT).write_bytes(markdown.encode('utf-8'))
    print(json.dumps({k: v for k, v in data['counts'].items() if k != 'by_section'}))


if __name__ == '__main__':
    main()
