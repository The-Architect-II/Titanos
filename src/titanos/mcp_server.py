"""Official MCP transport over Titanos's existing HTTP contracts and lifecycle.

Principal propagation (mounted mode, `app` given): every tool, resource template and helper threads the MCP tool
Context into request(). The principal is read per message from that message's outer HTTP request
(ctx.request_context.request.scope['titanos.principal'], written there by the Guard), and the in-process call carries it
on a titanos.internal scope with no Authorization header, so the bootstrap key is never what authorizes a call that
arrived over /mcp and every inner route still checks its own scope. A message with an outer request but no
authenticated principal is refused (ToolError). Static resources take no Context in mcp 2.2; they use the principal the
Guard bound for the current outer request (broker.context.PRINCIPAL). A direct in-process call with no MCP request at
all (an embedder or a test calling call_tool from Python) keeps the key create_mcp was given. Stdio mode (no `app`)
sends Bearer TITANOS_API_KEY over HTTP, as before.
"""

import base64
import binascii
import json
import os
import re
from typing import Annotated, Any, Literal
from urllib.parse import quote, urlsplit

import httpx
from mcp.server import MCPServer
from mcp.server.mcpserver import Audio, Context
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import BlobResourceContents, CallToolResult, EmbeddedResource, TextContent, ToolAnnotations
from pydantic import Field

from . import __version__
from .api_plus import (ExtensionRequest, SentimentRequest, SentimentCompositionRequest, RerankRequest, VADRequest, DiarizationRequest, AudioAnalysisRequest,
                       TranslationRequest, AudioTranslationRequest, PairClassificationRequest, DepthRequest, SegmentationRequest)
from .sentiment_contract import SentimentResult, SentimentComposition
from .retrieval_contract import RerankResult
from .retrieval_workflow_contract import RetrievalAnswerRequest, RetrievalAnswerResult
from .audio_contract import VADResult, DiarizationResult, AudioAnalysisResult
from .audio_gating import VADGateOptions
from .translation_contract import TextTranslationResult, AudioTranslationResult
from .pair_contract import PairClassificationResult
from .depth import DepthResult
from .segmentation import SegmentationResult
from .client import TitanosError, _check, _origin
from .providers import AcquisitionPlan, AcquisitionRequest
from pydantic.json_schema import SkipJsonSchema
from .server import AssetJob, Chat, Completion, Embeddings, ImageRequest, Registration, Responses, Speech, TensorRequest, VideoRequest, current_settings


class ChatCall(Chat):
    stream: Literal[False] = False


class CompletionCall(Completion):
    stream: Literal[False] = False


class ResponsesCall(Responses):
    stream: Literal[False] = False


ModelName = Annotated[str, Field(min_length=1, max_length=128)]
JobId = Annotated[str, Field(pattern=r'^[a-f0-9]{32}$')]
AcquisitionId = Annotated[str, Field(pattern=r'^[a-f0-9]{32}$')]
ArtifactName = Annotated[str, Field(pattern=r'^[a-f0-9]{32}\.(png|mp4|wav)$')]
InlineMedia = Annotated[str, Field(min_length=1, description='Base64-encoded media bytes, never a path or URL; limits come from the server settings.')]
Backend = Literal['nvidia', 'pytorch', 'openvino', 'onnx', 'system', 'all']
READ = ToolAnnotations(read_only_hint=True, open_world_hint=False)
CHANGE = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=False)
NO_PRINCIPAL = 'Mounted MCP call has no authenticated outer request'
CLOUD_SENTENCE = ('When the operator enabled cloud upstreams, a named model may be served by a cloud provider; call '
                  'broker_status or broker_explain to see where a model would run, and read each result\'s _meta.served_by.')
# provenance response headers -> _meta keys of inference tool results
PROVENANCE_META = (('x-titanos-served-by', 'served_by'), ('x-titanos-served-model', 'served_model'),
                   ('x-request-id', 'request_id'), ('x-titanos-decision-id', 'decision_id'),
                   ('x-titanos-fallback-count', 'fallback_count'))


def _decode_media(value, image=False):
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ToolError('Media must contain valid base64 bytes') from exc
    settings = current_settings()
    maximum = settings.max_input_image_bytes if image else settings.max_body_bytes
    if not raw or len(raw) > maximum:
        raise ToolError(f'Media must contain 1 byte to {maximum} bytes under the configured limit')
    return raw


def open_world_hint(broker=None) -> bool:
    """The static open_world_hint of the inference tools, fixed at create_mcp time: True unless the role is managed
    without delegation or broker.json defines no upstream. Tool annotations are per process; a reload does not change
    them. An unreadable broker.json counts as possibly open (True). Reads one file; no network, no directory creation."""
    from .broker.config import broker_path, read_broker_config
    from .broker.role import switches_for
    try:
        if broker is not None:
            settings, path, switches = broker.settings, broker.config_path, broker.switches
        else:
            from .settings import load_settings
            settings = load_settings()
            path, switches = broker_path(settings), switches_for(settings)
        if settings.role == 'managed' and not switches.delegation:
            return False
        return bool(read_broker_config(path).upstreams)
    except Exception:
        return True


def provenance_meta(response) -> dict[str, Any]:
    """The _meta block of an inference result, copied from its X-Titanos-* provenance headers (empty without them)."""
    meta = {}
    for header, name in PROVENANCE_META:
        value = response.headers.get(header)
        if value is None:
            continue
        if name == 'fallback_count':
            try:
                value = int(value)
            except ValueError:
                continue
        meta[name] = value
    return meta if 'served_by' in meta else {}


def with_provenance(value, response):
    """value unchanged when the response carries no provenance; otherwise the same result with _meta provenance."""
    meta = provenance_meta(response)
    if not meta:
        return value
    if isinstance(value, Audio):
        return CallToolResult(content=[value.to_audio_content()], _meta=meta)
    try:
        from mcp.server.mcpserver.utilities.func_metadata import _convert_to_content
        content = _convert_to_content(value)
    except ImportError:  # pragma: no cover - the pinned mcp exposes it
        content = [TextContent(type='text', text=json.dumps(value, indent=2))]
    return CallToolResult(content=content, structured_content=value if isinstance(value, dict) else None, _meta=meta)


def create_mcp(base_url=None, api_key=None, app=None, broker=None) -> MCPServer:
    """Build stdio or mounted MCP; `app` dispatches through the same ASGI routes.

    A mounted owner must call streamable_http_app() before entering
    `mcp.session_manager.run()` from its top-level ASGI lifespan.
    """
    base_url = (base_url or os.environ.get('TITANOS_URL') or 'http://127.0.0.1:11435').rstrip('/')
    _origin(base_url)
    if urlsplit(base_url).query or urlsplit(base_url).fragment:
        raise ValueError('Server URL cannot contain a query or fragment')
    key = os.environ.get('TITANOS_API_KEY', '') if api_key is None else api_key
    INFER = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=False,
                            open_world_hint=open_world_hint(broker))
    mcp = MCPServer('Titanos', version=__version__, instructions=(
        'Discover registered models and their capabilities before inference. Titanos runs locally '
        'and independently of Pair or HiveMind. Choose an explicit registered model. For long asset '
        'generation submit a job, retain its ID, inspect status, then read its artifact. Implemented '
        'capabilities do not certify model quality or hardware qualification. Media inputs are inline '
        'base64; artifact reads are limited to Titanos-generated identifiers. Model discovery and '
        'acquisition use separate durable acquisition tools; plan an immutable revision, monitor/cancel '
        'that acquisition, then activate it before inference. ' + CLOUD_SENTENCE))

    def outer_request(ctx):
        """The outer HTTP request of this MCP message, or None (no Context, or a direct in-process call)."""
        if ctx is None:
            return None
        try:
            return ctx.request_context.request
        except (ValueError, AttributeError, LookupError):
            return None

    def outer_principal(ctx):
        """(principal or None, mounted message) for a mounted call; see the module docstring."""
        from .broker.context import PRINCIPAL
        from .broker.types import Principal
        req = outer_request(ctx)
        if req is not None:
            principal = req.scope.get('titanos.principal')
            if not isinstance(principal, Principal):
                raise ToolError(NO_PRINCIPAL)
            return principal
        principal = PRINCIPAL.get()  # the Guard-bound principal of the current outer request (static resources)
        return principal if isinstance(principal, Principal) else None

    async def request(method, route, *, ctx=None, **kwargs):
        public_origin = kwargs.pop('public_origin', None)
        headers = dict(kwargs.pop('headers', {}))
        principal = outer_principal(ctx) if app is not None else None
        if principal is None and key:
            # stdio mode, or a direct in-process call with no MCP request at all (never a message over /mcp)
            headers = {'Authorization': 'Bearer ' + key, **headers}
        async def internal_app(scope, receive, send):
            # This marker is set by trusted in-process dispatch only. A client
            # cannot manufacture an ASGI scope field through HTTP headers.
            inner = {**scope, 'titanos.internal': True}
            if principal is not None:
                inner['titanos.principal'] = principal
            await app(inner, receive, send)
        transport = httpx.ASGITransport(app=internal_app) if app is not None else None
        try:
            async with httpx.AsyncClient(base_url=public_origin or base_url, headers=headers, transport=transport,
                                         timeout=current_settings().client_timeout, trust_env=False, follow_redirects=False) as client:
                return _check(await client.request(method, route, **kwargs))
        except (httpx.HTTPError, TitanosError) as exc:
            raise ToolError(str(exc)) from exc

    async def json_request(method, route, *, ctx=None, **kwargs):
        return (await request(method, route, ctx=ctx, **kwargs)).json()

    def artifact_origin(context):
        req = outer_request(context) if app is not None else None
        if req is not None:
            url = urlsplit(str(req.url))
            return url.scheme + '://' + url.netloc
        return base_url

    async def infer(route, payload, context=None, *, origin=False):
        # Omitted defaults stay omitted, preserving max_completion_tokens semantics.
        response = await request('POST', route, ctx=context, json=payload.model_dump(mode='json', exclude_unset=True),
                                 public_origin=artifact_origin(context) if origin else None)
        return response.json()

    async def inference(route, payload, context=None, *, origin=False):
        """An inference tool's result: the JSON answer with _meta provenance when the server stamped it."""
        response = await request('POST', route, ctx=context, json=payload.model_dump(mode='json', exclude_unset=True),
                                 public_origin=artifact_origin(context) if origin else None)
        return with_provenance(response.json(), response)

    async def artifact_bytes(name, ctx=None):
        if not re.fullmatch(r'[a-f0-9]{32}\.(png|mp4|wav)', name):
            raise ToolError('Use a Titanos artifact filename, not a path or URL')
        response = await request('GET', '/v1/files/' + name, ctx=ctx)
        if len(response.content) > current_settings().max_body_bytes:
            raise ToolError('Artifact exceeds TITANOS_MAX_BODY_BYTES for inline MCP; download through the HTTP files route')
        return response.content

    @mcp.tool(annotations=READ)
    async def server_info(ctx: Context = None) -> dict[str, Any]:
        """Discover API version, task routes, limits, job support and compatibility boundaries."""
        return await json_request('GET', '/api/info', ctx=ctx)

    @mcp.tool(annotations=READ)
    async def list_models(loaded_only: bool = False, ctx: Context = None) -> dict[str, Any]:
        """List configured model IDs, tasks, and actual residency state."""
        return await json_request('GET', '/api/models/loaded' if loaded_only else '/v1/models', ctx=ctx)

    @mcp.tool(annotations=READ)
    async def model_capabilities(model: ModelName, ctx: Context = None) -> dict[str, Any]:
        """Inspect operations and advisory local setup issues without loading weights; readiness is not runtime qualification."""
        return await json_request('POST', '/api/models/capabilities', ctx=ctx, json={'model': model})

    @mcp.tool(annotations=READ)
    async def list_model_providers(ctx: Context = None) -> dict[str, Any]:
        """List model discovery/acquisition providers and their supported lifecycle operations."""
        return await json_request('GET','/api/model-providers', ctx=ctx)

    @mcp.tool(annotations=READ)
    async def discover_models(provider:str,task:str,query:str='',limit:Annotated[int,Field(ge=1,le=100)]=20, ctx: Context = None) -> dict[str,Any]:
        """Search one provider by Titanos task. Candidate metadata is not runtime or quality proof."""
        return await json_request('GET','/api/model-providers/'+quote(provider,safe='')+'/search', ctx=ctx,
                                  params={'task':task,'query':query,'limit':limit})

    @mcp.tool(annotations=READ)
    async def inspect_model_source(provider:str,identifier:str,revision:str|None=None, ctx: Context = None) -> dict[str,Any]:
        """Resolve provider metadata and an exact revision without downloading or loading weights."""
        return await json_request('POST','/api/model-providers/inspect', ctx=ctx,
                                  json={'provider':provider,'identifier':identifier,'revision':revision})

    @mcp.tool(annotations=READ)
    async def import_model_page(url:str, ctx: Context = None) -> dict[str,Any]:
        """Normalize a permitted HTTPS model page/link. Metadata-only results are not executable compatibility."""
        return await json_request('POST','/api/model-providers/import', ctx=ctx, json={'url':url})

    @mcp.tool(annotations=READ)
    async def plan_model_acquisition(payload:AcquisitionRequest, ctx: Context = None) -> dict[str,Any]:
        """Resolve a secret-free immutable file/runtime plan; this does not download or register a model."""
        return await json_request('POST','/api/acquisitions/plan', ctx=ctx,
                                  json=payload.model_dump(mode='json',exclude_unset=True))

    @mcp.tool(annotations=CHANGE)
    async def submit_model_acquisition(payload:AcquisitionPlan,
                                       idempotency_key:Annotated[str,Field(min_length=1,max_length=128)]|None=None, ctx: Context = None) -> dict[str,Any]:
        """Start a durable model acquisition separate from inference jobs; retain its ID for progress/cancel/retry."""
        headers={'Idempotency-Key':idempotency_key} if idempotency_key else {}
        return await json_request('POST','/api/acquisitions', ctx=ctx, json=payload.model_dump(mode='json'),headers=headers)

    @mcp.tool(annotations=READ)
    async def list_model_acquisitions(limit:Annotated[int,Field(ge=1,le=1000)]=100, ctx: Context = None) -> dict[str,Any]:
        """List durable model acquisition/setup records."""
        return await json_request('GET','/api/acquisitions', ctx=ctx, params={'limit':limit})

    @mcp.tool(annotations=READ)
    async def get_model_acquisition(identifier:AcquisitionId, ctx: Context = None) -> dict[str,Any]:
        """Read one acquisition's provider, revision, progress, license and activation state."""
        return await json_request('GET','/api/acquisitions/'+identifier, ctx=ctx)

    @mcp.tool(annotations=CHANGE)
    async def cancel_model_acquisition(identifier:AcquisitionId, ctx: Context = None) -> dict[str,Any]:
        """Request cooperative cancellation; partial data remains inactive and may support explicit retry."""
        return await json_request('DELETE','/api/acquisitions/'+identifier, ctx=ctx)

    @mcp.tool(annotations=CHANGE)
    async def retry_model_acquisition(identifier:AcquisitionId, ctx: Context = None) -> dict[str,Any]:
        """Retry a failed/cancelled/interrupted acquisition with the same immutable plan."""
        return await json_request('POST','/api/acquisitions/'+identifier+'/retry', ctx=ctx)

    @mcp.tool(annotations=CHANGE)
    async def activate_model_acquisition(identifier:AcquisitionId,
                                         accepted_license_sha256:str|None=None,load:bool=True,runtime_profile:str|None=None, ctx: Context = None) -> dict[str,Any]:
        """Atomically register a prepared model and optionally load it; unknown licenses require exact hash acknowledgement."""
        return await json_request('POST','/api/acquisitions/'+identifier+'/activate', ctx=ctx,
                                  json={'accepted_license_sha256':accepted_license_sha256,'load':load,
                                        **({'runtime_profile':runtime_profile} if runtime_profile is not None else {})})

    @mcp.tool(annotations=CHANGE)
    async def model_details(model: ModelName, worker: str | None = None, ctx: Context = None) -> dict[str, Any]:
        """Load if necessary and report the actual model runtime and selected placement."""
        return await json_request('POST', '/api/models/details', ctx=ctx, json={'model': model, 'worker': worker})

    @mcp.tool(annotations=CHANGE)
    async def load_model(model: ModelName, worker: str | None = None, ctx: Context = None) -> dict[str, Any]:
        """Warm a registered model; ordinary local loading can replace another resident model."""
        return await json_request('POST', '/api/models/load', ctx=ctx, json={'model': model, 'worker': worker})

    @mcp.tool(annotations=CHANGE)
    async def unload_model(model: ModelName, worker: str | None = None, ctx: Context = None) -> dict[str, Any]:
        """Release a model's resident runtime. Registration and checkpoint files remain."""
        return await json_request('POST', '/api/models/unload', ctx=ctx, json={'model': model, 'worker': worker})

    @mcp.tool(annotations=CHANGE)
    async def register_model(payload: Registration, ctx: Context = None) -> dict[str, Any]:
        """Register an existing local checkpoint through the server's validated registry API and reload it. HTTP registration does not enable remote code or legacy pickle trust."""
        return await infer('/api/models/register', payload, ctx)

    @mcp.tool(annotations=CHANGE)
    async def reload_registry(ctx: Context = None) -> dict[str, Any]:
        """Reload the operator-managed registry while idle; active work prevents reconfiguration."""
        return await json_request('POST', '/api/models/reload', ctx=ctx)

    @mcp.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=True, idempotent_hint=False, open_world_hint=False))
    async def unregister_model(model: ModelName, ctx: Context = None) -> dict[str, Any]:
        """Remove a model registration while idle, retaining checkpoint files."""
        return await json_request('DELETE', '/api/models/registrations/' + quote(model, safe=''), ctx=ctx)

    @mcp.tool(annotations=READ)
    async def list_devices(backend: Backend = 'all',refresh:bool=False, ctx: Context = None) -> dict[str, Any]:
        """Inspect installed hardware and runtime visibility; bounded refresh never loads a model."""
        return await json_request('GET', '/api/devices', ctx=ctx, params={'backend': backend,'refresh':refresh})

    @mcp.tool(annotations=READ)
    async def list_workers(ctx: Context = None) -> dict[str, Any]:
        """Inspect configured worker placement, availability and observed runtime state."""
        return await json_request('GET', '/api/workers', ctx=ctx)

    @mcp.tool(annotations=INFER)
    async def chat(payload: ChatCall, ctx: Context = None) -> dict[str, Any]:
        """Generate text or vision chat; inline image data URLs are accepted by vision models. Tools and structured output require a capable adapter. Use HTTP for token streaming."""
        return await inference('/v1/chat/completions', payload, ctx)

    @mcp.tool(annotations=INFER)
    async def complete(payload: CompletionCall, ctx: Context = None) -> dict[str, Any]:
        """Complete one text prompt through a registered chat model."""
        return await inference('/v1/completions', payload, ctx)

    @mcp.tool(annotations=INFER)
    async def responses(payload: ResponsesCall, ctx: Context = None) -> dict[str, Any]:
        """Run stateless text/image Responses on local chat/vision models. Tools and structured output require native llama.cpp support. Use HTTP for streaming."""
        return await inference('/v1/responses', payload, ctx)

    @mcp.tool(annotations=INFER)
    async def embed(payload: Embeddings, ctx: Context = None) -> dict[str, Any]:
        """Generate normalized embeddings; overlength text is rejected unless truncation is explicitly enabled, and token loss is reported."""
        return await inference('/v1/embeddings', payload, ctx)

    @mcp.tool(annotations=CHANGE)
    async def rerank(payload: RerankRequest, ctx: Context = None) -> RerankResult:
        """Rank documents using a local trained cross-encoder; preserve original indices and report uncalibrated logits and per-document truncation."""
        return RerankResult.model_validate(await infer('/v1/rerank', payload, ctx))

    @mcp.tool(annotations=CHANGE)
    async def retrieval_answer(payload: RetrievalAnswerRequest, ctx: Context = None) -> RetrievalAnswerResult:
        """Embed/rerank inline passages and answer with literal source quotations. Inspect failed/abstained status; citation integrity is not semantic entailment."""
        return RetrievalAnswerResult.model_validate(await infer('/v1/retrieval/answer',payload, ctx))

    @mcp.tool(annotations=CHANGE)
    async def classify_pairs(payload: PairClassificationRequest, ctx: Context = None) -> PairClassificationResult:
        """Classify premise/hypothesis pairs with native model labels/logits and uncalibrated scores; no sentiment/rerank mapping or truth guarantee."""
        return PairClassificationResult.model_validate(await infer('/v1/classifications/text-pairs',payload, ctx))

    @mcp.tool(annotations=INFER)
    async def speak(payload: Speech, ctx: Context = None) -> Audio:
        """Synthesize speech and return a native MCP WAV audio content block."""
        if payload.stream_format is not None or payload.response_format != 'wav':
            raise ToolError('unsupported_capability: speak returns one WAV block; stream_format and response_format pcm are POST /v1/audio/speech options')
        response = await request('POST', '/v1/audio/speech', ctx=ctx, json=payload.model_dump(mode='json', exclude_unset=True))
        return with_provenance(Audio(data=response.content, format='wav'), response)

    @mcp.tool(annotations=INFER)
    async def transcribe(model: ModelName, audio_base64: InlineMedia,
                         language: str | None = None, worker: str | None = None,
                         prompt: Annotated[str,Field(min_length=1,strict=True)] | None = None,
                         max_tokens: Annotated[int,Field(gt=0)] | None = None,
                         response_format: Literal['json', 'verbose_json'] = 'json',
                         timestamp_granularities: list[Literal['segment', 'word']] | None = None,
                         vad_enabled: bool | None = None, vad_model: str | None = None,
                         vad_options: VADGateOptions | None = None,
                         timeout: Annotated[int,Field(gt=0)] | None = None, ctx: Context = None) -> dict[str, Any]:
        """Transcribe inline audio bytes. Optional prompt conditions builtin Whisper with vocabulary/context; it never rewrites the transcript. No server filesystem paths or remote media URLs are accepted."""
        data = {k: v for k, v in {'model': model, 'language': language, 'worker': worker, 'max_tokens':max_tokens, 'prompt':prompt,
                'response_format': response_format, 'timestamp_granularities[]': timestamp_granularities, 'timeout': timeout,
                'vad_enabled':vad_enabled, 'vad_model':vad_model,
                'vad_options':json.dumps(vad_options.model_dump(mode='json', exclude_none=True)) if vad_options is not None else None}.items() if v is not None}
        response = await request('POST', '/v1/audio/transcriptions', ctx=ctx, data=data,
                                 files={'file': ('audio.wav', _decode_media(audio_base64), 'application/octet-stream')})
        return with_provenance(response.json(), response)

    @mcp.tool(annotations=CHANGE)
    async def diarize(payload: DiarizationRequest, ctx: Context = None) -> DiarizationResult:
        """Anonymous request-local speaker diarization; preserves simultaneous intervals, not identity or confidence."""
        return DiarizationResult.model_validate(await json_request('POST', '/v1/audio/diarizations', ctx=ctx, json=payload.model_dump(mode='json', exclude_unset=True)))

    @mcp.tool(annotations=CHANGE)
    async def analyze_audio(payload: AudioAnalysisRequest, ctx: Context = None) -> AudioAnalysisResult:
        """One deadline for ASR, anonymous speaker alignment and optional transcript sentiment; no audio tone/personality inference."""
        return AudioAnalysisResult.model_validate(await json_request('POST', '/v1/audio/analysis', ctx=ctx, json=payload.model_dump(mode='json', exclude_unset=True)))

    @mcp.tool(annotations=CHANGE)
    async def vad(payload: VADRequest, ctx: Context = None) -> VADResult:
        """Detect speech activity from inline audio with explicit frame/state/reset semantics; scores are not calibrated confidence."""
        return VADResult.model_validate(await json_request('POST', '/v1/audio/vad', ctx=ctx, json=payload.model_dump(mode='json', exclude_unset=True)))

    @mcp.tool(annotations=CHANGE)
    async def translate(payload: TranslationRequest, ctx: Context = None) -> TextTranslationResult | dict[str, Any]:
        """Translate local text; include_provenance opts into loaded NMT identity, tokenizer choices and measurements."""
        result = await infer('/v1/translations', payload, ctx)
        return TextTranslationResult.model_validate(result) if payload.include_provenance else result

    @mcp.tool(annotations=CHANGE)
    async def translate_audio(payload: AudioTranslationRequest, ctx: Context = None) -> AudioTranslationResult | dict[str, Any]:
        """Translate inline audio to English directly or through explicit NMT; optional stages distinguish observed ASR language from tokenizer IDs."""
        result = await json_request('POST', '/v1/extensions/infer', ctx=ctx,
            json={**payload.model_dump(mode='json', exclude_unset=True), 'operation':'audio_translation'})
        return AudioTranslationResult.model_validate(result) if payload.include_provenance else result

    @mcp.tool(annotations=INFER)
    async def generate_image(payload: ImageRequest, ctx: Context) -> dict[str, Any]:
        """Generate PNG images. Choose response_format=url for a retrievable Titanos artifact; b64_json returns inline PNG data."""
        return await inference('/v1/images/generations', payload, ctx, origin=True)

    @mcp.tool(annotations=CHANGE)
    async def edit_image(model: ModelName, image_base64: InlineMedia,
                         prompt: Annotated[str, Field(min_length=1)],
                         ctx: Context,
                         size: str | None = None, steps: Annotated[int, Field(ge=1)] | None = None,
                         n: Annotated[int, Field(ge=1, strict=True)] | None = None,
                         strength: Annotated[float, Field(gt=0, le=1)] | None = None,
                         seed: Annotated[int, Field(ge=0, le=2**32-1)] | None = None,
                         negative_prompt: str | None = None,
                         guidance_scale: Annotated[float, Field(ge=0)] | None = None,
                         image_guidance_scale: Annotated[float,Field(ge=1,strict=True,allow_inf_nan=False)] | None = None,
                         response_format: Literal['b64_json', 'url'] | None = None,
                         worker: str | None = None,
                         mask_base64: InlineMedia | None = None,
                         preserve_unmasked: Annotated[bool, Field(strict=True)] | None = None,
                         mask: SkipJsonSchema[Any] = None,
                         mask_image: SkipJsonSchema[Any] = None,
                         width: SkipJsonSchema[Any] = None,
                         height: SkipJsonSchema[Any] = None) -> dict[str, Any]:
        """Edit with a compatible image model. InstructPix2Pix uses image_guidance_scale and requires a source image; masks and strength are unsupported for that family. Optional n requests a count within server limits. Compatible SD/SDXL models accept mask_base64 as same-size RGBA PNG: alpha0 edits, alpha255 keeps; preserve_unmasked copies original RGB pixels outside the mask."""
        # Flat MCP tools ignore unknown arguments. Reserve mask aliases to reject
        # them, while keeping unsupported controls out of the advertised schema.
        if any(value is not None for value in (mask, mask_image)):
            raise ToolError('unsupported_capability: Use mask_base64; mask and mask_image are not supported aliases')
        if width is not None or height is not None:
            raise ToolError('unsupported_capability: width/height aliases are unsupported; use size for compatible models, and the source image size for InstructPix2Pix')
        data = {'model': model, 'prompt': prompt}
        if n is not None:data['n']=str(n)
        if preserve_unmasked is not None:data['preserve_unmasked']=str(preserve_unmasked).lower()
        if strength is not None:data['strength']=str(strength)
        for key,value in {'seed':seed,'negative_prompt':negative_prompt,'guidance_scale':guidance_scale,'image_guidance_scale':image_guidance_scale,'response_format':response_format}.items():
            if value is not None:data[key]=str(value)
        if size is not None:
            data['size'] = size
        if steps is not None:
            data['steps'] = str(steps)
        if worker is not None:
            data['worker'] = worker
        files = {'image': ('image.png', _decode_media(image_base64, image=True), 'application/octet-stream')}
        if mask_base64 is not None:
            files['mask'] = ('mask.png', _decode_media(mask_base64, image=True), 'image/png')
        return await json_request('POST', '/v1/images/edits', ctx=ctx, data=data,
                                  files=files,
                                  public_origin=artifact_origin(ctx))

    @mcp.tool(annotations=CHANGE)
    async def generate_video(payload: VideoRequest, ctx: Context) -> dict[str, Any]:
        """Generate an MP4 synchronously; prefer submit_job for long-running video inference."""
        return await infer('/v1/videos/generations', payload, ctx, origin=True)

    @mcp.tool(annotations=CHANGE)
    async def infer_tensor(payload: TensorRequest, ctx: Context = None) -> dict[str, Any]:
        """Run a prepared ONNX/OpenVINO graph with explicit typed tensor inputs."""
        return await infer('/api/infer', payload, ctx)

    @mcp.tool(annotations=CHANGE)
    async def sentiment(payload: SentimentRequest, ctx: Context = None) -> SentimentResult:
        """Classify expressed text or ASR-transcript sentiment with uncalibrated scores and explicit abstention; does not infer audio affect or actual feelings."""
        return SentimentResult.model_validate(await infer('/v1/sentiment', payload, ctx))

    @mcp.tool(annotations=CHANGE)
    async def compose_sentiment(payload: SentimentCompositionRequest, ctx: Context = None) -> SentimentComposition:
        """Run classifier and LLM under one deadline; preserve stage errors, abstention and disagreement. Inspect status before using the label."""
        return SentimentComposition.model_validate(await json_request('POST', '/v1/sentiment/compose', ctx=ctx,
            json=payload.model_dump(mode='json', exclude_unset=True, by_alias=True)))

    @mcp.tool(annotations=CHANGE)
    async def estimate_depth(payload: DepthRequest, ctx: Context = None) -> DepthResult:
        """Estimate relative inverse depth aligned to the image. Float32 values have no distance units; larger values indicate nearer surfaces."""
        return DepthResult.model_validate(await infer('/v1/images/depth', payload, ctx))

    @mcp.tool(annotations=CHANGE)
    async def segment_image(payload: SegmentationRequest, ctx: Context = None) -> SegmentationResult:
        """Segment one image using a box or labeled points in original image coordinates. Return aligned RGBA PNG edit masks (selected pixels transparent) and uncalibrated predicted IoU scores."""
        return SegmentationResult.model_validate(await infer('/v1/images/segment', payload, ctx))

    @mcp.tool(annotations=CHANGE)
    async def extension_infer(payload: ExtensionRequest, ctx: Context = None) -> dict[str, Any]:
        """Run standalone inference, including prompted image segmentation, with an explicit registered local model. Discover operations and exact schemas at titanos://extensions."""
        return await json_request('POST', '/v1/extensions/infer', ctx=ctx, json=payload.model_dump(mode='json', exclude_unset=True, by_alias=True))

    @mcp.tool(annotations=CHANGE)
    async def submit_job(payload: AssetJob, ctx: Context,
                         idempotency_key: Annotated[str, Field(min_length=1, max_length=128)] | None = None) -> dict[str, Any]:
        """Submit persistent image, speech or video generation. Save the returned job ID; repeat a request with the same idempotency key to avoid duplicate work."""
        headers = {'Idempotency-Key': idempotency_key} if idempotency_key is not None else {}
        return await json_request('POST', '/api/jobs', ctx=ctx, json=payload.model_dump(mode='json', exclude_unset=True), headers=headers,
                                  public_origin=artifact_origin(ctx))

    @mcp.tool(annotations=READ)
    async def list_jobs(limit: Annotated[int, Field(ge=1, le=100)] = 100, ctx: Context = None) -> dict[str, Any]:
        """Rediscover recent persistent jobs and their completed, running, failed or interrupted state."""
        return await json_request('GET', '/api/jobs', ctx=ctx, params={'limit': limit})

    @mcp.tool(annotations=READ)
    async def get_job(identifier: JobId, ctx: Context = None) -> dict[str, Any]:
        """Read one saved job status and its result/artifact references."""
        return await json_request('GET', '/api/jobs/' + identifier, ctx=ctx)

    @mcp.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=True, idempotent_hint=True, open_world_hint=False))
    async def cancel_job(identifier: JobId, ctx: Context = None) -> dict[str, Any]:
        """Request cancellation of a queued/running job and preserve its status record."""
        return await json_request('DELETE', '/api/jobs/' + identifier, ctx=ctx)

    @mcp.tool(annotations=READ)
    async def read_artifact(name: ArtifactName, ctx: Context = None) -> EmbeddedResource:
        """Read a generated PNG, WAV or MP4 as an MCP binary resource under TITANOS_MAX_BODY_BYTES. Only Titanos artifact identifiers are accepted."""
        data = await artifact_bytes(name, ctx)
        mime = {'png': 'image/png', 'wav': 'audio/wav', 'mp4': 'video/mp4'}[name.rsplit('.', 1)[1]]
        return EmbeddedResource(resource=BlobResourceContents(uri='titanos://artifacts/' + name,
                                mime_type=mime, blob=base64.b64encode(data).decode('ascii')))

    # Static resources take no Context in mcp 2.2: request() uses the Guard-bound principal of the outer request.
    @mcp.resource('titanos://info', mime_type='application/json')
    async def info_resource() -> str:
        return json.dumps(await server_info())

    @mcp.resource('titanos://models', mime_type='application/json')
    async def models_resource() -> str:
        return json.dumps(await list_models())

    @mcp.resource('titanos://jobs', mime_type='application/json')
    async def jobs_resource() -> str:
        return json.dumps(await list_jobs())

    @mcp.resource('titanos://acquisitions', mime_type='application/json')
    async def acquisitions_resource() -> str:
        return json.dumps(await list_model_acquisitions())

    @mcp.resource('titanos://schema', mime_type='application/json')
    async def schema_resource() -> str:
        return json.dumps(await json_request('GET', '/openapi.json'))

    @mcp.resource('titanos://extensions', mime_type='application/json')
    async def extensions_resource() -> str:
        return json.dumps(await json_request('GET', '/v1/extensions'))

    @mcp.resource('titanos://models/{model}/capabilities', mime_type='application/json')
    async def model_resource(model: str, ctx: Context) -> str:
        return json.dumps(await model_capabilities(model, ctx))

    @mcp.resource('titanos://jobs/{identifier}', mime_type='application/json')
    async def job_resource(identifier: str, ctx: Context) -> str:
        if not re.fullmatch(r'[a-f0-9]{32}', identifier):
            raise ToolError('Invalid job identifier')
        return json.dumps(await get_job(identifier, ctx))

    @mcp.resource('titanos://acquisitions/{identifier}', mime_type='application/json')
    async def acquisition_resource(identifier:str, ctx: Context) -> str:
        if not re.fullmatch(r'[a-f0-9]{32}',identifier):
            raise ToolError('Invalid acquisition identifier')
        return json.dumps(await get_model_acquisition(identifier, ctx))

    @mcp.resource('titanos://artifacts/{identifier}.png', mime_type='image/png')
    async def png_resource(identifier: str, ctx: Context) -> bytes:
        return await artifact_bytes(identifier + '.png', ctx)

    @mcp.resource('titanos://artifacts/{identifier}.wav', mime_type='audio/wav')
    async def wav_resource(identifier: str, ctx: Context) -> bytes:
        return await artifact_bytes(identifier + '.wav', ctx)

    @mcp.resource('titanos://artifacts/{identifier}.mp4', mime_type='video/mp4')
    async def video_resource(identifier: str, ctx: Context) -> bytes:
        return await artifact_bytes(identifier + '.mp4', ctx)

    @mcp.prompt()
    def create_asset(description: str, modality: str = 'image') -> str:
        """Plan and execute a generated asset using currently registered models."""
        return (f'Create a {modality} asset for this request: {description}\n'
                'Read titanos://info and titanos://models, inspect the chosen model capabilities, '
                'then submit supported image/speech/video work with submit_job. Retain the ID, '
                'inspect status with get_job, and read the completed artifact. Report actual '
                'model and result; do not describe failed or unqualified output as successful.')

    @mcp.prompt()
    def diagnose_model(model: str) -> str:
        """Inspect a registered model and explain an inference/setup failure."""
        return (f'Inspect model {model!r} using list_models, model_capabilities, list_devices and '
                'list_workers. Distinguish configuration, runtime availability and proven inference. '
                'Explain the smallest next action supported by evidence; use model_details only '
                'when loading weights is appropriate.')

    @mcp.tool(annotations=READ)
    async def list_runtime_profiles(ctx: Context = None) -> dict[str,Any]:
        """List server-packaged runtime profiles compatible with this host and their human-readable readiness scope."""
        return await json_request('GET','/api/runtime-profiles', ctx=ctx)

    @mcp.tool(annotations=READ)
    async def list_runtime_setups(ctx: Context = None) -> dict[str,Any]:
        """List durable native runtime setup records so another agent/session can recover their IDs."""
        return await json_request('GET','/api/runtime-setups', ctx=ctx)

    from .runtime_api import register_mcp
    register_mcp(mcp,json_request)
    from .assignment_api import register_mcp as register_assignment_mcp
    register_assignment_mcp(mcp,json_request)
    from .broker.mcp_tools import register_mcp as register_broker_mcp
    register_broker_mcp(mcp, json_request, request=request, open_world=INFER.open_world_hint)
    return mcp


def run_stdio(base_url=None):
    """Run the MCP stdio bridge against an already-running standalone server."""
    create_mcp(base_url).run(transport='stdio')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Titanos MCP stdio bridge')
    parser.add_argument('--url')
    run_stdio(parser.parse_args().url)
