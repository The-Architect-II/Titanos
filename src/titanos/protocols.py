"""Small protocol translations over the same local inference implementations."""
import asyncio
import codecs
from datetime import datetime, timezone
import inspect
import json
import logging
from pathlib import Path
import re
import time
import uuid
from typing import Literal

from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError, model_validator

from .engine import EngineError

# Ollama keep_alive: seconds (number or numeric string) or a Go duration ("5m", "1h30m"); 0 unloads after the answer,
# a negative value keeps the model resident (Titanos's own default), a positive duration has no Titanos timer and is
# named in X-Titanos-Dropped (never a 422). Wrong types and unparseable strings are 400 naming the field.
DURATION_UNITS={'ns':1e-9,'us':1e-6,'µs':1e-6,'ms':1e-3,'s':1.0,'m':60.0,'h':3600.0}
DROPPED_HEADER='X-Titanos-Dropped'
# Titanos's model acquisition path, named by the refusals of Ollama's library routes (/api/pull, /api/create, /api/copy,
# /api/delete) and by /api/show: models are local directories, never pulled from an Ollama library.
ACQUISITION_PATH=('Titanos has no Ollama library: models are local directories acquired and registered through its own path '
                  '(search GET /api/model-providers/{provider}/search, plan POST /api/acquisitions/plan, download POST /api/acquisitions, '
                  'activate POST /api/acquisitions/{identifier}/activate, or register an existing directory with POST /api/models/register; '
                  'remove with DELETE /api/models/registrations/{name} and DELETE /api/acquisitions/{identifier}/weights). '
                  'GET /api/tags lists what is registered')


def keep_alive_policy(value):
    """'unload' (0), 'resident' (negative), 'dropped' (a positive duration) or None (absent); EngineError 400 otherwise."""
    if value is None:return None
    if isinstance(value,bool) or not isinstance(value,(int,float,str)):
        raise EngineError('keep_alive must be a number of seconds or a duration string such as 5m, 0 or -1')
    if isinstance(value,str):
        text=value.strip();sign=1.0
        if text.startswith('-'):sign,text=-1.0,text[1:]
        if re.fullmatch(r'\d+(\.\d+)?',text):seconds=sign*float(text)
        else:
            parts=re.findall(r'(\d+(?:\.\d+)?)(ns|us|µs|ms|s|m|h)',text)
            if not parts or ''.join(n+u for n,u in parts)!=text:
                raise EngineError(f'keep_alive {value!r} is not a number of seconds or a Go duration such as 5m, 1h30m, 0 or -1')
            seconds=sign*sum(float(n)*DURATION_UNITS[u] for n,u in parts)
    else:seconds=float(value)
    if seconds!=seconds:raise EngineError('keep_alive must be a finite number')
    return 'unload' if seconds==0 else 'resident' if seconds<0 else 'dropped'


class Options(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    num_ctx: int | None=Field(default=None,ge=1)
    num_predict: int | None=Field(default=None,gt=0)
    temperature: float | None=Field(default=None,ge=0,le=2)
    top_p: float | None=Field(default=None,gt=0,le=1)
    top_k: int | None=Field(default=None,ge=0)
    min_p: float | None=Field(default=None,ge=0,le=1)
    repeat_penalty: float | None=Field(default=None,gt=0)
    # Ollama's presence/frequency penalties: forwarded to a llama-cpp model; typed 400 on the in-process backends (openai_params)
    presence_penalty: float | None=Field(default=None,ge=-2,le=2)
    frequency_penalty: float | None=Field(default=None,ge=-2,le=2)
    seed: int | None=None
    stop: str | list[str] | None=None


class Ollama(BaseModel):
    model_config=ConfigDict(extra='forbid')
    model: str
    messages: list[dict] | None=None
    prompt: str | None=None
    system: str | None=None
    images: list[str] | None=None
    options: Options=Field(default_factory=Options)
    stream: bool=True
    format: str | dict | None=None
    tools: list[dict] | None=None
    keep_alive: int | float | str | None=None  # typed by keep_alive_policy (400), never a 422
    _ignored: tuple=PrivateAttr(default=())  # unknown top-level fields under TITANOS_UNKNOWN_FIELDS=ignore (managed role)

    @model_validator(mode='wrap')
    @classmethod
    def _unknown_fields(cls,data,handler):
        from .request_compat import unknown_fields
        data,ignored=unknown_fields('ollama',cls,data)
        instance=handler(data)
        if ignored:instance._ignored=ignored
        return instance


class Anthropic(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    model: str
    messages: list[dict]
    max_tokens: int=Field(gt=0)
    system: str | list[dict] | None=None
    temperature: float | None=Field(default=None,ge=0,le=2)
    top_p: float | None=Field(default=None,gt=0,le=1)
    top_k: int | None=Field(default=None,ge=0)
    stop_sequences: list[str] | None=None
    tools: list[dict] | None=None
    tool_choice: dict | None=None
    stream: bool=False
    # standard client fields (metadata, service_tier, thinking ...) accepted and named in X-Titanos-Ignored
    # (titanos.request_compat); Claude Code sends metadata on every call
    _ignored: tuple=PrivateAttr(default=())

    @model_validator(mode='wrap')
    @classmethod
    def _compat_fields(cls,data,handler):
        from .request_compat import split
        data,ignored,_=split('anthropic',data)
        instance=handler(data)
        if ignored:instance._ignored=tuple(ignored)
        return instance


async def events(response):
    """Consume real SSE incrementally, including fragmented native-server frames."""
    decoder=codecs.getincrementaldecoder('utf-8')();pending=''
    iterator=response.body_iterator
    try:
        async for chunk in response.body_iterator:
            pending+=decoder.decode(chunk) if isinstance(chunk,bytes) else chunk
            while '\n' in pending:
                line,pending=pending.split('\n',1)
                if line.startswith('data:'):
                    value=line[5:].strip()
                    if value and value!='[DONE]':yield json.loads(value)
    finally:
        if hasattr(iterator,'aclose'):await iterator.aclose()


def register_protocols(app,config,engine,generate,embed,speech,broker=None):
    """Ollama, Anthropic and ElevenLabs shapes over the local implementations. With a broker, /api/tags and /api/ps
    append the broker entries the caller may use (cloud entries need a cloud scope; nothing is ever 'loaded' remotely),
    and /api/ps also lists the loaded native (llama-cpp) models.

    Ollama client surface: GET /api/version (here outside managed role; broker/routes.py registers the managed role's
    public copy with the same body), POST /api/show from the registry and the passive model inspection, keep_alive and
    options.num_ctx on the inference routes (keep_alive_policy; num_ctx is admitted against a llama-cpp model's
    registered per-slot context and named in X-Titanos-Dropped on the other backends, whose context is the checkpoint's
    own), and typed 400 refusals on /api/pull, /api/create, /api/copy and /api/delete that name ACQUISITION_PATH."""
    from . import __version__
    from .server import Chat, Completion, Embeddings, Payload, Speech
    from .settings import current_settings
    role=getattr(getattr(broker,'settings',None),'role',None) or current_settings().role

    def broker_entries(request,loaded_only=False):
        if broker is None:return []
        return broker.catalog_for(request.scope.get('titanos.principal'),loaded_only=loaded_only,family='ollama')

    def visible(request,name):
        # a key sees the local models its model globs and target classes allow (keyless and bootstrap: all)
        return broker is None or broker.visible_local(request.scope.get('titanos.principal'),name)

    def validate(cls,values):
        try:return cls.model_validate(values)
        except ValidationError as exc:raise EngineError(str(exc)) from exc

    embed_wants_request=embed is not None and 'request' in inspect.signature(embed).parameters  # None: a test with no embed handler
    embed_is_async=inspect.iscoroutinefunction(embed);embed_takes_response=embed is not None and 'response' in inspect.signature(embed).parameters

    async def run_embed(values,request,response):
        """The local /v1/embeddings handler over an Ollama body: sync or async, with or without the request argument
        (server.embeddings takes it for the native GGUF embedding branch), answering the OpenAI embeddings document."""
        payload=validate(Embeddings,values);carrier=Response()  # the handler names its ignored fields (user, dimensions null) on it
        args=(payload,request,*((carrier,) if embed_takes_response else ())) if embed_wants_request else (payload,)
        result=await embed(*args) if embed_is_async else await asyncio.to_thread(embed,*args)
        return native_result(carry_ignored(result,carrier,response))  # before the native Response becomes a dict and loses its headers

    def native_result(result):
        if isinstance(result,Response) and not isinstance(result,StreamingResponse):
            if not 200<=result.status_code<300:return result
            try:
                value=json.loads(result.body)
                if not isinstance(value,dict):raise ValueError()
                return value
            except (ValueError,TypeError) as exc:
                raise EngineError('Native backend returned an invalid JSON response',502,'invalid_backend_response') from exc
        return result

    def sampling(options):
        data=options.model_dump(exclude_none=True)
        data.pop('num_ctx',None)  # context_window(): admitted or dropped by backend, never a sampling parameter
        if 'num_predict' in data:data['max_tokens']=data.pop('num_predict')
        return data

    def note_dropped(result,response,names):
        """Name the accepted-but-not-honoured Ollama fields on the answer (a Response directly, a dict through FastAPI's
        injected response object), as the broker's translated paths do."""
        if not names:return result
        value=', '.join(dict.fromkeys(names))
        if isinstance(result,Response):result.headers[DROPPED_HEADER]=value
        elif response is not None:response.headers[DROPPED_HEADER]=value
        return result

    def context_window(model,num_ctx):
        """options.num_ctx against the backend's context: a llama-cpp runtime is sized per slot when it starts, so a num_ctx
        up to the model's largest per-slot context is honoured (titanos.native_context: the leased runtime starts, or
        restarts once idle, at the next power of two) and a larger one is a 400 context_limit; the other backends have no
        context option (the checkpoint's own length bounds prompt plus max_tokens), so the field is named in X-Titanos-Dropped."""
        spec=config.models.get(model)
        if num_ctx is None or spec is None:return []
        if spec.backend=='llama-cpp' and spec.native is not None:
            # The largest per-slot context is native.max_slot_context, TITANOS_NATIVE_MAX_SLOT_CONTEXT or the GGUF trained
            # context_length, never below the registered context_size over slots; the request itself carries num_ctx to
            # NativePool.lease through server.generate (ollama_run sets it on the validated body), where the runtime is
            # started or resized. The refusal is native_context.context_limit; the OpenAI routes name such a num_ctx instead.
            from .native_context import needed_slot_context
            needed_slot_context(spec,num_ctx,model)
            return []
        return ['options.num_ctx']

    async def release(model):
        """keep_alive 0: unload after the answer, the same operation as POST /api/models/unload. Best effort, as Ollama's
        own timer is: a model another request holds (409 model_busy) or that is already unloaded stays as it is."""
        try:
            native=getattr(app.state,'native',None)
            if native is not None and model in native.models:await native.unload(model)
            else:await asyncio.to_thread(engine.unload,model)
        except EngineError as exc:
            logging.info('keep_alive 0: %s was not unloaded (%s)',model,exc)

    async def lenient_body(request):
        """The JSON object of a request whose body may be absent (Ollama library routes); 400 for other JSON values."""
        raw=await request.body()
        if not raw.strip():return {}
        try:values=json.loads(raw)
        except ValueError as exc:raise EngineError('Request body must be a JSON object') from exc
        if not isinstance(values,dict):raise EngineError('Request body must be a JSON object')
        return values

    def refusal(field,message):
        # the one Titanos error envelope (server.error) with param naming the Ollama field the refusal is about
        return JSONResponse({'error':{'message':message,'type':'unsupported_capability','param':field,'code':'unsupported_capability'}},status_code=400)

    def ollama_message(message):
        msg=dict(message)
        images=msg.pop('images',[]) or []
        if images:
            msg['content']=[{'type':'text','text':msg.get('content','')},
                            *[{'type':'image_url','image_url':{'url':'data:image/png;base64,'+v}} for v in images]]
        for call in msg.get('tool_calls',[]) or []:
            call.setdefault('id','call_'+uuid.uuid4().hex)
            call.setdefault('type','function')
            fn=call.get('function',{})
            if isinstance(fn.get('arguments'),dict):fn['arguments']=json.dumps(fn['arguments'])
        return msg

    def ollama_messages(messages):
        # Ollama's tool loop has no ids: assistant tool_calls get one above and a `role: tool` result carries tool_name
        # instead of tool_call_id. Each result is bound to the unanswered call with that tool_name (the next one in order
        # without a tool_name); a result with nothing to answer is a 400 naming the message. tool_name travels as name.
        out=[];pending=[]
        for index,message in enumerate(messages):
            msg=ollama_message(message)
            if msg.get('role')=='assistant' and msg.get('tool_calls'):
                pending.extend((call['id'],call.get('function',{}).get('name')) for call in msg['tool_calls'])
            elif msg.get('role')=='tool':
                name=msg.pop('tool_name',None)
                if name:msg['name']=name
                if not msg.get('tool_call_id'):
                    hit=next((i for i,(_,fn) in enumerate(pending) if not name or fn==name),None)
                    if hit is None:
                        raise EngineError(f'messages[{index}]: the tool result'+(f' for tool_name {name!r}' if name else '')
                                          +' answers no assistant tool call in the preceding messages (Ollama tool results are bound by tool_name to the pending tool_calls)')
                    msg['tool_call_id']=pending.pop(hit)[0]
            out.append(msg)
        return out

    async def ollama_run(payload,request,response=None,completion=False):
        started=time.monotonic_ns()
        policy=keep_alive_policy(payload.keep_alive)
        dropped=(['keep_alive'] if policy=='dropped' else [])+context_window(payload.model,payload.options.num_ctx)
        values={'model':payload.model,'stream':payload.stream,**sampling(payload.options)}
        if payload.stream:values['stream_options']={'include_usage':True}
        if payload.format:
            values['response_format']={'type':'json_object'} if payload.format=='json' else {'type':'json_schema','json_schema':{'name':'response','schema':payload.format}}
        if payload.tools:values['tools']=payload.tools
        if completion:
            if payload.messages is not None or payload.images or payload.format or payload.tools:
                raise EngineError('Use /api/chat for image, structured-output or tool requests')
            if payload.prompt is None:raise EngineError('prompt is required')
            values['prompt']=(payload.system+'\n' if payload.system else '')+payload.prompt
            body=validate(Completion,values);body._num_ctx=payload.options.num_ctx
        else:
            if payload.prompt is not None or payload.images:raise EngineError('Use messages for /api/chat')
            values['messages']=ollama_messages(payload.messages or [])
            if payload.system:values['messages'].insert(0,{'role':'system','content':payload.system})
            body=validate(Chat,values);body._num_ctx=payload.options.num_ctx
        result=native_result(await generate(body,'completion' if completion else 'chat',request))
        if isinstance(result,Response) and not isinstance(result,StreamingResponse):return result
        def convert(item,stream=False):
            if 'error' in item:return {'error':item['error'].get('message','Inference failed')}
            choice=item['choices'][0] if item.get('choices') else {}
            final=not stream or choice.get('finish_reason') is not None
            message=choice.get('delta' if stream else 'message',{})
            out={'model':payload.model,'created_at':datetime.now(timezone.utc).isoformat(),'done':final}
            if completion:out['response']=choice.get('text','')
            else:
                out['message']={'role':'assistant','content':message.get('content') or ''}
                if message.get('tool_calls'):
                    calls=[]
                    for call in message['tool_calls']:
                        fn=dict(call.get('function',{}))
                        if isinstance(fn.get('arguments'),str):  # streaming: only whole calls reach here (ToolCallBuffer)
                            try:fn['arguments']=json.loads(fn['arguments'])
                            except ValueError:raise EngineError('Model returned invalid tool arguments',500,'inference_error')
                        calls.append({'function':fn})
                    out['message']['tool_calls']=calls
            if final:
                usage=item.get('usage',{})
                out.update(done_reason=choice.get('finish_reason','stop'),total_duration=time.monotonic_ns()-started,
                           prompt_eval_count=usage.get('prompt_tokens',0),eval_count=usage.get('completion_tokens',0))
            return out
        if not payload.stream:
            out=convert(result)
            if policy=='unload':await release(payload.model)
            return note_dropped(out,response,dropped)
        async def stream():
            terminal=None;usage={};calls=ToolCallBuffer()  # Ollama streams whole tool_calls, not OpenAI argument fragments
            async for item in events(result):
                if item.get('usage'):usage=item['usage']
                if item.get('choices') and item['choices'][0].get('finish_reason'):
                    terminal=item;calls.absorb(item)  # a last fragment may travel with the finish frame
                elif (item.get('choices') or item.get('error')) and not calls.absorb(item):
                    yield json.dumps(convert(item,True))+'\n'
            for item in calls.frames():yield json.dumps(convert(item,True))+'\n'
            if terminal is not None:yield json.dumps(convert({**terminal,'usage':usage},True))+'\n'
            if policy=='unload':await release(payload.model)  # after the inner iterator closed (events) and the last frame
        return note_dropped(StreamingResponse(stream(),media_type='application/x-ndjson'),response,dropped)

    @app.post('/api/chat')
    async def ollama_chat(payload: Ollama,request: Request,response: Response=None):
        from .request_compat import attach;return attach(await ollama_run(payload,request,response),response,payload._ignored)

    @app.post('/api/generate')
    async def ollama_generate(payload: Ollama,request: Request,response: Response=None):
        from .request_compat import attach;return attach(await ollama_run(payload,request,response,True),response,payload._ignored)

    if role!='managed':
        @app.get('/api/version')
        def ollama_version():
            # Ollama clients read `version`; the body equals the managed role's public identity route (broker/routes.py)
            return {'name':'Titanos','version':__version__,'role':role}

    def model_status(name):
        native=getattr(app.state,'native',None)
        rows=(native.catalog() if native is not None and name in native.models else engine.catalog()['data'])
        return next((row.get('status') for row in rows if row.get('id')==name),None)

    def local_show(name,spec):
        from .capabilities import describe
        info=describe(spec)
        setup=info.get('setup') if isinstance(info,dict) and isinstance(info.get('setup'),dict) else {}
        implemented=info.get('implemented') if isinstance(info,dict) and isinstance(info.get('implemented'),list) else []
        capabilities=[]
        if spec.task in {'chat','vision'}:capabilities.append('completion')
        if 'tools' in implemented:capabilities.append('tools')
        if spec.task=='vision':capabilities.append('vision')
        if spec.task=='embeddings':capabilities.append('embedding')
        parameters=[f'num_ctx {spec.native.context_size}'] if spec.native is not None else []
        for key,value in sorted(spec.defaults.items()):
            parameters.append(f"{ {'max_tokens':'num_predict','repetition_penalty':'repeat_penalty'}.get(key,key)} {value if isinstance(value,str) else json.dumps(value)}")
        model_info={'titanos.task':spec.task,'titanos.backend':spec.backend,'titanos.device':spec.device,'titanos.dtype':spec.dtype,
                    'titanos.source_repo':spec.source_repo,'titanos.source_revision':spec.source_revision}
        architecture=setup.get('architecture') if isinstance(setup.get('architecture'),dict) else {}
        names=architecture.get('architectures')
        if isinstance(names,list) and names and isinstance(names[0],str):model_info['general.architecture']=names[0]
        if isinstance(architecture.get('pipeline_class'),str):model_info['general.architecture']=architecture['pipeline_class']
        if spec.native is not None:
            model_info.update({'llama-cpp.model_file':spec.native.model_file,'llama-cpp.context_size':spec.native.context_size,
                               'llama-cpp.slots':spec.native.slots,'llama-cpp.gpu_layers':spec.native.gpu_layers})
        try:modified=datetime.fromtimestamp(Path(spec.path).stat().st_mtime,timezone.utc).isoformat()
        except OSError:modified=None
        return {'model':name,'modified_at':modified,'template':'',
                'modelfile':'# Titanos has no Modelfile: '+name+' is a registry entry (models.json) over a local model directory. '+ACQUISITION_PATH,
                'parameters':'\n'.join(parameters),
                'details':{'parent_model':spec.source_repo or '','format':spec.backend,'family':spec.task,'families':[spec.task],
                           'parameter_size':'','quantization_level':''},
                'model_info':model_info,'capabilities':capabilities,
                'titanos':{'task':spec.task,'backend':spec.backend,'device':spec.device,'status':model_status(name),'setup':setup}}

    @app.post('/api/show')
    async def ollama_show(request: Request,response: Response=None):
        values=await lenient_body(request)
        name=values.get('model',values.get('name'))
        if not isinstance(name,str) or not name:raise EngineError('model is required: the registered model name (Ollama clients send model or name)')
        unknown=sorted(set(values)-{'model','name','verbose'})
        if unknown:raise EngineError('Unsupported /api/show fields: '+', '.join(unknown))
        if 'verbose' in values and not isinstance(values['verbose'],bool):raise EngineError('verbose must be a boolean')
        spec=config.models.get(name)
        if spec is not None and visible(request,name):
            document=await asyncio.to_thread(local_show,name,spec)
        else:
            entry=broker.model_entry(name,request.scope.get('titanos.principal'),protocol=None) if broker is not None else None
            if entry is None:raise EngineError('Unknown model',404,'model_not_found')
            document={'model':name,'modified_at':None,'template':'','modelfile':'# cloud model served by the Titanos broker; nothing is stored locally',
                      'parameters':'','details':{'parent_model':'','format':'cloud','family':'cloud','families':['cloud'],'parameter_size':'','quantization_level':''},
                      'model_info':{'titanos.served_by':'cloud'},'capabilities':[],'titanos':entry}
        if 'verbose' in values and response is not None:response.headers['X-Titanos-Ignored']='verbose'  # no tokenizer arrays to add
        return document

    @app.post('/api/pull')
    async def ollama_pull(request: Request):
        values=await lenient_body(request)
        return refusal('model',f"/api/pull cannot fetch model {values.get('model',values.get('name'))!r}: "+ACQUISITION_PATH)

    @app.post('/api/create')
    async def ollama_create(request: Request):
        values=await lenient_body(request)
        return refusal('model',f"/api/create cannot build model {values.get('model',values.get('name'))!r}: Titanos has no Modelfile, "
                       'from, files or adapters; request-time defaults belong to the registration (POST /api/models/register defaults). '+ACQUISITION_PATH)

    @app.post('/api/copy')
    async def ollama_copy(request: Request):
        values=await lenient_body(request)
        return refusal('destination',f"/api/copy cannot copy {values.get('source')!r} to {values.get('destination')!r}: a Titanos model is one "
                       'registry entry over one local directory; register the directory under another name with POST /api/models/register. '+ACQUISITION_PATH)

    @app.delete('/api/delete')
    async def ollama_delete(request: Request):
        values=await lenient_body(request)
        return refusal('model',f"/api/delete cannot remove model {values.get('model',values.get('name'))!r}: Titanos never deletes weights from an "
                       'Ollama request; unregister with DELETE /api/models/registrations/{name} and remove downloaded weights with '
                       'DELETE /api/acquisitions/{identifier}/weights')

    @app.get('/api/tags')
    def ollama_tags(request: Request):
        return {'models':[{'name':name,'model':name,'details':{'format':spec.backend,'family':spec.task},
                           'titanos':{'task':spec.task,'device':spec.device}} for name,spec in config.models.items()
                          if visible(request,name)]
                +broker_entries(request)}

    @app.get('/api/ps')
    def ollama_ps(request: Request):
        native=getattr(app.state,'native',None)
        loaded=native.catalog(True) if native is not None else []
        # Titanos catalog rows plus Ollama's name/model (= id), so stock clients see which model is resident
        return {'models':[{**m,'name':m.get('id'),'model':m.get('id')} for m in engine.catalog(loaded_only=True)['data']+loaded if visible(request,m.get('id'))]
                +broker_entries(request,True)}

    @app.post('/api/embed')
    async def ollama_embed(request: Request,response: Response=None):
        values=await request.json()
        if not isinstance(values,dict):raise EngineError('Request must be an object')
        policy=keep_alive_policy(values.pop('keep_alive',None))
        result=await run_embed(values,request,response)
        if isinstance(result,Response):return result
        if policy=='unload':await release(result['model'])
        return note_dropped({'model':result['model'],'embeddings':[x['embedding'] for x in result['data']],
                             'prompt_eval_count':result['usage'].get('prompt_tokens',0),
                             **{name:result[name] for name in ('input_details','truncation') if name in result}},
                            response,['keep_alive'] if policy=='dropped' else [])

    @app.post('/api/embeddings')
    async def ollama_embeddings(request: Request,response: Response=None):
        values=await request.json()
        if not isinstance(values,dict):raise EngineError('Request must be an object')
        policy=keep_alive_policy(values.pop('keep_alive',None))
        if 'prompt' in values:values['input']=values.pop('prompt')
        if not isinstance(values.get('input'),str):raise EngineError('Legacy embeddings accepts one prompt string')
        result=await run_embed(values,request,response)
        if isinstance(result,Response):return result
        if len(result['data'])!=1:raise EngineError('Legacy embeddings accepts one prompt')
        if policy=='unload':await release(result['model'])
        return note_dropped({'embedding':result['data'][0]['embedding'],
                             **{name:result[name] for name in ('input_details','truncation') if name in result}},
                            response,['keep_alive'] if policy=='dropped' else [])

    def required(block,key,kind):
        # Upstream-shaped blocks missing a required key are client errors (400), never a server KeyError (500).
        if not isinstance(block,dict) or key not in block:raise EngineError(f'{kind} block requires {key}')
        return block[key]

    @app.post('/v1/messages')
    async def anthropic_messages(payload: Anthropic,request: Request,response: Response=None):
        from .request_compat import attach
        return attach(await anthropic_answer(payload,request),response,payload._ignored)

    async def anthropic_answer(payload: Anthropic,request: Request):
        values={'model':payload.model,'max_tokens':payload.max_tokens,'stream':payload.stream,'messages':[]}
        if payload.stream:values['stream_options']={'include_usage':True}
        for key in ('temperature','top_p','top_k'):
            if getattr(payload,key) is not None:values[key]=getattr(payload,key)
        if payload.stop_sequences is not None:values['stop']=payload.stop_sequences
        if payload.system:
            system=payload.system if isinstance(payload.system,str) else '\n'.join(required(p,'text','system text') for p in payload.system if isinstance(p,dict) and p.get('type')=='text')
            values['messages'].append({'role':'system','content':system})
        for message in payload.messages:
            content=message.get('content','');role=message.get('role')
            if isinstance(content,str):values['messages'].append({'role':role,'content':content});continue
            if not isinstance(content,list) or any(not isinstance(block,dict) for block in content):
                raise EngineError('Message content must be text or a list of typed content blocks')
            parts=[];calls=[];results=[]
            for block in content:
                kind=block.get('type')
                if kind=='text':parts.append({'type':'text','text':required(block,'text','text')})
                elif kind=='image':
                    source=required(block,'source','image')
                    if not isinstance(source,dict):raise EngineError('image block source must be an object')
                    if source.get('type')!='base64':raise EngineError('Images require inline base64 data')
                    parts.append({'type':'image_url','image_url':{'url':f"data:{required(source,'media_type','image source')};base64,{required(source,'data','image source')}"}})
                elif kind=='tool_use':calls.append({'id':required(block,'id','tool_use'),'type':'function','function':{'name':required(block,'name','tool_use'),'arguments':json.dumps(required(block,'input','tool_use'))}})
                elif kind=='tool_result':results.append({'role':'tool','tool_call_id':required(block,'tool_use_id','tool_result'),'content':block.get('content','')})
                else:raise EngineError(f'Unsupported content block {kind}',400,'unsupported_capability')
            if parts or calls:values['messages'].append({'role':role,'content':parts or None,**({'tool_calls':calls} if calls else {})})
            values['messages'].extend(results)
        if payload.tools:
            values['tools']=[{'type':'function','function':{'name':required(tool,'name','tool'),'description':tool.get('description',''),'parameters':required(tool,'input_schema','tool')}} for tool in payload.tools]
        if payload.tool_choice:
            choice=payload.tool_choice
            if choice.get('disable_parallel_tool_use') is True:values['parallel_tool_calls']=False
            values['tool_choice']={'type':'function','function':{'name':required(choice,'name','tool_choice')}} if choice.get('type')=='tool' else {'auto':'auto','any':'required','none':'none'}.get(choice.get('type'),'auto')
        result=native_result(await generate(validate(Chat,values),'chat',request))
        if isinstance(result,Response) and not isinstance(result,StreamingResponse):return result
        ident='msg_'+uuid.uuid4().hex
        if not payload.stream:
            choice=result['choices'][0];message=choice['message'];content=[]
            if message.get('content'):content.append({'type':'text','text':message['content']})
            for call in message.get('tool_calls',[]) or []:
                fn=call['function']
                try:arguments=json.loads(fn['arguments'])
                except ValueError:raise EngineError('Model returned invalid tool arguments',500,'inference_error')
                content.append({'type':'tool_use','id':call['id'],'name':fn['name'],'input':arguments})
            usage=result.get('usage',{})
            return {'id':ident,'type':'message','role':'assistant','model':payload.model,'content':content,
                    'stop_reason':{'length':'max_tokens','tool_calls':'tool_use'}.get(choice.get('finish_reason'),'end_turn'),
                    'stop_sequence':None,'usage':{'input_tokens':usage.get('prompt_tokens',0),'output_tokens':usage.get('completion_tokens',0)}}
        # text deltas, tool_use blocks with input_json_delta fragments and stop_reason tool_use (agent_capability.anthropic_stream)
        from .agent_capability import anthropic_stream
        return StreamingResponse(anthropic_stream(result,ident,payload.model),media_type='text/event-stream')

    @app.post('/v1/text-to-speech/{voice_id}')
    async def elevenlabs_speech(voice_id: str,request: Request):
        if request.query_params.get('output_format','wav')!='wav':
            raise EngineError('This adapter produces WAV; request output_format=wav',400,'unsupported_capability')
        try:values=await request.json()
        except ValueError as exc:raise EngineError('Request body must be a JSON object with text and model_id') from exc
        if not isinstance(values,dict) or set(values)-{'text','model_id'}:raise EngineError('Supported fields: text and model_id')
        names=[name for name,spec in config.models.items() if spec.task=='speech']
        model=values.get('model_id') or (names[0] if len(names)==1 else None)
        if model is None:raise EngineError('Specify the registered speech model_id')
        return await asyncio.to_thread(speech,validate(Speech,{'model':model,'input':values.get('text'),'voice':voice_id}))

    # the local text-to-speech handler, reused by POST /v1/text-to-speech/{voice_id}/stream (broker/routes.py)
    app.state.elevenlabs_speech=elevenlabs_speech


def carry_ignored(result,carrier,response):
    """The X-Titanos-Ignored header the local embeddings handler set (on the Response it returned for a llama-cpp model, or on
    the carrier it was handed in place of FastAPI's injected response) copied onto the Ollama route's own response. run_embed
    turns that Response into a dict for the Ollama document, which would otherwise lose the header: the field would then be
    removed from the request without being named, a silent drop."""
    from .request_compat import IGNORED_HEADER
    value=(result.headers.get(IGNORED_HEADER) if isinstance(result,Response) else None) or carrier.headers.get(IGNORED_HEADER)
    if value and response is not None:response.headers[IGNORED_HEADER]=value
    return result


class ToolCallBuffer:
    """OpenAI streams a tool call as delta.tool_calls fragments (an index, the name once, the arguments as string pieces);
    Ollama streams whole tool_calls with object arguments, and the stock client validates every frame (a fragment is a
    pydantic error on its side). The fragments are gathered here by index and emitted as one complete frame ahead of the
    final frame (ollama_run.stream); convert() parses the complete argument text. Text deltas pass through untouched."""
    def __init__(self):self.calls={}

    def absorb(self,item):
        """Take the tool_calls fragments out of one chat.completion.chunk; True when nothing is left in it to emit."""
        choice=item['choices'][0] if isinstance(item.get('choices'),list) and item['choices'] else {}
        delta=choice.get('delta') if isinstance(choice,dict) else None
        fragments=delta.pop('tool_calls',None) if isinstance(delta,dict) else None
        for fragment in fragments or []:
            if not isinstance(fragment,dict):continue
            call=self.calls.setdefault(fragment.get('index',len(self.calls)),{'function':{'name':'','arguments':''}})
            fn=fragment.get('function') or {}
            if fn.get('name'):call['function']['name']=fn['name']
            if isinstance(fn.get('arguments'),str):call['function']['arguments']+=fn['arguments']
        return bool(fragments) and not delta.get('content') and 'error' not in item

    def frames(self):
        """One chunk carrying every gathered call with its complete JSON argument text; an invalid text is one error frame
        (the Ollama string envelope), never a fragment. Nothing when no call was streamed."""
        if not self.calls:return []
        calls=[self.calls[index] for index in sorted(self.calls)]
        for call in calls:
            try:json.loads(call['function']['arguments'] or '{}')
            except ValueError:return [{'error':{'message':'Model returned invalid tool arguments for '+(call['function']['name'] or 'a tool call')}}]
            call['function']['arguments']=call['function']['arguments'] or '{}'
        return [{'choices':[{'index':0,'delta':{'role':'assistant','content':'','tool_calls':calls}}]}]
