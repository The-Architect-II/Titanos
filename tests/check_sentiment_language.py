"""Language/composition contracts; controlled backends, no models or held-out cases."""
import asyncio
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient
from pydantic import ValidationError

from titanos.api_plus import SentimentRequest
from titanos.config import Config, ModelSpec
from titanos.engine import EngineError
from titanos.sentiment_contract import (LABEL_OUTPUTS, SentimentResult, SentimentComposition,
    label_prefix_allowed_tokens, parse_llm_output)
from titanos.server import create_app
from titanos.settings import Settings


def decoding():
    class Tokens(list):
        def __getitem__(self, value):
            out = super().__getitem__(value)
            return Tokens(out) if isinstance(value, slice) else out
        def tolist(self): return list(self)
    class Tokenizer:
        def encode(self, text, add_special_tokens=False): return list(text.encode())
        def decode(self, tokens, skip_special_tokens=True): return bytes(tokens).decode()
    allowed = label_prefix_allowed_tokens(Tokenizer(), 2, [300, 301])
    for text in LABEL_OUTPUTS:
        tokens = list(text.encode())
        for index, token in enumerate(tokens):
            assert token in allowed(0, Tokens([9, 9] + tokens[:index]))
        assert allowed(0, Tokens([9, 9] + tokens)) == [300, 301]
    try: allowed(0, Tokens([9, 9, 999]))
    except ValueError: pass
    else: raise AssertionError('Out-of-language generation must fail, never become unconstrained')
    for version in ['label-v2', 'label-v3']:
        for label in ['negative', 'neutral', 'positive', 'mixed', 'unknown']:
            raw = parse_llm_output(json.dumps({'label': label}), 'stop', version)
            assert raw['scores'] is None and raw['model_label'] == label
    for text, finish in [('{}','stop'), ('{"label":"positive","abstain":true}','stop'),
                         ('{"label":"positive","label":"negative"}','stop'),
                         ('{"label":"positive"}','length'), ('{"label":[]}','stop')]:
        assert parse_llm_output(text, finish, 'label-v2')['invalid_model_output']


def contracts():
    with tempfile.TemporaryDirectory() as folder:
        cfg = Config(models={'c': ModelSpec(path=folder, task='sentiment'),
            'l': ModelSpec(path=folder, task='chat', defaults={'max_tokens': 64, 'temperature': 0}),
            'g': ModelSpec(path=folder, task='chat', backend='openvino-genai', device='CPU'),
            'wrong': ModelSpec(path=folder, task='embeddings')})
        app = create_app(cfg, output_dir=Path(folder)/'outputs', settings=Settings(client_timeout=1))
        seen = []; outcome = {'label': 'positive', 'scores': {'negative': .1, 'neutral': .1, 'positive': .8}, 'error': None, 'delay': 0}
        def infer(model, task, data, emit=None, cancel=None):
            seen.append({'model': model, 'task': task, 'data': copy.deepcopy(data), 'cancel': cancel})
            if outcome['delay'] and task == 'sentiment': time.sleep(outcome['delay'])
            if outcome['error'] == model: raise EngineError('Controlled stage failure', 503, 'fixture_failure')
            if task == 'chat':
                return {'text': json.dumps({'label': outcome['label']}), 'finish_reason': 'stop',
                    'provenance': {'device': 'cpu'}, 'usage': {'prompt_tokens': 20, 'completion_tokens': 5, 'total_tokens': 25}}
            return {'scores': outcome['scores'], 'score_kind': 'softmax_uncalibrated', 'input_tokens': 4,
                    'provenance': {'backend': 'transformers', 'device': 'cpu', 'checkpoint_digest': 'a'*64,
                                   'label_mapping': {'0':'negative','1':'neutral','2':'positive'}}}
        app.state.engine.target.infer = infer
        with TestClient(app, base_url='http://127.0.0.1') as client:
            base = {'classifier_model': 'c', 'llm_model': 'l', 'text': 'A development-only sample.'}
            def compose(**kwargs): return client.post('/v1/sentiment/compose', json={**base, **kwargs})
            result = client.post('/v1/sentiment', json={'model':'l','text':base['text']}).json()
            SentimentResult.model_validate(result)
            assert result['label'] == 'positive' and result['scores'] is None and result['score_kind'] == 'none'
            assert result['decision_basis'] == 'label' and result['score_thresholds_applied'] is False
            assert result['schema_version'] == '2'
            assert result['constraint_mode'] == 'label_choices' and result['provenance']['prompt_version'] == 'label-v2'
            assert seen[-1]['data']['_sentiment_label_choices'] is True
            v3 = client.post('/v1/sentiment', json={'model':'l','text':base['text'],'prompt_version':'label-v3'}).json()
            assert v3['provenance']['prompt_version'] == 'label-v3' and v3['constraint_mode'] == 'label_choices'
            assert v3['provenance']['prompt_sha256'] != result['provenance']['prompt_sha256']
            body = compose().json(); SentimentComposition.model_validate(body)
            assert body['status'] == 'ok' and body['label'] == 'positive' and body['agreement'] == 'agree'
            assert body['scores'] is None and body['classifier']['result']['scores'] == outcome['scores']
            assert body['classifier']['result']['schema_version'] == '2'
            bad_version = copy.deepcopy(body['classifier']['result']); bad_version['schema_version'] = '1'
            try: SentimentResult.model_validate(bad_version)
            except ValidationError: pass
            else: raise AssertionError('Expanded classifier wire output must not impersonate schema 1')
            assert body['llm']['result']['model_label'] == 'positive'
            outcome['label'] = 'negative'
            body = compose().json(); assert body['label'] == 'unknown' and body['agreement'] == 'disagree'
            assert body['abstention_reasons'] == ['stage_disagreement']
            outcome['label'] = 'mixed'
            body = compose().json(); assert body['status'] == 'abstained' and body['agreement'] == 'undetermined'
            assert body['llm']['result']['model_label'] == 'mixed' and body['llm']['result']['abstention_reasons'] == ['mixed_polarity']
            outcome['label'] = 'positive'; outcome['scores'] = {'negative': .3, 'neutral': .3, 'positive': .4}
            body = compose().json(); assert body['label'] == 'unknown' and body['abstention_reasons'] == ['classifier_abstained']
            outcome['scores'] = {'negative': .1, 'neutral': .1, 'positive': .8}
            body = compose(input_modality='asr_transcript', asr_uncertain=True).json()
            assert body['status'] == 'abstained' and body['abstention_reasons'] == ['classifier_abstained','llm_abstained']
            before = len(seen); outcome['error'] = 'c'
            body = compose().json(); assert body['status'] == 'failed' and body['classifier']['error']['code'] == 'fixture_failure'
            assert body['llm']['status'] == 'not_run' and len(seen) == before+1
            outcome['error'] = 'l'
            body = compose().json(); assert body['status'] == 'failed' and body['classifier']['result']['label'] == 'positive'
            assert body['llm']['error']['http_status'] == 503
            outcome['error'] = None; outcome['delay'] = .03
            import threading
            actual_timer = threading.Timer
            intervals = []
            def timer(interval, *args, **kwargs):
                intervals.append(interval)
                return actual_timer(interval, *args, **kwargs)
            with patch('titanos.api_plus.threading.Timer', side_effect=timer):
                assert compose(timeout=150).json()['status'] == 'ok'
            assert len(intervals) == 2 and 0 < intervals[1] < intervals[0] - .015, intervals
            before = len(seen); body = compose(timeout=10).json()
            assert body['status'] == 'failed' and body['classifier']['error']['code'] == 'deadline_exceeded'
            # The short budget can expire while queued or during the first stage.
            assert body['llm']['status'] == 'not_run' and len(seen) <= before+1
            assert all(call['model'] == 'c' for call in seen[before:])
            outcome['delay'] = 0
            for fields in [{'llm_model':'missing'},{'classifier_model':'wrong'},{'policy':'prefer_llm'}]:
                before = len(seen); response = compose(**fields)
                assert response.status_code in {400,404,422} and len(seen) == before
            before = len(seen)
            assert compose(llm_model='g', decoding='label_choices').status_code == 400
            assert len(seen) == before, 'Known incompatible LLM controls must fail before the classifier runs'
            from titanos.sentiment_contract import llm_instruction
            def slow_preparation(version):
                time.sleep(.03)
                return llm_instruction(version)
            with patch('titanos.api_plus.llm_instruction',side_effect=slow_preparation):
                before = len(seen)
                body = compose(timeout=5).json()
                assert body['status']=='failed' and body['classifier']['status']=='not_run'
                assert body['classifier']['error']['code']=='deadline_exceeded'
                direct = client.post('/v1/sentiment',json={'model':'l','text':'Sample','timeout':5})
                assert direct.status_code==504 and direct.json()['error']['code']=='deadline_exceeded'
                assert len(seen)==before, 'Preparation time must not renew the absolute deadline'
            for key in ['sentiment_prompt_version', 'sentiment_decoding']:
                for bad_default in [{}, [], None, 42, 'unsupported']:
                    cfg.models['l'].defaults[key] = bad_default
                    before = len(seen)
                    assert compose().status_code == 400
                    assert client.post('/v1/sentiment',json={'model':'l','text':'Sample'}).status_code == 400
                    assert len(seen) == before, 'Invalid model defaults must fail before either stage runs'
                del cfg.models['l'].defaults[key]
            body = compose().json(); bad = copy.deepcopy(body); bad['label'] = 'negative'
            try: SentimentComposition.model_validate(bad)
            except ValidationError: pass
            else: raise AssertionError('Forged composition outcome accepted')
            bad = copy.deepcopy(body); bad['llm']['result']['input']['sha256'] = 'b'*64
            try: SentimentComposition.model_validate(bad)
            except ValidationError: pass
            else: raise AssertionError('Mismatched stage input accepted')
            extension = client.post('/v1/extensions/infer', json={**base,'operation':'sentiment_compose'})
            assert extension.status_code == 200 and extension.json()['label'] == 'positive', extension.text
            before = len(seen)
            assert client.post('/v1/sentiment',json={'model':'g','text':'Sample','decoding':'label_choices'}).status_code == 400
            assert len(seen) == before
            assert client.post('/v1/sentiment',json={'model':'g','text':'Sample','decoding':'validate'}).json()['constraint_mode'] == 'prompt_then_validation'
            async def mcp():
                result = await app.state.mcp.call_tool('compose_sentiment', {'payload':base})
                assert not result.is_error, result
                value = result.structured_content
                if value is None: value=json.loads(next(x.text for x in result.content if getattr(x,'type','')=='text'))
                value=value.get('result',value); assert value['label']=='positive' and value['classifier']['result']
            asyncio.run(mcp())


def client_cli():
    from titanos.client import Client
    from titanos import cli
    import io
    from contextlib import redirect_stdout
    seen=[]
    def reply(request):
        seen.append((request.url.path,json.loads(request.content)))
        return httpx.Response(200,json={'sample':True})
    with Client('http://127.0.0.1',transport=httpx.MockTransport(reply)) as client:
        client.compose_sentiment('c','l','Development text',prompt_version='label-v2')
        client.extension('sentiment_compose','c',llm_model='l',text='Development text')
    assert seen[0][0]=='/v1/sentiment/compose' and seen[0][1]['classifier_model']=='c'
    assert seen[1][1]['classifier_model']=='c'
    class Stub:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def compose_sentiment(self,c,l,text,**kw):seen.append((c,l,text,kw));return {'sample':True}
    with patch('titanos.client.Client',return_value=Stub()),redirect_stdout(io.StringIO()):
        cli.main(['sentiment-compose','c','l','Development text','--prompt-version','label-v2','--decoding','label_choices','--deadline-ms','50'])
    assert seen[-1]==('c','l','Development text',{'prompt_version':'label-v2','decoding':'label_choices','timeout':50})


async def caller_cancellation():
    """Actual ASGI disconnect and MCP task cancellation reach an active worker."""
    import threading
    for mode in ['http_disconnect', 'mcp_task_cancel', 'mcp_extension_cancel']:
        for stop_stage in ['c', 'l']:
            started, stopped, observed = threading.Event(), threading.Event(), threading.Event()
            seen = []
            def infer(model, task, data, emit=None, cancel=None):
                seen.append(model)
                if model == stop_stage:
                    started.set(); until = time.monotonic() + 2
                    try:
                        while not cancel.is_set() and time.monotonic() < until: time.sleep(.005)
                        assert cancel.is_set(), 'Caller cancellation did not reach backend'
                        observed.set()
                        # Deliberately return a normal result; the route must reject it after cancellation.
                    finally: stopped.set()
                if task == 'chat':
                    return {'text':'{"label":"positive"}', 'finish_reason':'stop', 'provenance':{'device':'cpu'}}
                return {'scores':{'negative':.1,'neutral':.1,'positive':.8}, 'score_kind':'softmax_uncalibrated',
                    'provenance':{'backend':'transformers','device':'cpu'}}
            with tempfile.TemporaryDirectory() as folder:
                app = create_app(Config(models={'c':ModelSpec(path=folder,task='sentiment'),
                    'l':ModelSpec(path=folder,task='chat')}),output_dir=Path(folder))
                app.state.engine.target.infer = infer
                payload = {'classifier_model':'c','llm_model':'l','text':'A development cancellation fixture.','timeout':5000}
                try:
                    if mode == 'http_disconnect':
                        content = json.dumps(payload).encode(); first = True; sent = []
                        async def receive():
                            nonlocal first
                            if first:
                                first = False
                                return {'type':'http.request','body':content,'more_body':False}
                            while not started.is_set(): await asyncio.sleep(.005)
                            return {'type':'http.disconnect'}
                        async def send(message): sent.append(message)
                        scope = {'type':'http','asgi':{'version':'3.0'},'http_version':'1.1','method':'POST',
                            'scheme':'http','path':'/v1/sentiment/compose','raw_path':b'/v1/sentiment/compose',
                            'root_path':'','query_string':b'','client':('127.0.0.1',50000),'server':('127.0.0.1',80),
                            'headers':[(b'host',b'127.0.0.1'),(b'content-type',b'application/json')]}
                        await asyncio.wait_for(app(scope,receive,send),3)
                        body = json.loads(b''.join(m.get('body',b'') for m in sent))
                        assert body['status']=='failed' and body['label']=='unknown'
                        stage = body['classifier' if stop_stage=='c' else 'llm']
                        assert stage['status']=='failed' and stage['error']['code']=='cancelled'
                        if stop_stage=='l': assert body['classifier']['result']['label']=='positive'
                        else: assert body['llm']['status']=='not_run' and body['llm']['error']['code']=='cancelled'
                    else:
                        tool = 'compose_sentiment' if mode=='mcp_task_cancel' else 'extension_infer'
                        if mode=='mcp_extension_cancel': payload['operation']='sentiment_compose'
                        pending = asyncio.create_task(app.state.mcp.call_tool(tool,{'payload':payload}))
                        assert await asyncio.to_thread(started.wait,1)
                        pending.cancel()
                        try: await asyncio.wait_for(pending,3)
                        except asyncio.CancelledError: pass
                        else: raise AssertionError('MCP caller cancellation was swallowed')
                    assert observed.is_set() and stopped.is_set(), (mode,stop_stage)
                    assert seen == (['c'] if stop_stage=='c' else ['c','l'])
                finally:
                    await app.state.native.close()
                    await asyncio.to_thread(app.state.engine.close)

async def queued_deadline():
    with tempfile.TemporaryDirectory() as folder:
        app = create_app(Config(models={'c':ModelSpec(path=folder,task='sentiment'),
            'l':ModelSpec(path=folder,task='chat')}),output_dir=Path(folder))
        calls=[]
        def infer(*args,**kwargs): calls.append(args);raise AssertionError('Expired request ran a model')
        app.state.engine.target.infer=infer
        actual_to_thread=asyncio.to_thread
        async def delayed(function,*args,**kwargs):
            await asyncio.sleep(.03)
            return await actual_to_thread(function,*args,**kwargs)
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://127.0.0.1') as client:
                with patch('titanos.api_plus.asyncio.to_thread',side_effect=delayed):
                    for route,fields in [('/v1/sentiment',{'model':'l'}),('/v1/sentiment/compose',{'classifier_model':'c','llm_model':'l'})]:
                        response=await client.post(route,json={**fields,'text':'A queue deadline fixture.','timeout':5})
                        assert response.status_code==504 and response.json()['error']['code']=='deadline_exceeded'
            assert not calls
        finally:
            await app.state.native.close()
            await asyncio.to_thread(app.state.engine.close)


if __name__=='__main__':
    decoding();contracts();client_cli();asyncio.run(caller_cancellation());asyncio.run(queued_deadline())
    print('PASS: versioned categorical labels, finite decode language, consensus/abstention/errors/shared deadline, ASGI disconnect and MCP task cancellation in either stage, CLI/Python/MCP contracts; no model inference')
