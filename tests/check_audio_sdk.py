"""Existing OpenAI ASR JSON/text clients and opt-in metadata, no model or listener."""
import asyncio
from pathlib import Path
import sys
import tempfile
import httpx
from openai import AsyncOpenAI

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from titanos.config import Config,ModelSpec
from titanos.server import create_app

async def main():
    with tempfile.TemporaryDirectory() as tmp:
        app=create_app(Config(models={'asr':ModelSpec(path=tmp,task='transcription')}),output_dir=Path(tmp)/'out')
        seen=[]
        def infer(name,task,data,**kwargs):
            seen.append(data)
            if data.get('response_format')!='verbose_json':return 'hello'
            return {'task':'transcribe','text':'hello','duration':1.,'sample_rate':16000,'decoded_samples':16000,
                'language':None,'language_source':'unavailable','requested_language':None,'segments':None,
                'words':[{'id':0,'start':0.,'end':None,'word':'hello'}],
                'timestamps':{'granularity':'word','source':'backend_alignment','supported':['word']},
                'quality':{'confidence':None,'avg_logprob':None,'no_speech_prob':None,'status':'unavailable'},'backend':'transformers'}
        app.state.engine.target.infer=infer
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),trust_env=False) as transport:
                async with AsyncOpenAI(base_url='http://127.0.0.1/v1',api_key='test',http_client=transport) as client:
                    assert (await client.audio.transcriptions.create(model='asr',file=('test.wav',b'fixture'),response_format='json')).text=='hello'
                    assert await client.audio.transcriptions.create(model='asr',file=('test.wav',b'fixture'),response_format='text')=='hello'
                    result=await client.audio.transcriptions.create(model='asr',file=('test.wav',b'fixture'),response_format='verbose_json',timestamp_granularities=['word'])
                    assert result.words[0].end is None and result.language is None
                    assert result.model_extra['quality']['confidence'] is None
                    assert seen[-1]['timestamp_granularities']==['word']
                    result=await client.audio.transcriptions.create(model='asr',file=('test.wav',b'fixture'),prompt='Names and vocabulary')
                    assert result.text=='hello' and seen[-1]['prompt']=='Names and vocabulary'
                    await client.audio.transcriptions.create(model='asr',file=('test.wav',b'fixture'))
                    assert 'prompt' not in seen[-1]
    print('PASS: OpenAI SDK JSON/text compatibility and ordinary verbose_json with explicit missing measurements. Strict verbose schema is not claimed.')

if __name__=='__main__':asyncio.run(main())
