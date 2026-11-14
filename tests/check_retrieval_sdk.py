"""Official OpenAI SDK accepts embedding fidelity additions; controlled inference."""
import asyncio
import hashlib
from pathlib import Path
import sys
import tempfile

import httpx
from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from titanos.config import Config, ModelSpec
from titanos.server import create_app


async def main():
    with tempfile.TemporaryDirectory(prefix='titanos-retrieval-sdk-') as temp:
        app = create_app(Config(models={'encoder': ModelSpec(path=temp, task='embeddings', device='cpu')}), output_dir=Path(temp))
        seen = []
        def infer(model, task, data, **unused):
            seen.append(data)
            return {'vectors': [[0.25, 0.75]], 'usage': {'prompt_tokens': 2, 'total_tokens': 2},
                    'input_details': [{'index': 0, 'input_tokens': 3, 'processed_tokens': 2, 'truncated': True,
                                       'input_sha256': hashlib.sha256(b'example').hexdigest()}],
                    'truncation': {'enabled': True, 'side': 'right', 'strategy': 'single_sequence', 'max_tokens': 2}}
        app.state.engine.target.infer = infer
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), trust_env=False) as transport:
                async with AsyncOpenAI(base_url='http://127.0.0.1/v1', api_key='controlled-test-key',
                                       http_client=transport, _strict_response_validation=True) as client:
                    result = await client.embeddings.create(model='encoder', input='example', encoding_format='float',
                                                           extra_body={'truncate': True, 'max_tokens': 2})
                    assert result.object == 'list' and result.data[0].object == 'embedding'
                    assert result.usage.prompt_tokens == 2 and result.model == 'encoder'
                    assert result.data[0].embedding == [0.25, 0.75]
                    assert result.model_extra['input_details'][0]['truncated'] is True
                    assert result.model_extra['truncation']['max_tokens'] == 2
                    assert seen[-1]['truncate'] is True and seen[-1]['max_tokens'] == 2
            # SDK debug strict validation models only float embeddings. Its normal
            # default path requests base64 and then decodes it in a post-parser.
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), trust_env=False) as transport:
                async with AsyncOpenAI(base_url='http://127.0.0.1/v1', api_key='controlled-test-key', http_client=transport) as client:
                    result = await client.embeddings.create(model='encoder', input='example', extra_body={'truncate': True, 'max_tokens': 2})
                    assert result.data[0].embedding == [0.25, 0.75]
                    assert result.model_extra['input_details'][0]['truncated'] is True
    print('PASS: official OpenAI SDK strict float validation and default base64 decoding retain envelope/fidelity metadata; no model or network endpoint')


if __name__ == '__main__': asyncio.run(main())
