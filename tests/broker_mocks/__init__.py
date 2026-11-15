"""Protocol-faithful loopback mock providers for the broker checks (base-tier dependencies only: fastapi/starlette,
uvicorn, httpx).

    from broker_mocks import OpenAIMock, SENTINELS, netguard
    netguard.install()
    with OpenAIMock() as mock:                      # MockServer on 127.0.0.1:<free port>
        url = mock.api_base                         # http://127.0.0.1:<port>/v1
        mock.faults.push('status', status=429)      # next request gets OpenAI's native 429 with Retry-After
        mock.hold(frame=2)                          # streams wait before frame 2 until mock.gate.set()
        ...
        mock.hits                                   # [RecordedRequest], credentials as '<sentinel:openai>'

Importing this package starts nothing and patches nothing; netguard.install() is explicit.
"""
from . import netguard
from .advisor_mock import AdvisorMock
from .anthropic_mock import AnthropicMock
from .azure_mock import AzureMock
from .elevenlabs_mock import ElevenLabsMock
from .faults import KINDS as FAULT_KINDS
from .faults import Fault, FaultQueue, wait_until
from .fixtures import SENTINEL_NAMES, SENTINELS, Fixture, all_fixtures, load as load_fixture, parse_sse
from .gemini_mock import GeminiMock
from .ollama_mock import OllamaCloudMock
from .openai_mock import OpenAIMock
from .router_mock import RouterUpstreamMock
from .server import BaseMock, MockServer, RecordedRequest, Reply, SinkMock, fingerprint, unused_loopback_url

__all__ = ['AdvisorMock', 'AnthropicMock', 'AzureMock', 'BaseMock', 'ElevenLabsMock', 'FAULT_KINDS', 'Fault', 'FaultQueue',
           'Fixture', 'GeminiMock', 'MockServer', 'OllamaCloudMock', 'OpenAIMock', 'RecordedRequest', 'Reply',
           'RouterUpstreamMock', 'SENTINELS', 'SENTINEL_NAMES', 'SinkMock', 'all_fixtures', 'fingerprint', 'load_fixture',
           'netguard', 'parse_sse', 'unused_loopback_url', 'wait_until']
