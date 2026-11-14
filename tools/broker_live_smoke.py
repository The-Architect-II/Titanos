"""Broker live smoke (validation lane V14): one minimal request per approved modality through a real `titanos serve`.

Manual tier, never in CI. It spends money, so it refuses unless every gate holds:
- --live and --i-understand-this-spends-money are both given (the `titanos broker live-smoke` wrapper passes them);
- --approval names an owner approval file OUTSIDE this repository that parses, is unexpired and passes the checks below;
- --home is a directory outside this repository (a fresh TITANOS_HOME is created under it for every run).

    python tools/broker_live_smoke.py --live --i-understand-this-spends-money --approval FILE --home DIR \
        [--providers openai,nvidia_api_catalog,hf_router] [--modalities chat_stream,embeddings,...]
    python tools/broker_live_smoke.py --plan --approval FILE          # validate and print the plan; no server, no network
    python tools/broker_live_smoke.py --dry-run [--home DIR]          # offline self-test against the loopback mocks
    python tools/broker_live_smoke.py --approval-template             # print an approval skeleton to fill in

Approval file (JSON, schema titanos-broker-live-approval/v1; unknown keys are refused, secret-shaped strings too):
    {"approved_by": "owner", "expires_at": "2026-10-03T00:00:00Z", "spend_cap_usd": "0.50",
     "providers": ["openai", "nvidia_api_catalog", "hf_router"],
     "credential_env": {"openai": "OPENAI_API_KEY", "nvidia_api_catalog": "NGC_API_KEY", "hf_router": "HF_TOKEN"},
     "models": {"openai": {"chat": "...", "responses": "...", "embeddings": "...", "speech": "...",
                           "transcription": "...", "moderation": "...", "images": "..."},
                "nvidia_api_catalog": {"chat": "...", "embeddings": "..."}, "hf_router": {"chat": "..."}},
     "optional": ["responses", "chat_nonstream", "images"],
     "prices": [{"match": "<model>@openai", "input_per_mtok": "0.15", "output_per_mtok": "0.6"}, ...],
     "output_dir": "<absolute directory outside the repository>",
     "params": {"openai.images": {"size": "1024x1024", "quality": "low"}, "openai.speech": {"voice": "alloy"}}}
- credential_env must name exactly the authorized variables (OPENAI_API_KEY, NGC_API_KEY, HF_TOKEN) for the listed
  providers; spend_cap_usd is at most the authorized USD 0.50; every probed model needs a price row (budgets fail closed
  on unpriced targets, so a free surface such as moderation needs a zero row); params never replace the probe fields.
  --approval-template prints one placeholder row per default model with the fields its answer is settled on; the
  token-billed transcription models (gpt-4o-transcribe, gpt-4o-mini-transcribe: the answer reports tokens, no seconds)
  get input_per_mtok and output_per_mtok plus the per_audio_minute fallback. The NVIDIA defaults are current catalog
  models (google/gemma-3-4b-it; nvidia/llama-3.2-nv-embedqa-1b-v1, asymmetric: the probe sends input_type query).

What one live run does:
1. Creates <home>/run-<id>/ with a fresh TITANOS_HOME whose broker and vault directories get an explicit protected
   (owner-only) DACL before the server starts, and a registry made by the same code path as `titanos init`.
2. Starts `python -m titanos serve --role broker` on 127.0.0.1 with a generated bootstrap key passed in TITANOS_API_KEY,
   TITANOS_BROKER_VAULT_BACKEND=refs-only (no credential value is stored anywhere) and an egress allowlist: broker
   egress mode internet, plus a sitecustomize guard in the server process that lets name lookups and connections reach
   only loopback and exactly the approved providers' profile hosts (see GUARD_SOURCE). The server environment drops
   every TITANOS_* variable and every other credential-shaped variable; only the approved credential variables pass.
3. Configures, through the server's own admin routes with the bootstrap key, one upstream per approved provider (profile
   default base URL, static catalog of exactly the approved models), credentials as env references
   (env:OPENAI_API_KEY, env:NGC_API_KEY, env:HF_TOKEN; validated with the provider's free models list), the approval's
   price rows and cloud_policy explicit. It then mints a tk_ key (expiring in two hours, scopes inference, broker:read and
   cloud:<provider>, constrained to the approved refs) whose USD budget equals the cap minus earlier spend under the
   same approval (output_dir/spend-ledger.jsonl; a run that never settled counts its whole reservation).
4. Sends one request per approved modality with that key (OpenAI: moderation, embeddings, chat streamed with
   include_usage, non-stream chat, Responses, speech as WAV, transcription of that WAV, one image; NVIDIA API catalog:
   chat streamed, embeddings with input_type query; HF router: chat) and checks HTTP status, response shape, usage,
   the provenance headers (X-Request-ID echo, X-Titanos-Decision-Id, X-Titanos-Served-By, X-Titanos-Served-Model) and the
   matching decision-log entry (served upstream and ref, key id, cost). Before every request it reads the key's
   remaining budget and stops at the cap; a 402 from the server's pre-dispatch budget also stops the run.
5. Revokes the key, verifies the audit chain, stops the server and scans every file of the run (server log, TITANOS_HOME,
   receipts) for the credential values, the bootstrap key and the minted key. The credential values are read in-process
   for this comparison only; they are never printed, logged, stored or sent anywhere by this tool, and the scan reports
   counts and file names only.
6. Writes to <output_dir>/<run id>/: receipt.json (redacted: fingerprints, statuses, provenance headers, usage, decision
   metadata, sizes and hashes of outputs, a failed probe's error code and its error message scrubbed with redact.scrub
   and cut to 400 characters (so a live failure is diagnosable without another call; for a 200 answer that failed its
   checks, the in-body or in-stream error event, or none), one matrix-shaped cell per probe
   with its state; never prompts other than the
   fixed probe strings below, never outputs or secrets) and broker_live_evidence.jsonl, the file
   `python tools/build_broker_matrix.py --live PATH` reads: one line {evidence_id, outcome: pass, profile, surface,
   stream: on|off, observed_on, note} per live probe whose every check passed (cell <surface>|<profile>|<stream>). A
   failed, skipped or capped probe is never evidence, and a secret found by the scan voids every record of the run.

--dry-run runs the same code path against the WP4 loopback mocks (tests/broker_mocks) with synthetic credentials in the
same environment variable names, broker egress loopback-only, the test netguard in both processes and the allowlist
guard with no approved host, and asserts the gates, the approval checks, the ledger, the guard, a full run (every probe
mock-pass, one mock hit per probe carrying the referenced credential, zero leaks) and a cap run (402, nothing sent
upstream). It makes no non-loopback connection. Exit codes: 0 pass, 1 a probe, check or scan failed, 2 refused.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import math
import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
TESTS = ROOT / 'tests'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import httpx  # noqa: E402

SCHEMA_APPROVAL = 'titanos-broker-live-approval/v1'
SCHEMA_RECEIPT = 'titanos-broker-live-receipt/v1'
EVIDENCE_FILE = 'broker_live_evidence.jsonl'
DRY_EVIDENCE_FILE = 'dry_run_evidence_format.jsonl'
# The owner's authorization (2026-10-01) covers these three variables only, by env reference, and USD 0.50 in total.
CREDENTIAL_ENV = {'openai': 'OPENAI_API_KEY', 'nvidia_api_catalog': 'NGC_API_KEY', 'hf_router': 'HF_TOKEN'}
PROVIDERS = tuple(CREDENTIAL_ENV)
AUTHORIZED_CAP_USD = Decimal('0.50')
MODALITIES = {'openai': ('chat', 'responses', 'embeddings', 'speech', 'transcription', 'moderation', 'images'),
              'nvidia_api_catalog': ('chat', 'embeddings'),
              'hf_router': ('chat',)}
OPTIONAL = ('responses', 'chat_nonstream', 'images')
APPROVAL_KEYS = frozenset({'schema', 'approved_by', 'approved_at', 'expires_at', 'spend_cap_usd', 'providers',
                           'credential_env', 'models', 'optional', 'prices', 'output_dir', 'params', 'note'})
APPROVAL_REQUIRED = ('approved_by', 'expires_at', 'spend_cap_usd', 'providers', 'credential_env', 'models', 'prices',
                     'output_dir')
# fixed probe strings: the only request text that reaches a provider or a receipt
PROBE_CHAT = 'Reply with the single word: ok'
PROBE_TEXT = 'titanos broker live smoke'
PROBE_SPEECH = 'Titanos broker live smoke test.'
PROBE_IMAGE = 'a plain blue square'
PROBE_FIELDS = frozenset({'model', 'messages', 'input', 'stream', 'stream_options', 'prompt', 'file', 'n', 'store',
                          'response_format', 'input_type', 'encoding_format'})
DEFAULT_OUTPUT_TOKENS = 16
KEY_HOURS = 2
REQUEST_TIMEOUT = 120.0
LEDGER = 'spend-ledger.jsonl'
GUARD_MODULE = 'titanos_smoke_egress_guard'
EXIT_OK, EXIT_FAIL, EXIT_REFUSED = 0, 1, 2
_SECRETISH = re.compile(r'(API_?KEY|TOKEN|SECRET|PASSWORD|PASSPHRASE|CREDENTIAL|ACCESS_?KEY)', re.IGNORECASE)
_RECORDED_HEADERS = ('content-type', 'x-request-id')


class Refused(Exception):
    """A gate refused the run before anything was started or sent."""


@dataclass(frozen=True)
class Probe:
    provider: str
    name: str            # probe name, unique per provider (chat_stream, chat, embeddings ...)
    modality: str        # key of approval.models[provider]
    surface: str         # the native surface the upstream serves
    path: str            # inbound Titanos route
    stream: bool = False
    optional: str | None = None  # the approval.optional item that enables it
    needs: str | None = None     # probe id whose output this one consumes

    @property
    def id(self) -> str:
        return f'{self.provider}.{self.name}'


# Run order: free and cheap first, the image last.
PROBES: tuple[Probe, ...] = (
    Probe('openai', 'moderation', 'moderation', 'openai.moderations', '/v1/moderations'),
    Probe('openai', 'embeddings', 'embeddings', 'openai.embeddings', '/v1/embeddings'),
    Probe('openai', 'chat_stream', 'chat', 'openai.chat', '/v1/chat/completions', stream=True),
    Probe('openai', 'chat', 'chat', 'openai.chat', '/v1/chat/completions', optional='chat_nonstream'),
    Probe('openai', 'responses', 'responses', 'openai.responses', '/v1/responses', optional='responses'),
    Probe('nvidia_api_catalog', 'chat_stream', 'chat', 'openai.chat', '/v1/chat/completions', stream=True),
    Probe('nvidia_api_catalog', 'embeddings', 'embeddings', 'openai.embeddings', '/v1/embeddings'),
    Probe('hf_router', 'chat', 'chat', 'openai.chat', '/v1/chat/completions'),
    Probe('openai', 'speech', 'speech', 'openai.audio.speech', '/v1/audio/speech'),
    Probe('openai', 'transcription', 'transcription', 'openai.audio.transcriptions', '/v1/audio/transcriptions',
          needs='openai.speech'),
    Probe('openai', 'images', 'images', 'openai.images.generations', '/v1/images/generations', optional='images'),
)
PROBE_IDS = frozenset(p.id for p in PROBES)
PROBE_NAMES = frozenset(p.name for p in PROBES) | {'models'}


# ---------------------------------------------------------------------------------------------------------------------
# egress allowlist guard (written next to the run as sitecustomize.py; also loaded into this process with no host)

GUARD_SOURCE = r'''"""Egress allowlist of a broker live-smoke process (written by tools/broker_live_smoke.py). Not a sandbox.

Installed at interpreter start when TITANOS_SMOKE_EGRESS_GUARD=1 and this directory is on PYTHONPATH. Name lookups
succeed only for 'localhost', loopback literals and the host names listed in TITANOS_SMOKE_EGRESS_ALLOW (comma list);
the addresses those lookups return become the only non-loopback addresses a socket may connect or send to. Covered:
socket.getaddrinfo, gethostbyname, gethostbyname_ex, gethostbyaddr, socket.connect, connect_ex and sendto, and on
Windows asyncio's IocpProactor.connect. Every refusal, and the install itself, is appended as one JSON line to
TITANOS_SMOKE_EGRESS_REPORT (with TITANOS_SMOKE_EGRESS_NONCE). The next sitecustomize after this directory on sys.path
still runs.
"""
import ipaddress
import json
import os
import socket
import sys
import threading

MARK = '__titanos_smoke_guard__'
_lock = threading.Lock()
_names = frozenset()
_ips = set()
_originals = {}


def _text(host):
    if isinstance(host, bytes):
        host = host.decode('idna', 'replace')
    return str(host).strip().strip('[]').split('%', 1)[0]


def _ip(host):
    try:
        return ipaddress.ip_address(_text(host))
    except ValueError:
        return None


def _loopback(host):
    if _text(host).lower().rstrip('.') == 'localhost':
        return True
    address = _ip(host)
    if address is None:
        return False
    mapped = getattr(address, 'ipv4_mapped', None)
    return bool((mapped or address).is_loopback)


def _known_ip(host):
    address = _ip(host)
    if address is None:
        return False
    mapped = getattr(address, 'ipv4_mapped', None)
    with _lock:
        return (mapped or address).compressed in _ips


def _report(event):
    event = dict(event, pid=os.getpid(), nonce=os.environ.get('TITANOS_SMOKE_EGRESS_NONCE', ''))
    path = os.environ.get('TITANOS_SMOKE_EGRESS_REPORT')
    if path:
        try:
            with open(path, 'a', encoding='utf-8') as handle:
                handle.write(json.dumps(event, sort_keys=True) + '\n')
        except OSError:
            pass


def _remember(addresses):
    with _lock:
        for value in addresses:
            address = _ip(value)
            if address is not None:
                mapped = getattr(address, 'ipv4_mapped', None)
                _ips.add((mapped or address).compressed)


def _lookup_allowed(host):
    return host is None or _loopback(host) or _known_ip(host) or _text(host).lower().rstrip('.') in _names


def _refuse_lookup(host, port, via):
    _report({'event': 'refused', 'host': _text(host), 'port': port, 'via': via})
    raise socket.gaierror(socket.EAI_NONAME, 'live-smoke egress guard: lookup of %s refused' % _text(host))


def _check_address(address, via):
    if isinstance(address, (str, bytes)) or not isinstance(address, tuple) or not address:
        return
    host = address[0]
    if _loopback(host) or _known_ip(host):
        return
    port = address[1] if len(address) > 1 else None
    _report({'event': 'refused', 'host': _text(host), 'port': port, 'via': via})
    raise PermissionError('live-smoke egress guard: connection to %s:%s refused' % (_text(host), port))


def installed():
    return bool(_originals)


def refusals():
    return list(_refused)


_refused = []


def install(allowed=None):
    """Patch every lookup and connect path (idempotent). allowed: host names; None reads the environment."""
    global _names
    if _originals:
        return
    if allowed is None:
        allowed = os.environ.get('TITANOS_SMOKE_EGRESS_ALLOW', '').split(',')
    _names = frozenset(h.strip().lower().rstrip('.') for h in allowed if h and h.strip())
    original_getaddrinfo = socket.getaddrinfo
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_sendto = socket.socket.sendto
    original_byname = socket.gethostbyname
    original_byname_ex = socket.gethostbyname_ex
    original_byaddr = socket.gethostbyaddr

    def getaddrinfo(host, port, *args, **kwargs):
        if not _lookup_allowed(host):
            _refused.append(_text(host))
            _refuse_lookup(host, port, 'getaddrinfo')
        infos = original_getaddrinfo(host, port, *args, **kwargs)
        if host is not None and not _loopback(host):
            _remember(info[4][0] for info in infos if isinstance(info, tuple) and len(info) == 5)
        return infos

    def gethostbyname(host):
        if not _lookup_allowed(host):
            _refused.append(_text(host))
            _refuse_lookup(host, None, 'gethostbyname')
        address = original_byname(host)
        _remember([address])
        return address

    def gethostbyname_ex(host):
        if not _lookup_allowed(host):
            _refused.append(_text(host))
            _refuse_lookup(host, None, 'gethostbyname_ex')
        result = original_byname_ex(host)
        _remember(result[2])
        return result

    def gethostbyaddr(host):
        if not _loopback(host):  # a reverse lookup is a DNS query of its own
            _refused.append(_text(host))
            _refuse_lookup(host, None, 'gethostbyaddr')
        return original_byaddr(host)

    def connect(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6):
            _check_address(address, 'socket.connect')
        return original_connect(self, address)

    def connect_ex(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6):
            _check_address(address, 'socket.connect_ex')
        return original_connect_ex(self, address)

    def sendto(self, data, *args):
        if self.family in (socket.AF_INET, socket.AF_INET6) and args:
            _check_address(args[-1], 'socket.sendto')
        return original_sendto(self, data, *args)

    for fn in (getaddrinfo, gethostbyname, gethostbyname_ex, gethostbyaddr, connect, connect_ex, sendto):
        setattr(fn, MARK, True)
    _originals.update(getaddrinfo=original_getaddrinfo, connect=original_connect, connect_ex=original_connect_ex,
                      sendto=original_sendto, gethostbyname=original_byname, gethostbyname_ex=original_byname_ex,
                      gethostbyaddr=original_byaddr)
    socket.getaddrinfo = getaddrinfo
    socket.gethostbyname = gethostbyname
    socket.gethostbyname_ex = gethostbyname_ex
    socket.gethostbyaddr = gethostbyaddr
    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    socket.socket.sendto = sendto
    if sys.platform == 'win32':
        from asyncio import windows_events
        original_iocp = windows_events.IocpProactor.connect

        def iocp_connect(self, conn, address):
            if getattr(conn, 'family', None) in (socket.AF_INET, socket.AF_INET6):
                _check_address(address, 'IocpProactor.connect')
            return original_iocp(self, conn, address)

        setattr(iocp_connect, MARK, True)
        _originals['iocp'] = original_iocp
        windows_events.IocpProactor.connect = iocp_connect
    _report({'event': 'installed', 'allowed': sorted(_names)})


def _chain():
    """Run the next sitecustomize AFTER this directory on sys.path (never an earlier one: no re-entry)."""
    here = os.path.normcase(os.path.dirname(os.path.abspath(__file__)))
    seen = False
    for entry in list(sys.path):
        try:
            current = os.path.normcase(os.path.abspath(entry or os.curdir))
        except Exception:
            continue
        if current == here:
            seen = True
            continue
        if not seen:
            continue
        candidate = os.path.join(entry, 'sitecustomize.py')
        if os.path.isfile(candidate):
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location('_titanos_smoke_chained_sitecustomize', candidate)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            except Exception:
                pass
            return


if __name__ in ('sitecustomize', '_titanos_chained_sitecustomize') and \
        os.environ.get('TITANOS_SMOKE_EGRESS_GUARD', '').strip() == '1':
    install()
    _chain()
'''


def write_guard(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'sitecustomize.py'
    path.write_text(GUARD_SOURCE, encoding='utf-8', newline='\n')
    return path


def load_guard(path: Path):
    """The guard module loaded into this process under its own name (it does not install itself on import)."""
    module = sys.modules.get(GUARD_MODULE)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(GUARD_MODULE, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[GUARD_MODULE] = module
    spec.loader.exec_module(module)
    return module


def read_report(path: Path) -> list[dict]:
    try:
        lines = path.read_text(encoding='utf-8').splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


# ---------------------------------------------------------------------------------------------------------------------
# gates: flags, paths, approval, ledger

def inside_repo(path) -> bool:
    try:
        Path(path).expanduser().resolve().relative_to(ROOT)
        return True
    except ValueError:
        return False


def check_gate(args) -> None:
    """The flag gate of a live run (Refused when any part is missing)."""
    if getattr(args, 'dry_run', False):
        raise Refused('--dry-run is the offline self-test; it cannot be combined with a live run')
    if not (getattr(args, 'live', False) and getattr(args, 'spends_money', False)):
        raise Refused('The live smoke spends money: pass both --live and --i-understand-this-spends-money')
    if not getattr(args, 'approval', None):
        raise Refused('The live smoke needs --approval FILE (the owner approval, outside the repository)')
    if not getattr(args, 'home', None):
        raise Refused('The live smoke needs --home DIR (outside the repository)')
    if inside_repo(args.home):
        raise Refused('--home must be outside the repository')


def _parse_time(value, what: str) -> datetime:
    try:
        moment = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    except ValueError:
        raise Refused(f'The approval needs an ISO {what}') from None
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=timezone.utc)


def _money(value, what: str) -> Decimal:
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise Refused(f'{what} must be a decimal amount') from None
    if not amount.is_finite():
        raise Refused(f'{what} must be finite')
    return amount


def validate_approval(doc, *, sha256: str, now: datetime | None = None) -> dict:
    """The normalized approval, or Refused naming the first failed rule (never echoing a value)."""
    from titanos.broker.config import PriceRow, check_no_secrets
    from titanos.broker.errors import BrokerError

    now = now or datetime.now(timezone.utc)
    if not isinstance(doc, dict):
        raise Refused('The approval file must be a JSON object')
    unknown = sorted(set(doc) - APPROVAL_KEYS)
    if unknown:
        raise Refused('The approval file has unknown keys: ' + ', '.join(k[:40] for k in unknown))
    missing = [k for k in APPROVAL_REQUIRED if k not in doc]
    if missing:
        raise Refused('The approval file lacks: ' + ', '.join(missing))
    try:
        check_no_secrets(doc)
    except (BrokerError, ValueError):
        raise Refused('The approval file contains a secret-shaped value; it may name variables only') from None
    if doc.get('schema', SCHEMA_APPROVAL) != SCHEMA_APPROVAL:
        raise Refused(f'The approval schema must be {SCHEMA_APPROVAL}')
    approved_by = doc['approved_by']
    if not isinstance(approved_by, str) or not approved_by.strip() or len(approved_by) > 128:
        raise Refused('approved_by must name the approver (1-128 characters)')
    expires = _parse_time(doc['expires_at'], 'expires_at')
    if expires <= now:
        raise Refused('The approval has expired; ask the owner for a new approval')
    cap = _money(doc['spend_cap_usd'], 'spend_cap_usd')
    if cap <= 0:
        raise Refused('spend_cap_usd must be positive')
    if cap > AUTHORIZED_CAP_USD:
        raise Refused(f'spend_cap_usd exceeds the authorized USD {AUTHORIZED_CAP_USD}')
    providers = doc['providers']
    if (not isinstance(providers, list) or not providers or len(set(providers)) != len(providers)
            or not all(isinstance(p, str) and p in PROVIDERS for p in providers)):
        raise Refused('providers must be a non-empty list of distinct ids from: ' + ', '.join(PROVIDERS))
    if doc['credential_env'] != {p: CREDENTIAL_ENV[p] for p in providers}:
        raise Refused('credential_env must map each listed provider to its authorized variable: '
                      + ', '.join(f'{p}={CREDENTIAL_ENV[p]}' for p in providers))
    models = doc['models']
    if not isinstance(models, dict) or not set(models) <= set(providers):
        raise Refused('models maps listed providers to {modality: model}')
    clean_models: dict[str, dict[str, str]] = {}
    for provider, table in models.items():
        if not isinstance(table, dict) or not set(table) <= set(MODALITIES[provider]):
            raise Refused(f'models.{provider} accepts the modalities ' + ', '.join(MODALITIES[provider]))
        for modality, model in table.items():
            if (not isinstance(model, str) or not model or len(model) > 200 or model != model.strip()
                    or any(c.isspace() for c in model) or '@' in model):
                raise Refused(f'models.{provider}.{modality} must be one upstream model id (no spaces, no @)')
        clean_models[provider] = dict(table)
    optional = doc.get('optional', [])
    if not isinstance(optional, list) or not set(optional) <= set(OPTIONAL):
        raise Refused('optional lists items from: ' + ', '.join(OPTIONAL))
    rows = doc['prices']
    if not isinstance(rows, list) or not rows:
        raise Refused('prices must be a non-empty list of price rows')
    try:
        prices = [PriceRow.model_validate(row) for row in rows]
    except Exception:
        raise Refused('prices holds a row that is not a valid price row') from None
    output_dir = doc['output_dir']
    if not isinstance(output_dir, str) or not Path(output_dir).is_absolute():
        raise Refused('output_dir must be an absolute directory')
    if inside_repo(output_dir):
        raise Refused('output_dir must be outside the repository')
    params = doc.get('params', {})
    if not isinstance(params, dict) or not set(params) <= PROBE_IDS:
        raise Refused('params maps probe ids to extra request fields')
    for probe_id, extra in params.items():
        if not isinstance(extra, dict) or set(extra) & PROBE_FIELDS:
            raise Refused(f'params.{probe_id} may add fields but never replace ' + ', '.join(sorted(PROBE_FIELDS)))
    return {'sha256': sha256, 'approved_by': approved_by.strip(), 'expires_at': expires, 'cap': cap,
            'providers': list(providers), 'models': clean_models, 'optional': list(optional), 'prices': prices,
            'price_rows': [row.model_dump(mode='json', exclude_none=True) for row in prices],
            'output_dir': Path(output_dir), 'params': params}


def load_approval(path, *, now: datetime | None = None) -> dict:
    path = Path(path).expanduser()
    if inside_repo(path):
        raise Refused('The approval file must live outside the repository')
    try:
        raw = path.read_bytes()
        doc = json.loads(raw.decode('utf-8'))
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        raise Refused(f'The approval file could not be read: {type(exc).__name__}') from None
    return validate_approval(doc, sha256=hashlib.sha256(raw).hexdigest(), now=now)


def prior_spend(output_dir: Path, approval_sha: str) -> Decimal:
    """USD already spent under this approval: settled runs count their spend, unsettled runs their whole reservation."""
    runs: dict[str, dict] = {}
    try:
        lines = (Path(output_dir) / LEDGER).read_text(encoding='utf-8').splitlines()
    except FileNotFoundError:
        return Decimal(0)
    for line in lines:
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if entry.get('approval_sha256') != approval_sha:
            continue
        run = runs.setdefault(str(entry.get('run_id')), {})
        if entry.get('state') == 'started':
            run['reserved'] = Decimal(str(entry.get('reserved_usd', '0')))
        elif entry.get('state') == 'settled':
            run['spent'] = Decimal(str(entry.get('spent_usd', '0')))
    total = Decimal(0)
    for run in runs.values():
        total += run['spent'] if 'spent' in run else run.get('reserved', Decimal(0))
    return total


def ledger_append(output_dir: Path, entry: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / LEDGER).open('a', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(entry, sort_keys=True) + '\n')


def select_probes(approval: dict, providers_arg: str | None,
                  modalities_arg: str | None) -> tuple[list, list, list, bool]:
    """(providers, probes to run, [(probe id, skip reason)], models-list probe selected)."""
    providers = list(approval['providers'])
    if providers_arg:
        wanted = [p.strip() for p in providers_arg.split(',') if p.strip()]
        bad = [p for p in wanted if p not in approval['providers']]
        if bad:
            raise Refused('--providers names providers the approval does not list: ' + ', '.join(b[:40] for b in bad))
        providers = [p for p in providers if p in wanted]
    tokens = None
    if modalities_arg:
        tokens = {t.strip() for t in modalities_arg.split(',') if t.strip()}
        bad = sorted(t for t in tokens if t not in PROBE_NAMES and t not in PROBE_IDS)
        if bad:
            raise Refused('--modalities accepts probe names or ids: ' + ', '.join(sorted(PROBE_NAMES)))
    chosen, skipped = [], []
    for probe in PROBES:
        if probe.provider not in providers:
            continue
        if tokens is not None and probe.name not in tokens and probe.id not in tokens:
            continue
        if probe.optional and probe.optional not in approval['optional']:
            skipped.append((probe.id, f'optional item {probe.optional} is not approved'))
            continue
        if probe.modality not in approval['models'].get(probe.provider, {}):
            skipped.append((probe.id, f'the approval names no {probe.modality} model'))
            continue
        chosen.append(probe)
    models_probe = tokens is None or 'models' in tokens
    if not models_probe:  # an upstream is configured only when one of its probes runs
        providers = [p for p in providers if any(probe.provider == p for probe in chosen)]
    return providers, chosen, skipped, models_probe


def probe_ref(approval: dict, probe: Probe) -> str:
    return f"{approval['models'][probe.provider][probe.modality]}@{probe.provider}"


def unpriced(approval: dict, probes) -> list[str]:
    """Refs whose matched price row (if any) cannot price the probe's quantities: the server would refuse them with
    402 budget_unpriced_target, so the run refuses first."""
    return sorted({probe_ref(approval, p) for p in probes if estimate_usd(approval, p) is None})


def estimate_usd(approval: dict, probe: Probe) -> str | None:
    """A rough pre-run estimate for the plan (the server's own pre-dispatch reservation is what enforces the cap)."""
    from titanos.broker.limits import Estimate
    from titanos.broker.pricing import PriceTable, money
    body = build_body(approval, probe, b'')
    tokens = 0 if isinstance(body, tuple) else math.ceil(len(json.dumps(body).encode('utf-8')) / 3)
    estimate = {'chat': Estimate(input_tokens=tokens, max_output_tokens=DEFAULT_OUTPUT_TOKENS),
                'responses': Estimate(input_tokens=tokens, max_output_tokens=DEFAULT_OUTPUT_TOKENS),
                'embeddings': Estimate(input_tokens=tokens, max_output_tokens=0),
                'moderation': Estimate(input_tokens=tokens, max_output_tokens=0),
                'speech': Estimate(characters=len(PROBE_SPEECH)),
                'transcription': Estimate(audio_seconds=5.0),
                'images': Estimate(images=1)}[probe.modality]
    return money(PriceTable(approval['prices']).estimate(probe_ref(approval, probe), estimate))


# ---------------------------------------------------------------------------------------------------------------------
# request bodies and response checks

def build_body(approval: dict, probe: Probe, audio: bytes | None):
    """JSON body (dict) or (form fields, files) for transcription. Only the fixed probe strings are sent. The model is
    the qualified ref <model>@<upstream>, so routing never depends on a catalog or a local name."""
    model = probe_ref(approval, probe)
    extra = dict(approval['params'].get(probe.id, {}))
    if probe.modality == 'chat':
        body = {'model': model, 'messages': [{'role': 'user', 'content': PROBE_CHAT}], 'max_tokens': DEFAULT_OUTPUT_TOKENS}
        if probe.stream:
            body.update(stream=True, stream_options={'include_usage': True})
    elif probe.modality == 'responses':
        body = {'model': model, 'input': PROBE_CHAT, 'max_output_tokens': DEFAULT_OUTPUT_TOKENS, 'store': False}
    elif probe.modality == 'embeddings':
        body = {'model': model, 'input': PROBE_TEXT, 'encoding_format': 'float'}
        if probe.provider == 'nvidia_api_catalog':
            body['input_type'] = 'query'
    elif probe.modality == 'moderation':
        body = {'model': model, 'input': PROBE_TEXT}
    elif probe.modality == 'speech':
        body = {'model': model, 'input': PROBE_SPEECH, 'voice': 'alloy', 'response_format': 'wav'}
    elif probe.modality == 'images':
        body = {'model': model, 'prompt': PROBE_IMAGE, 'n': 1}
    elif probe.modality == 'transcription':
        fields = {'model': model, 'response_format': 'json', **{k: str(v) for k, v in extra.items()}}
        return fields, {'file': ('probe.wav', audio or b'', 'audio/wav')}
    else:  # pragma: no cover - table is closed
        raise ValueError(probe.modality)
    for name, value in extra.items():
        body[name] = value
    return body


def sse_events(text: str) -> tuple[list, bool]:
    events, done = [], False
    for block in re.split(r'\r?\n\r?\n', text):
        data = [line[5:].lstrip() for line in block.splitlines() if line.startswith('data:')]
        if not data:
            continue
        payload = '\n'.join(data)
        if payload.strip() == '[DONE]':
            done = True
            continue
        try:
            events.append(json.loads(payload))
        except ValueError:
            events.append({'_unparsed': True})
    return events, done


def _int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def check_shape(probe: Probe, response: httpx.Response, content: bytes) -> tuple[dict, dict, dict | None]:
    """(checks, shape facts, usage) of a 200 answer. Facts are sizes, counts and hashes, never the content."""
    ctype = response.headers.get('content-type', '')
    checks, facts, usage = {}, {}, None
    if probe.stream:
        checks['content_type_sse'] = ctype.startswith('text/event-stream')
        events, done = sse_events(content.decode('utf-8', 'replace'))
        chunks = [e for e in events if isinstance(e, dict) and e.get('object') == 'chat.completion.chunk']
        deltas = [c for c in chunks for choice in (c.get('choices') or []) if isinstance(choice, dict)
                  and isinstance(choice.get('delta'), dict)]
        usage_chunks = [c for c in chunks if isinstance(c.get('usage'), dict)]
        usage = usage_chunks[-1]['usage'] if usage_chunks else None
        checks['stream_done'] = done
        checks['stream_chunks'] = bool(deltas) and not any(e.get('_unparsed') for e in events if isinstance(e, dict))
        checks['usage'] = bool(usage) and _int(usage.get('prompt_tokens')) and _int(usage.get('completion_tokens'))
        facts.update(events=len(events), chunks=len(chunks),
                     text_chars=sum(len(str(choice.get('delta', {}).get('content') or ''))
                                    for c in chunks for choice in (c.get('choices') or []) if isinstance(choice, dict)))
        return checks, facts, usage
    if probe.modality == 'speech':
        checks['audio_wav'] = content[:4] == b'RIFF' and content[8:12] == b'WAVE' and len(content) > 44
        facts.update(bytes=len(content), sha256=hashlib.sha256(content).hexdigest()[:16], content_type=ctype)
        return checks, facts, None
    try:
        body = json.loads(content.decode('utf-8'))
    except (ValueError, UnicodeDecodeError):
        return {'json': False}, {'bytes': len(content)}, None
    checks['json'] = isinstance(body, dict)
    if not isinstance(body, dict):
        return checks, facts, None
    usage = body.get('usage') if isinstance(body.get('usage'), dict) else None
    if probe.modality == 'chat':
        choices = body.get('choices') or []
        message = choices[0].get('message') if choices and isinstance(choices[0], dict) else None
        checks['chat_message'] = body.get('object') == 'chat.completion' and isinstance(message, dict)
        checks['usage'] = bool(usage) and _int(usage.get('prompt_tokens')) and _int(usage.get('completion_tokens'))
        facts.update(choices=len(choices), text_chars=len(str((message or {}).get('content') or '')),
                     finish_reason=(choices[0].get('finish_reason') if choices else None))
    elif probe.modality == 'responses':
        checks['response_object'] = body.get('object') == 'response' and body.get('status') in ('completed', 'incomplete')
        checks['usage'] = bool(usage) and _int(usage.get('input_tokens')) and _int(usage.get('output_tokens'))
        facts.update(status=body.get('status'), output_items=len(body.get('output') or []))
    elif probe.modality == 'embeddings':
        data = body.get('data') or []
        vector = data[0].get('embedding') if data and isinstance(data[0], dict) else None
        checks['embedding'] = isinstance(vector, list) and bool(vector) and all(isinstance(v, (int, float))
                                                                               for v in vector)
        checks['usage'] = bool(usage) and _int(usage.get('prompt_tokens'))
        facts.update(vectors=len(data), dimensions=len(vector) if isinstance(vector, list) else None)
    elif probe.modality == 'moderation':
        results = body.get('results') or []
        checks['moderation_result'] = bool(results) and isinstance(results[0], dict) and \
            isinstance(results[0].get('flagged'), bool)
        facts.update(results=len(results))
    elif probe.modality == 'transcription':
        text = body.get('text')
        checks['transcript'] = isinstance(text, str)
        facts.update(text_chars=len(text) if isinstance(text, str) else None,
                     mentions_smoke=bool(isinstance(text, str) and 'smoke' in text.lower()))
    elif probe.modality == 'images':
        data = body.get('data') or []
        item = data[0] if data and isinstance(data[0], dict) else {}
        kind = 'b64_json' if item.get('b64_json') else ('url' if item.get('url') else None)
        checks['image'] = len(data) == 1 and kind is not None
        if kind == 'b64_json':
            try:
                raw = base64.b64decode(item['b64_json'], validate=False)
            except (ValueError, TypeError):
                raw = b''
            facts.update(bytes=len(raw), png=raw[:8] == b'\x89PNG\r\n\x1a\n', sha256=hashlib.sha256(raw).hexdigest()[:16])
        facts.update(images=len(data), kind=kind)
    return checks, facts, usage


def target_class(provider: str) -> str:
    from titanos.broker.profiles import PROFILES
    return 'router' if PROFILES[provider].kind == 'router' else 'cloud'


def check_provenance(probe: Probe, ref: str, request_id: str, response: httpx.Response) -> dict:
    h = response.headers
    return {'request_id_echo': h.get('x-request-id') == request_id,
            'decision_id_header': bool(h.get('x-titanos-decision-id')),
            'served_by': h.get('x-titanos-served-by') == f'{target_class(probe.provider)}:{probe.provider}',
            'served_model': h.get('x-titanos-served-model') == ref,
            'fallback_count_zero': h.get('x-titanos-fallback-count') == '0'}


# Surfaces whose answer carries the billed quantity (tokens, characters, images) or whose upload is measured (the
# transcription probe sends the speech probe's WAV): the decision must hold a cost and an explained basis, never the
# reservation. A moderation answer has no usage (free); it settles at its (zero-priced) reservation in the usage ledger,
# which the end-of-run usage check covers.
PRICED_IN_DECISION = frozenset({'chat', 'responses', 'embeddings', 'speech', 'images', 'transcription'})


def check_decision(probe: Probe, ref: str, key_id: str, decision: dict | None) -> dict:
    if not isinstance(decision, dict):
        return {'decision_found': False}
    served = decision.get('served') if isinstance(decision.get('served'), dict) else {}
    attempts = decision.get('attempts') or []
    checks = {'decision_found': True,
              'decision_served': served.get('ref') == ref and served.get('upstream') == probe.provider
              and served.get('klass') == target_class(probe.provider),
              'decision_key': decision.get('key_id') == key_id,
              'decision_committed': decision.get('committed_at_attempt') is not None and len(attempts) == 1
              and (attempts[0] or {}).get('outcome') == 'success'}
    if probe.modality in PRICED_IN_DECISION:
        checks['decision_priced'] = decision.get('cost_usd') is not None
        checks['decision_price_basis'] = decision.get('price_basis') not in (None, 'reservation', 'unmeasured',
                                                                             'unpriced')
    return checks


def check_usage_rows(rows, probes: list[dict]) -> dict:
    """The key's usage ledger after the run: one request per passed probe ref and surface, nothing unpriced, no
    errors."""
    if rows is None:
        return {'usage_rows': False}
    counted: dict[tuple, int] = {}
    for row in rows:
        counted[(row.get('model'), row.get('surface'))] = counted.get((row.get('model'), row.get('surface')), 0)             + int(row.get('requests') or 0)
    expected: dict[tuple, int] = {}
    for probe in probes:
        if probe['state'] == 'pass':
            expected[(probe['ref'], probe['surface'])] = expected.get((probe['ref'], probe['surface']), 0) + 1
    return {'usage_rows': True,
            'usage_requests_match': all(counted.get(k, 0) == v for k, v in expected.items()),
            'usage_nothing_unpriced': all(int(r.get('unpriced_requests') or 0) == 0 for r in rows),
            'usage_no_errors': all(int(r.get('errors') or 0) == 0 for r in rows)}


def decision_summary(decision: dict | None) -> dict | None:
    if not isinstance(decision, dict):
        return None
    keep = ('decision_id', 'request_id', 'ts', 'role', 'surface', 'inbound_protocol', 'requested_model', 'served',
            'translation', 'attempts', 'fallback_count', 'committed_at_attempt', 'usage', 'cost_usd', 'latency_ms',
            'notes', 'key_id')
    return {k: decision.get(k) for k in keep if k in decision}


# ---------------------------------------------------------------------------------------------------------------------
# the server

def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def protect(directory: Path) -> bool:
    """mkdir + an explicit protected owner-only DACL (0700 on POSIX); True when the verdict reads back owner-only."""
    from titanos.broker.vault import acl
    directory.mkdir(parents=True, exist_ok=True)
    acl.make_owner_only(directory, directory=True)
    return acl.is_owner_only(directory)


def server_env(*, home: Path, bootstrap: str, egress_mode: str, allow_hosts: list[str], guard_dir: Path,
               report: Path, nonce: str, keep: set[str], overrides: dict[str, str], netguard_site: Path | None,
               netguard_report: Path | None) -> dict[str, str]:
    from titanos.broker.redact import PROVIDER_ENV_DENY, PROVIDER_ENV_DENY_SUFFIXES
    deny = {name.upper() for name in PROVIDER_ENV_DENY}
    env = {}
    for name, value in os.environ.items():
        upper = name.upper()
        if upper in keep:
            env[name] = value
            continue
        if upper.startswith('TITANOS_') or upper in deny or upper.endswith(PROVIDER_ENV_DENY_SUFFIXES) \
                or _SECRETISH.search(upper) or upper in {'PYTHONPATH', 'PYTHONSTARTUP', 'PYTHONHOME'}:
            continue
        env[name] = value
    paths = ([str(netguard_site)] if netguard_site else []) + [str(guard_dir), str(SRC)]
    env.update(overrides)
    env.update({
        'PYTHONPATH': os.pathsep.join(paths), 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONIOENCODING': 'utf-8',
        'TITANOS_HOME': str(home), 'TITANOS_API_KEY': bootstrap, 'TITANOS_ROLE': 'broker',
        'TITANOS_BROKER_EGRESS': egress_mode, 'TITANOS_BROKER_VAULT_BACKEND': 'refs-only',
        'TITANOS_BROKER_DECISION_LOG_FILE': 'true', 'TITANOS_BROKER_MAX_FALLBACKS': '0',
        'TITANOS_SMOKE_EGRESS_GUARD': '1', 'TITANOS_SMOKE_EGRESS_ALLOW': ','.join(allow_hosts),
        'TITANOS_SMOKE_EGRESS_REPORT': str(report), 'TITANOS_SMOKE_EGRESS_NONCE': nonce,
    })
    if netguard_site is not None:
        env['TITANOS_TEST_NETGUARD'] = '1'
        env['TITANOS_TEST_NETGUARD_REPORT'] = str(netguard_report)
    return env


class Serve:
    def __init__(self, run_dir: Path, env: dict[str, str], home: Path, timeout: float):
        self.run_dir, self.env, self.home, self.timeout = run_dir, env, home, timeout
        self.port = free_port()
        self.base = f'http://127.0.0.1:{self.port}'
        self.log_path = run_dir / 'serve.log'
        self.proc = None
        self._log = None

    def start(self) -> None:
        registry = self.home / 'models.json'
        command = [sys.executable, '-B', '-m', 'titanos', '--config', str(registry), 'serve', '--host', '127.0.0.1',
                   '--port', str(self.port), '--role', 'broker', '--limit-concurrency', '64', '--log-level', 'warning']
        self._log = self.log_path.open('ab')
        self.proc = subprocess.Popen(command, cwd=str(self.run_dir), env=self.env, stdin=subprocess.DEVNULL,
                                     stdout=self._log, stderr=subprocess.STDOUT)
        deadline = time.perf_counter() + self.timeout
        with httpx.Client(trust_env=False, timeout=2.0) as client:
            while time.perf_counter() < deadline:
                if self.proc.poll() is not None:
                    raise RuntimeError(f'titanos serve exited with {self.proc.returncode} before it was ready '
                                       f'(see {self.log_path.name})')
                try:
                    if client.get(self.base + '/v1/health/ready').status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                time.sleep(0.25)
        raise RuntimeError(f'titanos serve was not ready within {self.timeout:.0f} s')

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self) -> bool:
        """Terminate the server (the venv launcher's job object takes the interpreter with it); True when the port no
        longer answers."""
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=20)
        if self._log is not None:
            self._log.close()
            self._log = None
        stopped = False
        for _ in range(40):
            try:
                with socket.create_connection(('127.0.0.1', self.port), timeout=0.5):
                    pass
            except OSError:
                stopped = True
                break
            time.sleep(0.25)
        return stopped


# ---------------------------------------------------------------------------------------------------------------------
# one run

def _http(base: str) -> httpx.Client:
    return httpx.Client(base_url=base, trust_env=False, follow_redirects=False,
                        timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=10.0))


def _bearer(token: str) -> dict:
    return {'Authorization': 'Bearer ' + token}


def _error_code(response: httpx.Response, content: bytes | None = None) -> str | None:
    try:
        body = json.loads(content if content is not None else response.content)
    except (ValueError, UnicodeDecodeError):
        return None
    error = body.get('error') if isinstance(body, dict) else None
    if isinstance(error, dict):
        return str(error.get('code') or error.get('type') or '')[:80] or None
    return None


def _error_message(response: httpx.Response) -> str:
    try:
        error = response.json().get('error')
    except (ValueError, AttributeError):
        return ''
    return _safe(str(error.get('message') or ''))[:300] if isinstance(error, dict) else ''


def _failure_message(content: bytes | None) -> str | None:
    """The error message of a failed probe's answer (the server's envelope already carries the provider's scrubbed
    message), scrubbed again with redact.scrub and cut to 400 characters, so a live failure is diagnosable from the
    receipt without another call. Envelopes: OpenAI {error: {message}} and the in-stream {error: {message}} event,
    Ollama {error: str}, ElevenLabs {detail: {message}}; any other body is scrubbed text."""
    text = (content or b'').decode('utf-8', 'replace')
    try:
        body = json.loads(text)
    except ValueError:
        events, _ = sse_events(text) if 'data:' in text else ([], False)
        body = next((e for e in events if isinstance(e, dict) and e.get('error')), None)
    message = None
    if isinstance(body, dict):
        error = body.get('error')
        detail = body.get('detail')
        if isinstance(error, dict):
            message = error.get('message')
        elif isinstance(error, str):
            message = error
        elif isinstance(detail, dict):
            message = detail.get('message')
        elif isinstance(detail, str):
            message = detail
    if message is None:
        message = text
    message = _safe(str(message)).strip()
    return message or None


def _body_error(content: bytes | None) -> tuple[str | None, str | None]:
    """(code, message) of the first error envelope inside a 200 answer that failed its checks: an in-stream error event
    (OpenAI chat {error: {...}} and the terminal frame after a mid-stream failure, Responses response.failed
    {response: {error}} and the 'error' event {type: 'error', code, message}, Anthropic {type: 'error', error}), an
    NDJSON line {error: ...} or a JSON body {error: ...}. Both are scrubbed with redact.scrub; the message is cut to 400
    characters. (None, None) when the answer carries no error envelope: unlike _failure_message, the raw body is never
    used as a message, because a 200 answer may hold model output."""
    text = (content or b'').decode('utf-8', 'replace')
    try:
        items = [json.loads(text)]
    except ValueError:
        if 'data:' in text:
            items = sse_events(text)[0]
        else:
            items = []
            for line in text.splitlines():
                try:
                    items.append(json.loads(line))
                except ValueError:
                    continue
    for item in items:
        if not isinstance(item, dict):
            continue
        error = item.get('error')
        if error is None and item.get('type') == 'response.failed' and isinstance(item.get('response'), dict):
            error = item['response'].get('error')
        if error is None and item.get('type') == 'error' and (item.get('message') or item.get('code')):
            error = item
        if isinstance(error, dict) and (error.get('message') or error.get('code') or error.get('type')):
            code = error.get('code') or error.get('type')
            message = _safe(str(error.get('message') or '')).strip() or None
            return (_safe(str(code))[:80] if code else None), message
        if isinstance(error, str) and error.strip():
            return None, _safe(error).strip()
    return None, None


def _admin(client: httpx.Client, boot: str, method: str, path: str, **kw) -> httpx.Response:
    response = client.request(method, path, headers=_bearer(boot), **kw)
    if response.status_code >= 400:
        raise RuntimeError(f'{method} {path} answered {response.status_code} {_error_code(response) or ""}: '
                           + _error_message(response))
    return response


def find_decision(client: httpx.Client, boot: str, decision_id: str | None) -> dict | None:
    if not decision_id:
        return None
    rows = client.get('/api/broker/decisions', params={'limit': 200}, headers=_bearer(boot))
    if rows.status_code != 200:
        return None
    for row in rows.json().get('data') or []:
        if row.get('decision_id') == decision_id:
            return row
    return None


def close_out(base: str, bootstrap: str, key_id: str | None, receipt: dict) -> Decimal:
    """After the probes (or an error): the key's usage rows and their check, the key revoked, the audit chain verified.
    Returns the USD the usage ledger settled for the key."""
    total = Decimal(0)
    with _http(base) as client:
        if key_id:
            rows = client.get('/api/broker/usage', headers=_bearer(bootstrap),
                              params={'group_by': 'upstream,model,surface', 'key_id': key_id})
            receipt['usage'] = rows.json().get('rows') if rows.status_code == 200 else None
            receipt['usage_check'] = check_usage_rows(receipt['usage'], receipt['probes'])
            total = sum((Decimal(r.get('cost_usd') or '0') for r in receipt['usage'] or []), Decimal(0))
            revoked = client.post(f'/api/keys/{key_id}/revoke', headers=_bearer(bootstrap))
            receipt['key']['revoked'] = revoked.status_code == 200
        verify = client.get('/api/audit/verify', headers=_bearer(bootstrap))
        receipt['audit_verify'] = verify.json() if verify.status_code == 200 else {'status': verify.status_code}
    return total


def leak_scan(paths, values: dict[str, str]) -> dict:
    """Search every file under paths for each value (raw and JSON-escaped). Reports names and counts, never values."""
    needles = {}
    for name, value in values.items():
        if isinstance(value, str) and len(value) >= 8:
            forms = {value.encode('utf-8'), json.dumps(value)[1:-1].encode('utf-8')}
            needles[name] = forms
    files, size, hits = 0, 0, []
    for root in paths:
        root = Path(root)
        if not root.exists():
            continue
        for path in ([root] if root.is_file() else sorted(p for p in root.rglob('*') if p.is_file())):
            try:
                data = path.read_bytes()
            except OSError:
                continue
            files += 1
            size += len(data)
            for name, forms in needles.items():
                if any(form in data for form in forms):
                    hits.append({'file': path.name, 'secret': name})
    return {'files': files, 'bytes': size, 'secrets_checked': sorted(needles), 'hits': hits}


def run_smoke(approval: dict, *, home_root: Path, mode: str, providers_arg: str | None = None,
              modalities_arg: str | None = None, base_urls: dict[str, str] | None = None,
              env_overrides: dict[str, str] | None = None, netguard_site: Path | None = None,
              serve_timeout: float = 180.0, log=print) -> dict:
    """One run (live or dry-run). Returns the receipt; probe failures are recorded, never raised. Refused is raised
    only by preflight, before the server starts."""
    live = mode == 'live'
    providers, probes, skipped, models_probe = select_probes(approval, providers_arg, modalities_arg)
    if not providers or (not probes and not models_probe):
        raise Refused('Nothing to run: the filters select no probe')
    missing_price = unpriced(approval, probes)
    if missing_price:
        raise Refused('No price row matches: ' + ', '.join(missing_price) + ' (budgets fail closed on unpriced targets)')
    output_dir = approval['output_dir']
    if inside_repo(home_root) or inside_repo(output_dir):
        raise Refused('--home and output_dir must be outside the repository')
    spent_before = prior_spend(output_dir, approval['sha256'])
    remaining = approval['cap'] - spent_before
    if remaining <= 0:
        raise Refused(f'The approval cap is used up (spent USD {spent_before} of {approval["cap"]})')
    if live:
        absent = [CREDENTIAL_ENV[p] for p in providers if not os.environ.get(CREDENTIAL_ENV[p])]
        if absent:
            raise Refused('Set the approved credential variables first: ' + ', '.join(absent))

    from titanos.broker.profiles import PROFILES
    from titanos.config import update_config
    import titanos

    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
    evidence_id = ('live-' if live else 'dry-') + run_id
    run_dir = Path(home_root).expanduser().resolve() / f'run-{run_id}'
    out_dir = Path(output_dir) / run_id
    home = run_dir / 'titanos-home'
    acl_ok = {'run_dir': protect(run_dir), 'titanos_home': protect(home), 'broker': protect(home / 'broker'),
              'vault': protect(home / 'broker' / 'vault'), 'output': protect(out_dir)}
    update_config(home / 'models.json', lambda current: None)  # what `titanos init` does
    guard_dir = run_dir / 'egress_guard'
    write_guard(guard_dir)
    guard_report = run_dir / 'egress-guard.jsonl'
    netguard_report = run_dir / 'netguard.jsonl'
    hosts = sorted({httpx.URL(PROFILES[p].default_base_url).host for p in providers}) if live else []
    nonce = uuid.uuid4().hex
    bootstrap = 'smoke-boot-' + secrets.token_urlsafe(32)
    keep = {CREDENTIAL_ENV[p] for p in providers} if live else set()
    env = server_env(home=home, bootstrap=bootstrap, egress_mode='internet' if live else 'loopback-only',
                     allow_hosts=hosts, guard_dir=guard_dir, report=guard_report, nonce=nonce, keep=keep,
                     overrides=env_overrides or {}, netguard_site=netguard_site, netguard_report=netguard_report)
    receipt = {
        'schema': SCHEMA_RECEIPT, 'run_id': run_id, 'evidence_id': evidence_id, 'mode': mode,
        'started_at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        'titanos_version': titanos.__version__, 'source_commit': _commit(),
        'approval': {'sha256': approval['sha256'], 'approved_by': approval['approved_by'],
                     'expires_at': approval['expires_at'].isoformat().replace('+00:00', 'Z'),
                     'spend_cap_usd': str(approval['cap']), 'spent_before_usd': str(spent_before),
                     'providers': approval['providers'], 'optional': approval['optional'],
                     'price_rows': approval['price_rows']},
        'filters': {'providers': providers_arg, 'modalities': modalities_arg},
        'server': {'role': 'broker', 'bind': '127.0.0.1', 'egress_mode': env['TITANOS_BROKER_EGRESS'],
                   'egress_allowlist': hosts, 'vault_backend': 'refs-only', 'protected_dirs': acl_ok,
                   'base_urls': 'profile defaults' if live else 'loopback mocks'},
        'credentials': [], 'key': None, 'probes': [], 'skipped': [{'probe': p, 'reason': r} for p, r in skipped],
        'spend': None, 'usage': None, 'usage_check': None, 'audit_verify': None, 'egress_guard': None, 'secret_scan': None,
        'result': 'fail', 'errors': [],
    }
    ledger_append(output_dir, {'state': 'started', 'run_id': run_id, 'approval_sha256': approval['sha256'],
                               'reserved_usd': str(remaining), 'mode': mode})
    key_secret, key_id, spent = None, None, Decimal(0)
    serve = Serve(run_dir, env, home, serve_timeout)
    stopped_at_cap = False
    try:
        log(f'[smoke] starting titanos serve (role broker, egress {env["TITANOS_BROKER_EGRESS"]}) on {serve.base}')
        serve.start()
        installs = [e for e in read_report(guard_report) if e.get('event') == 'installed' and e.get('nonce') == nonce]
        receipt['server']['guard_installed'] = bool(installs) and all(e.get('allowed') == hosts for e in installs)
        if not receipt['server']['guard_installed']:
            raise RuntimeError('The egress allowlist guard did not report its install in the server process')
        with _http(serve.base) as client:
            # upstreams (created disabled: a credential binds to an existing upstream's origin), prices, policy
            upstreams = {}
            for provider in providers:
                models = sorted({m for m in approval['models'].get(provider, {}).values()})
                upstreams[provider] = {'profile': provider, 'enabled': False, 'models': models, 'catalog': 'static',
                                       'label': 'broker live smoke'}
                if base_urls:
                    upstreams[provider]['base_url'] = base_urls[provider]
                _admin(client, bootstrap, 'PUT', f'/api/broker/upstreams/{provider}', json=upstreams[provider])
            _admin(client, bootstrap, 'PUT', '/api/broker/prices', json=approval['price_rows'])
            _admin(client, bootstrap, 'PUT', '/api/broker/policy',
                   json={'cloud_policy': 'explicit', 'default_location': 'auto', 'anonymous_cloud': False,
                         'allow_http_lan': False, 'max_fallbacks': 0})
            usable = set()
            for provider in providers:
                started = time.perf_counter()
                answer = client.put(f'/api/broker/credentials/smoke-{provider}', headers=_bearer(bootstrap),
                                    json={'upstream': provider, 'env': CREDENTIAL_ENV[provider], 'validate': models_probe})
                meta = answer.json().get('credential', answer.json()) if answer.headers.get(
                    'content-type', '').startswith('application/json') else {}
                record = {'id': f'smoke-{provider}', 'upstream': provider, 'source': 'env',
                          'env_var': CREDENTIAL_ENV[provider], 'origin': meta.get('origin'),
                          'status': answer.status_code,
                          'fingerprint': meta.get('fingerprint'), 'validation': meta.get('validation'),
                          'models_probe': models_probe, 'ms': round((time.perf_counter() - started) * 1000, 1)}
                receipt['credentials'].append(record)
                if answer.status_code in (200, 201):
                    usable.add(provider)
                    # bind the credential and enable the upstream (same base_url: the validation stands)
                    _admin(client, bootstrap, 'PUT', f'/api/broker/upstreams/{provider}',
                           json=dict(upstreams[provider], credential=f'smoke-{provider}', enabled=True))
            # the inbound key: budget = the cap minus earlier spend under this approval
            expires = datetime.now(timezone.utc) + timedelta(hours=KEY_HOURS)
            refs = sorted({probe_ref(approval, p) for p in probes})
            spec = {'name': f'live-smoke-{run_id[-8:]}',
                    'scopes': ['inference', 'broker:read'] + [f'cloud:{p}' for p in providers],
                    'constraints': {'upstreams': providers, 'models': refs or ['*'],
                                    'target_classes': sorted({target_class(p) for p in providers})},
                    'limits': {'rpm': 30, 'concurrent': 1},
                    'budget': {'usd_per_day': str(remaining), 'usd_per_month': str(remaining),
                               'max_request_usd': str(remaining), 'hard': True},
                    'expires_at': expires.isoformat().replace('+00:00', 'Z'), 'note': 'broker live smoke'}
            minted = _admin(client, bootstrap, 'POST', '/api/keys', json=spec).json()
            key_secret, key_id = minted.pop('secret'), minted['id']
            receipt['key'] = {'id': key_id, 'scopes': minted.get('scopes'), 'budget': minted.get('budget'),
                              'constraints': minted.get('constraints'), 'limits': minted.get('limits'),
                              'expires_at': minted.get('expires_at')}
            outputs: dict[str, bytes] = {}
            for index, probe in enumerate(probes):
                ref = probe_ref(approval, probe)
                entry = {'probe': probe.id, 'provider': probe.provider, 'modality': probe.modality,
                         'surface': probe.surface, 'stream': probe.stream, 'ref': ref,
                         'inbound': f'POST {probe.path}', 'state': 'skipped'}
                receipt['probes'].append(entry)
                if stopped_at_cap:
                    entry['reason'] = 'spend cap reached'
                    continue
                if probe.provider not in usable:
                    entry['reason'] = 'the credential did not validate'
                    continue
                if probe.needs and probe.needs not in outputs:
                    entry['reason'] = f'needs the output of {probe.needs}'
                    continue
                left = client.get('/api/keys/self', headers=_bearer(key_secret))
                budget_left = (left.json().get('budget_remaining') or {}) if left.status_code == 200 else {}
                entry['budget_remaining_before'] = budget_left.get('usd_today')
                if budget_left.get('usd_today') is not None and Decimal(budget_left['usd_today']) <= 0:
                    stopped_at_cap = True
                    entry['reason'] = 'spend cap reached'
                    continue
                request_id = f'smoke-{run_id[-8:]}-{index:02d}'
                headers = {**_bearer(key_secret), 'X-Request-ID': request_id}
                body = build_body(approval, probe, outputs.get(probe.needs or '', b''))
                started = time.perf_counter()
                first_byte = None
                try:
                    if isinstance(body, tuple):
                        response = client.post(probe.path, headers=headers, data=body[0], files=body[1])
                        content = response.content
                    else:
                        with client.stream('POST', probe.path, headers=headers, json=body) as response:
                            chunks = []
                            for chunk in response.iter_bytes():
                                if first_byte is None:
                                    first_byte = time.perf_counter()
                                chunks.append(chunk)
                            content = b''.join(chunks)
                except httpx.HTTPError as exc:
                    entry.update(state='fail', reason=f'transport error {type(exc).__name__}')
                    continue
                elapsed = time.perf_counter() - started
                entry.update(status=response.status_code, ms=round(elapsed * 1000, 1),
                             ms_first_byte=round((first_byte - started) * 1000, 1) if first_byte else None,
                             request_id=request_id,
                             headers={k: v for k, v in response.headers.items()
                                      if k.lower().startswith('x-titanos-') or k.lower() in _RECORDED_HEADERS})
                if response.status_code != 200:
                    code = _error_code(response, content)
                    entry.update(state='fail', reason=f'HTTP {response.status_code} {code or ""}'.strip(),
                                 error_code=code, error_message=_failure_message(content))
                    if response.status_code == 402 or code in ('budget_exceeded', 'budget_unpriced_target'):
                        entry['state'] = 'stopped_at_cap'
                        stopped_at_cap = True
                    log(f'[smoke] {probe.id}: {entry["state"]} ({entry["reason"]})')
                    continue
                try:
                    checks, facts, usage = check_shape(probe, response, content)
                    checks.update(check_provenance(probe, ref, request_id, response))
                    decision = find_decision(client, bootstrap, response.headers.get('x-titanos-decision-id'))
                    checks.update(check_decision(probe, ref, key_id, decision))
                except (ValueError, KeyError, TypeError, AttributeError, IndexError, httpx.HTTPError) as exc:
                    code, message = _body_error(content)
                    entry.update(state='fail', reason=f'checks could not run: {type(exc).__name__}',
                                 error_code=code, error_message=message)
                    continue
                cost = (decision or {}).get('cost_usd')
                if cost is not None:
                    spent += Decimal(str(cost))
                entry.update(checks=checks, shape=facts, usage=usage, decision=decision_summary(decision),
                             cost_usd=cost, state='pass' if all(checks.values()) else 'fail')
                if entry['state'] == 'fail':
                    entry['reason'] = 'failed checks: ' + ', '.join(k for k, v in checks.items() if not v)
                    # a 200 answer that failed its checks: keep the in-body or in-stream error, scrubbed (None if none)
                    code, message = _body_error(content)
                    entry.update(error_code=code, error_message=message)
                if probe.modality == 'speech' and checks.get('audio_wav'):
                    outputs[probe.id] = content
                log(f'[smoke] {probe.id}: {entry["state"]} ({response.status_code}, {entry["ms"]} ms)')
    except (RuntimeError, httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        receipt['errors'].append(f'{type(exc).__name__}: {_safe(str(exc))}')
    finally:
        try:
            if serve.running():
                spent = max(spent, close_out(serve.base, bootstrap, key_id, receipt))
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            receipt['errors'].append(f'close-out {type(exc).__name__}: {_safe(str(exc))}')
        receipt['server']['stopped'] = serve.stop()
    conservative = receipt['usage'] is None and any(p.get('status') == 200 for p in receipt['probes'])
    if conservative:  # the usage ledger could not be read after calls went out: settle the whole reservation
        spent = max(spent, remaining)
    events = read_report(guard_report)
    receipt['egress_guard'] = {'installs': sum(1 for e in events if e.get('event') == 'installed'),
                               'refusals': [{k: e.get(k) for k in ('host', 'port', 'via')} for e in events
                                            if e.get('event') == 'refused']}
    if netguard_site is not None:
        receipt['netguard'] = {'blocked': [{k: e.get(k) for k in ('host', 'port', 'via')}
                                           for e in read_report(netguard_report)]}
    receipt['spend'] = {'spent_usd': str(spent), 'cap_usd': str(approval['cap']), 'budget_usd': str(remaining),
                        'stopped_at_cap': stopped_at_cap, 'settled_at_reservation': conservative}
    ledger_append(output_dir, {'state': 'settled', 'run_id': run_id, 'approval_sha256': approval['sha256'],
                               'spent_usd': str(spent), 'mode': mode})
    executed = [p for p in receipt['probes'] if p['state'] != 'skipped']
    receipt['result'] = 'pass' if (executed and all(p['state'] == 'pass' for p in executed)
                                   and not receipt['errors'] and not receipt['egress_guard']['refusals']
                                   and all((receipt.get('usage_check') or {'usage_rows': False}).values())
                                   and all(c['status'] in (200, 201) for c in receipt['credentials'])) else 'fail'
    receipt['finished_at'] = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    receipt['cells'] = evidence_cells(receipt, live=live)
    secret_values = {'bootstrap_key': bootstrap}
    if key_secret:
        secret_values['inbound_key'] = key_secret
    secret_values.update(_credential_values(providers, live=live, overrides=env_overrides or {}))
    records = matrix_records(receipt)
    records_path = out_dir / (EVIDENCE_FILE if live else DRY_EVIDENCE_FILE)
    _write_json(out_dir / 'receipt.json', receipt)
    _write_jsonl(records_path, records)
    scan = leak_scan([run_dir, out_dir, output_dir / LEDGER], secret_values)
    receipt['secret_scan'] = scan
    if scan['hits']:  # a leak voids the run: no cell may cite it
        receipt['result'] = 'fail'
        records = []
        for cell in receipt['cells']:
            if cell['state'] in ('live-pass', 'mock-pass'):
                cell['state'] = 'fail'
    receipt['evidence_records'] = {'file': records_path.name, 'records': len(records)}
    _write_json(out_dir / 'receipt.json', receipt)
    _write_jsonl(records_path, records)
    final = leak_scan([out_dir / 'receipt.json', records_path], secret_values)
    if final['hits']:  # pragma: no cover - a leak in the counts-only rewrite would be a bug in this tool
        receipt['result'] = 'fail'
        _write_jsonl(records_path, [])
    secret_values.clear()
    receipt['paths'] = {'run_dir': str(run_dir), 'out_dir': str(out_dir), 'evidence': str(records_path)}
    return receipt


def matrix_records(receipt: dict) -> list[dict]:
    """Records in the live evidence format tools/build_broker_matrix.py reads (--live PATH, JSONL): one per passed
    probe, {evidence_id, outcome: pass, profile, surface, stream: on|off, observed_on, note}; the cell is
    <surface>|<profile>|<stream>. A live run writes them to broker_live_evidence.jsonl; a dry run writes the same shape
    with dry- ids to dry_run_evidence_format.jsonl, which is never live evidence."""
    from titanos.broker.surfaces import SURFACES
    observed = str(receipt['started_at'])[:10]
    out = []
    for probe in receipt['probes']:
        if probe['state'] != 'pass':
            continue
        surface = SURFACES[probe['surface']]
        stream = 'on' if probe['stream'] or str(surface.stream_from).startswith('path:') else 'off'
        decision = (probe.get('decision') or {}).get('decision_id') or 'none'
        out.append({'evidence_id': f"{receipt['evidence_id']}.{probe['probe']}", 'outcome': 'pass',
                    'profile': probe['provider'], 'surface': probe['surface'], 'stream': stream,
                    'observed_on': observed,
                    'note': f"broker live smoke {probe['ref']} via titanos serve role broker, decision {decision}"})
    return out


def evidence_cells(receipt: dict, *, live: bool) -> list[dict]:
    """Every probe of the run as a matrix-shaped cell (surface, adapter, profile, upstream, target class, stream, error
    class, role, state), the failed and untried ones included, for the receipt."""
    from titanos.broker.profiles import PROFILES
    passed = 'live-pass' if live else 'mock-pass'
    cells = []
    for cred in receipt['credentials']:
        if not cred['models_probe']:
            continue
        ok = cred['status'] in (200, 201) and (cred.get('validation') or '') == 'ok'
        cells.append({'evidence_id': f"{receipt['evidence_id']}.{cred['upstream']}.models",
                      'probe': f"{cred['upstream']}.models", 'kind': 'credential_validation', 'surface': 'models.list',
                      'profile': cred['upstream'], 'adapter': PROFILES[cred['upstream']].adapter,
                      'upstream': cred['upstream'], 'target_class': target_class(cred['upstream']), 'stream': False,
                      'error_class': 'none', 'role': 'broker', 'model': None, 'state': passed if ok else 'fail'})
    for probe in receipt['probes']:
        decision = probe.get('decision') or {}
        state = {'pass': passed, 'fail': 'fail', 'stopped_at_cap': 'untried', 'skipped': 'untried'}[probe['state']]
        cells.append({'evidence_id': f"{receipt['evidence_id']}.{probe['probe']}", 'probe': probe['probe'],
                      'kind': 'inference', 'surface': decision.get('surface') or probe['surface'],
                      'inbound_protocol': decision.get('inbound_protocol'), 'inbound': probe['inbound'],
                      'profile': probe['provider'], 'adapter': PROFILES[probe['provider']].adapter,
                      'upstream': probe['provider'], 'target_class': target_class(probe['provider']),
                      'stream': probe['stream'], 'error_class': 'none', 'role': 'broker', 'model': probe['ref'],
                      'state': state, 'reason': probe.get('reason'),
                      'decision_id': decision.get('decision_id'), 'cost_usd': probe.get('cost_usd')})
    for item in receipt['skipped']:
        probe = next(p for p in PROBES if p.id == item['probe'])
        cells.append({'evidence_id': f"{receipt['evidence_id']}.{item['probe']}", 'probe': item['probe'],
                      'kind': 'inference', 'surface': probe.surface, 'inbound': f'POST {probe.path}',
                      'profile': probe.provider, 'adapter': PROFILES[probe.provider].adapter, 'upstream': probe.provider,
                      'target_class': target_class(probe.provider), 'stream': probe.stream, 'error_class': 'none',
                      'role': 'broker', 'model': None, 'state': 'untried', 'reason': item['reason']})
    return cells


def _credential_values(providers, *, live: bool, overrides: dict) -> dict[str, str]:
    """The credential values for the leak scan only (never printed, logged or stored)."""
    out = {}
    for provider in providers:
        var = CREDENTIAL_ENV[provider]
        value = os.environ.get(var) if live else overrides.get(var)
        if value:
            out[f'credential:{provider}'] = value
    return out


def _safe(text: str) -> str:
    from titanos.broker.redact import scrub
    return scrub(text)[:400]


def _commit() -> str | None:
    try:
        out = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT), capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    value = out.stdout.strip()
    return value if re.fullmatch(r'[0-9a-f]{40}', value) else None


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows), encoding='utf-8', newline='\n')
    os.replace(tmp, path)


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, sort_keys=False, default=str) + '\n', encoding='utf-8', newline='\n')
    os.replace(tmp, path)


def install_local_guard(directory: Path):
    """This process talks only to loopback: the allowlist guard with no approved host."""
    guard = load_guard(write_guard(directory))
    guard.install(allowed=[])
    return guard


# ---------------------------------------------------------------------------------------------------------------------
# --dry-run: the offline self-test

DRY_SECRETS = {'OPENAI_API_KEY': 'test-openai-not-a-key', 'NGC_API_KEY': 'test-nvidia-not-a-key',
               'HF_TOKEN': 'test-hf-not-a-key'}
DRY_MODELS = {'openai': {'chat': 'gpt-4o-mini', 'responses': 'gpt-4o-mini', 'embeddings': 'text-embedding-3-small',
                         'speech': 'gpt-4o-mini-tts', 'transcription': 'gpt-4o-mini-transcribe',
                         'moderation': 'omni-moderation-latest', 'images': 'gpt-image-1-mini'},
              'nvidia_api_catalog': {'chat': 'google/gemma-3-4b-it', 'embeddings': 'nvidia/llama-3.2-nv-embedqa-1b-v1'},
              'hf_router': {'chat': 'meta-llama/Llama-3.1-8B-Instruct'}}
# The image probe's default: the smallest current OpenAI image model at its cheapest documented setting. dall-e-2 is gone
# (live 2026-10-02: first 400 unknown_parameter for response_format, then "The model 'dall-e-2' does not exist."), and
# gpt-image models answer base64 only; a 1024x1024 low-quality gpt-image-1-mini image passed live on 2026-10-02.
DRY_IMAGE_PARAMS = {'size': '1024x1024', 'quality': 'low'}
# Transcription models OpenAI bills by tokens (the answer reports usage.input_tokens/output_tokens and no seconds): their
# price rows need input_per_mtok and output_per_mtok, or settlement can only price them by the measured upload duration
# at per_audio_minute.
TOKEN_BILLED_TRANSCRIPTION = ('gpt-4o-transcribe*', 'gpt-4o-mini-transcribe*')
# the price fields each modality's answer is settled on (approval template)
PRICE_FIELDS = {'chat': ('input_per_mtok', 'output_per_mtok'), 'responses': ('input_per_mtok', 'output_per_mtok'),
                'embeddings': ('input_per_mtok', 'output_per_mtok'), 'moderation': ('input_per_mtok', 'output_per_mtok'),
                'speech': ('per_1k_chars',), 'transcription': ('per_audio_minute',), 'images': ('per_image',)}


def template_prices(models: dict) -> list[dict]:
    """One placeholder price row per approved model, with the fields its answer is settled on (token-billed
    transcription models get token prices plus the per-minute fallback; free surfaces such as moderation still need a
    zero row because budgets fail closed on unpriced targets)."""
    import fnmatch
    rows: dict[str, dict] = {}
    for provider, by_modality in models.items():
        for modality, model in by_modality.items():
            fields = PRICE_FIELDS[modality]
            if modality == 'transcription' and any(fnmatch.fnmatchcase(model, g) for g in TOKEN_BILLED_TRANSCRIPTION):
                fields = ('input_per_mtok', 'output_per_mtok', 'per_audio_minute')
            row = rows.setdefault(f'{model}@{provider}', {'match': f'{model}@{provider}'})
            for name in fields:
                row.setdefault(name, f'<USD, {modality}>' if modality != 'moderation' else '0')
    return list(rows.values())


def dry_approval_doc(output_dir: Path, *, cap='0.50', expensive: bool = False) -> dict:
    scale = '1000000' if expensive else None
    rows = [{'match': '*@openai', 'input_per_mtok': scale or '0.15', 'output_per_mtok': scale or '0.6',
             'per_1k_chars': '0.015', 'per_audio_minute': '0.003', 'per_image': '0.016'},
            {'match': 'omni-moderation-*@openai', 'input_per_mtok': '0', 'output_per_mtok': '0'},
            {'match': '*@nvidia_api_catalog', 'input_per_mtok': scale or '0', 'output_per_mtok': scale or '0'},
            {'match': '*@hf_router', 'input_per_mtok': scale or '0.1', 'output_per_mtok': scale or '0.1'}]
    return {'schema': SCHEMA_APPROVAL, 'approved_by': 'dry-run self-test',
            'expires_at': (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat().replace('+00:00', 'Z'),
            'spend_cap_usd': cap, 'providers': list(PROVIDERS), 'credential_env': dict(CREDENTIAL_ENV),
            'models': json.loads(json.dumps(DRY_MODELS)), 'optional': list(OPTIONAL), 'prices': rows,
            'output_dir': str(output_dir), 'params': {'openai.images': dict(DRY_IMAGE_PARAMS)}}


_EVIDENCE_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9._:-]{2,95}')
_EVIDENCE_FIELDS = frozenset({'evidence_id', 'outcome', 'cell', 'profile', 'surface', 'stream', 'observed_on', 'note'})


def matrix_format_check(path: Path) -> tuple[str, bool, object]:
    """The evidence records parse with the matrix builder's own reader (tools/build_broker_matrix.read_live) and name
    provider cells that exist there (an intercept surface and one of its stream variants). Without the builder in this
    checkout the record shape is checked against the same rules locally."""
    from titanos.broker.surfaces import SURFACES
    name = 'evidence: the records parse as matrix live evidence and name existing provider cells'
    rows = read_report(path)
    builder = ROOT / 'tools' / 'build_broker_matrix.py'
    try:
        if builder.is_file():
            spec = importlib.util.spec_from_file_location('titanos_build_broker_matrix_for_smoke', builder)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            parsed = module.read_live(path)
            variants = getattr(module, '_variants')
            how = 'build_broker_matrix.read_live'
        else:
            parsed = [{'evidence_id': r['evidence_id'], 'cell': f"{r['surface']}|{r['profile']}|{r['stream']}"}
                      for r in rows if set(r) <= _EVIDENCE_FIELDS and r.get('outcome') == 'pass'
                      and _EVIDENCE_ID.fullmatch(str(r.get('evidence_id')))]
            variants = None
            how = 'local rules (tools/build_broker_matrix.py is not in this checkout)'
    except SystemExit as exc:
        return name, False, f'the matrix builder refused the records: {exc}'
    bad = []
    for record in parsed:
        surface, profile, stream = record['cell'].split('|')
        row = SURFACES.get(surface)
        allowed = variants(row) if (variants and row) else ('on', 'off')
        if row is None or row.kind != 'intercept' or stream not in allowed or profile not in PROVIDERS:
            bad.append(record['cell'])
    return name, len(parsed) == len(rows) and bool(rows) and not bad, {'reader': how, 'records': len(parsed), 'bad': bad}


class SelfTest:
    def __init__(self, log=print):
        self.results: list[dict] = []
        self.log = log

    def check(self, name: str, ok: bool, detail=None) -> bool:
        self.results.append({'check': name, 'ok': bool(ok), 'detail': detail})
        self.log(f'[dry-run] {"ok  " if ok else "FAIL"} {name}' + (f' ({detail})' if detail and not ok else ''))
        return bool(ok)

    def refuses(self, name: str, fn) -> bool:
        try:
            fn()
        except Refused as exc:
            return self.check(name, True, str(exc)[:160])
        except Exception as exc:  # noqa: BLE001 - recorded as a failed check
            return self.check(name, False, f'{type(exc).__name__} instead of Refused')
        return self.check(name, False, 'accepted')


def run_dry(home: Path, serve_timeout: float, log=print) -> int:
    sys.path.insert(0, str(TESTS))
    from broker_mocks import OpenAIMock, netguard  # noqa: E402  (tests/broker_mocks, the WP4 mocks)
    from broker_mocks.server import redact_value as mock_spelling
    netguard.install()
    local_guard = install_local_guard(home / 'local-guard')
    t = SelfTest(log)
    now = datetime.now(timezone.utc)

    # 1. flag gate and the `titanos broker live-smoke` wrapper
    ns = lambda **kw: SimpleNamespace(**{'live': True, 'spends_money': True, 'approval': str(home / 'a.json'),  # noqa: E731
                                         'home': str(home), 'dry_run': False, 'providers': None, 'modalities': None,
                                         **kw})
    t.refuses('gate: no --live', lambda: check_gate(ns(live=False)))
    t.refuses('gate: no --i-understand-this-spends-money', lambda: check_gate(ns(spends_money=False)))
    t.refuses('gate: no --approval', lambda: check_gate(ns(approval=None)))
    t.refuses('gate: --home inside the repository', lambda: check_gate(ns(home=str(ROOT / 'tools'))))
    t.refuses('gate: --dry-run with --live', lambda: check_gate(ns(dry_run=True)))
    from titanos.broker.cli import live_smoke_command
    expired = home / 'expired-approval.json'
    expired.write_text(json.dumps({'expires_at': '2020-01-01T00:00:00Z'}), encoding='utf-8')
    for label, kw in (('wrapper refuses without both flags', {'spends_money': False}),
                      ('wrapper refuses an expired approval', {'approval': expired})):
        try:
            live_smoke_command(SimpleNamespace(**{'live': True, 'spends_money': True, 'approval': home / 'none.json',
                                                  'home': home, 'providers': None, 'modalities': None, **kw}))
            t.check(label, False, 'accepted')
        except ValueError:
            t.check(label, True)

    # 2. approval rules
    out_ok = home / 'scenario-full' / 'out'
    base = dry_approval_doc(out_ok)
    good = validate_approval(base, sha256='0' * 64, now=now)
    t.check('approval: the dry approval validates', good['cap'] == Decimal('0.50'))
    variants = {
        'approval: expired': dict(base, expires_at='2020-01-01T00:00:00Z'),
        'approval: cap above the authorized USD 0.50': dict(base, spend_cap_usd='0.51'),
        'approval: zero cap': dict(base, spend_cap_usd='0'),
        'approval: another credential variable': dict(base, credential_env=dict(CREDENTIAL_ENV, openai='MY_KEY')),
        'approval: unknown provider': dict(base, providers=['openai', 'anthropic']),
        'approval: unknown key': dict(base, extra=True),
        'approval: secret-shaped value': dict(base, note='sk-proj-' + 'a' * 40),
        'approval: output_dir inside the repository': dict(base, output_dir=str(ROOT / 'docs')),
        'approval: params replace a probe field': dict(base, params={'openai.chat': {'messages': []}}),
        'approval: model with @': dict(base, models={'openai': {'chat': 'gpt@other'}}),
        'approval: no prices': dict(base, prices=[]),
    }
    for label, doc in variants.items():
        t.refuses(label, lambda doc=doc: validate_approval(doc, sha256='0' * 64, now=now))
    t.refuses('approval: file inside the repository', lambda: load_approval(ROOT / 'tools' / 'no-such-approval.json'))
    unpriced_doc = dict(base, prices=[{'match': '*@openai', 'input_per_mtok': '1', 'output_per_mtok': '1'}])
    t.refuses('preflight: unpriced probe refused before the server starts',
              lambda: run_smoke(validate_approval(unpriced_doc, sha256='1' * 64, now=now), home_root=home,
                                mode='dry-run', modalities_arg='images', base_urls={}, log=lambda *_: None))
    t.refuses('preflight: unknown modality filter', lambda: select_probes(good, None, 'chat,video'))
    t.refuses('preflight: a provider the approval does not list',
              lambda: select_probes(validate_approval(dict(base, providers=['openai'], credential_env={
                  'openai': 'OPENAI_API_KEY'}, models={'openai': DRY_MODELS['openai']}), sha256='2' * 64, now=now),
                  'hf_router', None))

    # 2b. the approval template: current NVIDIA catalog defaults (asymmetric embeddings get input_type) and token price
    # fields for token-billed transcription models
    from titanos.broker.adapters.common import matches as profile_glob
    from titanos.broker.profiles import PROFILES
    embed = DRY_MODELS['nvidia_api_catalog']['embeddings']
    t.check('template: NVIDIA defaults are current catalog models',
            DRY_MODELS['nvidia_api_catalog'] == {'chat': 'google/gemma-3-4b-it',
                                                 'embeddings': 'nvidia/llama-3.2-nv-embedqa-1b-v1'})
    t.check('template: the NVIDIA embeddings default is asymmetric (the probe sends input_type)',
            any(profile_glob(embed, g) for g in PROFILES['nvidia_api_catalog'].extras['asymmetric_embedding_globs']))
    rows = {row['match']: row for row in template_prices(DRY_MODELS)}
    stt = rows[f"{DRY_MODELS['openai']['transcription']}@openai"]
    t.check('template: the token-billed transcription row carries token prices and the per-minute fallback',
            {'input_per_mtok', 'output_per_mtok', 'per_audio_minute'} <= set(stt), stt)
    t.check('template: one price row per probed model',
            set(rows) == {f'{m}@{p}' for p, by in DRY_MODELS.items() for m in by.values()}, sorted(rows))
    image_row = rows.get(f"{DRY_MODELS['openai']['images']}@openai") or {}
    t.check('template: the image default is gpt-image-1-mini, 1024x1024, quality low, with a per_image placeholder',
            DRY_MODELS['openai']['images'] == 'gpt-image-1-mini'
            and dry_approval_doc(home)['params'] == {'openai.images': {'size': '1024x1024', 'quality': 'low'}}
            and set(image_row) == {'match', 'per_image'} and image_row['per_image'].startswith('<USD'), image_row)

    # 3. spend ledger
    ledger_dir = home / 'ledger-test'
    ledger_append(ledger_dir, {'state': 'started', 'run_id': 'a', 'approval_sha256': 'x', 'reserved_usd': '0.50'})
    ledger_append(ledger_dir, {'state': 'settled', 'run_id': 'a', 'approval_sha256': 'x', 'spent_usd': '0.10'})
    ledger_append(ledger_dir, {'state': 'started', 'run_id': 'b', 'approval_sha256': 'x', 'reserved_usd': '0.40'})
    ledger_append(ledger_dir, {'state': 'started', 'run_id': 'c', 'approval_sha256': 'y', 'reserved_usd': '0.50'})
    t.check('ledger: settled spend plus unsettled reservations', prior_spend(ledger_dir, 'x') == Decimal('0.50'))
    used = validate_approval(dict(base, output_dir=str(ledger_dir)), sha256='x', now=now)
    t.refuses('ledger: a used-up approval is refused before the server starts',
              lambda: run_smoke(used, home_root=home, mode='dry-run', base_urls={}, log=lambda *_: None))

    # 4. leak scanner positive control
    planted = home / 'leak-control'
    planted.mkdir(parents=True, exist_ok=True)
    (planted / 'a.log').write_text('prefix ' + DRY_SECRETS['HF_TOKEN'] + ' suffix', encoding='utf-8')
    (planted / 'b.json').write_text(json.dumps({'k': 'clean'}), encoding='utf-8')
    found = leak_scan([planted], {'credential:hf_router': DRY_SECRETS['HF_TOKEN']})
    t.check('leak scan: finds a planted value and reports names only',
            found['hits'] == [{'file': 'a.log', 'secret': 'credential:hf_router'}]
            and DRY_SECRETS['HF_TOKEN'] not in json.dumps(found))

    # 5. the allowlist guard in a child interpreter (the test netguard sits under it as the backstop)
    t.check('guard: this process carries the allowlist guard', local_guard.installed())
    guard_dir = home / 'guard-test'
    write_guard(guard_dir)
    report, backstop = guard_dir / 'report.jsonl', guard_dir / 'netguard.jsonl'
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(('127.0.0.1', 0))
    listener.listen(1)
    code = ('import socket\n'
            'r = []\n'
            'for host in ("example.invalid", "192.0.2.1"):\n'
            '    try:\n'
            '        socket.getaddrinfo(host, 443); r.append("resolved")\n'
            '    except OSError: r.append("refused")\n'
            's = socket.socket()\n'
            'try:\n'
            '    s.connect(("192.0.2.1", 9)); r.append("connected")\n'
            'except OSError: r.append("refused")\n'
            'finally: s.close()\n'
            f's = socket.create_connection(("127.0.0.1", {listener.getsockname()[1]}), timeout=5); s.close()\n'
            'r.append("loopback")\n'
            'print(",".join(r))\n')
    env = server_env(home=guard_dir, bootstrap='x' * 16, egress_mode='loopback-only', allow_hosts=['api.example.com'],
                     guard_dir=guard_dir, report=report, nonce='selftest', keep=set(), overrides={},
                     netguard_site=TESTS / 'broker_mocks' / 'netguard_site', netguard_report=backstop)
    child = subprocess.run([sys.executable, '-B', '-c', code], env=env, capture_output=True, text=True, timeout=60,
                           cwd=str(guard_dir))
    listener.close()
    events = read_report(report)
    t.check('guard: unapproved lookups and connects refused, loopback allowed',
            child.stdout.strip() == 'refused,refused,refused,loopback', child.stdout.strip() or child.stderr[-300:])
    t.check('guard: the refusals came from the allowlist guard (the backstop saw none)',
            sum(1 for e in events if e.get('event') == 'refused') == 3 and not read_report(backstop),
            {'guard': len(events), 'backstop': len(read_report(backstop))})
    t.check('guard: installed with exactly the approved host list',
            any(e.get('event') == 'installed' and e.get('allowed') == ['api.example.com'] for e in events))
    # the allow path: an approved name resolves (a stub resolver answers with a TEST-NET address, so no DNS query is
    # sent) and its address passes the allowlist guard; the netguard backstop then stops the connect in this test
    report2, backstop2 = guard_dir / 'report-allow.jsonl', guard_dir / 'netguard-allow.jsonl'
    code2 = ('import importlib.util, socket\n'
             'real = socket.getaddrinfo\n'
             'def stub(host, port, *a, **k):\n'
             '    if host == "api.example.com":\n'
             '        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.0.2.10", port))]\n'
             '    return real(host, port, *a, **k)\n'
             'socket.getaddrinfo = stub\n'
             f'spec = importlib.util.spec_from_file_location("allow_guard", {str(guard_dir / "sitecustomize.py")!r})\n'
             'guard = importlib.util.module_from_spec(spec); spec.loader.exec_module(guard)\n'
             'guard.install(allowed=["api.example.com"])\n'
             'r = [socket.getaddrinfo("api.example.com", 443)[0][4][0]]\n'
             'try:\n'
             '    socket.getaddrinfo("other.example.com", 443); r.append("resolved")\n'
             'except OSError: r.append("refused")\n'
             's = socket.socket()\n'
             'try:\n'
             '    s.connect(("192.0.2.10", 443)); r.append("connected")\n'
             'except PermissionError as exc: r.append("guard" if "live-smoke" in str(exc) else "backstop")\n'
             'except OSError: r.append("oserror")\n'
             'finally: s.close()\n'
             'print(",".join(r))\n')
    env2 = dict(env, TITANOS_SMOKE_EGRESS_GUARD='0', TITANOS_SMOKE_EGRESS_REPORT=str(report2),
                TITANOS_TEST_NETGUARD_REPORT=str(backstop2))
    child2 = subprocess.run([sys.executable, '-B', '-c', code2], env=env2, capture_output=True, text=True, timeout=60,
                            cwd=str(guard_dir))
    t.check('guard: an approved host resolves and its address passes the guard (the backstop stops it in this test)',
            child2.stdout.strip() == '192.0.2.10,refused,backstop'
            and [e.get('host') for e in read_report(report2) if e.get('event') == 'refused'] == ['other.example.com']
            and [e.get('host') for e in read_report(backstop2)] == ['192.0.2.10'],
            child2.stdout.strip() or child2.stderr[-300:])

    # 6. full run against the mocks
    mocks = {'openai': OpenAIMock(), 'nvidia_api_catalog': OpenAIMock(credential=DRY_SECRETS['NGC_API_KEY']),
             'hf_router': OpenAIMock(credential=DRY_SECRETS['HF_TOKEN'])}
    for mock in mocks.values():
        mock.start()
    try:
        bases = {p: m.api_base for p, m in mocks.items()}
        doc = dry_approval_doc(out_ok)
        path = home / 'scenario-full' / 'approval.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc, indent=2), encoding='utf-8')
        approval = load_approval(path)
        receipt = run_smoke(approval, home_root=home / 'scenario-full', mode='dry-run', base_urls=bases,
                            env_overrides=dict(DRY_SECRETS), netguard_site=TESTS / 'broker_mocks' / 'netguard_site',
                            serve_timeout=serve_timeout, log=log)
        states = {p['probe']: p['state'] for p in receipt['probes']}
        t.check('full: every probe mock-pass', receipt['result'] == 'pass' and set(states.values()) == {'pass'}
                and len(states) == len(PROBES), {k: v for k, v in states.items() if v != 'pass'} or receipt['errors'])
        for probe in receipt['probes']:
            if probe['state'] != 'pass':
                t.check(f'full: {probe["probe"]}', False, probe.get('reason'))
        t.check('full: credentials validated through the models list',
                len(receipt['credentials']) == len(PROVIDERS) and all(c['status'] == 201 and c['validation'] == 'ok' and c['fingerprint'] for c in receipt['credentials']),
                receipt['credentials'])
        inference = {p: [h for h in m.hits if h.surface and h.surface != 'openai.models'] for p, m in mocks.items()}
        expected = {p: sum(1 for pr in PROBES if pr.provider == p) for p in mocks}
        t.check('full: exactly one upstream hit per probe', {p: len(h) for p, h in inference.items()} == expected,
                {p: len(h) for p, h in inference.items()})
        presented = all(hits for hits in inference.values()) and all(h.presented(DRY_SECRETS[CREDENTIAL_ENV[p]]) for p, hits in inference.items() for h in hits)
        t.check('full: every upstream hit carried the env-referenced credential', presented)
        # each upstream's own env-referenced credential, spelled as the mock recorder spells it ('<sentinel:NAME>' when the dry
        # secret is a broker_mocks sentinel, '<foreign:FP>' otherwise)
        spelling = {p: mock_spelling('Bearer ' + DRY_SECRETS[CREDENTIAL_ENV[p]])[0] for p in mocks}
        t.check('full: the only credential an upstream saw is its own (no inbound or bootstrap key forwarded)',
                all(hits for hits in inference.values())
                and all([c[1] for c in h.credentials] == [spelling[p]] for p, hits in inference.items() for h in hits),
                sorted({c[1] for hits in inference.values() for h in hits for c in h.credentials}))
        t.check('full: no secret in any run file (server log, TITANOS_HOME, receipts)',
                receipt['secret_scan']['hits'] == [] and receipt['secret_scan']['files'] > 5,
                receipt['secret_scan'])
        t.check('full: zero non-loopback attempts (allowlist guard and netguard)',
                receipt['egress_guard']['refusals'] == [] and receipt['netguard']['blocked'] == []
                and receipt['server']['guard_installed'], receipt['egress_guard'])
        t.check('full: usage ledger has one priced request per probe and no errors',
                bool(receipt['usage_check']) and all(receipt['usage_check'].values()), receipt['usage_check'])
        t.check('full: protected owner-only directories', all(receipt['server']['protected_dirs'].values()),
                receipt['server']['protected_dirs'])
        t.check('full: key revoked, audit chain verified, server stopped',
                bool(receipt['key']) and receipt['key'].get('revoked') and receipt['server']['stopped']
                and (receipt['audit_verify'] or {}).get('ok') is True, receipt['audit_verify'])
        t.check('full: receipt cells are mock-pass, never live-pass',
                {c['state'] for c in receipt['cells']} == {'mock-pass'}
                and len(receipt['cells']) == len(PROBES) + len(PROVIDERS))
        records_path = Path(receipt['paths']['evidence'])
        t.check('full: a dry run never writes broker_live_evidence.jsonl',
                records_path.name == DRY_EVIDENCE_FILE
                and not (Path(receipt['paths']['out_dir']) / EVIDENCE_FILE).exists())
        t.check('full: one evidence record per passed probe', len(read_report(records_path)) == len(PROBES))
        t.check(*matrix_format_check(records_path))
        spent = Decimal(receipt['spend']['spent_usd'])
        t.check('full: spend settled in the ledger and within the cap',
                Decimal(0) < spent <= Decimal('0.50') and prior_spend(out_ok, approval['sha256']) == spent,
                receipt['spend'])
        text = (Path(receipt['paths']['out_dir']) / 'receipt.json').read_text('utf-8')
        t.check('full: the receipt holds no request text beyond the fixed probe strings',
                'Reply with' not in text and 'plain blue' not in text)

        # 7. the cap: prices so high that the first reservation exceeds the budget -> 402, nothing sent upstream
        for mock in mocks.values():
            mock.reset()
        out_cap = home / 'scenario-cap' / 'out'
        cap_doc = dry_approval_doc(out_cap, cap='0.01', expensive=True)
        cap_path = home / 'scenario-cap' / 'approval.json'
        cap_path.parent.mkdir(parents=True, exist_ok=True)
        cap_path.write_text(json.dumps(cap_doc), encoding='utf-8')
        capped = run_smoke(load_approval(cap_path), home_root=home / 'scenario-cap', mode='dry-run', base_urls=bases,
                           modalities_arg='embeddings,chat_stream,chat', providers_arg='openai',
                           env_overrides=dict(DRY_SECRETS), netguard_site=TESTS / 'broker_mocks' / 'netguard_site',
                           serve_timeout=serve_timeout, log=log)
        first, rest = (capped['probes'] or [{}])[0], capped['probes'][1:]
        t.check('cap: the first request is refused 402 by the pre-dispatch budget',
                first['state'] == 'stopped_at_cap' and first.get('status') == 402, first.get('reason'))
        t.check('cap: every later probe is skipped at the cap',
                rest and all(p['state'] == 'skipped' and p['reason'] == 'spend cap reached' for p in rest))
        t.check('cap: nothing reached an upstream inference surface',
                all(not [h for h in m.hits if h.surface and h.surface != 'openai.models'] for m in mocks.values()))
        t.check('cap: run marked stopped_at_cap with zero spend',
                capped['spend']['stopped_at_cap'] and capped['spend']['spent_usd'] == '0', capped['spend'])

        # 8. upstream failures: a 400 whose body echoes the presented credential (as real providers echo key
        # fragments) and a stream whose first event is an error; neither may become evidence or leak into any file
        for mock in mocks.values():
            mock.reset()
        mocks['openai'].faults.push('status', status=400, leak=True, surface='openai.embeddings')
        mocks['openai'].faults.push('error_first_event', surface='openai.chat')
        out_err = home / 'scenario-errors' / 'out'
        err_path = home / 'scenario-errors' / 'approval.json'
        err_path.parent.mkdir(parents=True, exist_ok=True)
        err_path.write_text(json.dumps(dry_approval_doc(out_err)), encoding='utf-8')
        failed_run = run_smoke(load_approval(err_path), home_root=home / 'scenario-errors', mode='dry-run',
                               base_urls=bases, providers_arg='openai', modalities_arg='moderation,embeddings,chat_stream',
                               env_overrides=dict(DRY_SECRETS), netguard_site=TESTS / 'broker_mocks' / 'netguard_site',
                               serve_timeout=serve_timeout, log=log)
        states = {p['probe']: p['state'] for p in failed_run['probes']}
        t.check('errors: failed upstream calls fail their probes and the run',
                states == {'openai.moderation': 'pass', 'openai.embeddings': 'fail', 'openai.chat_stream': 'fail'}
                and failed_run['result'] == 'fail', states)
        records = read_report(Path(failed_run['paths']['evidence']))
        t.check('errors: only the passed probe is an evidence record',
                [r['evidence_id'].rsplit('.', 1)[-1] for r in records] == ['moderation'], records)
        failed_probes = [p for p in failed_run['probes'] if p['state'] == 'fail']
        messages = {p['probe']: p.get('error_message') for p in failed_probes}
        t.check('errors: each failed probe keeps the scrubbed provider message (at most 400 characters)',
                bool(failed_probes) and all(isinstance(m, str) and 0 < len(m) <= 400 for m in messages.values()),
                [{k: p.get(k) for k in ('probe', 'status', 'error_code', 'error_message')} for p in failed_probes])
        embed_message = messages.get('openai.embeddings') or ''
        t.check('errors: the 400 message echoing the credential is kept, the credential itself is redacted',
                'provided key' in embed_message and not any(v in embed_message for v in DRY_SECRETS.values()),
                messages)
        t.check('errors: the echoed credential reached no file (server log, TITANOS_HOME, receipts)',
                failed_run['secret_scan']['hits'] == [] and 'credential:openai' in
                failed_run['secret_scan']['secrets_checked'], failed_run['secret_scan'])
        t.check('errors: the key is still revoked and the server stopped',
                failed_run['key']['revoked'] and failed_run['server']['stopped'])

        # 9. a 200 answer that fails its checks: the upstream drops the connection after two frames, the server ends the
        # committed stream with its terminal error frame, and the probe keeps that in-stream error, scrubbed
        for mock in mocks.values():
            mock.reset()
        mocks['openai'].faults.push('disconnect_after_frames', frames=2, surface='openai.chat')
        out_mid = home / 'scenario-midstream' / 'out'
        mid_path = home / 'scenario-midstream' / 'approval.json'
        mid_path.parent.mkdir(parents=True, exist_ok=True)
        mid_path.write_text(json.dumps(dry_approval_doc(out_mid)), encoding='utf-8')
        mid_run = run_smoke(load_approval(mid_path), home_root=home / 'scenario-midstream', mode='dry-run',
                            base_urls=bases, providers_arg='openai', modalities_arg='chat_stream',
                            env_overrides=dict(DRY_SECRETS), netguard_site=TESTS / 'broker_mocks' / 'netguard_site',
                            serve_timeout=serve_timeout, log=log)
        mid = (mid_run['probes'] or [{}])[0]
        mid_message = mid.get('error_message')
        t.check('errors: a 200 stream that fails its checks keeps the scrubbed in-stream error code and message',
                mid.get('probe') == 'openai.chat_stream' and mid.get('state') == 'fail' and mid.get('status') == 200
                and mid.get('reason', '').startswith('failed checks') and bool(mid.get('error_code'))
                and isinstance(mid_message, str) and 0 < len(mid_message) <= 400
                and not any(v in mid_message for v in DRY_SECRETS.values()),
                {k: mid.get(k) for k in ('probe', 'state', 'status', 'reason', 'error_code', 'error_message')})
        t.check('errors: the failed 200 probe is not an evidence record and nothing leaked',
                read_report(Path(mid_run['paths']['evidence'])) == [] and mid_run['result'] == 'fail'
                and mid_run['secret_scan']['hits'] == [], mid_run['secret_scan'])
        t.check('errors: no in-body error is invented for a 200 answer without one',
                _body_error(b'{"object": "list", "data": []}') == (None, None)
                and _body_error(b'RIFF\x00\x00WAVEdata') == (None, None))
    finally:
        for mock in mocks.values():
            mock.stop()
    t.check('netguard: zero blocked attempts in this process', netguard.blocked_count() == 0, netguard.attempts())
    failed = [r for r in t.results if not r['ok']]
    _write_json(home / 'dry-run-summary.json', {'checks': len(t.results), 'failed': failed, 'results': t.results})
    log(f'[dry-run] {len(t.results) - len(failed)}/{len(t.results)} checks passed; summary: '
        f'{home / "dry-run-summary.json"}')
    return EXIT_OK if not failed else EXIT_FAIL


# ---------------------------------------------------------------------------------------------------------------------

def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Broker live smoke (manual tier; spends money; never in CI)')
    p.add_argument('--live', action='store_true', help='Run against the live providers')
    p.add_argument('--i-understand-this-spends-money', dest='spends_money', action='store_true')
    p.add_argument('--approval', type=Path, help='Owner approval file (outside the repository)')
    p.add_argument('--home', type=Path, help='Directory for the fresh TITANOS_HOME and server log of each run')
    p.add_argument('--providers', help='Comma list (subset of the approval): ' + ','.join(PROVIDERS))
    p.add_argument('--modalities', help='Comma list of probe names or ids: ' + ','.join(sorted(PROBE_NAMES)))
    p.add_argument('--plan', action='store_true', help='Validate the approval and print the plan; start nothing')
    p.add_argument('--dry-run', action='store_true', help='Offline self-test against the loopback mocks')
    p.add_argument('--approval-template', action='store_true',
                   help='Print an approval skeleton with the default cheap models (the owner fills it in)')
    p.add_argument('--serve-timeout', type=float, default=180.0, help='Seconds to wait for titanos serve readiness')
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.dry_run:
            if args.live or args.spends_money or args.plan:
                raise Refused('--dry-run is the offline self-test; it takes no --live, spend or --plan flag')
            temporary = not args.home
            home = Path(args.home).expanduser() if args.home else Path(tempfile.mkdtemp(prefix='titanos-smoke-dry-'))
            if inside_repo(home):
                raise Refused('--home must be outside the repository')
            home.mkdir(parents=True, exist_ok=True)
            code = run_dry(home.resolve(), args.serve_timeout)
            if temporary and code == EXIT_OK:  # a passing self-test leaves nothing behind; a failing one keeps its files
                shutil.rmtree(home, ignore_errors=True)
            return code
        if args.approval_template:
            template = dry_approval_doc(Path('<absolute directory outside the repository>'))
            template.update(approved_by='<owner>', expires_at='<ISO time, e.g. 2026-10-03T00:00:00Z>',
                            prices=template_prices(template['models']))
            print(json.dumps(template, indent=2))
            return EXIT_OK
        if args.plan:
            if not args.approval:
                raise Refused('--plan needs --approval FILE')
            approval = load_approval(args.approval)
            providers, probes, skipped, models_probe = select_probes(approval, args.providers, args.modalities)
            plan = {'providers': providers, 'approval_sha256': approval['sha256'], 'approved_by': approval['approved_by'],
                    'expires_at': approval['expires_at'].isoformat(), 'spend_cap_usd': str(approval['cap']),
                    'spent_before_usd': str(prior_spend(approval['output_dir'], approval['sha256'])),
                    'models_list_probe': models_probe,
                    'probes': [{'probe': p.id, 'ref': probe_ref(approval, p), 'estimate_usd': estimate_usd(approval, p)}
                               for p in probes],
                    'skipped': [{'probe': p, 'reason': r} for p, r in skipped],
                    'unpriced': unpriced(approval, probes)}
            print(json.dumps(plan, indent=2))
            return EXIT_OK if not plan['unpriced'] else EXIT_REFUSED
        check_gate(args)
        approval = load_approval(args.approval)
        install_local_guard(Path(args.home).expanduser() / 'local-guard')
        receipt = run_smoke(approval, home_root=Path(args.home), mode='live', providers_arg=args.providers,
                            modalities_arg=args.modalities, serve_timeout=args.serve_timeout)
    except Refused as exc:
        print(f'broker live smoke refused: {exc}', file=sys.stderr)
        return EXIT_REFUSED
    summary = {'result': receipt['result'], 'evidence_id': receipt['evidence_id'], 'spend': receipt['spend'],
               'probes': {p['probe']: p['state'] for p in receipt['probes']}, 'errors': receipt['errors'],
               'secret_scan_hits': len(receipt['secret_scan']['hits']), 'receipts': receipt['paths']['out_dir']}
    print(json.dumps(summary, indent=2))
    return EXIT_OK if receipt['result'] == 'pass' else EXIT_FAIL


if __name__ == '__main__':
    sys.exit(main())
