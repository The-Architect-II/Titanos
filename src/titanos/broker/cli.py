"""`titanos broker ...`, `titanos keys ...`, `titanos audit ...` and `titanos hivemind-manifest` (register_cli, run_cli).

Two modes:
- Offline (no --url): the command edits TITANOS_HOME files directly (broker.json, keys.json, the vault and the audit
  log) under the same FileLocks the server uses. Filesystem ownership is the authority, so this works in keyless mode
  too. Run it as the same OS user as the server (DPAPI and Keychain are user-scoped). Every change is audited before it
  is written, exactly like the HTTP admin routes. A running server picks up keys.json and the vault index on their next
  mtime check; broker.json changes need `titanos broker reload --url URL` (with a broker:admin key once the server
  enforces keys; Offline.apply_hint says which case applies) or a restart.
- --url URL: the command goes through titanos.client.Client to a running server, authenticated with the key in
  TITANOS_API_KEY or in the variable named by --api-key-env. A key is never accepted as an argument.

Secrets: there is no secret-valued option anywhere (argparse introspection is a test). A credential value comes only
from a hidden getpass prompt (the default), one line of stdin (--from-stdin) or an environment reference
(--from-env VAR: the value stays in the environment; only the name is recorded). A minted or rotated key secret is
printed once (or written to --output-file with an owner-only ACL) and never again.

Offline commands that need egress (credential validation, upstream test, advisor test) make exactly the one probe the
HTTP routes make, through the same guarded EgressClient and role switches; explain and catalog are zero-egress.
"""
from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import os
import re
import sys
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit

from .scopes import DEFAULT_KEY_SCOPES

COMMANDS = frozenset({'broker', 'keys', 'audit', 'hivemind-manifest'})
LOCATIONS = ('local_only', 'prefer_local', 'auto', 'cloud_only')
TARGET_CLASSES = ('local', 'cloud', 'router')
CAPABILITIES = ('chat', 'tools', 'vision', 'responses', 'embeddings', 'speech', 'transcription', 'images', 'moderation')
VIRTUAL_CAPABILITIES = ('chat', 'embeddings', 'speech', 'transcription', 'images', 'moderation')
STRATEGIES = ('ordered', 'weighted', 'cheapest', 'lowest_latency', 'advisor')
ADVISOR_PROFILES = ('generic-v1', 'hivemind-gim-router', 'pending-contract')
USAGE_GROUPS = ('key', 'upstream', 'model', 'surface', 'day')
HIVEMIND_PLACEHOLDERS = frozenset({'bind', 'port', 'bin', 'cli'})
RESET_CONFIRMATION = 'DELETE KEYS'
ACTOR = 'cli'
RELOAD_NOTE = 'Saved {path} (keys.json and the vault index are picked up automatically). {apply}.'
RELOAD_COMMAND = '`titanos broker reload --url URL`'
ADMIN_MINT = '`titanos keys create --name admin --scope broker:admin --expires 1d`'


# -- parser ------------------------------------------------------------------------------------------------------------
_OPTION_WORD = re.compile(r'--?[A-Za-z][A-Za-z0-9-]{0,40}')
_COMMAND_WORD = re.compile(r'(?!(?:sk|rk|pk|tk|hf|gsk|xai|nvapi|key|api)[-_])(?!.*\d{4})[a-z][a-z0-9-]{0,23}')
# Every argparse message that quotes (repr) a value typed on the command line: invalid choice, invalid <type> value
# (int/float/any type callable raising TypeError or ValueError), ignored explicit argument ('--flag=value' on a flag that
# takes none) and unknown parser. Group 1 is the fixed lead-in, group 2 the quoted value.
_QUOTED_VALUE = re.compile(r"(invalid choice: |invalid [^'\"\n]{1,120}? value: |ignored explicit argument |unknown parser )"
                           r"('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")")
# argparse prints the whole ambiguous argument unquoted, including any '=value' part
_AMBIGUOUS = re.compile(r'(ambiguous option: )(.*?)( could match )')
# FileType: "can't open '<name>': <OSError text>", where the OSError text repeats the name (repr)
_CANT_OPEN = re.compile(r"(can't open )'(.*?)'(: .*)?$", re.S)


def _redact_word(word: str) -> str:
    """An option name stays and its '=value' becomes '=<hidden>'; any other word becomes '<hidden>'."""
    name, equals, _ = word.partition('=')
    return name + ('=<hidden>' if equals else '') if _OPTION_WORD.fullmatch(name) else '<hidden>'


def redact_parser_message(message: str) -> str:
    """An argparse error never echoes a value typed on the command line: unrecognized arguments keep only their option
    names (a value, a positional extra or '--opt=value' becomes '<hidden>'), an ambiguous option keeps only its name,
    and a quoted value (invalid choice, invalid <type> value, ignored explicit argument, unknown parser, can't open)
    that does not look like a command word is hidden. A secret mistakenly put in argv is refused without being printed."""
    prefix = 'unrecognized arguments: '
    if message.startswith(prefix):
        return prefix + ' '.join(_redact_word(word) for word in message[len(prefix):].split())

    def quoted(match):
        value = match.group(2)[1:-1]
        return match.group(0) if _COMMAND_WORD.fullmatch(value) else match.group(1) + "'<hidden>'"
    def cant_open(match):
        value, rest = match.group(2), match.group(3) or ''
        if _COMMAND_WORD.fullmatch(value):
            return match.group(0)
        for spelling in {value, repr(value)[1:-1]} - {''}:
            rest = rest.replace(spelling, '<hidden>')
        return match.group(1) + "'<hidden>'" + rest
    message = _QUOTED_VALUE.sub(quoted, message)
    message = _CANT_OPEN.sub(cant_open, message)
    return _AMBIGUOUS.sub(lambda match: match.group(1) + _redact_word(match.group(2)) + match.group(3), message)


class RedactingArgumentParser(argparse.ArgumentParser):
    """The `titanos` parser (and, by inheritance, every subparser): error() redacts typed values (redact_parser_message)."""

    def error(self, message):
        super().error(redact_parser_message(message))


def _remote(p, *, url_flag='--url'):
    """Server mode options. The key comes from an environment variable, never from argv."""
    p.add_argument(url_flag, dest='url', metavar='URL', help='Send the command to a running server instead of editing local files')
    p.add_argument('--api-key-env', default='TITANOS_API_KEY', metavar='VAR',
                   help='Environment variable holding the server key for server mode (default TITANOS_API_KEY)')
    p.add_argument('--timeout', type=float, help='HTTP timeout in seconds for server mode')


def _sources(p, *, default_hint):
    group = p.add_mutually_exclusive_group()
    # --env and --file are accepted spellings because the vault's own refusal message names them
    group.add_argument('--from-env', '--env', metavar='VAR', help='Reference an environment variable on the server host; the value is never copied')
    group.add_argument('--from-file', '--file', metavar='PATH', help='Reference an owner-only file (absolute path) on the server host')
    group.add_argument('--from-stdin', action='store_true', help='Read the value from one line of standard input')
    p.set_defaults(secret_hint=default_hint)


def _key_flags(p, *, create):
    p.add_argument('--name', required=create)
    p.add_argument('--scope', dest='scopes', action='append', metavar='SCOPE',
                   help='Repeat for each scope (inference, cloud:openai, broker:read, ...); create without --scope mints the '
                        'SDK default key: ' + ' and '.join(DEFAULT_KEY_SCOPES) + ' (stock SDK models.list and inference)')
    p.add_argument('--models', metavar='GLOBS', help="Comma list of model globs, e.g. 'gpt-*@openai,qwen3-*'")
    p.add_argument('--upstreams', metavar='GLOBS', help='Comma list of upstream id globs')
    p.add_argument('--target-classes', metavar='CLASSES', help='Comma list of local, cloud, router')
    p.add_argument('--locations', metavar='LOCATIONS', help='Comma list of ' + ', '.join(LOCATIONS))
    p.add_argument('--rpm', type=int)
    p.add_argument('--tpm', type=int)
    p.add_argument('--concurrent', type=int)
    p.add_argument('--usd-per-day')
    p.add_argument('--usd-per-month')
    p.add_argument('--tokens-per-day', type=int)
    p.add_argument('--max-request-usd')
    p.add_argument('--expires', metavar='WHEN', help='30d, 12h or an ISO date/time such as 2026-12-31')
    p.add_argument('--default-location', choices=LOCATIONS)
    p.add_argument('--note')


def register_cli(sub) -> None:
    """Add the broker, keys, audit and hivemind-manifest subcommands to the main `titanos` parser."""
    broker = sub.add_parser('broker', help='Cloud broker: upstreams, credentials, aliases, policy, catalog and decisions')
    actions = broker.add_subparsers(dest='broker_action', required=True)

    for name, text in (('status', 'Broker summary'), ('profiles', 'Upstream profiles shipped with Titanos'),
                       ('reload', 'Re-read broker.json in a running server (offline: validate the file)')):
        _remote(actions.add_parser(name, help=text))

    up = actions.add_parser('upstream', help='Configure broker upstreams')
    ups = up.add_subparsers(dest='upstream_action', required=True)
    p = ups.add_parser('add', help='Create or replace an upstream')
    p.add_argument('id')
    p.add_argument('--profile', required=True)
    p.add_argument('--base-url')
    p.add_argument('--credential', metavar='ID', help='An existing credential id (never a secret)')
    models = p.add_mutually_exclusive_group()
    models.add_argument('--models', metavar='M1,M2')
    models.add_argument('--allow-any-model', action='store_true')
    p.add_argument('--surfaces', metavar='S1,S2')
    p.add_argument('--enable', action='store_true')
    p.add_argument('--anthropic-version')
    p.add_argument('--anthropic-beta', action='append', default=[], metavar='FLAG')
    p.add_argument('--azure-resource')
    p.add_argument('--azure-endpoint-host')
    p.add_argument('--azure-api-version')
    p.add_argument('--deployment', action='append', default=[], metavar='MODEL=DEPLOYMENT')
    p.add_argument('--header', action='append', default=[], metavar='NAME=VALUE', help='Non-secret static headers allowed by the profile')
    p.add_argument('--modalities', metavar='text,image,audio')
    p.add_argument('--allow-store', action='store_true')
    p.add_argument('--catalog', choices=['static', 'live'])
    p.add_argument('--auth', choices=['bearer', 'azure_api_key', 'none'])
    p.add_argument('--label')
    _remote(p)
    p = ups.add_parser('list')
    _remote(p)
    for name in ('show', 'enable', 'disable', 'remove', 'test', 'health', 'reset-breaker'):
        p = ups.add_parser(name)
        p.add_argument('id')
        _remote(p)

    cred = actions.add_parser('credential', help='Write-only provider credentials (values never printed)')
    creds = cred.add_subparsers(dest='credential_action', required=True)
    p = creds.add_parser('set', help='Store a credential bound to one upstream; prompts (hidden) by default')
    p.add_argument('id')
    p.add_argument('--upstream', required=True)
    _sources(p, default_hint='prompt')
    p.add_argument('--purpose', choices=['inference', 'advisor'], default='inference')
    p.add_argument('--no-validate', action='store_true', help='Skip the validation probe')
    _remote(p)
    p = creds.add_parser('list')
    _remote(p)
    for name in ('show', 'validate', 'remove'):
        p = creds.add_parser(name)
        p.add_argument('id')
        _remote(p)
    p = creds.add_parser('rotate', help='Replace a credential value or reference; prompts (hidden) by default')
    p.add_argument('id')
    _sources(p, default_hint='prompt')
    _remote(p)

    alias = actions.add_parser('alias', help='Published model aliases')
    aliases = alias.add_subparsers(dest='alias_action', required=True)
    p = aliases.add_parser('set')
    p.add_argument('name')
    p.add_argument('ref', help='<upstream model>@<upstream id>')
    p.add_argument('--capability', dest='capabilities', action='append', choices=CAPABILITIES)
    p.add_argument('--inject', action='append', default=[], metavar='KEY=VALUE')
    _remote(p)
    p = aliases.add_parser('remove')
    p.add_argument('name')
    _remote(p)
    _remote(aliases.add_parser('list'))

    virtual = actions.add_parser('virtual', help='Virtual models (ordered, weighted, cheapest, latency or advisor routing)')
    virtuals = virtual.add_subparsers(dest='virtual_action', required=True)
    p = virtuals.add_parser('set')
    p.add_argument('name')
    p.add_argument('--capability', required=True, choices=VIRTUAL_CAPABILITIES)
    p.add_argument('--strategy', default='ordered', choices=STRATEGIES)
    p.add_argument('--target', dest='targets', action='append', required=True, metavar='REF[:WEIGHT]')
    p.add_argument('--fallback-on', metavar='T1,T2')
    p.add_argument('--max-attempts', type=int)
    p.add_argument('--advisor')
    p.add_argument('--hidden', action='store_true')
    _remote(p)
    _remote(virtuals.add_parser('list'))
    p = virtuals.add_parser('remove')
    p.add_argument('name')
    _remote(p)

    policy = actions.add_parser('policy', help='Cloud policy and default placement')
    policies = policy.add_subparsers(dest='policy_action', required=True)
    _remote(policies.add_parser('show'))
    p = policies.add_parser('set')
    p.add_argument('--cloud-policy', choices=['off', 'explicit', 'fallback'])
    p.add_argument('--default-location', choices=LOCATIONS)
    p.add_argument('--anonymous-cloud', choices=['true', 'false'])
    p.add_argument('--allow-http-lan', choices=['true', 'false'])
    p.add_argument('--max-fallbacks', type=int)
    p.add_argument('--auto-model')
    _remote(p)

    advisor = actions.add_parser('advisor', help='Routing advisors')
    advisors = advisor.add_subparsers(dest='advisor_action', required=True)
    p = advisors.add_parser('set')
    p.add_argument('id')
    p.add_argument('--profile', required=True, choices=ADVISOR_PROFILES)
    p.add_argument('--url', dest='advisor_url', metavar='URL', help='The advisor endpoint (http only to loopback)')
    p.add_argument('--credential', metavar='ID')
    p.add_argument('--timeout-ms', type=int)
    p.add_argument('--share-excerpt', action='store_true')
    p.add_argument('--label', dest='labels', action='append', default=[], metavar='LABEL=REF')
    p.add_argument('--display-label')
    _remote(p, url_flag='--server')
    p = advisors.add_parser('test')
    p.add_argument('id')
    _remote(p)
    _remote(advisors.add_parser('list'))

    prices = actions.add_parser('prices', help='Operator price rows (none ship with Titanos)')
    price = prices.add_subparsers(dest='prices_action', required=True)
    p = price.add_parser('import')
    p.add_argument('file', type=Path)
    _remote(p)
    _remote(price.add_parser('list'))

    p = actions.add_parser('catalog', help='Cloud, router and virtual models a principal may use')
    p.add_argument('--refresh', action='store_true', help='Refresh stale live catalogs (server mode)')
    _remote(p)
    p = actions.add_parser('explain', help='Dry-run routing (zero egress)')
    p.add_argument('--surface', default='openai.chat')
    p.add_argument('--model', required=True)
    p.add_argument('--location', choices=LOCATIONS)
    p.add_argument('--target', metavar='REF')
    p.add_argument('--body', type=Path, help='A JSON request body; --model replaces its model field')
    p.add_argument('--key-id', help='Explain as this key (admin)')
    _remote(p)
    p = actions.add_parser('decisions', help='Recent routing decisions (metadata only)')
    p.add_argument('--limit', type=int, default=100)
    p.add_argument('--key-id')
    p.add_argument('--model')
    _remote(p)
    p = actions.add_parser('usage', help='Usage and cost rollups')
    p.add_argument('--group-by', default='key,upstream,model,day')
    p.add_argument('--since')
    p.add_argument('--until')
    p.add_argument('--key-id')
    _remote(p)
    p = actions.add_parser('connect', help='One step: add a provider upstream, store and validate its credential, '
                                           'enable it and list its models')
    p.add_argument('provider', help='A profile id (titanos broker profiles), e.g. openai, anthropic, gemini, elevenlabs')
    _sources(p, default_hint='prompt')
    p.add_argument('--no-credential', action='store_true', help='Connect a profile that needs no credential (auth none)')
    p.add_argument('--upstream', metavar='ID', help='Upstream id (default: the provider id)')
    p.add_argument('--credential', dest='credential_id', metavar='ID', help='Credential id (default: <upstream>-key)')
    p.add_argument('--models', metavar='M1,M2', help='Static models to list besides the live catalog')
    p.add_argument('--base-url', help='Base URL for profiles without a default (custom_openai, litellm, nim_custom ...)')
    p.add_argument('--no-enable', action='store_true', help='Store and validate, but leave the upstream disabled')
    p.add_argument('--no-validate', action='store_true', help='Skip the validation probe')
    p.add_argument('--aliases', action='store_true',
                   help="Publish an alias for every discovered model id that collides with nothing, so SDKs can send "
                        "the provider's own model names")
    p.add_argument('--anonymous-cloud', choices=['true', 'false'],
                   help='Set policy.anonymous_cloud (standalone role: keyless loopback callers may use the cloud; '
                        'default: true when no key is enforced on a loopback bind, else unchanged)')
    p.add_argument('--allow-store', action='store_true',
                   help="Set the upstream's data_policy.allow_store (provider-side files, batches, stored Responses)")
    _remote(p)
    p = actions.add_parser('quickstart', help='Find provider keys in the environment (names only) and connect them')
    p.add_argument('--yes', action='store_true', help='Connect every provider whose key variable is set (by env '
                                                     'reference; the value is never copied)')
    p.add_argument('--provider', action='append', default=[], metavar='ID', help='Limit to these providers')
    p.add_argument('--aliases', action='store_true', help='Publish aliases for the discovered models (see connect)')
    p.add_argument('--anonymous-cloud', choices=['true', 'false'], help='Also set policy.anonymous_cloud')
    _remote(p)
    p = actions.add_parser('live-smoke', help='Owner-approved live provider smoke (spends money; never in CI)')
    p.add_argument('--approval', type=Path, required=True)
    p.add_argument('--home', type=Path, required=True)
    p.add_argument('--live', action='store_true')
    p.add_argument('--i-understand-this-spends-money', dest='spends_money', action='store_true')
    p.add_argument('--providers')
    p.add_argument('--modalities')

    keys = sub.add_parser('keys', help='Inbound API keys (tk_ tokens): create, list, update, revoke, rotate')
    key_actions = keys.add_subparsers(dest='keys_action', required=True)
    p = key_actions.add_parser('create', help='Mint a key; the secret is shown once')
    _key_flags(p, create=True)
    p.add_argument('--output-file', type=Path, help='Write the secret to this file (owner-only) instead of printing it')
    _remote(p)
    _remote(key_actions.add_parser('list'))
    for name in ('show', 'revoke'):
        p = key_actions.add_parser(name)
        p.add_argument('id')
        _remote(p)
    p = key_actions.add_parser('update')
    p.add_argument('id')
    _key_flags(p, create=False)
    _remote(p)
    p = key_actions.add_parser('rotate', help='A new secret for the same key id (shown once)')
    p.add_argument('id')
    p.add_argument('--output-file', type=Path)
    _remote(p)
    _remote(key_actions.add_parser('whoami', help='The principal of the key in TITANOS_API_KEY (server mode)'))
    p = key_actions.add_parser('reset', help='Offline only: delete keys.json and return to keyless mode (typed confirmation)')
    p.add_argument('--keyless', action='store_true', required=True)

    audit = sub.add_parser('audit', help='Read or verify the hash-chained broker audit log')
    audit.add_argument('audit_action', nargs='?', choices=['list', 'verify'], default='list')
    audit.add_argument('--since', help='UTC date, e.g. 2026-10-01')
    audit.add_argument('--kind')
    audit.add_argument('--limit', type=int, default=500)
    _remote(audit)

    p = sub.add_parser('hivemind-manifest', help='Generate a HiveMind engine-manager manifest (managed role)')
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--job-type', required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--register-with-gateway', metavar='URL', help='Also register the job type with this gateway at start')
    for name in sorted(COMMANDS):
        _no_abbreviations(sub.choices[name])


def _no_abbreviations(parser) -> None:
    """Refuse abbreviated options on every parser of these commands: an abbreviation such as --api-key must never resolve
    to --api-key-env (which would put whatever follows it on the command line)."""
    parser.allow_abbrev = False
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for child in action.choices.values():
                _no_abbreviations(child)


# -- helpers -----------------------------------------------------------------------------------------------------------
def _print(value) -> None:
    print(json.dumps(value, indent=2, default=str))


def _csv(value):
    return [item.strip() for item in value.split(',') if item.strip()] if value else None


def _pairs(values, what):
    out = {}
    for item in values or []:
        name, sep, value = item.partition('=')
        if not sep or not name.strip():
            raise ValueError(f'{what} takes NAME=VALUE, got {item!r}')
        out[name.strip()] = value.strip()
    return out


def _json_value(text):
    try:
        return json.loads(text)
    except ValueError:
        return text


def _bool(text):
    return None if text is None else text == 'true'


def parse_target(text: str) -> dict:
    """A virtual-model --target: '<ref>' or '<ref>:<weight>'. A trailing ':<integer>' is a weight only when the rest is
    still a valid ref; 'gpt-oss:120b@ollama_cloud' therefore stays one ref (its ':120b' sits before the '@')."""
    from .naming import parse_ref
    head, sep, tail = text.rpartition(':')
    if sep and tail.isdigit() and head:
        try:
            parse_ref(head)
        except Exception:
            pass
        else:
            if '@' not in tail:
                return {'ref': head, 'weight': int(tail)}
    parse_ref(text)
    return {'ref': text}


def parse_expiry(text: str | None, *, now: datetime | None = None) -> str | None:
    if text is None:
        return None
    now = now or datetime.now(timezone.utc)
    found = re.fullmatch(r'(\d+)([dh])', text.strip())
    if found:
        amount = int(found.group(1))
        delta = timedelta(days=amount) if found.group(2) == 'd' else timedelta(hours=amount)
        return (now + delta).replace(microsecond=0).isoformat().replace('+00:00', 'Z')
    try:
        if re.fullmatch(r'\d{4}-\d{2}-\d{2}', text.strip()):
            day = date.fromisoformat(text.strip())
            return datetime(day.year, day.month, day.day, tzinfo=timezone.utc).isoformat().replace('+00:00', 'Z')
        moment = datetime.fromisoformat(text.strip().replace('Z', '+00:00'))
    except ValueError:
        raise ValueError('--expires takes 30d, 12h or an ISO date such as 2026-12-31') from None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')


def key_fields(args, *, create: bool) -> dict:
    """KeySpec/KeyPatch fields from the key flags (only the ones given, for a patch)."""
    out = {}
    if args.name is not None:
        out['name'] = args.name
    if args.scopes:
        out['scopes'] = args.scopes
    elif create:
        # the documented SDK default key: inference plus the model listing stock SDKs call first (models.list)
        out['scopes'] = list(DEFAULT_KEY_SCOPES)
    constraints = {k: v for k, v in {'models': _csv(args.models), 'upstreams': _csv(args.upstreams),
                                    'target_classes': _csv(args.target_classes),
                                    'locations': _csv(args.locations)}.items() if v}
    if constraints:
        out['constraints'] = constraints
    limits = {k: v for k, v in {'rpm': args.rpm, 'tpm': args.tpm, 'concurrent': args.concurrent}.items() if v is not None}
    if limits:
        out['limits'] = limits
    budget = {k: v for k, v in {'usd_per_day': args.usd_per_day, 'usd_per_month': args.usd_per_month,
                               'tokens_per_day': args.tokens_per_day, 'max_request_usd': args.max_request_usd}.items()
              if v is not None}
    if budget:
        out['budget'] = budget
    if args.expires is not None:
        out['expires_at'] = parse_expiry(args.expires)
    if args.default_location is not None:
        out['default_location'] = args.default_location
    if args.note is not None:
        out['note'] = args.note
    return out


def read_source(args, prompt: str) -> tuple[str, str]:
    """(kind, value) of a credential source: ('env', VAR), ('file', PATH) or ('secret', value). The value is read from a
    hidden prompt or one line of stdin and is never echoed or kept beyond the call."""
    if args.from_env:
        return 'env', args.from_env
    if args.from_file:
        return 'file', args.from_file
    if args.from_stdin:
        value = sys.stdin.readline().rstrip('\r\n')
    else:
        value = getpass.getpass(prompt)
    value = value.strip()
    if not value:
        raise ValueError('No credential value was entered; nothing changed')
    return 'secret', value


def _client(args):
    from ..client import Client
    return Client(args.url, api_key=os.environ.get(args.api_key_env or 'TITANOS_API_KEY', ''), timeout=args.timeout)


def write_secret_file(path: Path, token: str) -> None:
    from .keys import write_private
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    write_private(path, (token + '\n').encode('utf-8'))


def _secret_out(token: str, output_file, value: dict) -> dict:
    """The key answer with the secret shown once: printed in the JSON, or written to an owner-only file."""
    if output_file is not None:
        write_secret_file(output_file, token)
        return {**value, 'secret_file': str(output_file)}
    return {**value, 'secret': token}


class Offline:
    """The TITANOS_HOME files a running server reads, through a Broker facade built for this command (lazy)."""

    def __init__(self, registry_path):
        from ..settings import load_settings
        self.settings = load_settings()
        self.registry_path = Path(registry_path) if registry_path is not None else None
        self._broker = None

    def local_config(self):
        from ..config import Config, read_config
        if self.registry_path is not None and self.registry_path.exists():
            return read_config(self.registry_path)
        return Config()

    @property
    def broker(self):
        if self._broker is None:
            from .facade import Broker
            self._broker = Broker(settings=self.settings, config=self.local_config,
                                  bootstrap_key=os.environ.get('TITANOS_API_KEY') or None)
        return self._broker

    @property
    def config_path(self) -> Path:
        return self.broker.config_path

    def config(self):
        from .config import read_broker_config
        return read_broker_config(self.config_path)

    def apply_hint(self) -> str:
        """How a running server applies a broker.json change. Once the server enforces keys (keys.json holds any record,
        or serve was started with TITANOS_API_KEY) POST /api/broker/reload needs broker:admin, which the inference keys of
        the quick start lack; when keys.json holds no active key with it, the hint names the mint (or a restart)."""
        try:
            store = self.broker.keys
            keyed = store.has_records()
            admin = any(r.active() and {'admin', 'broker:admin'} & set(r.scopes) for r in store.list())
        except Exception:  # an unreadable keys.json: the generic hint stays true
            keyed, admin = True, True
        if not keyed:
            return (f'A running server applies it after {RELOAD_COMMAND} or a restart (if serve was started with '
                    'TITANOS_API_KEY, run the reload with that key in TITANOS_API_KEY)')
        if admin:
            return (f'A running server applies it after a restart, or after {RELOAD_COMMAND} with a key holding '
                    'broker:admin in TITANOS_API_KEY (keys.json holds keys, so the server enforces them)')
        return ('A running server applies it after a restart. keys.json holds keys, so the server enforces them, and '
                f'none of the active ones holds broker:admin: to reload instead, mint one ({ADMIN_MINT}), put it in '
                f'TITANOS_API_KEY and run {RELOAD_COMMAND}')

    def write(self, kind: str, subject: str, change, detail: dict | None = None):
        """Apply change to broker.json under the lock with the HTTP admin write's checks; audited before the replace."""
        from .config import BrokerConfig, check_name_conflicts, check_no_secrets, update_broker_config
        from .catalog import local_models_from
        broker = self.broker
        old = self.config()

        def checked(cfg):
            change(cfg)
            candidate = BrokerConfig.model_validate(cfg.model_dump(mode='json'))
            check_name_conflicts(candidate, local_models_from(self.local_config()).keys())
            if self.settings.role == 'managed' and not broker.switches.delegation and broker._enables_egress(old, candidate):
                raise ValueError('This change would enable egress in managed role without delegation (role_forbids)')
            check_no_secrets(candidate.model_dump(mode='json'))
            broker.audit.event(kind, ACTOR, subject, detail or {})
        updated = update_broker_config(self.config_path, checked)
        print(RELOAD_NOTE.format(path=self.config_path, apply=self.apply_hint()), file=sys.stderr)
        return updated

    def principal(self, key_id=None):
        from .scopes import BOOTSTRAP_SCOPES
        from .types import Principal
        if key_id:
            return self.broker.keys.to_principal(self.broker.keys.get(key_id))
        return Principal(id='cli', kind='bootstrap', scopes=BOOTSTRAP_SCOPES, name='cli')

    def run(self, coroutine):
        """Run one async broker operation and close the egress pools of its loop."""
        async def wrapped():
            try:
                return await coroutine
            finally:
                egress = self.broker._components.get('egress')
                if egress is not None:
                    await egress.aclose()
        return asyncio.run(wrapped())

    async def probe(self, upstream_id: str, purpose: str, credential=None):
        from .errors import BrokerError
        broker = self.broker
        snap = broker.snapshot()
        adapter = snap.adapters.get(upstream_id)
        if adapter is None:
            raise BrokerError(f'Upstream {upstream_id} has no adapter (unknown upstream or slot profile)',
                              'upstream_unconfigured')
        deadline = time.monotonic() + float(self.settings.broker_connect_timeout) + float(
            self.settings.broker_first_byte_timeout)
        grant = snap.issuer.issue(purpose, upstream_id=upstream_id, decision_id=uuid.uuid4().hex,
                                  request_id=uuid.uuid4().hex, deadline=deadline, credential=credential)
        return await adapter.probe(grant, broker.egress)

    def origin_for(self, upstream_id: str, purpose: str) -> str:
        from .config import parse_origin
        cfg = self.config()
        if purpose == 'advisor':
            advisor = cfg.advisors.get(upstream_id)
            if advisor is None or not advisor.url:
                raise ValueError(f'Advisor {upstream_id} has no url to bind a credential to')
            return parse_origin(advisor.url, allow_http_lan=cfg.policy.allow_http_lan)
        upstream = cfg.upstreams.get(upstream_id)
        if upstream is None:
            raise ValueError(f'Upstream {upstream_id} does not exist; add it first (titanos broker upstream add)')
        origin = upstream.origin(allow_http_lan=cfg.policy.allow_http_lan)
        if origin is None:
            raise ValueError(f'Upstream {upstream_id} has no base_url')
        return origin

    def validate(self, cred_id: str, upstream_id: str, *, staged: bool = False) -> tuple[bool, str | None]:
        cfg = self.config()
        upstream = cfg.upstreams[upstream_id]
        origin = self.origin_for(upstream_id, 'inference')
        handle = self.broker.vault.validation_handle(cred_id, origin=origin, template=upstream.effective_auth(),
                                                     staged=staged)
        result = self.run(self.probe(upstream_id, 'validate', credential=handle))
        return bool(result.ok), (None if result.ok else (result.code or f'status_{result.status}'))


def _meta(record) -> dict:
    return record.to_dict()


# -- commands ----------------------------------------------------------------------------------------------------------
def run_cli(args, registry_path) -> int:
    """Run one broker/keys/audit/hivemind-manifest command. Broker refusals become ValueError (`titanos: <message>`)."""
    from ..engine import EngineError
    try:
        if args.command == 'hivemind-manifest':
            return _hivemind(args, registry_path)
        if args.command == 'keys':
            return _keys(args, registry_path)
        if args.command == 'audit':
            return _audit(args, registry_path)
        return _broker(args, registry_path)
    except EngineError as exc:
        code = getattr(exc, 'code', None)
        raise ValueError(f'{code}: {exc}' if code else str(exc)) from None
    except RuntimeError as exc:  # AuditChainError: nothing was changed
        raise ValueError(f'audit_unavailable: {exc}') from None
    except Exception as exc:
        if type(exc).__name__ == 'Timeout':  # filelock: another writer holds the file
            raise ValueError('A broker file is locked by another writer; retry shortly') from None
        raise


def _hivemind(args, registry_path) -> int:
    import shutil
    from ..config import config_path
    executable = str(Path(sys.executable).parent / ('titanos.exe' if os.name == 'nt' else 'titanos'))
    if not Path(executable).exists():
        executable = shutil.which('titanos') or executable
    manifest = hivemind_manifest(registry_path if registry_path is not None else config_path(None), executable,
                                 args.port, args.job_type, register_with_gateway=args.register_with_gateway)
    text = json.dumps(manifest, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding='utf-8')
        print(args.output)
    else:
        print(text, end='')
    return 0


def placeholders_in(text: str) -> list[str]:
    """HiveMind's placeholder extraction (manifest.rs placeholders_in): {token} with [A-Za-z0-9_]+."""
    out, i = [], 0
    while i < len(text):
        if text[i] == '{':
            end = text.find('}', i + 1)
            if end >= 0:
                token = text[i + 1:end]
                if token and all(c.isascii() and (c.isalnum() or c == '_') for c in token):
                    out.append(token)
                i = end + 1
                continue
        i += 1
    return out


def hivemind_manifest(path, executable, port, job_type, *, register_with_gateway=None) -> dict:
    """A HiveMind engine-manager manifest for a managed Titanos. Only the {bind}, {port}, {bin} and {cli} placeholders are
    used (HiveMind's loader rejects unknown tokens such as {host}); serve runs with --role managed on an explicit,
    non-reserved port; health and readiness are GET /v1/health/ready; identity is GET /api/info name Titanos."""
    from ..hardware import discover
    from .role import RESERVED_PORTS
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError('HiveMind Titanos port must be an integer in 1..65535')
    if port in RESERVED_PORTS:
        raise ValueError('HiveMind Titanos port must avoid reserved Ollama, LM Studio and vLLM ports '
                         f'({", ".join(map(str, sorted(RESERVED_PORTS)))})')
    if not isinstance(job_type, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,63}', job_type):
        raise ValueError('--job-type must be 1-64 characters [A-Za-z0-9._-]')
    args = ['--config', str(path), 'serve', '--role', 'managed', '--host', '{bind}', '--port', '{port}']
    if register_with_gateway:
        gateway_url(register_with_gateway)
        args += ['--register-with-gateway', register_with_gateway, '--job-type', job_type]
    host = discover()
    platform_key = host['os'] + '/' + {'x86_64': 'amd64', 'aarch64': 'arm64'}.get(host['arch'], host['arch'])
    ready = 'http://127.0.0.1:{port}/v1/health/ready'
    manifest = {
        'engine': 'titanos', 'display_name': 'Titanos', 'manifest_version': 1,
        'integration': {'job_type': job_type, 'enabled_by_default': False,
                        'identity': {'path': '/api/info', 'name': 'Titanos'}},
        'runtime': {'mode': 'process', 'port': port, 'bind': '127.0.0.1', 'args': args,
                    'ready': {'http': ready, 'status': 200, 'timeout_s': 60},
                    'health': {'http': ready, 'status': 200, 'interval_s': 10},
                    'stop': {'signal': 'term', 'grace_s': 10}},
        'platforms': {platform_key: {'detect': [executable], 'runtime': {'bin': executable, 'cli': executable}}},
        'actions': {
            'list_models': {'http': {'method': 'GET', 'path': '/v1/models'}},
            'loaded_models': {'http': {'method': 'GET', 'path': '/api/models/loaded'}},
            'server_info': {'http': {'method': 'GET', 'path': '/api/info'}},
            'chat': {'http': {'method': 'POST', 'path': '/v1/chat/completions',
                              'body_schema': {'model': 'string', 'messages': 'array'}}},
        },
    }
    strings = [*args, executable, ready, ready]
    unknown = sorted({t for s in strings for t in placeholders_in(s)} - HIVEMIND_PLACEHOLDERS)
    if unknown:
        raise ValueError('HiveMind manifest would contain unknown placeholders: ' + ', '.join(unknown))
    return manifest


def gateway_url(url: str) -> str:
    parts = urlsplit(url)
    if parts.scheme not in ('http', 'https') or not parts.hostname or parts.username or parts.password \
            or parts.query or parts.fragment:
        raise ValueError('--register-with-gateway takes an http(s) gateway URL without credentials, query or fragment')
    return url.rstrip('/')


def gateway_url_arg(url: str) -> str:
    """argparse type for serve --register-with-gateway: an unusable URL is refused at parse time, before the server
    starts, and the message never repeats the URL (it may carry credentials)."""
    try:
        return gateway_url(url)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


# -- broker ------------------------------------------------------------------------------------------------------------
def _broker(args, registry_path) -> int:
    action = args.broker_action
    if action == 'live-smoke':
        raise ValueError('broker live-smoke is run by titanos.cli (live_smoke_command)')
    remote = getattr(args, 'url', None)
    if remote:
        with _client(args) as client:
            _print(_broker_remote(client, args))
        return 0
    _print(_broker_offline(Offline(registry_path), args))
    return 0


def _upstream_spec(args, current: dict | None = None) -> dict:
    spec = dict(current or {})
    spec['profile'] = args.profile
    for name, value in (('base_url', args.base_url), ('credential', args.credential),
                        ('anthropic_version', args.anthropic_version), ('catalog', args.catalog), ('auth', args.auth),
                        ('label', args.label)):
        if value is not None:
            spec[name] = value
    if args.models is not None:
        spec['models'] = _csv(args.models) or []
    if args.allow_any_model:
        spec['allow_any_model'] = True
    if args.surfaces is not None:
        spec['surfaces'] = _csv(args.surfaces)
    if args.enable:
        spec['enabled'] = True
    if args.anthropic_beta:
        spec['anthropic_betas'] = list(args.anthropic_beta)
    if args.azure_resource or args.azure_api_version or args.deployment or args.azure_endpoint_host:
        azure = dict(spec.get('azure') or {})
        for name, value in (('resource', args.azure_resource), ('endpoint_host', args.azure_endpoint_host),
                            ('api_version', args.azure_api_version)):
            if value is not None:
                azure[name] = value
        if args.deployment:
            azure['deployments'] = {**azure.get('deployments', {}), **_pairs(args.deployment, '--deployment')}
        spec['azure'] = azure
    if args.header:
        spec['headers'] = {**spec.get('headers', {}), **_pairs(args.header, '--header')}
    if args.modalities is not None or args.allow_store:
        policy = dict(spec.get('data_policy') or {})
        if args.modalities is not None:
            policy['modalities'] = _csv(args.modalities)
        if args.allow_store:
            policy['allow_store'] = True
        spec['data_policy'] = policy
    return spec


def _virtual_spec(args) -> dict:
    spec = {'capability': args.capability, 'strategy': args.strategy, 'targets': [parse_target(t) for t in args.targets],
            'expose': not args.hidden}
    if args.fallback_on is not None:
        spec['fallback_on'] = _csv(args.fallback_on) or []
    if args.max_attempts is not None:
        spec['max_attempts'] = args.max_attempts
    if args.advisor is not None:
        spec['advisor'] = args.advisor
    return spec


def _advisor_spec(args) -> dict:
    spec = {'profile': args.profile}
    for name, value in (('url', args.advisor_url), ('credential', args.credential), ('timeout_ms', args.timeout_ms),
                        ('label', args.display_label)):
        if value is not None:
            spec[name] = value
    if args.share_excerpt:
        spec['share_excerpt'] = True
    if args.labels:
        spec['label_map'] = _pairs(args.labels, '--label')
    return spec


def _policy_fields(args) -> dict:
    return {k: v for k, v in {'cloud_policy': args.cloud_policy, 'default_location': args.default_location,
                              'anonymous_cloud': _bool(args.anonymous_cloud), 'allow_http_lan': _bool(args.allow_http_lan),
                              'max_fallbacks': args.max_fallbacks, 'auto_model': args.auto_model}.items() if v is not None}


def _explain_input(args):
    body = {}
    if args.body is not None:
        body = json.loads(args.body.read_text(encoding='utf-8-sig'))
        if not isinstance(body, dict):
            raise ValueError('--body must contain a JSON object')
    from .surfaces import SURFACES
    surface = SURFACES.get(args.surface)
    if surface is None:
        raise ValueError(f'Unknown surface {args.surface!r}')
    field = surface.model_from.split(':', 1)[1] if surface.model_from.startswith('json:') else 'model'
    body[field] = args.model
    headers = {k: v for k, v in {'X-Titanos-Location': args.location, 'X-Titanos-Target': args.target}.items() if v}
    return body, headers


def _connect_source(args) -> dict:
    """The credential source keywords of `broker connect` (never echoed): {} for --no-credential."""
    if args.no_credential:
        if args.from_env or args.from_file or args.from_stdin:
            raise ValueError('--no-credential takes no credential source')
        return {}
    kind, value = read_source(args, f'API key for {args.provider} (hidden): ')
    return {kind: value}


def _quickstart_plan(rows, args) -> list[dict]:
    """The providers quickstart would connect: every listed row with a set variable (first set name), limited to
    --provider when given."""
    wanted = set(args.provider or ())
    plan = []
    for row in rows:
        if wanted and row['provider'] not in wanted:
            continue
        name = next((v['name'] for v in row['variables'] if v['set']), None)
        if name is not None:
            plan.append({'provider': row['provider'], 'env': name,
                         'command': f"titanos broker connect {row['provider']} --from-env {name}"})
    return plan


def _broker_remote(client, args):
    action = args.broker_action
    if action == 'connect':
        source = _connect_source(args)
        try:
            return client.broker_connect(args.provider, upstream=args.upstream, credential_id=args.credential_id,
                                         models=_csv(args.models) if args.models else None, base_url=args.base_url,
                                         enable=not args.no_enable, validate=not args.no_validate,
                                         aliases=args.aliases, anonymous_cloud=_bool(args.anonymous_cloud),
                                         allow_store=True if args.allow_store else None, **source)
        finally:
            source.clear()
    if action == 'quickstart':
        rows = client.broker_env_candidates().get('data', [])
        plan = _quickstart_plan(rows, args)
        if not args.yes:
            return {'object': 'quickstart', 'candidates': rows, 'plan': plan,
                    'note': 'Variable names only (the server process environment). Re-run with --yes to connect them.'}
        results = []
        for item in plan:
            try:
                results.append({'provider': item['provider'], 'ok': True, 'result': client.broker_connect(
                    item['provider'], env=item['env'], aliases=args.aliases,
                    anonymous_cloud=_bool(args.anonymous_cloud))})
            except Exception as exc:  # one provider's refusal never stops the others
                results.append({'provider': item['provider'], 'ok': False, 'error': str(exc)[:300]})
        return {'object': 'quickstart', 'connected': results}
    if action == 'status':
        return client.broker_status()
    if action == 'profiles':
        return client.broker_profiles()
    if action == 'reload':
        return client.broker_reload()
    if action == 'upstream':
        sub = args.upstream_action
        if sub == 'add':
            current = None
            try:
                current = {k: v for k, v in client.upstream(args.id).items()
                           if k not in {'id', 'ready', 'readiness', 'credential_validation', 'catalog_status'}}
            except Exception:
                current = None
            return client.put_upstream(args.id, **_upstream_spec(args, current))
        if sub == 'list':
            return client.upstreams()
        if sub == 'show':
            return client.upstream(args.id)
        if sub in ('enable', 'disable'):
            current = {k: v for k, v in client.upstream(args.id).items()
                       if k not in {'id', 'ready', 'readiness', 'credential_validation', 'catalog_status'}}
            current['enabled'] = sub == 'enable'
            return client.put_upstream(args.id, **current)
        if sub == 'remove':
            return client.delete_upstream(args.id)
        if sub == 'test':
            return client.test_upstream(args.id)
        if sub == 'health':
            return client.upstream_health(args.id)
        return client.reset_breaker(args.id)
    if action == 'credential':
        sub = args.credential_action
        if sub == 'set':
            kind, value = read_source(args, f'Credential value for {args.id} (hidden): ')
            try:
                return client.set_credential(args.id, upstream=args.upstream, purpose=args.purpose,
                                             validate=not args.no_validate, **{kind: value})
            finally:
                del value
        if sub == 'list':
            return client.credentials()
        if sub == 'show':
            rows = client.credentials().get('data', [])
            found = next((row for row in rows if row.get('id') == args.id), None)
            if found is None:
                raise ValueError(f'Credential {args.id} does not exist')
            return found
        if sub == 'validate':
            return client.validate_credential(args.id)
        if sub == 'rotate':
            kind, value = read_source(args, f'New credential value for {args.id} (hidden): ')
            try:
                return client.rotate_credential(args.id, **{kind: value})
            finally:
                del value
        return client.delete_credential(args.id)
    if action == 'alias':
        sub = args.alias_action
        if sub == 'set':
            inject = {k: _json_value(v) for k, v in _pairs(args.inject, '--inject').items()}
            return client.put_alias(args.name, args.ref, capabilities=tuple(args.capabilities or ()), inject=inject or None)
        if sub == 'remove':
            return client.delete_alias(args.name)
        return client.aliases()
    if action == 'virtual':
        sub = args.virtual_action
        if sub == 'set':
            return client.put_virtual_model(args.name, **_virtual_spec(args))
        if sub == 'remove':
            return client.delete_virtual_model(args.name)
        return client.virtual_models()
    if action == 'policy':
        return client.policy() if args.policy_action == 'show' else client.set_policy(**_policy_fields(args))
    if action == 'advisor':
        sub = args.advisor_action
        if sub == 'set':
            return client.put_advisor(args.id, **_advisor_spec(args))
        if sub == 'test':
            return client.test_advisor(args.id)
        return client.advisors()
    if action == 'prices':
        if args.prices_action == 'import':
            return client.put_prices(_price_rows(args.file))
        return client.prices()
    if action == 'catalog':
        return client.broker_catalog(refresh=args.refresh)
    if action == 'explain':
        body, headers = _explain_input(args)
        return client.broker_explain(args.surface, body, headers, key_id=args.key_id)
    if action == 'decisions':
        return client.broker_decisions(limit=args.limit, key_id=args.key_id, model=args.model)
    if action == 'usage':
        return client.broker_usage(group_by=args.group_by, since=args.since, until=args.until, key_id=args.key_id)
    raise ValueError(f'Unknown broker command {action}')


def _price_rows(path: Path) -> list:
    rows = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if isinstance(rows, dict):
        rows = rows.get('prices', rows.get('data'))
    if not isinstance(rows, list):
        raise ValueError('The prices file must hold a JSON array of price rows (or {"prices": [...]})')
    return rows


def _status_offline(off: Offline) -> dict:
    broker = off.broker
    cfg = off.config()
    try:
        vault = broker.vault.summary()
    except Exception as exc:
        vault = {'backend': 'unavailable', 'error': type(exc).__name__}
    try:
        report = broker.audit.verify().to_dict()
    except Exception as exc:
        report = {'ok': False, 'reason': type(exc).__name__}
    keys = broker.keys.summary()
    return {'mode': 'offline', 'config': str(off.config_path), 'role': off.settings.role,
            'cloud_policy': {'file': cfg.policy.cloud_policy, 'ceiling': off.settings.broker_cloud_ceiling},
            'egress': off.settings.broker_egress, 'vault': vault,
            'upstreams': {'total': len(cfg.upstreams), 'enabled': len(cfg.enabled_upstreams())},
            'aliases': sorted(cfg.models), 'virtual_models': sorted(cfg.virtual_models), 'advisors': sorted(cfg.advisors),
            'keys': {'records': keys.has_records, 'active': keys.active, 'active_admin': keys.active_admin},
            'audit': report}


def _connect_offline(off: Offline, provider: str, source: dict, args) -> dict:
    """`broker connect` without --url: the same steps as POST /api/broker/connect (broker/connect.py) on the TITANOS_HOME
    files, as the operator on the host (env: and file: sources and anonymous_cloud allowed)."""
    from .connect import ConnectRequest, connect
    from .errors import BrokerError
    credential = None
    if source:
        (kind, value), = source.items()
        credential = {('value' if kind == 'secret' else kind): value}
    request = ConnectRequest.model_validate({
        'provider': provider, 'credential': credential, 'upstream': getattr(args, 'upstream', None),
        'credential_id': getattr(args, 'credential_id', None),
        'models': _csv(args.models) if getattr(args, 'models', None) else None,
        'base_url': getattr(args, 'base_url', None), 'enable': not getattr(args, 'no_enable', False),
        'validate': not getattr(args, 'no_validate', False), 'aliases': bool(getattr(args, 'aliases', False)),
        'anonymous_cloud': _bool(getattr(args, 'anonymous_cloud', None)),
        'allow_store': True if getattr(args, 'allow_store', False) else None})
    broker = off.broker

    def write(kind, subject, change, detail):
        updated = off.write(kind, subject, change, detail)
        broker.reload(actor=ACTOR)
        return updated

    def audit(kind, subject, detail):
        broker.audit.event(kind, ACTOR, subject, detail)
    try:
        auth = getattr(broker, 'authenticator', None)
        keyless_owner = auth is not None and not auth.enforced and bool(getattr(auth, 'bind_loopback', False))
        result = off.run(connect(broker, request, write=write, audit=audit, host_sources=True,
                                 may_set_anonymous_cloud=True, keyless_owner=keyless_owner,
                                 local_names=lambda: broker.local_models().keys()))
    except BrokerError as exc:
        raise ValueError(f'{exc.code}: {exc}') from None
    if isinstance(result, dict) and isinstance(result.get('next_steps'), list):
        # this CLI validated from ITS environment and wrote the TITANOS_HOME files; the server is another process
        if credential and 'env' in credential:
            result['next_steps'].append(
                f"The server reads {credential['env']} from its own environment: start titanos serve where it is set "
                '(a server already running without it cannot use this upstream until restarted with it)')
        result['next_steps'].append(off.apply_hint())
    return result


def _broker_offline(off: Offline, args):
    from .config import AdvisorSpec, BrokerPolicy, PriceRow, PublishedModel, Upstream, VirtualModel
    from .errors import BrokerError
    action = args.broker_action
    if action == 'connect':
        source = _connect_source(args)
        try:
            return _connect_offline(off, args.provider, source, args)
        finally:
            source.clear()
    if action == 'quickstart':
        from .connect import env_candidates
        rows = env_candidates()
        plan = _quickstart_plan(rows, args)
        if not args.yes:
            return {'object': 'quickstart', 'candidates': rows, 'plan': plan,
                    'note': 'Variable names only, from this shell; the server reads the same names from its own '
                            'environment. Re-run with --yes to connect them.'}
        results = []
        for item in plan:
            try:
                results.append({'provider': item['provider'], 'ok': True,
                                'result': _connect_offline(off, item['provider'], {'env': item['env']}, args)})
            except ValueError as exc:
                results.append({'provider': item['provider'], 'ok': False, 'error': str(exc)[:300]})
        return {'object': 'quickstart', 'connected': results}
    if action == 'status':
        return _status_offline(off)
    if action == 'profiles':
        from .profiles import PROFILES
        return {'object': 'list', 'data': [{'id': p.id, 'kind': p.kind, 'adapter': p.adapter, 'auth': p.auth,
                                            'default_base_url': p.default_base_url, 'surfaces': sorted(p.surfaces),
                                            'static_headers': list(p.static_headers)}
                                           for p in sorted(PROFILES.values(), key=lambda p: p.id)]}
    if action == 'reload':
        cfg = off.config()
        return {'status': 'valid', 'config': str(off.config_path), 'upstreams': len(cfg.upstreams),
                'note': 'Offline mode validated broker.json. ' + off.apply_hint() + '.'}
    if action == 'upstream':
        sub = args.upstream_action
        cfg = off.config()
        if sub == 'list':
            return {'object': 'list', 'data': [{'id': uid, **u.model_dump(mode='json')} for uid, u in sorted(cfg.upstreams.items())]}
        if sub in ('show', 'enable', 'disable', 'remove', 'test') and args.id not in cfg.upstreams:
            raise ValueError(f'Upstream {args.id} does not exist')
        if sub == 'show':
            return {'id': args.id, **cfg.upstreams[args.id].model_dump(mode='json')}
        if sub == 'add':
            current = cfg.upstreams[args.id].model_dump(mode='json', exclude_unset=True) if args.id in cfg.upstreams else None
            upstream = Upstream.model_validate(_upstream_spec(args, current))
            if upstream.credential:
                record = off.broker.vault.record(upstream.credential)
                if record.upstream_id != args.id:
                    raise ValueError(f'Credential {upstream.credential} is bound to another upstream')
            previous = cfg.upstreams.get(args.id)

            def change(c):
                c.upstreams[args.id] = upstream
            updated = off.write('upstream.write', args.id, change, {'profile': upstream.profile, 'enabled': upstream.enabled})
            new_origin = updated.upstreams[args.id].origin(allow_http_lan=updated.policy.allow_http_lan)
            old_origin = previous.origin(allow_http_lan=updated.policy.allow_http_lan) if previous is not None else None
            reset = []
            if previous is None or old_origin != new_origin:
                try:
                    reset = off.broker.vault.invalidate_for_upstream(args.id, origin=new_origin)
                except BrokerError:
                    reset = []
            return {'id': args.id, **updated.upstreams[args.id].model_dump(mode='json'), 'credentials_reset': reset}
        if sub in ('enable', 'disable'):
            def change(c):
                c.upstreams[args.id].enabled = sub == 'enable'
            updated = off.write('upstream.write', args.id, change, {'enabled': sub == 'enable'})
            return {'id': args.id, **updated.upstreams[args.id].model_dump(mode='json')}
        if sub == 'remove':
            def change(c):
                c.upstreams.pop(args.id, None)
            off.write('upstream.write', args.id, change, {'deleted': True})
            return {'id': args.id, 'deleted': True}
        if sub == 'test':
            if off.settings.role == 'managed':
                raise ValueError('role_forbids: upstream tests are not available in managed role')
            started = time.perf_counter()
            result = off.run(off.probe(args.id, 'test'))
            return {'ok': bool(result.ok), 'status': result.status, 'latency_ms': result.latency_ms
                    if result.latency_ms is not None else round((time.perf_counter() - started) * 1000, 3), 'code': result.code}
        raise ValueError(f'`broker upstream {sub}` reads breaker state, which lives in the running server: add --url URL')
    if action == 'credential':
        return _credential_offline(off, args)
    if action == 'alias':
        sub = args.alias_action
        if sub == 'list':
            return {'object': 'list', 'data': [{'name': n, **a.model_dump(mode='json')} for n, a in sorted(off.config().models.items())]}
        if sub == 'remove':
            if args.name not in off.config().models:
                raise ValueError(f'Published model {args.name} does not exist')

            def change(c):
                c.models.pop(args.name, None)
            off.write('alias.write', args.name, change, {'deleted': True})
            return {'name': args.name, 'deleted': True}
        inject = {k: _json_value(v) for k, v in _pairs(args.inject, '--inject').items()}
        alias = PublishedModel.model_validate({'target': args.ref, **({'capabilities': args.capabilities} if args.capabilities else {}),
                                              **({'inject': inject} if inject else {})})

        def change(c):
            c.models[args.name] = alias
        off.write('alias.write', args.name, change, {'target': alias.target})
        return {'name': args.name, **alias.model_dump(mode='json')}
    if action == 'virtual':
        sub = args.virtual_action
        if sub == 'list':
            return {'object': 'list', 'data': [{'name': n, **v.model_dump(mode='json')} for n, v in sorted(off.config().virtual_models.items())]}
        if sub == 'remove':
            if args.name not in off.config().virtual_models:
                raise ValueError(f'Virtual model {args.name} does not exist')

            def change(c):
                c.virtual_models.pop(args.name, None)
            off.write('virtual.write', args.name, change, {'deleted': True})
            return {'name': args.name, 'deleted': True}
        virtual = VirtualModel.model_validate(_virtual_spec(args))

        def change(c):
            c.virtual_models[args.name] = virtual
        off.write('virtual.write', args.name, change, {'capability': virtual.capability, 'strategy': virtual.strategy})
        return {'name': args.name, **virtual.model_dump(mode='json')}
    if action == 'policy':
        cfg = off.config()
        if args.policy_action == 'show':
            return {**cfg.policy.model_dump(mode='json'), 'ceiling': off.settings.broker_cloud_ceiling,
                    'role': off.settings.role}
        policy = BrokerPolicy.model_validate({**cfg.policy.model_dump(mode='json'), **_policy_fields(args)})
        order = ('off', 'explicit', 'fallback')
        if order.index(policy.cloud_policy) > order.index(off.settings.broker_cloud_ceiling):
            raise ValueError(f'cloud_policy {policy.cloud_policy} exceeds TITANOS_BROKER_CLOUD_CEILING='
                             f'{off.settings.broker_cloud_ceiling}')

        def change(c):
            c.policy = policy
        off.write('policy.write', 'policy', change, {'cloud_policy': policy.cloud_policy,
                                                     'default_location': policy.default_location})
        return policy.model_dump(mode='json')
    if action == 'advisor':
        sub = args.advisor_action
        if sub == 'list':
            return {'object': 'list', 'data': [{'id': a, **s.model_dump(mode='json')} for a, s in sorted(off.config().advisors.items())]}
        if sub == 'test':
            return _advisor_test_offline(off, args.id)
        advisor = AdvisorSpec.model_validate(_advisor_spec(args))

        def change(c):
            c.advisors[args.id] = advisor
        off.write('advisor.write', args.id, change, {'profile': advisor.profile})
        return {'id': args.id, **advisor.model_dump(mode='json')}
    if action == 'prices':
        if args.prices_action == 'list':
            return {'object': 'list', 'data': [row.model_dump(mode='json') for row in off.config().prices]}
        rows = [PriceRow.model_validate(row) for row in _price_rows(args.file)]

        def change(c):
            c.prices = rows
        off.write('prices.write', 'prices', change, {'rows': len(rows)})
        return {'object': 'list', 'data': [row.model_dump(mode='json') for row in rows]}
    if action == 'catalog':
        if args.refresh:
            raise ValueError('Live catalogs refresh in the running server: add --url URL (offline listing makes no requests)')
        snap = off.broker.snapshot()
        principal = off.principal()
        return {'object': 'list', 'data': snap.catalog.entries(principal), 'upstream_status': snap.catalog.status()}
    if action == 'explain':
        body, headers = _explain_input(args)
        return off.run(off.broker.explain(args.surface, body, headers, off.principal(args.key_id), advisor=False))
    if action == 'decisions':
        return {'object': 'list', 'data': _decisions_offline(off, args)}
    if action == 'usage':
        groups = [g.strip() for g in args.group_by.split(',') if g.strip()]
        return {'object': 'list', 'rows': off.broker.usage.rollup(groups, since=args.since, until=args.until,
                                                                  key_id=args.key_id)}
    raise ValueError(f'Unknown broker command {action}')


def _decisions_offline(off: Offline, args) -> list:
    directory = off.broker.directory / 'decisions'
    if not directory.is_dir():
        raise ValueError('Decisions are kept in the running server (TITANOS_BROKER_DECISION_LOG_FILE is off): add --url URL')
    rows = []
    for path in sorted(directory.glob('*.jsonl'), reverse=True):
        for line in reversed(path.read_text(encoding='utf-8').splitlines()):
            try:
                item = json.loads(line)
            except ValueError:
                continue
            if args.key_id and item.get('key_id') != args.key_id:
                continue
            served = (item.get('served') or {}).get('ref') if isinstance(item.get('served'), dict) else None
            if args.model and item.get('requested_model') != args.model and served != args.model:
                continue
            rows.append(item)
            if len(rows) >= args.limit:
                return rows
    return rows


def _advisor_test_offline(off: Offline, advisor_id: str) -> dict:
    if off.settings.role == 'managed':
        raise ValueError('role_forbids: advisor tests are not available in managed role')
    spec = off.config().advisors.get(advisor_id)
    if spec is None:
        raise ValueError(f'Advisor {advisor_id} does not exist')
    if spec.profile == 'pending-contract':
        raise ValueError('advisor_profile_unconfigured: the pending-contract advisor slot has no wire contract')
    advisor = off.broker.snapshot().advisors.get(advisor_id)
    if advisor is None or advisor.callable() is not None:
        raise ValueError(f'advisor_profile_unconfigured: advisor {advisor_id} cannot be called')
    from .advisor import AdvisorRequest, AdvisorSummary
    from .errors import BrokerError
    request = AdvisorRequest(request_id=uuid.uuid4().hex, surface='openai.chat', capability='chat',
                             requested_model='auto', summary=AdvisorSummary(), candidates=[])
    started = time.perf_counter()

    async def call():
        return await advisor.advise(request, deadline=time.monotonic() + advisor.timeout_seconds())
    try:
        answer = off.run(call())
        ok, reason = answer is not None, None if answer is not None else 'no_answer'
    except BrokerError as exc:
        ok, reason = False, exc.code
    except Exception as exc:
        ok, reason = False, type(exc).__name__
    return {'id': advisor_id, 'ok': ok, 'ms': round((time.perf_counter() - started) * 1000.0, 3), 'reason': reason}


def _credential_offline(off: Offline, args):
    from .errors import BrokerError
    vault = off.broker.vault
    audit = off.broker.audit
    sub = args.credential_action
    if sub == 'list':
        return {'object': 'list', 'data': [_meta(r) for r in vault.records()]}
    if sub == 'show':
        return _meta(vault.record(args.id))
    if off.settings.role == 'managed':
        raise ValueError('role_forbids: credential writes and validation are not available in managed role')
    if sub == 'remove':
        vault.record(args.id)
        audit.event('credential.delete', ACTOR, args.id, {})
        vault.delete(args.id)
        print(RELOAD_NOTE.format(path=vault.index_path, apply=off.apply_hint()), file=sys.stderr)
        return {'id': args.id, 'deleted': True}
    if sub == 'validate':
        record = vault.record(args.id)
        if record.purpose != 'inference':
            raise ValueError('Only inference credentials are validated against an upstream')
        origin = off.origin_for(record.upstream_id, 'inference')
        ok, code = off.validate(args.id, record.upstream_id)
        audit.event('credential.validate', ACTOR, args.id, {'ok': ok, 'code': code})
        record = vault.mark_validated(args.id, ok, code, None, origin=origin if ok else None)
        if not ok:
            raise ValueError(f'credential_validation_failed: the upstream refused the credential ({code})')
        return _meta(record)
    if sub == 'set':
        origin = off.origin_for(args.upstream, args.purpose)
        kind, value = read_source(args, f'Credential value for {args.id} (hidden): ')
        kw = {'upstream_id': args.upstream, 'origin': origin, 'purpose': args.purpose}
        audit.event('credential.put', ACTOR, args.id, {'upstream': args.upstream, 'source': 'vault' if kind == 'secret' else kind,
                                                       'purpose': args.purpose})
        try:
            if kind == 'secret':
                vault.put_secret(args.id, value, **kw)
            elif kind == 'env':
                vault.put_env_ref(args.id, value, **kw)
            else:
                vault.put_file_ref(args.id, value, **kw)
        except BrokerError as exc:
            try:
                audit.event('credential.put', ACTOR, args.id, {'ok': False, 'code': exc.code})
            except Exception:
                pass
            raise
        finally:
            del value
        if args.purpose == 'inference' and not args.no_validate:
            ok, code = off.validate(args.id, args.upstream)
            audit.event('credential.validate', ACTOR, args.id, {'ok': ok, 'code': code})
            record = vault.mark_validated(args.id, ok, code, None, origin=origin if ok else None)
            if not ok:
                raise ValueError(f'credential_validation_failed: the upstream refused the credential ({code}); the record '
                                 f'is kept as {record.validation} (titanos broker credential rotate/remove)')
        else:
            record = vault.mark_validated(args.id, True, 'skipped', None)
        print(RELOAD_NOTE.format(path=vault.index_path, apply=off.apply_hint()), file=sys.stderr)
        return _meta(record)
    # rotate
    current = vault.record(args.id)
    kind, value = read_source(args, f'New credential value for {args.id} (hidden): ')
    try:
        if current.source == 'vault' and kind == 'secret' and current.purpose == 'inference':
            vault.stage_rotation(args.id, value)
            ok, code = off.validate(args.id, current.upstream_id, staged=True)
            if not ok:
                vault.discard_rotation(args.id)
                audit.event('credential.rotate', ACTOR, args.id, {'ok': False, 'code': code})
                raise ValueError(f'credential_validation_failed: the upstream refused the new credential ({code}); '
                                 'the current one keeps serving')
            try:
                audit.event('credential.rotate', ACTOR, args.id, {'ok': True})
            except BaseException:
                vault.discard_rotation(args.id)
                raise
            return _meta(vault.commit_rotation(args.id))
        kw = {'upstream_id': current.upstream_id, 'origin': current.origin, 'purpose': current.purpose}
        audit.event('credential.rotate', ACTOR, args.id, {'source': 'vault' if kind == 'secret' else kind})
        if kind == 'secret':
            vault.put_secret(args.id, value, **kw)
        elif kind == 'env':
            vault.put_env_ref(args.id, value, **kw)
        else:
            vault.put_file_ref(args.id, value, **kw)
    finally:
        del value
    if current.purpose == 'inference':
        ok, code = off.validate(args.id, current.upstream_id)
        audit.event('credential.validate', ACTOR, args.id, {'ok': ok, 'code': code})
    else:
        ok, code = True, 'skipped'
    record = vault.mark_validated(args.id, ok, code, None)
    if not ok:
        raise ValueError(f'credential_validation_failed: the upstream refused the new credential ({code})')
    return _meta(record)


def live_smoke_command(args) -> list[str]:
    """The argv of tools/broker_live_smoke.py for `titanos broker live-smoke` (titanos.cli runs it): refuses without both
    flags and an unexpired approval file. The broker package itself never starts processes (source scan)."""
    if not (args.live and args.spends_money):
        raise ValueError('live-smoke spends money: pass both --live and --i-understand-this-spends-money')
    try:
        approval = json.loads(Path(args.approval).read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ValueError(f'The approval file could not be read: {type(exc).__name__}') from None
    expires = approval.get('expires_at') if isinstance(approval, dict) else None
    try:
        moment = datetime.fromisoformat(str(expires).replace('Z', '+00:00'))
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
    except ValueError:
        raise ValueError('The approval file needs an ISO expires_at') from None
    if moment <= datetime.now(timezone.utc):
        raise ValueError('The approval file has expired; ask the owner for a new approval')
    tool = Path(__file__).resolve().parents[3] / 'tools' / 'broker_live_smoke.py'
    if not tool.is_file():
        raise ValueError('tools/broker_live_smoke.py is not part of this installation (run it from a source checkout)')
    command = [sys.executable, '-B', str(tool), '--live', '--i-understand-this-spends-money',
               '--approval', str(args.approval), '--home', str(args.home)]
    if args.providers:
        command += ['--providers', args.providers]
    if args.modalities:
        command += ['--modalities', args.modalities]
    return command


# -- keys and audit ----------------------------------------------------------------------------------------------------
def _keys(args, registry_path) -> int:
    action = args.keys_action
    if action == 'reset':
        if getattr(args, 'url', None):
            raise ValueError('keys reset --keyless runs only offline')
        return _keys_reset(Offline(registry_path))
    if getattr(args, 'url', None):
        with _client(args) as client:
            if action == 'create':
                created = client.create_key(**key_fields(args, create=True))
                _print(_secret_out(created.secret, args.output_file, created.record))
            elif action == 'list':
                _print(client.keys())
            elif action == 'show':
                _print(client.key(args.id))
            elif action == 'update':
                _print(client.update_key(args.id, **key_fields(args, create=False)))
            elif action == 'revoke':
                _print(client.revoke_key(args.id))
            elif action == 'rotate':
                created = client.rotate_key(args.id)
                _print(_secret_out(created.secret, args.output_file, created.record))
            else:
                _print(client.whoami())
        return 0
    from .keys import KeyPatch, KeySpec
    off = Offline(registry_path)
    store = off.broker.keys
    if action == 'create':
        record, token = store.mint(KeySpec.model_validate(key_fields(args, create=True)), created_by=ACTOR)
        _print(_secret_out(token, args.output_file, record.public()))
    elif action == 'list':
        _print({'object': 'list', 'data': [r.public() for r in store.list()]})
    elif action == 'show':
        _print(store.get(args.id).public())
    elif action == 'update':
        _print(store.update(args.id, KeyPatch.model_validate(key_fields(args, create=False)), actor=ACTOR).public())
    elif action == 'revoke':
        _print(store.revoke(args.id, actor=ACTOR).public())
    elif action == 'rotate':
        record, token = store.rotate(args.id, actor=ACTOR)
        _print(_secret_out(token, args.output_file, {'id': record.id, 'name': record.name}))
    else:
        raise ValueError('keys whoami reports the principal of a key a server verified: add --url URL')
    if action in ('create', 'update', 'revoke', 'rotate'):
        print('Saved keys.json; a running server picks it up on its next check.', file=sys.stderr)
    return 0


def _keys_reset(off: Offline) -> int:
    from .keys import KeyStore
    broker = off.broker
    store = KeyStore(broker.directory / 'keys.json', off.settings, broker.audit, offline=True,
                     bootstrap_key_present=bool(os.environ.get('TITANOS_API_KEY')))
    print(f'This deletes {store.path} and returns Titanos to keyless loopback mode. Every key stops working.',
          file=sys.stderr)
    print(f'Type {RESET_CONFIRMATION} to continue: ', end='', file=sys.stderr, flush=True)
    answer = sys.stdin.readline().strip()
    if answer != RESET_CONFIRMATION:
        raise ValueError('Confirmation did not match; nothing changed')
    removed = store.reset_keyless(actor=ACTOR)
    _print({'reset': 'keyless', 'removed': removed})
    return 0


def _audit(args, registry_path) -> int:
    if getattr(args, 'url', None):
        with _client(args) as client:
            report = client.audit_verify() if args.audit_action == 'verify' else \
                client.audit(since=args.since, kind=args.kind, limit=args.limit)
    else:
        log = Offline(registry_path).broker.audit
        report = log.verify(since=args.since).to_dict() if args.audit_action == 'verify' else \
            {'object': 'list', 'data': log.query(since=args.since, kind=args.kind, limit=args.limit)}
    _print(report)
    if args.audit_action == 'verify' and not report.get('ok', False):
        if 'unparseable' in str(report.get('reason', '')):
            print('A torn last line (an interrupted append) is repaired by truncating the partial last line of the '
                  'newest audit day file named above; any other failure means the chain was altered.', file=sys.stderr)
        return 1
    return 0


# -- serve -------------------------------------------------------------------------------------------------------------
def serve_preconditions(settings, *, bootstrap_key, explicit_port: bool) -> list[str]:
    """role.check_serve_preconditions with this host's key table, vault posture and audit state. Raises ValueError naming
    the refused switch; returns notes for stderr. Reads files only: no directory is created outside broker role."""
    from .config import broker_dir
    from .keys import KeyStore
    from .role import check_serve_preconditions
    keystore = KeyStore(broker_dir() / 'keys.json', settings, bootstrap_key_present=bool(bootstrap_key)).summary()
    vault_backend, audit_ok = '', True
    if settings.role == 'broker':
        from ..config import home
        from .audit import AuditLog
        from .vault.store import open_vault
        vault_backend = open_vault(settings, home()).posture()
        try:
            log = AuditLog(broker_dir() / 'audit', settings.broker_audit_retention_days)
            audit_ok = bool(log.writable()) and bool(log.verify().ok)
        except Exception:
            audit_ok = False
    return check_serve_preconditions(settings, bootstrap_key=bootstrap_key, keystore=keystore,
                                     explicit_port=explicit_port, vault_backend=vault_backend, audit_ok=audit_ok)


__all__ = ['COMMANDS', 'gateway_url', 'hivemind_manifest', 'live_smoke_command', 'parse_target', 'placeholders_in',
           'register_cli', 'run_cli', 'serve_preconditions']
