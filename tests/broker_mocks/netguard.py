"""Socket-level network guard for broker tests: any non-loopback destination raises before a packet leaves the process.

install() patches, for the whole process:
- socket.socket.connect and socket.socket.connect_ex (every sync client, socket.create_connection, the selector event
  loop's sock_connect; a connected UDP socket too);
- socket.socket.sendto for AF_INET/AF_INET6 destinations (unconnected UDP, including the selector loop's datagram
  transport);
- socket.getaddrinfo, socket.gethostbyname, socket.gethostbyname_ex and socket.gethostbyaddr (each refuses any name other
  than 'localhost' and any non-loopback IP literal BEFORE resolving, so these resolver calls send no DNS query);
- on Windows, asyncio.windows_events.IocpProactor.connect: the Proactor event loop connects through ConnectEx, which bypasses
  socket.connect, and asyncio connects to IP literals without calling getaddrinfo.
Allowed destinations: 127.0.0.0/8, ::1, IPv4-mapped loopback, 'localhost' and AF_UNIX paths. 0.0.0.0, link-local,
metadata, RFC1918, multicast and every public address are refused.

Not covered: the Proactor loop's unconnected datagram path (IocpProactor.sendto through WSASendTo), raw sendmsg, and
native extensions that resolve or connect without Python's socket module. The broker uses none of them; the guard is a
test instrument for Python-level egress, not a sandbox.

Every refusal raises NetguardBlocked (a PermissionError, so httpx and httpcore surface it as a ConnectError) or, from
getaddrinfo, NetguardLookupBlocked (a socket.gaierror). Each is appended to ATTEMPTS ({host, port, via}) and, when
TITANOS_TEST_NETGUARD_REPORT names a file, written there as one JSON line (the subprocess counter the e2e test reads).

netguard_site/sitecustomize.py installs the same guard at interpreter start when TITANOS_TEST_NETGUARD=1 and the
netguard_site directory is on PYTHONPATH.
"""
from __future__ import annotations

import ipaddress
import json
import os
import socket
import sys
import threading

REPORT_ENV = 'TITANOS_TEST_NETGUARD_REPORT'
MARK = '__titanos_netguard__'
ALLOWED_NAMES = frozenset({'localhost'})
RESOLVERS = ('gethostbyname', 'gethostbyname_ex', 'gethostbyaddr')
ATTEMPTS: list[dict] = []
_lock = threading.Lock()
_originals: dict = {}


class NetguardBlocked(PermissionError):
    """A connect to a non-loopback address was refused by the test netguard."""


class NetguardLookupBlocked(socket.gaierror):
    """A name lookup for a non-loopback host was refused by the test netguard (no DNS query was sent by that call)."""


def _host_text(host) -> str:
    if isinstance(host, bytes):
        host = host.decode('idna', 'replace')
    return str(host).strip().strip('[]').split('%', 1)[0]


def is_loopback_host(host) -> bool:
    """True for loopback IP literals (IPv4-mapped included) and 'localhost'."""
    text = _host_text(host)
    if text.lower().rstrip('.') in ALLOWED_NAMES:
        return True
    try:
        address = ipaddress.ip_address(text)
    except ValueError:
        return False
    mapped = getattr(address, 'ipv4_mapped', None)
    return bool((mapped or address).is_loopback)


def _record(host, port, via: str) -> None:
    entry = {'host': _host_text(host), 'port': port, 'via': via, 'pid': os.getpid()}
    with _lock:
        ATTEMPTS.append(entry)
    path = os.environ.get(REPORT_ENV)
    if path:
        try:
            with open(path, 'a', encoding='utf-8') as handle:
                handle.write(json.dumps(entry) + '\n')
        except OSError:
            pass


def _check_address(address, via: str) -> None:
    if isinstance(address, (str, bytes)):  # AF_UNIX path
        return
    if not isinstance(address, tuple) or not address:
        return
    host, port = address[0], (address[1] if len(address) > 1 else None)
    if not is_loopback_host(host):
        _record(host, port, via)
        raise NetguardBlocked(f'netguard: connection to {_host_text(host)}:{port} refused (non-loopback, via {via})')


def blocked_count() -> int:
    with _lock:
        return len(ATTEMPTS)


def attempts() -> list[dict]:
    with _lock:
        return list(ATTEMPTS)


def reset_attempts() -> None:
    with _lock:
        ATTEMPTS.clear()


def installed() -> bool:
    return bool(_originals)


def install() -> None:
    """Patch every connect path (idempotent)."""
    if _originals:
        return
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_getaddrinfo = socket.getaddrinfo

    def connect(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6):
            _check_address(address, 'socket.connect')
        return original_connect(self, address)

    def connect_ex(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6):
            _check_address(address, 'socket.connect_ex')
        return original_connect_ex(self, address)

    def getaddrinfo(host, port, *args, **kwargs):
        if host is not None and not is_loopback_host(host):
            _record(host, port, 'getaddrinfo')
            raise NetguardLookupBlocked(socket.EAI_NONAME, f'netguard: lookup of {_host_text(host)} refused (non-loopback)')
        return original_getaddrinfo(host, port, *args, **kwargs)

    def resolver(name, original):
        def guarded(host):
            if not is_loopback_host(host):
                _record(host, None, name)
                raise NetguardLookupBlocked(socket.EAI_NONAME,
                                            f'netguard: lookup of {_host_text(host)} refused (non-loopback)')
            return original(host)
        guarded.__name__ = name
        return guarded

    original_sendto = socket.socket.sendto

    def sendto(self, data, *args):
        if self.family in (socket.AF_INET, socket.AF_INET6) and args:
            _check_address(args[-1], 'socket.sendto')
        return original_sendto(self, data, *args)

    resolvers = {name: resolver(name, getattr(socket, name)) for name in RESOLVERS}
    for fn in (connect, connect_ex, getaddrinfo, sendto, *resolvers.values()):
        setattr(fn, MARK, True)
    _originals['connect'] = original_connect
    _originals['connect_ex'] = original_connect_ex
    _originals['getaddrinfo'] = original_getaddrinfo
    _originals['sendto'] = original_sendto
    for name in RESOLVERS:
        _originals[name] = getattr(socket, name)
    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    socket.socket.sendto = sendto
    socket.getaddrinfo = getaddrinfo
    for name, fn in resolvers.items():
        setattr(socket, name, fn)
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


def uninstall() -> None:
    """Restore the original functions (tests normally keep the guard for the whole process)."""
    if not _originals:
        return
    socket.socket.connect = _originals.pop('connect')
    socket.socket.connect_ex = _originals.pop('connect_ex')
    socket.getaddrinfo = _originals.pop('getaddrinfo')
    socket.socket.sendto = _originals.pop('sendto')
    for name in RESOLVERS:
        setattr(socket, name, _originals.pop(name))
    if 'iocp' in _originals:
        from asyncio import windows_events
        windows_events.IocpProactor.connect = _originals.pop('iocp')


__all__ = ['ATTEMPTS', 'NetguardBlocked', 'NetguardLookupBlocked', 'REPORT_ENV', 'attempts', 'blocked_count', 'install',
           'installed', 'is_loopback_host', 'reset_attempts', 'uninstall']
