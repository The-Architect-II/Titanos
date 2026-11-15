"""`titanos service install|start|stop|uninstall|status`: run `titanos serve` as a host service (docs/OPERATIONS.md).

`install` refuses before registering anything when the registry is missing (`titanos init` first) or the executable cannot
run titanos (`<executable> --version` fails), and after `install` and `start` it waits (--wait, default DEFAULT_WAIT
seconds) for GET /healthz of the server the environment file describes: exit 0 only when it answered, otherwise exit 3
with the reason and the log tail, so a service that is registered but not serving is never reported as installed.
`stop` stops the server (Windows: `schtasks /End` ends the launcher .cmd only, so the server is stopped by the id it
recorded in TITANOS_HOME/service/<name>.pid; Linux: systemctl stop; macOS: launchctl bootout), `start` starts it again,
and on Windows `status` reads the pid file (a live process = running) rather than the launcher task's state alone.

One plan per platform (plan(): pure, no file or process touched, so every platform's output is checked on every CI leg):
- Windows: a Scheduled Task (schtasks.exe) that starts a small launcher .cmd at logon. The task is registered from a task
  XML bound to the installing account (logon trigger for that account, interactive token, least privilege, no run-time
  limit): `schtasks /SC ONLOGON` without a task owner is a machine-wide trigger that a non-elevated shell is refused
  ("Access is denied"), the XML form is what a standard user may register. `--system`: at boot as SYSTEM, which needs
  an elevated prompt. A Windows Service proper needs a program that speaks the Service Control Manager protocol; a
  console program cannot be registered with sc.exe from user space without such a wrapper, so the task is the
  documented form and `status` says so.
- Linux: a systemd unit; user instance at $XDG_CONFIG_HOME/systemd/user/<name>.service (boot-time start without a login
  needs `loginctl enable-linger`), `--system` at /etc/systemd/system/<name>.service.
- macOS: a launchd property list; ~/Library/LaunchAgents/com.titanos.<name>.plist, `--system` /Library/LaunchDaemons.
Every unit sets TITANOS_HOME explicitly and reads the rest of its environment from TITANOS_HOME/service/<name>.env,
written at install from the current environment's TITANOS_* variables (plus --host/--port/--role) and owner-only on
POSIX (0600). TITANOS_API_KEY therefore lives in that file only: never in the unit, the task definition, the launcher or a
command line. Logs go to TITANOS_HOME/logs/<name>.log on Windows and macOS and to the journal on Linux.
"""
from __future__ import annotations

import getpass
import json
import os
import platform
import plistlib
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath, PureWindowsPath
from xml.sax.saxutils import escape as xml_escape

from .config import home as titanos_home

SUPPORTED = ('Windows', 'Linux', 'Darwin')
NAME_PATTERN = re.compile(r'[A-Za-z][A-Za-z0-9._-]{0,63}')
# exit codes of `service status`: 0 running, 3 registered but not running, 4 not registered (1 is a command failure);
# `install` and `start` exit 3 too when the server did not answer GET /healthz within --wait seconds
STATUS_RUNNING, STATUS_STOPPED, STATUS_ABSENT = 0, 3, 4
DEFAULT_WAIT = 30.0  # seconds install/start wait for the server to answer; 0 skips the probe
WAIT_POLL, WAIT_STATUS_EVERY = 0.25, 2.0  # readiness poll interval; how often the service manager is asked whether the unit still runs


def default_executable() -> str:
    """The titanos console script beside this interpreter, else the first on PATH; ValueError when there is none."""
    candidate = Path(sys.executable).parent / ('titanos.exe' if os.name == 'nt' else 'titanos')
    if candidate.is_file():
        return str(candidate)
    found = shutil.which('titanos')
    if found:
        return found
    raise ValueError('no titanos executable was found beside this interpreter or on PATH; pass --executable PATH')


def service_environment(environ, *, host=None, port=None, role=None) -> dict:
    """The TITANOS_* variables the unit's environment file carries: every TITANOS_* of environ except TITANOS_HOME (the
    unit sets it explicitly), then the install flags. A value with a line break cannot be written and is refused."""
    env = {k: v for k, v in sorted(environ.items()) if k.startswith('TITANOS_') and k != 'TITANOS_HOME'}
    for key, value in (('TITANOS_HOST', host), ('TITANOS_PORT', port), ('TITANOS_ROLE', role)):
        if value is not None:
            env[key] = str(value)
    for key, value in env.items():
        if '\n' in value or '\r' in value:
            raise ValueError(f'{key} contains a line break and cannot be written to the service environment file')
    return env


def render_env(env: dict, *, quoted: bool) -> str:
    """KEY=VALUE lines. quoted (POSIX): single-quoted so `set -a; . file` and systemd EnvironmentFile read spaces and
    shell characters literally; unquoted (Windows): the launcher's `for /f` loop takes the text after the first '='."""
    lines = ['# Titanos service environment (titanos service install); TITANOS_HOME is set by the unit itself']
    for key, value in env.items():
        lines.append(f'{key}={_sh_quote(value) if quoted else value}')
    return '\n'.join(lines) + '\n'


def windows_account(environ) -> str:
    """DOMAIN\\user of the installing account (the owner of the per-user Scheduled Task)."""
    user = environ.get('USERNAME') or getpass.getuser()
    domain = environ.get('USERDOMAIN')
    return f'{domain}\\{user}' if domain else user


def render_task_xml(name: str, launcher: str, account: str, env_file: str) -> str:
    """The Task Scheduler definition `schtasks /Create /XML` registers for a standard user: a logon trigger for the
    installing account, run with that account's interactive token at least privilege, no execution time limit (the
    schtasks default would end the server after 72 hours) and one instance at a time. Written as UTF-16 with a byte
    order mark (plan()['file_encodings']): schtasks refuses a UTF-8 file ('unable to switch the encoding')."""
    return '\r\n'.join([
        '<?xml version="1.0" encoding="UTF-16"?>',
        '<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">',
        '  <RegistrationInfo>',
        f'    <Description>Titanos inference server ({xml_escape(name)}); written by "titanos service install", environment in {xml_escape(env_file)}</Description>',
        '  </RegistrationInfo>',
        '  <Triggers>',
        f'    <LogonTrigger><Enabled>true</Enabled><UserId>{xml_escape(account)}</UserId></LogonTrigger>',
        '  </Triggers>',
        '  <Principals>',
        f'    <Principal id="Author"><UserId>{xml_escape(account)}</UserId><LogonType>InteractiveToken</LogonType><RunLevel>LeastPrivilege</RunLevel></Principal>',
        '  </Principals>',
        '  <Settings>',
        '    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>',
        '    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>',
        '    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>',
        '    <StartWhenAvailable>true</StartWhenAvailable>',
        '    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>',
        '    <Enabled>true</Enabled>',
        '  </Settings>',
        '  <Actions Context="Author">',
        f'    <Exec><Command>{xml_escape(launcher)}</Command></Exec>',
        '  </Actions>',
        '</Task>',
    ]) + '\r\n'


def plan(system: str, *, name: str, home, executable: str, config, env: dict, system_wide: bool = False,
         uid: int | None = None, user_home=None, xdg_config_home=None, account: str | None = None) -> dict:
    """The files to write and the commands to run for one platform (pure). home/config: the TITANOS_HOME directory and
    the registry path; env: service_environment(); uid/user_home/xdg_config_home/account default to the current account."""
    if system not in SUPPORTED:
        raise ValueError(f'service units are written for Windows, Linux and macOS only; this host reports {system!r}')
    if not NAME_PATTERN.fullmatch(name or ''):
        raise ValueError('--name must be 1-64 characters [A-Za-z0-9._-] starting with a letter')
    if not executable:
        raise ValueError('the service needs the path of the titanos executable (--executable)')
    pure = PureWindowsPath if system == 'Windows' else PurePosixPath  # the target platform's spelling, not this host's
    home = str(home)
    config = str(config)
    service_dir = str(pure(home) / 'service')
    log_dir = str(pure(home) / 'logs')
    user_home = str(user_home if user_home is not None else Path.home())
    out = {'platform': system, 'name': name, 'home': home, 'system_wide': system_wide, 'executable': executable,
           'config': config, 'files': {}, 'install': [], 'start': [], 'stop': [], 'uninstall': [], 'status': [],
           'tolerated': [], 'remove_on_uninstall': [], 'notes': []}
    if system == 'Windows':
        env_file = str(PureWindowsPath(service_dir) / f'{name}.env')
        launcher = str(PureWindowsPath(service_dir) / f'{name}.cmd')
        pid_file = str(PureWindowsPath(service_dir) / f'{name}.pid')
        log = str(PureWindowsPath(log_dir) / f'{name}.log')
        out['files'][env_file] = render_env(env, quoted=False)
        # the server records its own process id: `schtasks /End` ends the launcher only, uninstall stops the server by that id
        out['files'][launcher] = '\r\n'.join([
            '@echo off',
            f'rem Titanos service launcher written by "titanos service install" (task {name}); edit {env_file} for the environment',
            f'set "TITANOS_HOME={home}"',
            f'if exist "{env_file}" for /f "usebackq eol=# tokens=1,* delims==" %%A in ("{env_file}") do set "%%A=%%B"',
            f'"{executable}" --config "{config}" serve --pid-file "{pid_file}" >> "{log}" 2>&1',
        ]) + '\r\n'
        out['pid_file'] = pid_file
        if system_wide:
            create = ['schtasks', '/Create', '/F', '/TN', name, '/TR', f'"{launcher}"', '/RL', 'LIMITED', '/SC', 'ONSTART', '/RU', 'SYSTEM']
        else:
            # a per-user logon task: schtasks /SC ONLOGON without a task owner is refused from a non-elevated shell
            task_xml = str(PureWindowsPath(service_dir) / f'{name}.xml')
            out['files'][task_xml] = render_task_xml(name, launcher, account or windows_account(os.environ), env_file)
            out['file_encodings'] = {task_xml: 'utf-16'}  # BOM + UTF-16, the encoding schtasks /XML reads
            out['task_xml'] = task_xml
            create = ['schtasks', '/Create', '/F', '/TN', name, '/XML', task_xml]
        out['start'] = [['schtasks', '/Run', '/TN', name]]
        # /End ends the launcher .cmd only: stop_server() then stops the server by the id in its pid file
        out['stop'] = [['schtasks', '/End', '/TN', name]]
        out['install'] = [create, *out['start']]
        out['uninstall'] = [['schtasks', '/Delete', '/F', '/TN', name]]
        out['status'] = [['schtasks', '/Query', '/TN', name, '/FO', 'LIST', '/V']]
        out['unit'] = f'Scheduled Task {name}'
        out['launcher'] = launcher
        out['log'] = log
        out['logs_command'] = f'Get-Content -Wait "{log}"'
        out['notes'] += ['A Scheduled Task, not a Windows Service: a console program cannot register with the Service '
                         'Control Manager without a wrapper (docs/OPERATIONS.md).',
                         'The task starts when this account logs on (a per-user task XML, registered without elevation); '
                         '--system (ONSTART as SYSTEM) needs an elevated prompt and a TITANOS_HOME the SYSTEM account can '
                         'read; a DPAPI vault sealed by this account is not readable by SYSTEM.']
        if not system_wide:
            out['notes'].append(f'{env_file} is protected only by the profile ACL; keep TITANOS_HOME inside the profile.')
    else:
        env_file = str(PurePosixPath(service_dir) / f'{name}.env')
        log = str(PurePosixPath(log_dir) / f'{name}.log')
        out['files'][env_file] = render_env(env, quoted=True)
        out['env_file_mode'] = 0o600
        if system == 'Linux':
            if system_wide:
                unit = f'/etc/systemd/system/{name}.service'
                ctl = ['systemctl']
                wanted = 'multi-user.target'
            else:
                base = xdg_config_home or os.environ.get('XDG_CONFIG_HOME') or str(PurePosixPath(user_home) / '.config')
                unit = str(PurePosixPath(base) / 'systemd' / 'user' / f'{name}.service')
                ctl = ['systemctl', '--user']
                wanted = 'default.target'
            out['files'][unit] = '\n'.join([
                '[Unit]',
                f'Description=Titanos inference server ({home})',
                'After=network-online.target',
                'Wants=network-online.target',
                '',
                '[Service]',
                'Type=simple',
                f'Environment=TITANOS_HOME={home}',
                f'EnvironmentFile=-{env_file}',
                f'ExecStart={_sd_quote(executable)} --config {_sd_quote(config)} serve',
                'Restart=on-failure',
                'RestartSec=5',
                'TimeoutStopSec=30',
                '',
                '[Install]',
                f'WantedBy={wanted}',
            ]) + '\n'
            out['install'] = [ctl + ['daemon-reload'], ctl + ['enable', '--now', name]]
            out['start'] = [ctl + ['start', name]]
            out['stop'] = [ctl + ['stop', name]]
            out['uninstall'] = [ctl + ['disable', '--now', name]]
            out['remove_on_uninstall'] = [unit]
            out['after_uninstall'] = [ctl + ['daemon-reload']]
            out['status'] = [ctl + ['is-active', name]]
            out['unit'] = unit
            out['logs_command'] = ' '.join(['journalctl'] + ctl[1:] + ['-u', name, '-f'])
            if not system_wide:
                out['notes'].append('A user unit starts at login; `loginctl enable-linger` keeps it running at boot and '
                                    'after logout.')
        else:
            label = f'com.titanos.{name}'
            if system_wide:
                unit = f'/Library/LaunchDaemons/{label}.plist'
                domain = 'system'
            else:
                unit = str(PurePosixPath(user_home) / 'Library' / 'LaunchAgents' / f'{label}.plist')
                if uid is None:
                    uid = os.getuid() if hasattr(os, 'getuid') else 501
                domain = f'gui/{uid}'
            script = (f"set -a; [ -f {_sh_quote(env_file)} ] && . {_sh_quote(env_file)}; set +a; "
                      f"exec {_sh_quote(executable)} --config {_sh_quote(config)} serve")
            document = {'Label': label, 'ProgramArguments': ['/bin/sh', '-c', script],
                        'EnvironmentVariables': {'TITANOS_HOME': home}, 'RunAtLoad': True,
                        'KeepAlive': {'SuccessfulExit': False}, 'StandardOutPath': log, 'StandardErrorPath': log}
            out['files'][unit] = plistlib.dumps(document, sort_keys=True).decode('utf-8')
            out['install'] = [['launchctl', 'bootstrap', domain, unit], ['launchctl', 'kickstart', '-k', f'{domain}/{label}']]
            # start after stop (bootout) loads the plist again; after install it is already loaded and bootstrap fails
            # harmlessly (tolerated) before kickstart restarts the job
            out['start'] = list(out['install'])
            out['tolerated'] = [['launchctl', 'bootstrap', domain, unit]]
            out['stop'] = [['launchctl', 'bootout', f'{domain}/{label}']]
            out['uninstall'] = [['launchctl', 'bootout', f'{domain}/{label}']]
            out['remove_on_uninstall'] = [unit]
            out['status'] = [['launchctl', 'print', f'{domain}/{label}']]
            out['unit'] = unit
            out['label'] = label
            out['log'] = log
            out['logs_command'] = f'tail -f {_sh_quote(log)}'
    out['env_file'] = env_file
    out['log_dir'] = log_dir
    return out


def _sd_quote(text: str) -> str:
    return '"' + text.replace('\\', '\\\\').replace('"', '\\"') + '"' if re.search(r'[\s"\'\\]', text) else text


def _sh_quote(text: str) -> str:
    return "'" + text.replace("'", "'\\''") + "'"


def interpret_status(system: str, completed) -> tuple[bool, bool]:
    """(registered, running) from the status command's return code and output."""
    text = (completed.stdout or '') + (completed.stderr or '')
    if completed.returncode != 0:
        # systemd: is-active exits 3 for a known inactive unit and prints 'inactive'/'failed'; 4 means no such unit
        if system == 'Linux' and completed.returncode == 3 and re.search(r'\b(inactive|failed|activating|deactivating)\b', text):
            return True, False
        return False, False
    if system == 'Windows':
        return True, bool(re.search(r'^Status:\s+Running\s*$', text, re.M | re.I))
    if system == 'Darwin':
        return True, bool(re.search(r'state = running', text))
    return True, True


def _run(command, *, check_message: str | None = None):
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if check_message is not None and completed.returncode != 0:
        raise ValueError(f'{check_message}: {command[0]} exited {completed.returncode}{_last_line(completed)}')
    return completed


def _last_line(completed) -> str:
    detail = (completed.stderr or completed.stdout or '').strip().splitlines()
    return f' ({detail[-1].strip()})' if detail else ''


def _run_all(commands, *, tolerated=(), check_message: str | None = None) -> list:
    """Run the plan's commands in order; a nonzero exit is fatal (check_message) unless the command is tolerated."""
    outcomes = []
    for command in commands:
        completed = _run(command, check_message=None if command in tolerated else check_message)
        outcomes.append({'command': command, 'returncode': completed.returncode})
    return outcomes


def process_alive(pid: int) -> bool:
    """Whether a process with this id exists (the server a pid file names). Windows: OpenProcess + GetExitCodeProcess,
    because os.kill(pid, 0) terminates the process there; POSIX: signal 0 (a permission error means it exists)."""
    if os.name == 'nt':
        import ctypes
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        handle = kernel32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.get_last_error() == 5  # ERROR_ACCESS_DENIED: another account's live process
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_pid(pid_file) -> int | None:
    """The process id a `serve --pid-file` wrote, or None when the file is absent or not a number."""
    try:
        text = Path(pid_file).read_text(encoding='utf-8', errors='replace').strip()
    except OSError:
        return None
    return int(text) if text.isdigit() else None


def host_path(current: dict, path) -> Path:
    """A plan path in this host's spelling, for the files `titanos service` itself reads and writes (the environment file,
    the launcher, the task XML, the pid file, the log). A plan spells its paths the way the target platform does, and the
    host that runs the command is that platform, so this is the identity everywhere but in the cross-platform checks: a
    Windows plan under a POSIX TITANOS_HOME still reads and writes under that home, never beside the current directory."""
    pure = PureWindowsPath if current['platform'] == 'Windows' else PurePosixPath
    try:
        return Path(current['home']).joinpath(*pure(path).relative_to(pure(current['home'])).parts)
    except ValueError:  # a unit outside TITANOS_HOME (systemd, launchd): already in this host's spelling
        return Path(path)


def read_env_file(path, *, quoted: bool) -> dict:
    """The KEY=VALUE lines render_env wrote (single quotes undone on POSIX); {} when the file is absent."""
    out = {}
    try:
        lines = Path(path).read_text(encoding='utf-8').splitlines()
    except OSError:
        return out
    for line in lines:
        if not line.strip() or line.startswith('#') or '=' not in line:
            continue
        key, _, value = line.partition('=')
        if quoted and len(value) >= 2 and value[0] == value[-1] == "'":
            value = value[1:-1].replace("'\\''", "'")
        out[key.strip()] = value
    return out


def health_url(env: dict) -> str:
    """GET /healthz of the server the service environment describes: TITANOS_HOST (a wildcard bind is probed on loopback),
    TITANOS_PORT (default 11435) and https when the native TLS pair is set."""
    host = (env.get('TITANOS_HOST') or '127.0.0.1').strip()
    if host in ('', '*', '0.0.0.0', '::'):
        host = '127.0.0.1'
    elif ':' in host and not host.startswith('['):
        host = f'[{host}]'
    scheme = 'https' if env.get('TITANOS_TLS_CERTFILE') else 'http'
    return f"{scheme}://{host}:{env.get('TITANOS_PORT') or 11435}/healthz"


def _http_status(url: str):
    """The HTTP status GET url answers, or None when nothing answers (connection refused, timeout)."""
    import ssl
    import urllib.error
    import urllib.request
    context = ssl._create_unverified_context() if url.startswith('https') else None  # a self-signed native pair; only the status is read
    try:
        with urllib.request.urlopen(url, timeout=3, context=context) as reply:
            return reply.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except (OSError, ValueError):
        return None


def wait_ready(system: str, current: dict, url: str, seconds: float, *, probe=None) -> dict:
    """Poll GET url until it answers, the unit stops running, or `seconds` pass. Every WAIT_STATUS_EVERY seconds the pid
    file (a dead process) and the service manager's status command (registered but not running: the server exited;
    on Windows the launcher ended with it) are consulted so a server that dies at once ends the wait early."""
    probe = probe or _http_status
    started = checked = time.monotonic()
    deadline = started + max(seconds, 0)
    while True:
        status = probe(url)
        if status is not None:
            return {'ready': status == 200, 'status': status, 'url': url, 'elapsed': round(time.monotonic() - started, 2)}
        now = time.monotonic()
        if now >= deadline:
            return {'ready': False, 'status': None, 'url': url, 'reason': 'timeout', 'elapsed': round(now - started, 2)}
        if now - checked >= WAIT_STATUS_EVERY:
            checked = now
            pid = read_pid(host_path(current, current['pid_file'])) if current.get('pid_file') else None
            exited = pid is not None and not process_alive(pid)
            if not exited and pid is None:
                registered, running = interpret_status(system, _run(current['status'][0]))
                exited = registered and not running
            if exited:
                return {'ready': False, 'status': None, 'url': url, 'reason': 'exited', 'elapsed': round(now - started, 2)}
        time.sleep(WAIT_POLL)


def log_tail(path, lines: int = 8) -> str:
    try:
        text = Path(path).read_text(encoding='utf-8', errors='replace') if path else ''
    except OSError:
        return ''
    return '\n'.join(text.strip().splitlines()[-lines:])


def stop_server(current: dict) -> tuple:
    """Run the plan's stop commands, then (Windows) stop the server by the id in its pid file and remove the file. Nothing
    is killed without that file. Returns (command outcomes, stopped pid or None)."""
    outcomes = _run_all(current['stop'])
    stopped = None
    pid_file = host_path(current, current['pid_file']) if current.get('pid_file') else None
    if pid_file is not None and pid_file.is_file():
        pid = read_pid(pid_file)
        if pid is not None:
            stopped = pid
            command = ['taskkill', '/PID', str(pid), '/T', '/F']
            completed = _run(command)
            outcomes.append({'command': command, 'returncode': completed.returncode})
        pid_file.unlink()
    return outcomes, stopped


def write_files(files: dict, *, env_file: str, env_mode: int | None, encodings: dict | None = None, resolve=Path) -> None:
    """Write the plan's files as given (UTF-8 unless encodings names another, e.g. the UTF-16 task XML); resolve maps a
    plan path to this host's spelling (host_path)."""
    encodings = encodings or {}
    for path, text in files.items():
        target = resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if path == env_file and env_mode is not None and os.name != 'nt':
            descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, env_mode)
            with os.fdopen(descriptor, 'w', encoding='utf-8', newline='') as handle:
                handle.write(text)
            os.chmod(target, env_mode)
        else:
            target.write_text(text, encoding=encodings.get(path, 'utf-8'), newline='')


def run_service_command(args, *, config_path, system: str | None = None, environ=None) -> int:
    """The `titanos service` subcommand. Prints JSON (--json) or a short report; ValueError for typed refusals."""
    system = system or platform.system()
    environ = os.environ if environ is None else environ
    home = titanos_home()
    action = args.service_action
    if action == 'install':
        executable = args.executable or default_executable()
        env = service_environment(environ, host=args.host, port=args.port, role=args.role)
    else:
        executable = args.__dict__.get('executable') or 'titanos'
        env = {}
    current = plan(system, name=args.name, home=home, executable=executable, config=config_path, env=env,
                   system_wide=args.system, account=windows_account(environ) if system == 'Windows' else None)
    if action != 'install':
        env = read_env_file(host_path(current, current['env_file']), quoted=system != 'Windows')  # the installed host/port/TLS for the readiness probe
    wait = getattr(args, 'wait', None)
    wait = DEFAULT_WAIT if wait is None else float(wait)
    if args.dry_run:
        _emit(args, {'action': action, 'dry_run': True, **{k: v for k, v in current.items() if k != 'files'},
                     'files': dict(current['files']) if action == 'install' else list(current['files']), 'health_url': health_url(env), 'wait': wait},
              'titanos service ' + action + ' (dry run): nothing was written or run')
        return 0
    if action == 'install':
        config = Path(config_path)
        if not config.is_file():
            raise ValueError(f'no registry at {config}: run `titanos init` (or `titanos add`) first; the service would exit at once '
                             'with that message')
        probe = _run([executable, '--version'])
        if probe.returncode != 0:
            raise ValueError(f'{executable} cannot run titanos (exit {probe.returncode}{_last_line(probe)}); pass --executable with the '
                             'titanos script of a Python environment where the package is installed')
        host_path(current, current['log_dir']).mkdir(parents=True, exist_ok=True)
        write_files(current['files'], env_file=current['env_file'], env_mode=current.get('env_file_mode'),
                    encodings=current.get('file_encodings'), resolve=lambda path: host_path(current, path))
        _run_all(current['install'], tolerated=current['tolerated'], check_message='service install failed')
        return _report_started(args, system, current, env, wait, 'install', 'installed',
                               {'env_file': current['env_file'], 'launcher': current.get('launcher'), 'log': current.get('log'),
                                'logs_command': current['logs_command'], 'notes': current['notes']})
    if action == 'start':
        _run_all(current['start'], tolerated=current['tolerated'], check_message='service start failed')
        return _report_started(args, system, current, env, wait, 'start', 'started', {'log': current.get('log'), 'logs_command': current['logs_command']})
    if action == 'stop':
        outcomes, stopped = stop_server(current)
        failed = [o for o in outcomes if o['returncode'] != 0]
        _emit(args, {'action': 'stop', 'result': 'stopped' if not failed else 'stop_failed', 'unit': current['unit'], 'commands': outcomes,
                     'stopped_pid': stopped},
              f"titanos service {args.name}: stopped ({current['unit']}" + (f'; server process {stopped} ended' if stopped else '') + ')'
              if not failed else f"titanos service {args.name}: stop failed ({failed[0]['command'][0]} exited {failed[0]['returncode']})")
        return 0 if not failed else 1
    if action == 'uninstall':
        outcomes, stopped = stop_server(current)
        outcomes += _run_all(current['uninstall'])
        removed = []
        for path in current['remove_on_uninstall']:
            if host_path(current, path).is_file():
                host_path(current, path).unlink()
                removed.append(path)
        for command in current.get('after_uninstall', []):
            _run(command)
        _emit(args, {'action': 'uninstall', 'result': 'uninstalled', 'unit': current['unit'], 'commands': outcomes, 'stopped_pid': stopped,
                     'removed': removed, 'kept': [current['env_file']] + [current[k] for k in ('launcher', 'task_xml') if current.get(k)]},
              f"titanos service {args.name}: unregistered ({current['unit']}); {current['env_file']} kept")
        return 0
    completed = _run(current['status'][0])
    registered, running = interpret_status(system, completed)
    pid = read_pid(host_path(current, current['pid_file'])) if current.get('pid_file') else None
    note = None
    if registered and pid is not None:
        # Windows: the task state is the launcher's; the server's own id says whether it is serving (a stale file: not running)
        running = process_alive(pid)
        note = f'server process {pid} ' + ('is running' if running else 'is not running (stale pid file)')
    code = STATUS_RUNNING if running else STATUS_STOPPED if registered else STATUS_ABSENT
    state = 'running' if running else 'registered, not running' if registered else 'not registered'
    _emit(args, {'action': 'status', 'unit': current['unit'], 'registered': registered, 'running': running, 'exit_code': code, 'pid': pid,
                 'server': note, 'output': (completed.stdout or '').strip()[-2000:], 'notes': current['notes']},
          f"titanos service {args.name}: {state} ({current['unit']}" + (f'; {note}' if note else '') + ')')
    return code


def _report_started(args, system, current, env, wait, action, result, extra) -> int:
    """After install/start: wait for GET /healthz (unless --wait 0) and report; exit 0 only when the server answered."""
    url = health_url(env)
    outcome = wait_ready(system, current, url, wait) if wait > 0 else {'ready': None, 'url': url, 'reason': 'not probed (--wait 0)'}
    document = {'action': action, 'result': result, 'unit': current['unit'], 'server': outcome, **extra}
    if outcome['ready']:
        _emit(args, document, f"titanos service {args.name}: {result} ({current['unit']}); GET {url} answered 200 after {outcome['elapsed']} s")
        return 0
    if outcome['ready'] is None:
        _emit(args, document, f"titanos service {args.name}: {result} ({current['unit']}); the server was not probed (--wait 0)")
        return 0
    tail = log_tail(host_path(current, current['log']) if current.get('log') else None)
    document['result'] = result + '_not_ready'
    document['log_tail'] = tail
    reason = outcome.get('reason') or f"status {outcome.get('status')}"
    _emit(args, document, f"titanos service {args.name}: {result} ({current['unit']}) but the server did not answer GET {url} within {wait} s "
          f"({reason}); " + (f'last log lines:\n{tail}' if tail else f"logs: {current['logs_command']}"))
    return STATUS_STOPPED


def _emit(args, document: dict, line: str) -> None:
    if getattr(args, 'json', False):
        print(json.dumps(document, indent=2))
    else:
        print(line)


__all__ = ['DEFAULT_WAIT', 'NAME_PATTERN', 'STATUS_ABSENT', 'STATUS_RUNNING', 'STATUS_STOPPED', 'SUPPORTED', 'default_executable',
           'health_url', 'host_path', 'interpret_status', 'plan', 'process_alive', 'read_env_file', 'read_pid', 'render_env', 'render_task_xml',
           'run_service_command', 'service_environment', 'stop_server', 'wait_ready', 'windows_account', 'write_files']
