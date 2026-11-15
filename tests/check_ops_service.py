"""OPS (c): `titanos service install|uninstall|status` (titanos/service.py, docs/OPERATIONS.md).

- plan() for Windows, Linux and macOS (pure, every platform checked on every host): the unit sets TITANOS_HOME
  explicitly, runs the given executable with the registry path, reads TITANOS_HOME/service/<name>.env; the Windows plan
  is a Scheduled Task registered from a per-user task XML (logon trigger and interactive token of the installing
  account, least privilege, no run-time limit: the form a non-elevated shell may register; --system ONSTART as SYSTEM)
  with a launcher .cmd, Linux a systemd unit under
  $XDG_CONFIG_HOME/systemd/user (--system /etc/systemd/system), macOS a launchd plist under ~/Library/LaunchAgents
  (--system /Library/LaunchDaemons) that plistlib parses back; TITANOS_API_KEY appears in the env file only, never in the
  unit, the launcher or a command; POSIX env values are single-quoted;
- service_environment() copies every TITANOS_* but TITANOS_HOME, applies --host/--port/--role, refuses line breaks;
- the files the command itself reads and writes (env file, launcher, task XML, pid file, log) live under TITANOS_HOME in
  this host's spelling (service.host_path): a Windows plan checked on a POSIX host writes nothing beside the current
  directory (every Commands test asserts that) and finds its pid file and log;
- install --dry-run writes and runs nothing; install writes the files (env file 0600 on POSIX) and runs the platform's
  commands in order; uninstall runs its commands, removes only the unit it wrote and keeps the env file; status maps the
  command outcome to running / registered / absent (exit 0 / 3 / 4);
- refusals name the field: an unknown platform, a bad --name, no executable;
- Windows uninstall stops the server by the id it recorded (the launcher passes `serve --pid-file`; `schtasks /End` ends
  the launcher .cmd only) and never kills anything without that file;
- install refuses before registering when the registry is missing (`titanos init`) or `<executable> --version` fails; install
  and start wait for GET /healthz of the installed host/port (exit 0 only when it answered; exit 3 with the reason and the
  log tail when it did not; --wait 0 skips the probe) and the wait ends early when the unit is registered but not running;
- start/stop plans per platform (Windows /Run and /End + the pid kill; systemctl start/stop; launchctl bootstrap+kickstart
  with a tolerated bootstrap / bootout), and Windows status reads the pid file (a live process = running, a stale one = not).
subprocess.run is replaced by a recorder: no schtasks, systemctl or launchctl runs here. ServePidFile alone starts a real
`python -m titanos serve --pid-file` on a free loopback port (49900..49949) over a weightless registry and stops it.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import plistlib
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from titanos import service  # noqa: E402
from titanos.cli import main  # noqa: E402

SECRET = 'tk_service_fixture_secret_0001'
ENV = {'TITANOS_API_KEY': SECRET, 'TITANOS_HOME': '/ignored', 'TITANOS_PORT': '49910', 'PATH': '/bin', 'TITANOS_ROLE': 'broker'}


def args(action, **kw):
    base = {'service_action': action, 'name': 'titanos', 'system': False, 'dry_run': False, 'json': True,
            'host': None, 'port': None, 'role': None, 'executable': None, 'wait': 0}
    base.update(kw)
    return SimpleNamespace(**base)


class Plans(unittest.TestCase):
    def setUp(self):
        self.env = service.service_environment(ENV, host='127.0.0.1')

    def plan(self, system, **kw):
        home = {'Windows': 'C:\\data\\titanos-home', 'Linux': '/var/lib/titanos', 'Darwin': '/opt/titanos'}[system]
        exe = {'Windows': 'C:\\tools\\titanos.exe', 'Linux': '/opt/venv/bin/titanos', 'Darwin': '/opt/venv/bin/titanos'}[system]
        config = home + ('\\' if system == 'Windows' else '/') + 'models.json'
        return service.plan(system, name='titanos', home=home, executable=exe, config=config, env=self.env,
                            uid=501, user_home='/account' if system != 'Windows' else 'C:\\profile',
                            xdg_config_home='/account/.config', account='DESKTOP\\operator', **kw)

    def assert_secret_only_in_env_file(self, current):
        for path, text in current['files'].items():
            if path == current['env_file']:
                self.assertIn(SECRET, text)
            else:
                self.assertNotIn(SECRET, text, path)
        for group in ('install', 'start', 'stop', 'uninstall', 'status'):
            for command in current[group]:
                self.assertNotIn(SECRET, ' '.join(command))
        self.assertNotIn('TITANOS_HOME', current['files'][current['env_file']].split('\n', 1)[1])

    def test_windows_scheduled_task(self):
        current = self.plan('Windows')
        launcher = current['files'][current['launcher']]
        self.assertIn('set "TITANOS_HOME=C:\\data\\titanos-home"', launcher)
        self.assertIn('for /f "usebackq eol=# tokens=1,* delims==" %%A in ("C:\\data\\titanos-home\\service\\titanos.env")', launcher)
        self.assertIn('"C:\\tools\\titanos.exe" --config "C:\\data\\titanos-home\\models.json" serve --pid-file "C:\\data\\titanos-home\\service\\titanos.pid" >> "C:\\data\\titanos-home\\logs\\titanos.log" 2>&1', launcher)
        self.assertEqual(current['pid_file'], 'C:\\data\\titanos-home\\service\\titanos.pid', 'uninstall stops the server by the id it recorded')
        self.assertTrue(launcher.endswith('\r\n') and '\n' not in launcher.replace('\r\n', ''))
        # a per-user logon task is registered from a task XML: `schtasks /SC ONLOGON` is refused from a non-elevated shell
        task_xml = 'C:\\data\\titanos-home\\service\\titanos.xml'
        self.assertEqual(current['install'][0], ['schtasks', '/Create', '/F', '/TN', 'titanos', '/XML', task_xml])
        self.assertEqual(current['task_xml'], task_xml)
        self.assertNotIn('ONLOGON', ' '.join(current['install'][0]))
        ns = {'t': 'http://schemas.microsoft.com/windows/2004/02/mit/task'}
        self.assertEqual(current['file_encodings'], {task_xml: 'utf-16'}, 'schtasks /XML reads a UTF-16 file with a byte order mark')
        self.assertTrue(current['files'][task_xml].startswith('<?xml version="1.0" encoding="UTF-16"?>'))
        task = ElementTree.fromstring(current['files'][task_xml].encode('utf-16'))
        self.assertEqual(task.findtext('t:Triggers/t:LogonTrigger/t:UserId', namespaces=ns), 'DESKTOP\\operator')
        principal = task.find('t:Principals/t:Principal', ns)
        self.assertEqual((principal.findtext('t:UserId', namespaces=ns), principal.findtext('t:LogonType', namespaces=ns), principal.findtext('t:RunLevel', namespaces=ns)),
                         ('DESKTOP\\operator', 'InteractiveToken', 'LeastPrivilege'))
        self.assertEqual(task.findtext('t:Actions/t:Exec/t:Command', namespaces=ns), current['launcher'])
        self.assertEqual(task.findtext('t:Settings/t:ExecutionTimeLimit', namespaces=ns), 'PT0S', 'the schtasks default would end the server after 72 hours')
        self.assertTrue(current['files'][task_xml].endswith('\r\n'))
        self.assertEqual(current['install'][1], ['schtasks', '/Run', '/TN', 'titanos'])
        self.assertEqual((current['start'], current['stop']), ([['schtasks', '/Run', '/TN', 'titanos']], [['schtasks', '/End', '/TN', 'titanos']]))
        self.assertEqual(current['uninstall'], [['schtasks', '/Delete', '/F', '/TN', 'titanos']], 'stop_server runs /End and the pid kill first')
        self.assertEqual(current['status'][0][:3], ['schtasks', '/Query', '/TN'])
        env_text = current['files'][current['env_file']]
        self.assertIn('TITANOS_API_KEY=' + SECRET + '\n', env_text)  # unquoted for the cmd loop
        self.assertIn('TITANOS_HOST=127.0.0.1\n', env_text)
        self.assertIn('TITANOS_PORT=49910\n', env_text)
        self.assertTrue(any('not a Windows Service' in note for note in current['notes']))
        self.assert_secret_only_in_env_file(current)
        system = self.plan('Windows', system_wide=True)
        self.assertEqual(system['install'][0][-4:], ['/SC', 'ONSTART', '/RU', 'SYSTEM'])
        self.assertNotIn('task_xml', system)
        self.assertTrue(any('DPAPI' in note for note in system['notes']))

    def test_linux_systemd(self):
        current = self.plan('Linux')
        self.assertEqual(current['unit'], '/account/.config/systemd/user/titanos.service')
        unit = current['files'][current['unit']]
        self.assertIn('Environment=TITANOS_HOME=/var/lib/titanos\n', unit)
        self.assertIn('EnvironmentFile=-/var/lib/titanos/service/titanos.env\n', unit)
        self.assertIn('ExecStart=/opt/venv/bin/titanos --config /var/lib/titanos/models.json serve\n', unit)
        self.assertIn('Restart=on-failure\n', unit)
        self.assertIn('WantedBy=default.target\n', unit)
        self.assertEqual(current['install'], [['systemctl', '--user', 'daemon-reload'],
                                              ['systemctl', '--user', 'enable', '--now', 'titanos']])
        self.assertEqual(current['uninstall'], [['systemctl', '--user', 'disable', '--now', 'titanos']])
        self.assertEqual((current['start'], current['stop']), ([['systemctl', '--user', 'start', 'titanos']], [['systemctl', '--user', 'stop', 'titanos']]))
        self.assertEqual(current['remove_on_uninstall'], [current['unit']])
        self.assertEqual(current['status'], [['systemctl', '--user', 'is-active', 'titanos']])
        self.assertEqual(current['logs_command'], 'journalctl --user -u titanos -f')
        self.assertIn("TITANOS_API_KEY='" + SECRET + "'\n", current['files'][current['env_file']])
        self.assertEqual(current['env_file_mode'], 0o600)
        self.assert_secret_only_in_env_file(current)
        system = self.plan('Linux', system_wide=True)
        self.assertEqual(system['unit'], '/etc/systemd/system/titanos.service')
        self.assertIn('WantedBy=multi-user.target\n', system['files'][system['unit']])
        self.assertEqual(system['install'][0], ['systemctl', 'daemon-reload'])

    def test_macos_launchd(self):
        current = self.plan('Darwin')
        self.assertEqual(current['unit'], '/account/Library/LaunchAgents/com.titanos.titanos.plist')
        document = plistlib.loads(current['files'][current['unit']].encode('utf-8'))
        self.assertEqual(document['Label'], 'com.titanos.titanos')
        self.assertEqual(document['EnvironmentVariables'], {'TITANOS_HOME': '/opt/titanos'})
        self.assertEqual(document['ProgramArguments'][:2], ['/bin/sh', '-c'])
        self.assertIn(". '/opt/titanos/service/titanos.env'", document['ProgramArguments'][2])
        self.assertIn("exec '/opt/venv/bin/titanos' --config '/opt/titanos/models.json' serve", document['ProgramArguments'][2])
        self.assertTrue(document['RunAtLoad'])
        self.assertEqual(document['StandardErrorPath'], '/opt/titanos/logs/titanos.log')
        self.assertEqual(current['install'], [['launchctl', 'bootstrap', 'gui/501', current['unit']],
                                              ['launchctl', 'kickstart', '-k', 'gui/501/com.titanos.titanos']])
        self.assertEqual(current['uninstall'], [['launchctl', 'bootout', 'gui/501/com.titanos.titanos']])
        self.assertEqual(current['stop'], [['launchctl', 'bootout', 'gui/501/com.titanos.titanos']])
        self.assertEqual(current['start'], current['install'], 'start after stop loads the plist again and kicks it')
        self.assertEqual(current['tolerated'], [current['install'][0]], 'bootstrap of an already loaded plist is not fatal')
        self.assertEqual(current['status'], [['launchctl', 'print', 'gui/501/com.titanos.titanos']])
        self.assert_secret_only_in_env_file(current)
        system = self.plan('Darwin', system_wide=True)
        self.assertEqual(system['unit'], '/Library/LaunchDaemons/com.titanos.titanos.plist')
        self.assertEqual(system['install'][0], ['launchctl', 'bootstrap', 'system', system['unit']])

    def test_environment_and_refusals(self):
        env = service.service_environment(ENV, port=49911, role='standalone')
        self.assertEqual(env, {'TITANOS_API_KEY': SECRET, 'TITANOS_PORT': '49911', 'TITANOS_ROLE': 'standalone'})
        with self.assertRaises(ValueError) as caught:
            service.service_environment({'TITANOS_HOST': 'a\nb'})
        self.assertIn('TITANOS_HOST', str(caught.exception))
        self.assertEqual(service.render_env({'A': "it's"}, quoted=True).splitlines()[1], "A='it'\\''s'")
        with self.assertRaises(ValueError) as caught:
            service.plan('FreeBSD', name='titanos', home='/h', executable='/x', config='/h/models.json', env={})
        self.assertIn('FreeBSD', str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            service.plan('Linux', name='9bad name', home='/h', executable='/x', config='/h/models.json', env={})
        self.assertIn('--name', str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            service.plan('Linux', name='titanos', home='/h', executable='', config='/h/models.json', env={})
        self.assertIn('--executable', str(caught.exception))

    def test_status_interpretation(self):
        done = lambda code, out='', err='': SimpleNamespace(returncode=code, stdout=out, stderr=err)  # noqa: E731
        self.assertEqual(service.interpret_status('Windows', done(0, 'TaskName: \\titanos\nStatus:        Running\n')), (True, True))
        self.assertEqual(service.interpret_status('Windows', done(0, 'Status:        Ready\n')), (True, False))
        self.assertEqual(service.interpret_status('Windows', done(1, '', 'ERROR: The system cannot find the file specified.')), (False, False))
        self.assertEqual(service.interpret_status('Linux', done(0, 'active\n')), (True, True))
        self.assertEqual(service.interpret_status('Linux', done(3, 'inactive\n')), (True, False))
        self.assertEqual(service.interpret_status('Linux', done(4, 'inactive\n')), (False, False))
        self.assertEqual(service.interpret_status('Darwin', done(0, 'com.titanos.titanos = {\n\tstate = running\n}')), (True, True))
        self.assertEqual(service.interpret_status('Darwin', done(0, 'state = not running')), (True, False))
        self.assertEqual(service.interpret_status('Darwin', done(113, '', 'Could not find service')), (False, False))


class Commands(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp(prefix='titanos-ops-service-'))
        self.addCleanup(shutil.rmtree, self.home, True)
        self.exe = self.home / ('titanos.exe' if os.name == 'nt' else 'titanos')
        self.exe.write_text('', encoding='utf-8')
        (self.home / 'models.json').write_text('{"models": {}}', encoding='utf-8')  # install refuses without a registry
        self.commands: list[list] = []
        self.answers: list = []

        def fake_run(command, **kw):
            self.commands.append(list(command))
            return self.answers.pop(0) if self.answers else SimpleNamespace(returncode=0, stdout='', stderr='')
        patcher = mock.patch.object(service.subprocess, 'run', fake_run)
        patcher.start()
        self.addCleanup(patcher.stop)
        env = mock.patch.dict(os.environ, {'TITANOS_HOME': str(self.home), 'TITANOS_API_KEY': SECRET, 'TITANOS_PORT': '49912'})
        env.start()
        self.addCleanup(env.stop)
        # a plan spelled for another platform must never leave files beside the current directory (a Windows plan on POSIX
        # once wrote literal '\\tmp\\...' names into the checkout)
        beside = set(os.listdir(os.getcwd()))
        self.addCleanup(lambda: self.assertEqual(sorted(set(os.listdir(os.getcwd())) - beside), [], 'files written beside the current directory'))

    def run_service(self, system, action, **kw):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = service.run_service_command(args(action, **kw), config_path=self.home / 'models.json', system=system)
        return code, json.loads(out.getvalue())

    def test_dry_run_writes_and_runs_nothing(self):
        code, result = self.run_service('Linux', 'install', dry_run=True, executable=str(self.exe), port=49913)
        self.assertEqual((code, result['dry_run'], result['action']), (0, True, 'install'))
        self.assertIn("TITANOS_PORT='49913'", result['files'][result['env_file']])
        self.assertFalse((self.home / 'service').exists())
        self.assertEqual(self.commands, [])
        # through the real parser too
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                main(['service', 'install', '--dry-run', '--json', '--executable', str(self.exe)])
            except SystemExit as exc:
                self.fail(f'exit {exc.code}: {err.getvalue()}')
        self.assertEqual(json.loads(out.getvalue())['dry_run'], True)
        self.assertFalse((self.home / 'service').exists())

    def test_install_uninstall_status(self):
        system = 'Linux'
        with mock.patch.dict(os.environ, {'XDG_CONFIG_HOME': str(self.home / 'xdg')}):
            code, result = self.run_service(system, 'install', executable=str(self.exe), host='127.0.0.1')
        self.assertEqual((code, result['result']), (0, 'installed'), result)
        env_file = Path(result['env_file'])
        unit = Path(result['unit'])
        self.assertTrue(env_file.is_file() and unit.is_file())
        self.assertIn("TITANOS_API_KEY='" + SECRET + "'", env_file.read_text(encoding='utf-8'))
        self.assertIn('TITANOS_HOST', env_file.read_text(encoding='utf-8'))
        self.assertNotIn(SECRET, unit.read_text(encoding='utf-8'))
        if os.name != 'nt':
            self.assertEqual(env_file.stat().st_mode & 0o777, 0o600)
        self.assertTrue((self.home / 'logs').is_dir())
        self.assertEqual(self.commands, [[str(self.exe), '--version'], ['systemctl', '--user', 'daemon-reload'], ['systemctl', '--user', 'enable', '--now', 'titanos']],
                         'the executable is probed before anything is registered')
        self.assertEqual(result['server'], {'ready': None, 'url': 'http://127.0.0.1:49912/healthz', 'reason': 'not probed (--wait 0)'})
        self.commands.clear()
        self.answers = [SimpleNamespace(returncode=0, stdout='active\n', stderr='')]
        code, result = self.run_service(system, 'status')
        self.assertEqual((code, result['registered'], result['running']), (service.STATUS_RUNNING, True, True))
        self.answers = [SimpleNamespace(returncode=3, stdout='inactive\n', stderr='')]
        code, result = self.run_service(system, 'status')
        self.assertEqual((code, result['running']), (service.STATUS_STOPPED, False))
        self.commands.clear()
        with mock.patch.dict(os.environ, {'XDG_CONFIG_HOME': str(self.home / 'xdg')}):
            code, result = self.run_service(system, 'uninstall')
        self.assertEqual((code, result['result']), (0, 'uninstalled'))
        self.assertEqual(self.commands, [['systemctl', '--user', 'stop', 'titanos'], ['systemctl', '--user', 'disable', '--now', 'titanos'], ['systemctl', '--user', 'daemon-reload']])
        self.assertFalse(unit.exists(), 'the unit this install wrote is removed')
        self.assertTrue(env_file.is_file(), 'the environment file under TITANOS_HOME is kept')
        self.assertEqual([Path(p) for p in result['removed']], [unit])
        self.answers = [SimpleNamespace(returncode=4, stdout='inactive\n', stderr='')]
        code, result = self.run_service(system, 'status')
        self.assertEqual((code, result['registered']), (service.STATUS_ABSENT, False))

    def test_failed_install_command_is_a_typed_error(self):
        self.answers = [SimpleNamespace(returncode=0, stdout='titanos 0.0\n', stderr=''), SimpleNamespace(returncode=1, stdout='', stderr='ERROR: Access is denied.')]
        with self.assertRaises(ValueError) as caught:
            self.run_service('Windows', 'install', executable=str(self.exe))
        self.assertIn('schtasks exited 1', str(caught.exception))
        self.assertIn('Access is denied', str(caught.exception))

    def test_install_refuses_without_a_registry(self):
        # the server would exit at once with "No registry at ...": refused before anything is written or registered
        (self.home / 'models.json').unlink()
        with self.assertRaises(ValueError) as caught:
            self.run_service('Windows', 'install', executable=str(self.exe))
        self.assertIn('titanos init', str(caught.exception))
        self.assertIn(str(self.home / 'models.json'), str(caught.exception))
        self.assertEqual(self.commands, [])
        self.assertFalse((self.home / 'service').exists())

    def test_install_refuses_an_executable_that_cannot_run_titanos(self):
        # a titanos.exe launcher beside an interpreter without the package (this host's venv) fails `--version`; the unit would die at once
        self.answers = [SimpleNamespace(returncode=1, stdout='', stderr="Traceback...\nModuleNotFoundError: No module named 'titanos'")]
        with self.assertRaises(ValueError) as caught:
            self.run_service('Linux', 'install', executable=str(self.exe))
        self.assertIn('--executable', str(caught.exception))
        self.assertIn("ModuleNotFoundError: No module named 'titanos'", str(caught.exception))
        self.assertEqual(self.commands, [[str(self.exe), '--version']])
        self.assertFalse((self.home / 'service').exists())

    def test_install_waits_for_the_server_and_exits_3_when_it_does_not_answer(self):
        probes = iter([None, None, 200])
        with mock.patch.object(service, '_http_status', lambda url: next(probes)), mock.patch.object(service, 'WAIT_POLL', 0.01):
            code, result = self.run_service('Windows', 'install', executable=str(self.exe), wait=5)
        self.assertEqual((code, result['result'], result['server']['ready'], result['server']['status']), (0, 'installed', True, 200), result)
        self.assertEqual(result['server']['url'], 'http://127.0.0.1:49912/healthz')
        for written in ('titanos.env', 'titanos.cmd', 'titanos.xml'):
            self.assertTrue((self.home / 'service' / written).is_file(), f'{written} is written under TITANOS_HOME in this host\'s spelling')
        # nothing answers: the install is reported as not ready with the log tail, exit 3 (registered, not running)
        (self.home / 'logs').mkdir(exist_ok=True)
        (self.home / 'logs' / 'titanos.log').write_text('starting\ntitanos: No registry at X. Run titanos init first.\n', encoding='utf-8')
        with mock.patch.object(service, '_http_status', lambda url: None), mock.patch.object(service, 'WAIT_POLL', 0.01):
            code, result = self.run_service('Windows', 'install', executable=str(self.exe), wait=0.3)
        self.assertEqual((code, result['result'], result['server']['ready'], result['server']['reason']), (service.STATUS_STOPPED, 'installed_not_ready', False, 'timeout'), result)
        self.assertIn('No registry at X', result['log_tail'])
        # a wildcard bind is probed on loopback; the native TLS pair makes it https
        self.assertEqual(service.health_url({'TITANOS_HOST': '0.0.0.0', 'TITANOS_PORT': '49913'}), 'http://127.0.0.1:49913/healthz')
        self.assertEqual(service.health_url({'TITANOS_HOST': '::1', 'TITANOS_TLS_CERTFILE': 'c.pem'}), 'https://[::1]:11435/healthz')
        self.assertEqual(service.health_url({}), 'http://127.0.0.1:11435/healthz')

    def test_start_reports_a_server_that_exited(self):
        # the unit was started but the server died: the service manager says registered-not-running and the wait ends early
        with mock.patch.dict(os.environ, {'XDG_CONFIG_HOME': str(self.home / 'xdg')}):
            self.run_service('Linux', 'install', executable=str(self.exe), host='127.0.0.1', port=49914)
        self.commands.clear()
        self.answers = [SimpleNamespace(returncode=0, stdout='', stderr=''), SimpleNamespace(returncode=3, stdout='inactive\n', stderr='')]
        with mock.patch.object(service, '_http_status', lambda url: None), mock.patch.object(service, 'WAIT_POLL', 0.01), \
                mock.patch.object(service, 'WAIT_STATUS_EVERY', 0.05):
            code, result = self.run_service('Linux', 'start', wait=5)
        self.assertEqual((code, result['result'], result['server']['reason']), (service.STATUS_STOPPED, 'started_not_ready', 'exited'), result)
        self.assertEqual(result['server']['url'], 'http://127.0.0.1:49914/healthz', 'the installed environment file names the port')
        self.assertEqual(self.commands, [['systemctl', '--user', 'start', 'titanos'], ['systemctl', '--user', 'is-active', 'titanos']])
        self.assertLess(result['server']['elapsed'], 4)

    def test_windows_stop_start_and_status_use_the_pid_file(self):
        pid_file = self.home / 'service' / 'titanos.pid'
        pid_file.parent.mkdir(parents=True)
        pid_file.write_text('4242\n', encoding='utf-8')
        code, result = self.run_service('Windows', 'stop')
        self.assertEqual((code, result['result'], result['stopped_pid']), (0, 'stopped', 4242), result)
        self.assertEqual(self.commands, [['schtasks', '/End', '/TN', 'titanos'], ['taskkill', '/PID', '4242', '/T', '/F']])
        self.assertFalse(pid_file.exists())
        self.commands.clear()
        code, result = self.run_service('Windows', 'stop')
        self.assertEqual((code, result['stopped_pid'], self.commands), (0, None, [['schtasks', '/End', '/TN', 'titanos']]), 'no pid file: nothing is killed')
        self.commands.clear()
        with mock.patch.object(service, '_http_status', lambda url: 200):
            code, result = self.run_service('Windows', 'start', wait=2)
        self.assertEqual((code, result['result'], result['server']['ready'], self.commands), (0, 'started', True, [['schtasks', '/Run', '/TN', 'titanos']]), result)
        # status: the task says Running (the launcher); the server's own id decides
        running = SimpleNamespace(returncode=0, stdout='TaskName: \\titanos\nStatus:        Running\n', stderr='')
        pid_file.write_text(str(os.getpid()), encoding='utf-8')
        self.answers = [running]
        code, result = self.run_service('Windows', 'status')
        self.assertEqual((code, result['running'], result['pid']), (service.STATUS_RUNNING, True, os.getpid()), result)
        dead = 4_000_000_000 if os.name == 'nt' else 2 ** 22 - 1
        pid_file.write_text(str(dead), encoding='utf-8')
        self.answers = [running]
        code, result = self.run_service('Windows', 'status')
        self.assertEqual((code, result['running']), (service.STATUS_STOPPED, False), result)
        self.assertIn('stale pid file', result['server'])
        self.assertTrue(service.process_alive(os.getpid()))
        self.assertFalse(service.process_alive(dead))
        env_text = service.render_env({'TITANOS_API_KEY': "it's", 'TITANOS_PORT': '1'}, quoted=True)
        self.assertEqual(service.read_env_file(self._write(env_text), quoted=True), {'TITANOS_API_KEY': "it's", 'TITANOS_PORT': '1'})
        self.assertEqual(service.read_env_file(self._write(service.render_env({'TITANOS_HOST': '0.0.0.0'}, quoted=False)), quoted=False), {'TITANOS_HOST': '0.0.0.0'})
        self.assertEqual(service.read_env_file(self.home / 'absent.env', quoted=True), {})

    def _write(self, text):
        path = self.home / 'probe.env'
        path.write_text(text, encoding='utf-8')
        return path

    def test_macos_start_tolerates_an_already_loaded_plist(self):
        self.answers = [SimpleNamespace(returncode=37, stdout='', stderr='Bootstrap failed: 37: Operation already in progress'), SimpleNamespace(returncode=0, stdout='', stderr='')]
        code, result = self.run_service('Darwin', 'start')
        self.assertEqual((code, result['result']), (0, 'started'), result)
        self.assertEqual([c[1] for c in self.commands], ['bootstrap', 'kickstart'])
        self.answers = [SimpleNamespace(returncode=0, stdout='', stderr=''), SimpleNamespace(returncode=113, stdout='', stderr='Could not find service')]
        with self.assertRaises(ValueError) as caught:
            self.run_service('Darwin', 'start')
        self.assertIn('service start failed', str(caught.exception))

    def test_windows_uninstall_stops_the_server_by_its_pid_file(self):
        # schtasks /End ends the launcher .cmd only; the server the launcher started recorded its id (serve --pid-file)
        pid_file = self.home / 'service' / 'titanos.pid'
        pid_file.parent.mkdir(parents=True)
        pid_file.write_text('4242\n', encoding='utf-8')
        code, result = self.run_service('Windows', 'uninstall')
        self.assertEqual((code, result['result'], result['stopped_pid']), (0, 'uninstalled', 4242), result)
        self.assertEqual(self.commands, [['schtasks', '/End', '/TN', 'titanos'], ['taskkill', '/PID', '4242', '/T', '/F'], ['schtasks', '/Delete', '/F', '/TN', 'titanos']])
        self.assertFalse(pid_file.exists(), 'the stale id is not kept for a later uninstall to kill an unrelated process')
        self.commands.clear()
        code, result = self.run_service('Windows', 'uninstall')
        self.assertEqual((result['stopped_pid'], self.commands[-1][0]), (None, 'schtasks'), 'no pid file: nothing is killed')

    def test_missing_executable_is_refused(self):
        with mock.patch.object(service.shutil, 'which', lambda name: None), \
                mock.patch.object(service.sys, 'executable', str(self.home / 'nowhere' / 'python')):
            with self.assertRaises(ValueError) as caught:
                self.run_service('Linux', 'install')
        self.assertIn('--executable', str(caught.exception))
        self.assertEqual(self.commands, [])


class ServePidFile(unittest.TestCase):
    """`titanos serve --pid-file PATH` writes the serving process id while the server runs and removes it on shutdown
    (a real `python -m titanos serve` on a free loopback port in 49900..49949 over a weightless registry; no model loads)."""

    def test_pid_file_lifecycle(self):
        root = Path(tempfile.mkdtemp(prefix='titanos-ops-pidfile-'))
        self.addCleanup(shutil.rmtree, root, True)
        (root / 'weightless').mkdir()
        (root / 'weightless' / 'config.json').write_text(json.dumps({'architectures': ['LlamaForCausalLM'], 'model_type': 'llama'}), encoding='utf-8')
        (root / 'models.json').write_text(json.dumps({'models': {'registry-only': {'path': str(root / 'weightless'), 'task': 'chat'}}}), encoding='utf-8')
        pid_file = root / 'service' / 'titanos.pid'
        port = next(p for p in range(49900, 49950) if _port_free(p))
        env = {k: v for k, v in os.environ.items() if not (k.startswith('TITANOS_') or k.endswith('_BASE_URL') or 'PROXY' in k.upper())}
        env.update({'TITANOS_HOME': str(root / 'home'), 'PYTHONPATH': str(ROOT / 'src'), 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONIOENCODING': 'utf-8',
                    'HF_HUB_OFFLINE': '1', 'CUDA_VISIBLE_DEVICES': ''})
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == 'win32' else 0
        with open(root / 'serve.log', 'wb') as log:
            process = subprocess.Popen([sys.executable, '-B', '-m', 'titanos', '--config', str(root / 'models.json'), 'serve', '--host', '127.0.0.1',
                                        '--port', str(port), '--log-level', 'warning', '--pid-file', str(pid_file)], cwd=str(root), env=env,
                                       stdout=log, stderr=subprocess.STDOUT, creationflags=flags, start_new_session=sys.platform != 'win32')
            try:
                deadline = time.monotonic() + 120
                while not (pid_file.is_file() and _healthy(port)):
                    if process.poll() is not None or time.monotonic() > deadline:
                        self.fail('serve did not come up with a pid file: ' + (root / 'serve.log').read_text(encoding='utf-8', errors='replace')[-1500:])
                    time.sleep(0.2)
                recorded = int(pid_file.read_text(encoding='utf-8').strip())
                self.assertGreater(recorded, 0)
                if sys.platform != 'win32':
                    self.assertEqual(recorded, process.pid)  # on Windows a venv launcher may sit between Popen and the interpreter
                process.send_signal(signal.CTRL_BREAK_EVENT if sys.platform == 'win32' else signal.SIGINT)
                process.wait(timeout=90)
                # on Windows the launcher stub exits at once while the interpreter behind it is still shutting down gracefully
                deadline = time.monotonic() + 90
                while (pid_file.exists() or not _port_free(port)) and time.monotonic() < deadline:
                    time.sleep(0.2)
                self.assertFalse(pid_file.exists(), 'the pid file is removed when the server exits')
                self.assertTrue(_port_free(port), 'the server released its port')
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=30)


def _port_free(port):
    sock = socket.socket()
    try:
        sock.bind(('127.0.0.1', port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _healthy(port):
    import urllib.request
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz', timeout=3) as reply:
            return reply.status == 200
    except OSError:
        return False


if __name__ == '__main__':
    unittest.main(verbosity=2)
