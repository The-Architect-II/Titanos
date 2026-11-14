"""Lab-neutral references for published records (tools/public_export.json, parameterization rules).

Generated documents that leave this repository (docs/execution_matrix.json and its renderings, the matrix-kit RAW reader input)
cite files by where the lab kept them and hosts by what their operating system called them. Those are operator circumstances,
not part of any claim, so the generators reduce them to stable labels before writing:

- a drive-rooted lab records root (`<letter>:/Titanos<...>/` or `<letter>:/Codex/`) becomes `<lab-records>/` and the remainder
  of the path is kept, so `.../runs/<run>/receipts/<file>` stays readable and comparable;
- an agent work tree under a Windows user profile (the account's Documents folder, then the agent's folder named in the rule
  below, a date and a tree name; already reduced to `<user-profile>/...` or still drive-rooted) is a lab records root as well
  and becomes `<lab-records>/`, keeping what follows the tree name;
- a Windows user-profile root (`<letter>:/Users/<account>/`) becomes `<user-profile>/` with the remainder kept, so no account
  name is written;
- a macOS home directory (`/Users/<account>`, no drive letter) becomes `<user-profile>` with the remainder kept;
- a Linux home directory (`/home/<account>`) becomes `<linux-home>` with the remainder kept; a folder literally named `home`
  below a label or another folder (`<lab-records>/native/home/x.json`) is not a home directory and is left alone;
- a bare drive-letter pair written as narrative (`the reachable <X>:/<Y>: roots`) becomes `lab records`;
- a host named by an inventory name, a hostname or a generated Windows machine name becomes its stable role label
  (HOST_ROLE_LABELS) when the private host map names it. The map (inventory name to role label, exact phrases to their neutral
  wording, names to deny) lives outside the repository: TITANOS_LAB_HOSTS_FILE, or tools/lab_hosts.private.json, which is
  gitignored and never committed. A phrase is matched after the path rules above, so a key may be spelled with the labels (a
  repeated `<lab-records>` list, a record folder named after a person or an agent). The public policy names only pattern
  classes and the role labels, never a machine;
- a generated Windows machine name (the policy's windows_generated_machine_name class: DESKTOP- or LAPTOP- followed by exactly
  seven letters/digits, WIN- followed by exactly eleven, any ASCII letter case, hyphen or underscore neighbours are boundaries)
  that no map entry names becomes `<windows-host>`. This is a last defence so the builder can never publish a generated name
  even when a run folder is named after one; it is not the intended spelling.

Role labels are the only accepted spelling of a host in a published document: `<hardware>-<form>-<os>-<arch>`, with a runtime
suffix when it disambiguates, chosen at kit launch (tools/matrix_kit/run_cells.py --host-label; docs/MATRIX_KIT.md, Host labels).

The rules are a text normalization applied at the source of truth (tools/build_execution_matrix.py over every string it writes,
tools/matrix_kit/ingest.py for the display receipt of every cell). They are not a secret scanner: tests/check_public_export_hygiene.py
asserts afterwards that no such reference survives in the public export set or in the tracked tree outside the private records.
docs/execution_matrix.json is rebuilt from the private reader input with the private map present (tools/public_export.json
one_off_passes records each rebuild), so the committed records are fixed points of every rule, with or without the map.
"""
import json
import os
import re
from pathlib import Path

LAB_RECORDS = '<lab-records>'
USER_PROFILE = '<user-profile>'
LINUX_HOME = '<linux-home>'
WINDOWS_HOST = '<windows-host>'
DRIVE_PAIR_PHRASE = 'lab records'

# The stable role labels of the hosts the execution matrix names: the single mapping every document uses (docs/MATRIX_KIT.md
# "Host labels", docs/CAPABILITY_MATRIX.md evidence rows, docs/roster.json, the kit's --host-label). A value describes the
# hardware role only. Add a label here first; the private map then points an inventory name at it.
HOST_ROLE_LABELS = {
    'blackwell-workstation-win-x64': 'the RTX PRO 6000 Blackwell workstation (Windows x64; Intel Core Ultra CPU with integrated graphics and AI Boost NPU) whose local kit runs are labelled matrix-kit-local-<...>; "this workstation" in prose',
    'turing-laptop-win-x64': 'the RTX 2070 Max-Q laptop with an Intel UHD Graphics 630 iGPU (Windows x64)',
    'rtx5070-desktop-win-x64': 'the RTX 5070 desktop that also carries the AMD RX 7900 XTX and RX 9070 XT (Windows x64)',
    'rtx5090-desktop-win-x64': 'the RTX 5090 desktop that also carries a GTX 1080 Ti (Windows x64)',
    'lunar-lake-laptop-win-x64': 'the Core Ultra 5 226V laptop: Arc 130V iGPU and AI Boost NPU (Windows x64)',
    'precision-5680-laptop-win-x64': 'the Dell Precision 5680 laptop: Iris Xe iGPU and RTX 3500 Ada (Windows x64)',
    'ryzen-ai-laptop-win-x64': 'the Ryzen AI 5 340 laptop: Radeon 840M iGPU and XDNA2 NPU (Windows x64)',
    'intel-arc-win-x64-openvino': 'the Intel Arc B580 + A770 desktop (Windows x64; OpenVINO, PyTorch XPU and llama.cpp SYCL runtimes)',
    'snapdragon-x-laptop-win-arm64': 'the Snapdragon X laptop: Adreno X1-45 GPU and Hexagon NPU (Windows ARM64)',
    'intel-macbook-pro-macos-x64': 'the Intel Core i9 MacBook Pro (macOS x86_64)',
    'apple-m1-macbook-air-macos-arm64': 'the Apple M1 MacBook Air (macOS 15 ARM64, 8 GB unified memory; CPU, MPS and CoreML/ANE slots)',
    'linux-cuda-server-x64': 'the Linux x64 server with two RTX 4090 and an unbound GTX 1080',
    'gb10-desktop-linux-arm64': 'a GB10 Linux ARM64 desktop (out of the current scope)',
}
ROLE_LABEL_FORM = re.compile(r'^[a-z][a-z0-9]*(?:-[a-z0-9]+)+$')

# A drive-rooted `Titanos<suffix>` or `Codex` folder with either separator, and the same roots without a trailing separator (a root
# named on its own, or followed by punctuation). The separator after the root, when present, is normalized to `/`. A root name
# carries no dots, so a truncation marker directly after a root name keeps its `...` after the label.
_LAB_ROOT = re.compile(r'(?<![A-Za-z0-9])[A-Za-z]:[\\/]+(?:Titanos[\w+-]*|Codex)(?P<sep>[\\/]+)?', re.IGNORECASE)
# `<letter>:/Users/<account>` with either separator: the account segment is the private part; what follows is kept.
_USER_ROOT = re.compile(r'(?<![A-Za-z0-9])[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s"\'<>|,;)]+', re.IGNORECASE)
# An agent work tree under a user profile: the Documents folder, the agent folder spelled in the pattern, a date and a tree segment
# (already reduced to `<user-profile>/...`, or still drive-rooted). The date and tree segments are private too; a separator after
# them is kept as `/` so the remainder of the path follows the label.
_CODEX_TREE = re.compile(r'(?:<user-profile>|(?<![A-Za-z0-9])[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s"\'<>|,;)]+)[\\/]+Documents[\\/]+Codex'
                         r'(?:[\\/]+[^\\/\s"\'<>|,;)]+[\\/]+[^\\/\s"\'<>|,;)]+)?(?P<sep>[\\/]+)?', re.IGNORECASE)
# `/home/<account>`: the account is the private part and what follows is kept. The look-behind excludes `>` and `/`, so a folder
# named `home` below a label or another folder is not a home directory. (Spelled `/home[/]` so this source never matches itself.)
_LINUX_HOME = re.compile(r'(?<![A-Za-z0-9>/])/home[/][^/\s"\'<>|,;)]+')
# `/Users/<account>` (macOS): the account is the private part and what follows is kept; the Windows `<letter>:/Users/<account>`
# form is handled above, this one has no drive letter and follows the Linux-home look-behind rules.
_MACOS_HOME = re.compile(r'(?<![A-Za-z0-9>/:])/Users[/][^/\s"\'<>|,;)]+')
# A bare drive-letter pair written as narrative (`the reachable <X>:/<Y>: roots`): the letters are the lab's drives.
_DRIVE_PAIR = re.compile(r'(?<![A-Za-z0-9])[A-Za-z]:/[A-Za-z]:(?![A-Za-z0-9])')
# The policy's windows_generated_machine_name class (tools/public_export.json detection_patterns), spelled identically so
# tests/check_public_export_hygiene.py can assert the two never drift: the demonstrated generated Windows machine-name forms, any
# ASCII letter case, with hyphen or underscore neighbours as boundaries (a run-folder label, a receipt id). Ordinary words
# (desktop), platform slot labels (win-x64-cpu), dated labels with eight digits and model identifiers do not match it.
WINDOWS_GENERATED_HOST = re.compile(r'(?ai)(?<![A-Za-z0-9])(?:(?:DESKTOP|LAPTOP)-[A-Z0-9]{7}|WIN-[A-Z0-9]{11})(?![A-Za-z0-9])')

# The private map: {"hosts": {inventory name: role label}, "phrases": {exact narrative phrase: neutral wording}, "deny": [names
# with no role label], "narrative": [coordination words the hygiene guard reports]}. Absent or unreadable, nothing is renamed.
HOSTS_FILE = Path(os.environ.get('TITANOS_LAB_HOSTS_FILE') or Path(__file__).with_name('lab_hosts.private.json'))


def load_private_map(path=HOSTS_FILE):
    """The private map's four lists, each empty when the file is absent, unreadable or lacks the key."""
    try:
        data = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    hosts = {str(k): str(v) for k, v in (data.get('hosts') or {}).items() if k and v}
    phrases = {str(k): str(v) for k, v in (data.get('phrases') or {}).items() if k}
    deny = [str(n) for n in (data.get('deny') or []) if n]
    narrative = [str(n) for n in (data.get('narrative') or []) if n]
    return {'hosts': hosts, 'phrases': phrases, 'deny': deny, 'narrative': narrative}


PRIVATE = load_private_map()
_HOST_LABELS = {name.casefold(): label for name, label in PRIVATE['hosts'].items()}
# whole-word, any ASCII case, hyphen or underscore neighbours count as boundaries; longer names first so a name that extends another
# (`<name>-0`) wins over its prefix
_HOST_RX = (re.compile(r'(?<![A-Za-z0-9])(?:' + '|'.join(re.escape(n) for n in sorted(PRIVATE['hosts'], key=len, reverse=True)) + r')(?![A-Za-z0-9])',
                       re.IGNORECASE | re.ASCII) if PRIVATE['hosts'] else None)
_PHRASES = sorted(PRIVATE['phrases'].items(), key=lambda item: len(item[0]), reverse=True)


def private_names():
    """Every name the private map denies in published text (host keys and the deny list); for the hygiene guard, never printed."""
    return sorted(set(PRIVATE['hosts']) | set(PRIVATE['deny']))


def narrative_words():
    """The coordination words and phrases the private map lists; the hygiene guard reports them, never printed."""
    return sorted(set(PRIVATE['narrative']) | set(PRIVATE['phrases']))


def neutralize_reference(text):
    """`text` with every lab records root, agent work tree, user-profile root and Linux home replaced by its label, every bare
    drive-letter pair by `lab records`, every privately mapped phrase and host by its neutral wording or role label, and every
    remaining generated Windows machine name by `<windows-host>`; non-strings are returned unchanged."""
    if not isinstance(text, str):
        return text
    text = _CODEX_TREE.sub(lambda m: LAB_RECORDS + ('/' if m.group('sep') else ''), text)
    text = _USER_ROOT.sub(USER_PROFILE, text)
    text = _LAB_ROOT.sub(lambda m: LAB_RECORDS + ('/' if m.group('sep') else ''), text)
    text = _LINUX_HOME.sub(LINUX_HOME, text)
    text = _MACOS_HOME.sub(USER_PROFILE, text)
    text = _DRIVE_PAIR.sub(DRIVE_PAIR_PHRASE, text)
    for phrase, neutral in _PHRASES:
        text = text.replace(phrase, neutral)
    if _HOST_RX is not None:
        text = _HOST_RX.sub(lambda m: _HOST_LABELS[m.group(0).casefold()], text)
    return WINDOWS_GENERATED_HOST.sub(WINDOWS_HOST, text)


def neutralize_records(value):
    """The same rule over every string inside a JSON-like tree (dicts, lists, strings); other values unchanged."""
    if isinstance(value, dict):
        return {key: neutralize_records(item) for key, item in value.items()}
    if isinstance(value, list):
        return [neutralize_records(item) for item in value]
    return neutralize_reference(value)


def records_reference(folder, name, depth=2):
    """A display reference for `name` inside the run folder `folder`: `<lab-records>/<last depth folder names>/<name>`.

    The matrix-kit RAW input names each cell's receipt for readers; the real path is resolved separately against the run folder
    (ingest.read_receipt) and never from this string. Keeping the last two folder names (`runs/<run id>` in the kit layout) makes
    the reference unique per run and comparable across hosts without recording any host location."""
    parts = [part for part in folder.resolve().parts[-depth:] if part and not re.fullmatch(r'[A-Za-z]:[\\/]*|[\\/]+', part)]
    return neutralize_reference('/'.join([LAB_RECORDS, *parts, str(name).replace('\\', '/')]))
