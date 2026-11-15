"""Which filesystem links Titanos refuses on model, bundle and staging paths. Standard library only.

Those paths must not pass through links or reparse points, so a checked path cannot be redirected afterwards. The one
exception is a POSIX ancestor only root can change: a symlink owned by root inside a root-owned directory that group and
others cannot write, as macOS /var, /tmp and /etc -> /private/... are (every macOS temporary directory sits under one).
Windows junctions and symlinks, the final path component itself, and every other link stay refused.
tools/music_runtime/ace_adapter.py carries its own copy of os_alias (the runtime cannot import titanos)."""
import os
import stat


def is_link(info):
    """lstat result of a symlink or a Windows reparse point (junction)."""
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, 'st_file_attributes', 0) & 0x400)


def is_link_path(path):
    """True when path exists and is a symlink or a reparse point. Path.is_junction and os.path.isjunction need Python
    3.12; lstat's reparse-point attribute also works on 3.11."""
    try:
        return is_link(os.lstat(path))
    except OSError:
        return False


def os_alias(path, info):
    """True for a POSIX symlink owned by root inside a root-owned directory that group and others cannot write."""
    if os.name == 'nt' or not stat.S_ISLNK(info.st_mode) or info.st_uid != 0:
        return False
    try:
        parent = os.lstat(os.path.dirname(os.fspath(path)) or os.sep)
    except OSError:
        return False
    return parent.st_uid == 0 and stat.S_ISDIR(parent.st_mode) and not parent.st_mode & (stat.S_IWGRP | stat.S_IWOTH)


def refused_link(path, *, ancestor):
    """True when path (which must exist) is a link Titanos refuses; ancestor=True allows OS-owned aliases."""
    info = os.lstat(path)
    return is_link(info) and not (ancestor and os_alias(path, info))


__all__ = ['is_link', 'is_link_path', 'os_alias', 'refused_link']
