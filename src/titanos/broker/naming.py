"""Model-reference grammar. There is no vendor-prefix guessing.

A qualified ref is '<upstream_model>@<upstream_id>', split on the LAST '@', so upstream model ids keep their own '/', ':'
and '@' characters ('accounts/fireworks/models/x@fireworks', 'meta-llama/Llama-3.1-8B-Instruct:fastest@hf_router',
'gpt-oss:120b@ollama_cloud'). Local registry names (config.py: [A-Za-z0-9][A-Za-z0-9._/-]{0,127}) exclude '@' and ':',
so a qualified ref can never collide with a local name.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from .errors import BrokerError

REF_MAX = 256
UPSTREAM_ID_PATTERN = r'[a-z][a-z0-9_-]{0,31}'
UPSTREAM_ID_RE = re.compile(UPSTREAM_ID_PATTERN)
LOCAL_NAME_RE = re.compile(r'[A-Za-z0-9][A-Za-z0-9._/-]{0,127}')  # config.py Config.valid_names
# 'store' would make the scope cloud:store ambiguous with cloud:<upstream_id>
RESERVED_UPSTREAM_IDS = frozenset({'store'})


@dataclass(frozen=True)
class ModelRef:
    kind: Literal['plain', 'qualified']
    model: str
    upstream: str | None = None

    def __str__(self) -> str:
        return format_ref(self.model, self.upstream) if self.kind == 'qualified' else self.model


def valid_upstream_id(text: str) -> bool:
    return isinstance(text, str) and UPSTREAM_ID_RE.fullmatch(text) is not None and text not in RESERVED_UPSTREAM_IDS


def is_local_name(text: str) -> bool:
    return isinstance(text, str) and LOCAL_NAME_RE.fullmatch(text) is not None


def check_model_text(text) -> str:
    """400 model_required for a missing or blank model; 400 invalid_request beyond REF_MAX characters."""
    if not isinstance(text, str) or not text.strip():
        raise BrokerError('A model is required', 'model_required')
    if len(text) > REF_MAX:
        raise BrokerError(f'Model reference exceeds {REF_MAX} characters', 'invalid_request')
    return text


def parse_ref(text: str) -> ModelRef:
    """Split on the last '@'. The part after it must be an upstream id and the part before it non-empty; otherwise the
    whole text is a plain name (which then resolves as a local, virtual or alias name, or 404s)."""
    check_model_text(text)
    model, at, upstream = text.rpartition('@')
    if at and model and valid_upstream_id(upstream):
        return ModelRef('qualified', model, upstream)
    return ModelRef('plain', text, None)


def format_ref(model: str, upstream: str) -> str:
    if not isinstance(model, str) or not model:
        raise ValueError('A qualified ref needs an upstream model id')
    if not valid_upstream_id(upstream):
        raise ValueError(f'Invalid upstream id: {upstream!r}')
    ref = model + '@' + upstream
    if len(ref) > REF_MAX:
        raise ValueError(f'Model reference exceeds {REF_MAX} characters')
    return ref


def expand_compat_suffix(name: str, compat) -> str | None:
    """Optional HiveMind-style suffix aliases ({'_ollama_cloud': 'ollama_cloud', '_cloud': 'openai'}), matched LONGEST
    suffix first, expanded only into a qualified ref (which re-enters qualified-ref resolution). None when nothing matches.
    compat is a broker.config.Compat, or the suffix mapping itself."""
    aliases = getattr(compat, 'hivemind_suffix_aliases', compat) if compat is not None else None
    if not aliases or not isinstance(name, str):
        return None
    if not isinstance(aliases, Mapping):
        raise TypeError('compat suffix aliases must be a mapping')
    for suffix in sorted(aliases, key=lambda value: (-len(value), value)):
        if suffix and name.endswith(suffix) and len(name) > len(suffix):
            return format_ref(name[:-len(suffix)], aliases[suffix])
    return None


__all__ = ['LOCAL_NAME_RE', 'ModelRef', 'REF_MAX', 'RESERVED_UPSTREAM_IDS', 'UPSTREAM_ID_PATTERN', 'UPSTREAM_ID_RE',
           'check_model_text', 'expand_compat_suffix', 'format_ref', 'is_local_name', 'parse_ref', 'valid_upstream_id']
