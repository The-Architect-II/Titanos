"""Per-model request rules of the Anthropic Messages API (data plus the pure functions that apply them).

Claude models differ in which request fields they accept, and a field a model refuses is an upstream 400 for the whole
request. The translated paths (OpenAI Chat, Responses, Gemini and Ollama clients and Studio reaching an Anthropic
upstream) and the advisor's reasoning injection (R19) therefore shape every body by the target model's row:

- thinking: 'none' (no extended thinking), 'budget' ({type: enabled, budget_tokens}; the turn holding a pending tool_use
  must start with a signed thinking block), 'adaptive' ({type: adaptive}; omitting thinking runs without it;
  budget_tokens is refused on 4.7+), 'default_on' (omitting thinking runs adaptive; {type: disabled} is accepted at effort
  high or below) and 'always' (thinking cannot be turned off; {type: disabled} and budget_tokens are a 400);
- efforts: the output_config.effort levels the model accepts (an empty tuple: the field is refused);
- sampling: 'allowed', 'exclusive' (temperature and top_p may not be sent together), 'removed' (temperature, top_p and
  top_k are a 400) or 'default_only' (non-default values are a 400, so Titanos never sends them);
- forced_tool_choice: whether tool_choice any / tool is accepted (Opus 5.5, Sonnet 5.5 and Fable 5.1 refuse it);
- structured_outputs: whether output_config.format (json_schema) is accepted;
- display_default_omitted: thinking blocks carry empty text unless thinking.display is 'summarized' (4.7 and later);
- disable: the spelling that turns thinking off where it can be turned off ('disabled', or 'between_tools' on Sonnet 5.5).

Rows are fnmatch globs over the upstream model id (a dated suffix matches its alias row); the first match wins. A model
no row names (a release newer than this table) gets the most conservative current rule (DEFAULT_RULES: thinking always on,
no sampling, no forced tool choice), which every current model accepts; the operator can still pin the provider's
behaviour per request through the native /v1/messages passthrough, which forwards every field untouched.

Facts transcribed from Anthropic's public model and migration documentation (cached 2026-09-25).
"""
from __future__ import annotations

import fnmatch
import math
from dataclasses import dataclass

ALL_EFFORTS = ('low', 'medium', 'high', 'xhigh', 'max')
EFFORT_ORDER = {name: index for index, name in enumerate(ALL_EFFORTS)}
# thinking budgets for budget-mode models, per requested effort (the IR effort vocabulary of every inbound protocol)
EFFORT_BUDGET = {'minimal': 1024, 'low': 1024, 'medium': 4096, 'high': 16384, 'xhigh': 32768, 'max': 65536}
MIN_THINKING_BUDGET = 1024
# the injected max_tokens floor for models whose thinking cannot be turned off: thinking shares max_tokens, so the
# broker's general default (4096) ends harder turns with stop_reason max_tokens
ALWAYS_THINKING_MIN_MAX_TOKENS = 16000
# the largest Claude context window; no input above it is ever billed (the API refuses it), so it bounds reservations
MAX_CONTEXT_TOKENS = 1_000_000
# per-image allowance of the input estimate (the largest image token count of current models, high-resolution vision)
IMAGE_TOKEN_ALLOWANCE = 4800


@dataclass(frozen=True)
class ModelRules:
    family: str
    thinking: str  # 'none' | 'budget' | 'adaptive' | 'default_on' | 'always'
    efforts: tuple = ()
    sampling: str = 'allowed'  # 'allowed' | 'exclusive' | 'removed' | 'default_only'
    forced_tool_choice: bool = True
    structured_outputs: bool = True
    display_default_omitted: bool = False
    disable: str | None = 'disabled'


_FIVE = ALL_EFFORTS
_ROWS: tuple[tuple[tuple[str, ...], ModelRules], ...] = (
    (('claude-fable-5-1*', 'claude-mythos-5-1*'),
     ModelRules('fable-5.1', 'always', _FIVE, 'removed', False, True, True, None)),
    (('claude-fable-5*', 'claude-mythos-5*', 'claude-mythos-preview*'),
     ModelRules('fable-5', 'always', _FIVE, 'removed', True, True, True, None)),
    (('claude-opus-5-5*',), ModelRules('opus-5.5', 'always', _FIVE, 'removed', False, True, True, None)),
    (('claude-sonnet-5-5*',), ModelRules('sonnet-5.5', 'always', _FIVE, 'default_only', False, True, True,
                                         'between_tools')),
    (('claude-opus-5*',), ModelRules('opus-5', 'default_on', _FIVE, 'removed', True, True, True)),
    (('claude-sonnet-5*',), ModelRules('sonnet-5', 'default_on', _FIVE, 'default_only', True, True, True)),
    (('claude-opus-4-8*', 'claude-opus-4-7*'), ModelRules('opus-4.7', 'adaptive', _FIVE, 'removed', True, True, True)),
    (('claude-opus-4-6*', 'claude-sonnet-4-6*'),
     ModelRules('4.6', 'adaptive', ('low', 'medium', 'high', 'max'), 'allowed', True, True, False)),
    (('claude-opus-4-5*',), ModelRules('opus-4.5', 'budget', ('low', 'medium', 'high'), 'exclusive')),
    (('claude-sonnet-4-5*', 'claude-haiku-4-5*', 'claude-opus-4-1*'),
     ModelRules('4.5', 'budget', (), 'exclusive')),
    (('claude-opus-4-0*', 'claude-opus-4-2025*', 'claude-sonnet-4-0*', 'claude-sonnet-4-2025*',
      'claude-3-7-sonnet*'), ModelRules('4.0', 'budget', (), 'allowed', True, False)),
    (('claude-3*', 'claude-2*', 'claude-instant*'), ModelRules('3', 'none', (), 'allowed', True, False)),
)
DEFAULT_RULES = ModelRules('unknown', 'always', _FIVE, 'removed', False, True, True, None)


def bare_model(model: str | None) -> str:
    """The model id without a provider prefix ('anthropic/' routers, 'anthropic.' cloud catalogs) or a '@' version."""
    name = (model or '').strip()
    for prefix in ('anthropic/', 'anthropic.'):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name.split('@', 1)[0]


def rules_for(model: str | None) -> ModelRules:
    name = bare_model(model)
    for globs, rules in _ROWS:
        if any(fnmatch.fnmatchcase(name, glob) for glob in globs):
            return rules
    return DEFAULT_RULES


def effort_for(rules: ModelRules, effort: str | None) -> str | None:
    """The supported output_config.effort for an IR effort ('minimal' -> 'low'; a level the model lacks -> the nearest
    lower one it has), or None (no effort to send)."""
    if not rules.efforts or not isinstance(effort, str):
        return None
    wanted = 'low' if effort == 'minimal' else effort
    if wanted not in EFFORT_ORDER:
        return None
    for name in reversed(ALL_EFFORTS[:EFFORT_ORDER[wanted] + 1]):
        if name in rules.efforts:
            return name
    return None


def effort_from_budget(budget) -> str | None:
    """An effort for a thinking token budget (Gemini thinkingBudget, an Anthropic budget sent to an adaptive model)."""
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 0:
        return None
    if budget <= 2048:
        return 'low'
    if budget <= 8192:
        return 'medium'
    if budget <= 32768:
        return 'high'
    return 'xhigh'


def _note(notes, text):
    if notes is not None and text not in notes:
        notes.append(text)


@dataclass(frozen=True)
class ThinkingPlan:
    thinking: dict | None = None  # the thinking field to send, or None (omit)
    effort: str | None = None  # output_config.effort to send, or None
    thinking_on: bool = False  # Titanos enabled thinking explicitly (sampling must then be dropped on 4.6 and older)


def plan_thinking(rules: ModelRules, reasoning, *, max_tokens: int | None, forced_tool: bool,
                  unsigned_tool_turn: bool, notes: list | None) -> ThinkingPlan:
    """The thinking and effort fields for an IR reasoning request (None: the client asked for nothing)."""
    if reasoning is None:
        return ThinkingPlan()
    enabled = bool(getattr(reasoning, 'enabled', True))
    effort = getattr(reasoning, 'effort', None)
    budget = getattr(reasoning, 'budget_tokens', None)
    if effort == 'none' or budget == 0:
        enabled = False
    if not enabled:
        if rules.thinking == 'default_on':
            return ThinkingPlan(thinking={'type': 'disabled'})
        if rules.thinking == 'always':
            if rules.disable == 'between_tools':
                return ThinkingPlan(thinking={'type': 'between_tools'})
            _note(notes, 'kept:thinking (always on for this model)')
            return ThinkingPlan(effort=effort_for(rules, 'low'))
        return ThinkingPlan()  # off by default
    if rules.thinking == 'none':
        _note(notes, 'dropped:reasoning (the model has no extended thinking)')
        return ThinkingPlan()
    if rules.thinking == 'budget':
        if forced_tool:
            _note(notes, 'dropped:thinking (forced tool_choice)')
            return ThinkingPlan()
        value = budget if isinstance(budget, int) and not isinstance(budget, bool) and budget > 0 else \
            EFFORT_BUDGET.get(effort or 'medium', 4096)
        if not isinstance(max_tokens, int) or max_tokens <= MIN_THINKING_BUDGET:
            _note(notes, 'dropped:thinking (max_tokens too small)')
            return ThinkingPlan()
        if unsigned_tool_turn:
            _note(notes, 'dropped:thinking (unsigned tool turn)')
            return ThinkingPlan()
        value = max(MIN_THINKING_BUDGET, min(int(value), max_tokens - 1))
        return ThinkingPlan(thinking={'type': 'enabled', 'budget_tokens': value}, thinking_on=True)
    level = effort_for(rules, effort if effort not in (None, 'none') else effort_from_budget(budget))
    if isinstance(budget, int) and budget > 0 and not effort:
        _note(notes, 'mapped:budget_tokens->effort')
    if rules.thinking == 'adaptive' and forced_tool:
        _note(notes, 'dropped:thinking (forced tool_choice)')
        return ThinkingPlan(effort=level)
    if rules.thinking in ('default_on', 'always') and forced_tool:
        return ThinkingPlan(effort=level)  # thinking stays the model default; the forced call decides the turn
    thinking = {'type': 'adaptive'}
    if rules.display_default_omitted:
        thinking['display'] = 'summarized'  # a client that asked for reasoning gets readable thinking text
    return ThinkingPlan(thinking=thinking, effort=level, thinking_on=rules.thinking == 'adaptive')


def shape_sampling(rules: ModelRules, sampling: dict, *, thinking_on: bool, notes: list | None) -> dict:
    """The sampling fields this model accepts (each dropped one is noted)."""
    out = {}
    for name, value in sampling.items():
        if name not in ('temperature', 'top_p', 'top_k'):
            _note(notes, f'dropped:{name}')
            continue
        if rules.sampling in ('removed', 'default_only'):
            _note(notes, f'dropped:{name} (the model refuses sampling parameters)')
            continue
        if thinking_on:
            _note(notes, f'dropped:{name} (thinking)')
            continue
        if name == 'temperature' and isinstance(value, (int, float)) and value > 1:
            _note(notes, 'clamped:temperature')
            value = 1.0
        out[name] = value
    if rules.sampling == 'exclusive' and 'temperature' in out and 'top_p' in out:
        out.pop('top_p')
        _note(notes, 'dropped:top_p (the model takes temperature or top_p, not both)')
    return out


def shape_tool_choice(rules: ModelRules, choice: dict, *, notes: list | None) -> dict:
    """tool_choice any / tool degrade to auto on models that refuse forced tool use (disable_parallel_tool_use kept)."""
    if choice.get('type') in ('any', 'tool') and not rules.forced_tool_choice:
        _note(notes, 'degraded:tool_choice=auto (the model refuses forced tool use)')
        out = {'type': 'auto'}
        if choice.get('disable_parallel_tool_use'):
            out['disable_parallel_tool_use'] = True
        return out
    return choice


def inject_thinking(document: dict, mode: str, model: str | None, notes: list) -> dict:
    """R19 on the native /v1/messages passthrough: the advisor's reasoning mode in a body that sets neither thinking nor
    output_config.effort, shaped for the target model (budget_tokens on budget models only when max_tokens leaves room,
    no sampling override is set and no tool loop lacks its signed thinking; adaptive thinking plus effort on 4.6 and
    later; effort alone where thinking is always on)."""
    from .anthropic_messages import _unsigned_tool_turn
    out = dict(document)
    config = out.get('output_config') if isinstance(out.get('output_config'), dict) else {}
    if 'thinking' in out or 'effort' in config:
        return out
    rules = rules_for(model)
    max_tokens = out.get('max_tokens')
    max_tokens = max_tokens if isinstance(max_tokens, int) and not isinstance(max_tokens, bool) else None
    messages = out.get('messages') if isinstance(out.get('messages'), list) else []
    choice = out.get('tool_choice') if isinstance(out.get('tool_choice'), dict) else {}
    forced = choice.get('type') in ('any', 'tool')
    if rules.thinking == 'budget' and any(key in out for key in ('temperature', 'top_k', 'top_p')):
        return out
    if rules.thinking == 'adaptive' and any(key in out for key in ('temperature', 'top_k')):
        return out
    from .ir import IRReasoning
    plan = plan_thinking(rules, IRReasoning(enabled=True, effort=mode), max_tokens=max_tokens, forced_tool=forced,
                         unsigned_tool_turn=_unsigned_tool_turn(messages), notes=None)
    if plan.thinking is not None:
        out['thinking'] = plan.thinking
        notes.append('injected:thinking')
    if plan.effort is not None:
        out['output_config'] = dict(config, effort=plan.effort)
        notes.append('injected:effort')
    return out


# -- input estimate ------------------------------------------------------------------------------------------------------
def _media_blocks(value, depth: int = 0):
    if depth > 32:
        return
    if isinstance(value, dict):
        source = value.get('source')
        if value.get('type') in ('image', 'document') and isinstance(source, dict) and source.get('type') == 'base64':
            yield value.get('type'), source
        for item in value.values():
            if isinstance(item, (dict, list)):
                yield from _media_blocks(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            yield from _media_blocks(item, depth + 1)


def estimate_input_tokens(body, raw_length: int) -> int:
    """The reservation estimate for an Anthropic Messages body: ceil(bytes / 3) as for every surface, except that each
    base64 image counts IMAGE_TOKEN_ALLOWANCE tokens instead of its encoded bytes, a base64 document counts its decoded
    bytes / 3, and the total never exceeds MAX_CONTEXT_TOKENS (the API refuses a larger input, so it is never billed)."""
    length = max(0, int(raw_length))
    extra = 0
    if isinstance(body, dict):
        for kind, source in _media_blocks({'messages': body.get('messages'), 'system': body.get('system')}):
            data = source.get('data')
            if not isinstance(data, str):
                continue
            length -= len(data)
            extra += IMAGE_TOKEN_ALLOWANCE if kind == 'image' else math.ceil(len(data) * 3 / 4 / 3)
    tokens = math.ceil(max(0, length) / 3) + extra
    return min(tokens, MAX_CONTEXT_TOKENS)


__all__ = ['ALL_EFFORTS', 'ALWAYS_THINKING_MIN_MAX_TOKENS', 'DEFAULT_RULES', 'EFFORT_BUDGET', 'IMAGE_TOKEN_ALLOWANCE',
           'MAX_CONTEXT_TOKENS', 'MIN_THINKING_BUDGET', 'ModelRules', 'ThinkingPlan', 'bare_model', 'effort_for',
           'effort_from_budget', 'estimate_input_tokens', 'inject_thinking', 'plan_thinking', 'rules_for',
           'shape_sampling', 'shape_tool_choice']
