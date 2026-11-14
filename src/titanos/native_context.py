"""Per-slot context of llama-cpp registrations, sized from Ollama's options.num_ctx.

llama-server allocates its KV cache when it starts: --ctx-size is shared by --parallel slots, so each request gets
native.context_size // native.slots tokens. HiveMind's gateway (and any Ollama client) states the context it wants per
request as options.num_ctx. A num_ctx within the running per-slot context needs nothing. Above it, up to max_slot_context,
the worker that serves the request starts (or, once its requests have drained, restarts) with a per-slot context of the
next power of two at or above num_ctx, capped at the maximum, and keeps that size until it is unloaded. Above the maximum
the request is a typed 400 context_limit on every route, never a 422 and never a silent cap.

The maximum per slot is native.max_slot_context, else TITANOS_NATIVE_MAX_SLOT_CONTEXT, else the GGUF header's trained
context_length; never below the registered per-slot context. SlotContextOptions holds the two registration fields
NativeOptions inherits. No process; titanos.engine is imported lazily (titanos.config imports this module).
"""
from pathlib import Path

from pydantic import BaseModel, Field, model_serializer, model_validator

from .gguf_header import cached_model_facts

SLOT_FIELDS = ('slot_context', 'max_slot_context')


class SlotContextOptions(BaseModel):
    """The per-slot context fields of NativeOptions. Unset (None) fields are left out of model_dump, so a registration that
    does not use them is saved exactly as before."""
    slot_context: int | None = Field(default=None, ge=256, description='Tokens per slot at a cold start; with context_size '
                                     'unset, context_size becomes slot_context x slots')
    max_slot_context: int | None = Field(default=None, ge=512, description='Largest per-slot context options.num_ctx may ask '
                                         'for; default TITANOS_NATIVE_MAX_SLOT_CONTEXT, else the GGUF trained context_length')

    @model_validator(mode='after')
    def sized_slots(self):
        if self.slot_context is not None:
            total = self.slot_context * self.slots
            if 'context_size' not in self.model_fields_set:
                self.context_size = total
            elif self.context_size != total:
                raise ValueError('native.context_size must equal slot_context x slots when both are set')
        if self.max_slot_context is not None and self.max_slot_context < self.context_size // self.slots:
            raise ValueError('native.max_slot_context is below the registered per-slot context (context_size / slots)')
        return self

    @model_serializer(mode='wrap')
    def _without_unset_slots(self, handler):
        data = handler(self)
        for name in SLOT_FIELDS:
            if isinstance(data, dict) and data.get(name) is None:
                data.pop(name, None)
        return data


def registered_slot_context(native):
    """Tokens per slot as registered: native.context_size over native.slots."""
    return native.context_size // native.slots


def trained_context(spec):
    """The GGUF header's context_length for spec's model file, or None when the file or the key is unreadable."""
    try:
        return cached_model_facts(Path(spec.path) / spec.native.model_file)['context_length']
    except (OSError, ValueError, KeyError, TypeError):
        return None


def max_slot_context(spec, settings=None):
    """The largest per-slot context options.num_ctx may ask spec's runtime for (see the module docstring)."""
    if settings is None:
        from .settings import current_settings
        settings = current_settings()
    native = spec.native
    ceiling = native.max_slot_context or settings.native_max_slot_context or trained_context(spec)
    return max(registered_slot_context(native), ceiling or 0)


def next_power_of_two(value):
    return 1 << max(0, value - 1).bit_length()


def context_limit(model, num_ctx, maximum, spec):
    from .engine import EngineError
    native = spec.native
    return EngineError(f'options.num_ctx {num_ctx} exceeds the largest per-slot context {maximum} of {model} (native.context_size '
                       f'{native.context_size} over {native.slots} slot(s); the maximum is native.max_slot_context, '
                       'TITANOS_NATIVE_MAX_SLOT_CONTEXT or the GGUF trained context_length); lower num_ctx', 400, 'context_limit')


def needed_slot_context(spec, num_ctx, model=None, settings=None):
    """The per-slot context a runtime needs for num_ctx: None when num_ctx is None or within the registered per-slot context,
    else the next power of two at or above num_ctx, capped at max_slot_context. EngineError 400 context_limit above the
    maximum. spec is a llama-cpp ModelSpec (its native options)."""
    if num_ctx is None or num_ctx <= registered_slot_context(spec.native):
        return None
    maximum = max_slot_context(spec, settings)
    if num_ctx > maximum:
        raise context_limit(model or spec.path, num_ctx, maximum, spec)
    return min(next_power_of_two(num_ctx), maximum)


__all__ = ['SlotContextOptions', 'context_limit', 'max_slot_context', 'needed_slot_context', 'next_power_of_two',
           'registered_slot_context', 'trained_context']
