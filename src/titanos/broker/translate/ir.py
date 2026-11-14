"""Canonical intermediate representation that every cross-protocol translation pivots through.

Each protocol has exactly one decoder and one encoder for requests, responses and streams (broker.translate), so every
translated cell is decode(inbound) -> IR -> encode(target). All IR records are frozen; sequences are built once by the
decoder and never mutated afterwards.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Union

from ..types import Usage

RequestKind = Literal['chat', 'embed', 'speech', 'transcribe', 'image', 'moderate', 'count_tokens']
StopReason = Literal['end_turn', 'max_tokens', 'tool_use', 'stop_sequence', 'content_filter']
STOP_REASONS = ('end_turn', 'max_tokens', 'tool_use', 'stop_sequence', 'content_filter')

_FROZEN = dict(frozen=True, slots=True)


# Content parts
@dataclass(**_FROZEN)
class Text:
    text: str


@dataclass(**_FROZEN)
class Image:
    media_type: str
    data: bytes | None = None
    url: str | None = None


@dataclass(**_FROZEN)
class Audio:
    media_type: str
    data: bytes


@dataclass(**_FROZEN)
class ToolUse:
    id: str
    name: str
    input_json: str


@dataclass(**_FROZEN)
class ToolResult:
    tool_use_id: str
    parts: list = field(default_factory=list)
    is_error: bool = False


@dataclass(**_FROZEN)
class Thinking:
    text: str
    signature: str | None = None


IRPart = Union[Text, Image, Audio, ToolUse, ToolResult, Thinking]


@dataclass(**_FROZEN)
class IRMessage:
    role: Literal['user', 'assistant', 'tool']
    parts: list = field(default_factory=list)


@dataclass(**_FROZEN)
class IRTool:
    name: str
    description: str | None = None
    parameters: dict = field(default_factory=dict)


@dataclass(**_FROZEN)
class IRToolChoice:
    mode: Literal['auto', 'required', 'none', 'tool']
    name: str | None = None


@dataclass(**_FROZEN)
class IRReasoning:
    enabled: bool = True
    effort: str | None = None
    budget_tokens: int | None = None


@dataclass(**_FROZEN)
class IRRequest:
    kind: RequestKind
    model: str
    system: list = field(default_factory=list)  # list[IRPart]
    messages: list = field(default_factory=list)  # list[IRMessage]
    tools: list = field(default_factory=list)  # list[IRTool]
    tool_choice: IRToolChoice | None = None
    sampling: dict = field(default_factory=dict)
    max_output_tokens: int | None = None
    stop: list | None = None
    response_format: dict | None = None
    reasoning: IRReasoning | None = None
    stream: bool = False
    extras_by_protocol: dict = field(default_factory=dict)
    inputs: list | None = None  # embed
    audio: bytes | None = None
    voice: str | None = None
    audio_format: str | None = None


# Stream events
@dataclass(**_FROZEN)
class Start:
    id: str
    model: str


@dataclass(**_FROZEN)
class TextDelta:
    index: int
    text: str


@dataclass(**_FROZEN)
class ThinkingDelta:
    index: int
    text: str
    signature: str | None = None


@dataclass(**_FROZEN)
class ToolCallStart:
    index: int
    id: str
    name: str


@dataclass(**_FROZEN)
class ToolArgsDelta:
    index: int
    partial_json: str


@dataclass(**_FROZEN)
class BlockStop:
    index: int


@dataclass(**_FROZEN)
class UsageEvent:
    usage: Usage


@dataclass(**_FROZEN)
class Stop:
    reason: StopReason


@dataclass(**_FROZEN)
class ErrorEvent:
    code: str
    message: str


IREvent = Union[Start, TextDelta, ThinkingDelta, ToolCallStart, ToolArgsDelta, BlockStop, UsageEvent, Stop, ErrorEvent]


@dataclass(**_FROZEN)
class IRResponse:
    id: str
    model: str
    content: list = field(default_factory=list)  # list[IRPart]
    stop: str = 'end_turn'
    usage: Usage = field(default_factory=Usage)
    extras: dict[str, Any] = field(default_factory=dict)


PART_TYPES = (Text, Image, Audio, ToolUse, ToolResult, Thinking)
EVENT_TYPES = (Start, TextDelta, ThinkingDelta, ToolCallStart, ToolArgsDelta, BlockStop, UsageEvent, Stop, ErrorEvent)

__all__ = ['Audio', 'BlockStop', 'EVENT_TYPES', 'ErrorEvent', 'IREvent', 'IRMessage', 'IRPart', 'IRReasoning', 'IRRequest',
           'IRResponse', 'IRTool', 'IRToolChoice', 'Image', 'PART_TYPES', 'RequestKind', 'STOP_REASONS', 'Start', 'Stop',
           'StopReason', 'Text', 'TextDelta', 'Thinking', 'ThinkingDelta', 'ToolArgsDelta', 'ToolCallStart', 'ToolResult',
           'ToolUse', 'UsageEvent']
