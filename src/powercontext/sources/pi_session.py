# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Capture the immutable visible events of one Pi session branch."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, JsonValue

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources import (
    TEXT_EVIDENCE_PROJECTION_KEY,
    AdapterSourceDefinition,
    ConnectorRunCompletion,
    ConnectorRunSession,
    ConnectorRunStatus,
    Source,
    SourceMaterialization,
    TextEvidence,
)

PI_EVENT_SOURCE_NAME: Final = "pi-event"
PI_EVENT_DEFINITION_VERSION: Final = "1"
PI_SESSION_CONNECTOR_NAME: Final = "pi-session-history"
PI_SESSION_CONNECTOR_VERSION: Final = "1"

_SESSION_TYPE: Final = "session"
_SESSION_FORMAT_VERSION: Final = 3
_CHECKPOINT_FORMAT_VERSION: Final = 1
_UNKNOWN_TOOL: Final = "unknown"
_TEXTUAL_EVENT_TYPES: Final = frozenset({"compaction", "branch_summary"})
_IDENTITY_SEPARATOR: Final = "\x00"


class PiSessionInput(BaseModel):
    """One UTF-8 Pi session archive and the leaf whose ancestry to select."""

    model_config = ConfigDict(frozen=True)

    content: bytes
    leaf_id: str | None = None


class PiEventInput(BaseModel):
    """One visible Pi session event together with its complete original record."""

    model_config = ConfigDict(frozen=True)

    session_id: str
    event_id: str
    parent_id: str | None = None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


class PiEventSource(Source):
    """Captured Pi event whose identity is only its session and event id."""

    session_id: str
    event_id: str
    parent_id: str | None = None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class _ParsedSession:
    """Fully validated session archive: selected branch plus the complete node graph."""

    session_id: str
    events: tuple[PiEventInput, ...]
    records: dict[str, dict[str, Any]]


def parse_pi_session(
    value: PiSessionInput | bytes | bytearray | memoryview | str,
    /,
) -> tuple[PiEventInput, ...]:
    """Return the visible events on the selected branch, ordered from root to leaf."""

    return _parse_session(value).events


class _PiEventAdapter:
    """Resolve and read the immutable captured Pi event Source."""

    input_class = PiEventInput
    name = PI_EVENT_SOURCE_NAME
    source_class = PiEventSource

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        return PiEventSource(
            name=_event_identity(value.session_id, value.event_id),
            materialization=SourceMaterialization.CAPTURED,
            session_id=value.session_id,
            event_id=value.event_id,
            parent_id=value.parent_id,
            event_type=value.event_type,
            role=value.role,
            content=value.content,
            record=value.record,
        )

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        if not isinstance(source, PiEventSource):
            raise TypeError("Pi event Sources must be read through their own adapter")  # noqa: TRY003
        return PiEventInput(
            session_id=source.session_id,
            event_id=source.event_id,
            parent_id=source.parent_id,
            event_type=source.event_type,
            role=source.role,
            content=source.content,
            record=source.record,
        )


class _PiEventTextEvidenceProjection:
    """Expose one captured Pi event as canonical textual evidence."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PiEventSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PiEventSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PI_EVENT_SOURCE_NAME,
            source_id=source.name,
            content=source.content,
            metadata={
                "session_id": source.session_id,
                "event_id": source.event_id,
                "parent_id": source.parent_id,
                "event_type": source.event_type,
                "role": source.role,
            },
        )


PI_EVENT_SOURCE_DEFINITION = AdapterSourceDefinition(
    _PiEventAdapter(),
    version=PI_EVENT_DEFINITION_VERSION,
    projections=(_PiEventTextEvidenceProjection(),),
)


class PiSessionConnector:
    """Import the visible events of one Pi session branch incrementally."""

    name = PI_SESSION_CONNECTOR_NAME
    version = PI_SESSION_CONNECTOR_VERSION
    source_definitions = frozenset({PI_EVENT_SOURCE_NAME})

    def __init__(self, path: str | Path, leaf_id: str | None = None) -> None:
        self._path = Path(path)
        self._leaf_id = leaf_id

    @property
    def path(self) -> Path:
        return self._path

    @property
    def leaf_id(self) -> str | None:
        return self._leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        """Fully re-parse the archive, then submit only newly visible branch events."""

        parsed = _parse_session(PiSessionInput(content=self._path.read_bytes(), leaf_id=self._leaf_id))
        imported = _imported_records(session.checkpoint, parsed.session_id)
        for event_id, digest in imported.items():
            record = parsed.records.get(event_id)
            if record is None:
                raise ValueError(f"imported Pi event {event_id!r} disappeared from the session archive")
            if _record_digest(record) != digest:
                raise ValueError(f"imported Pi event {event_id!r} was rewritten after it was captured")
        updated = dict(imported)
        for event in parsed.events:
            if event.event_id in updated:
                continue
            await session.submit(
                _event_identity(event.session_id, event.event_id),
                PI_EVENT_SOURCE_NAME,
                event,
            )
            updated[event.event_id] = _record_digest(event.record)
        return ConnectorRunCompletion(
            status=ConnectorRunStatus.COMPLETE,
            checkpoint={
                "version": _CHECKPOINT_FORMAT_VERSION,
                "session_id": parsed.session_id,
                "imported": updated,
            },
        )


def _parse_session(value: object, /) -> _ParsedSession:
    content, leaf_id = _coerce_input(value)
    records = _json_records(content)
    if not records:
        raise ValueError("Pi session archive must begin with a session header")
    header = records[0]
    session_id = _validate_header(header)

    known: set[str] = {session_id}
    nodes: dict[str, dict[str, Any]] = {}
    parents: dict[str, str | None] = {}
    order: list[str] = []
    for node in records[1:]:
        event_id = node.get("id")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("every Pi session node requires a non-empty string id")
        if event_id in known:
            raise ValueError(f"duplicate Pi session node id {event_id!r}")
        parent_id = node.get("parentId")
        if parent_id is not None and (not isinstance(parent_id, str) or parent_id not in known):
            raise ValueError(f"Pi session node {event_id!r} references a missing parent")
        known.add(event_id)
        nodes[event_id] = node
        parents[event_id] = parent_id
        order.append(event_id)

    selected = leaf_id if leaf_id is not None else (order[-1] if order else None)
    if selected is None:
        return _ParsedSession(session_id=session_id, events=(), records=nodes)
    if selected not in nodes:
        raise ValueError(f"selected Pi session leaf {selected!r} does not exist")

    chain: list[str] = []
    visited: set[str] = set()
    current: str | None = selected
    while current is not None and current in nodes:
        if current in visited:
            raise ValueError("Pi session graph must not contain a cycle")
        visited.add(current)
        chain.append(current)
        current = parents[current]
    chain.reverse()

    events: list[PiEventInput] = []
    for event_id in chain:
        event = _build_event(session_id, nodes[event_id])
        if event is not None:
            events.append(event)
    return _ParsedSession(session_id=session_id, events=tuple(events), records=nodes)


def _coerce_input(value: object) -> tuple[bytes, str | None]:
    if isinstance(value, PiSessionInput):
        return value.content, value.leaf_id
    if isinstance(value, str):
        return value.encode("utf-8"), None
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value), None
    raise TypeError("parse_pi_session expects a PiSessionInput")  # noqa: TRY003


def _json_records(content: bytes) -> list[dict[str, Any]]:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("Pi session content must be valid UTF-8") from error
    records: list[dict[str, Any]] = []
    for number, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Pi session line {number} is not valid JSON") from error
        if not isinstance(record, dict):
            raise ValueError(f"Pi session line {number} must be a JSON object")
        records.append(record)
    return records


def _validate_header(header: dict[str, Any]) -> str:
    if header.get("type") != _SESSION_TYPE:
        raise ValueError("Pi session archive must begin with a session header")
    version = header.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version != _SESSION_FORMAT_VERSION:
        raise ValueError(f"Pi session header must declare version {_SESSION_FORMAT_VERSION}")
    session_id = header.get("id")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("Pi session header requires a non-empty string id")
    return session_id


def _build_event(session_id: str, record: dict[str, Any]) -> PiEventInput | None:
    event_type = record.get("type")
    if not isinstance(event_type, str):
        return None
    if event_type == "message":
        message = record.get("message")
        if not isinstance(message, dict):
            return None
        role = message.get("role")
        if not isinstance(role, str) or not role:
            return None
        body = _message_body(role, message)
    elif event_type in _TEXTUAL_EVENT_TYPES:
        role = event_type
        summary = record.get("summary")
        body = summary if isinstance(summary, str) else ""
    elif event_type == "custom_message":
        role = "custom"
        body = _visible_text(record.get("content"), tool_calls=False)
    else:
        return None
    if not body:
        return None
    return PiEventInput(
        session_id=session_id,
        event_id=record["id"],
        parent_id=record.get("parentId"),
        event_type=event_type,
        role=role,
        content=f"{role}: {body}",
        record=record,
    )


def _message_body(role: str, message: dict[str, Any]) -> str:
    if role == "bashExecution":
        command = message.get("command")
        output = message.get("output")
        return f"$ {command if isinstance(command, str) else ''}\n{output if isinstance(output, str) else ''}"
    if role == "toolResult":
        return (
            f"[tool_result {_tool_name(message.get('toolName'))} "
            f"{_tool_name(message.get('toolCallId'))}] {_visible_text(message.get('content'))}"
        )
    content = message.get("content")
    if content is None:
        summary = message.get("summary")
        return summary if isinstance(summary, str) else ""
    return _visible_text(content)


def _visible_text(content: object, *, tool_calls: bool = True) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
        elif tool_calls and block_type == "toolCall":
            parts.append(_tool_call_text(block))
    return "\n".join(parts)


def _tool_call_text(block: dict[str, Any]) -> str:
    return (
        f"[tool_call {_tool_name(block.get('name'))} "
        f"{_tool_name(block.get('id'))}] {_compact_arguments(block.get('arguments'))}"
    )


def _tool_name(value: object) -> str:
    return value if isinstance(value, str) and value else _UNKNOWN_TOOL


def _compact_arguments(arguments: object) -> str:
    if arguments is None:
        arguments = {}
    try:
        return json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return "{}"


def _event_identity(session_id: str, event_id: str) -> str:
    """Derive the collision-free stable Source identity of one session event."""

    readable = (
        f"{PI_EVENT_SOURCE_NAME}:e:"
        f"{quote(session_id, safe='', errors='surrogatepass')}:"
        f"{quote(event_id, safe='', errors='surrogatepass')}"
    )
    if len(readable) <= MAX_SOURCE_ID_LENGTH:
        return readable
    digest = hashlib.sha256(f"{session_id}{_IDENTITY_SEPARATOR}{event_id}".encode("utf-8", "surrogatepass"))
    return f"{PI_EVENT_SOURCE_NAME}:h:{digest.hexdigest()}"


def _record_digest(record: dict[str, Any]) -> str:
    try:
        encoded = json.dumps(record, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise ValueError("Pi session records must be canonical JSON") from error
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _imported_records(checkpoint: JsonValue | None, session_id: str) -> dict[str, str]:
    if not isinstance(checkpoint, dict):
        return {}
    if checkpoint.get("version") != _CHECKPOINT_FORMAT_VERSION:
        return {}
    if checkpoint.get("session_id") != session_id:
        return {}
    imported = checkpoint.get("imported")
    if not isinstance(imported, dict):
        return {}
    return {
        event_id: digest
        for event_id, digest in imported.items()
        if isinstance(event_id, str) and isinstance(digest, str) and digest
    }


__all__ = [
    "PI_EVENT_DEFINITION_VERSION",
    "PI_EVENT_SOURCE_DEFINITION",
    "PI_EVENT_SOURCE_NAME",
    "PI_SESSION_CONNECTOR_NAME",
    "PI_SESSION_CONNECTOR_VERSION",
    "PiEventInput",
    "PiEventSource",
    "PiSessionConnector",
    "PiSessionInput",
    "parse_pi_session",
]
