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

"""Wire-stable context packs and prefix deltas for incremental context transfer.

A :class:`ContextPack` is the serializable form of one prepared context: an ordered tuple of
byte chunks, a SHA-256 digest over the chunk declaration, and the exact wire byte count. A
:class:`ContextPackDelta` reuses a run of leading chunks from a base pack and carries the
target suffix, so a receiver that only holds the previous pack can rebuild the next one
without any sender-side state.

The wire encoding is intentionally canonical: UTF-8 JSON with sorted keys and the compact
separators ``,`` and ``:``, and standard Base64 for every chunk. The pack digest covers only
the schema and the chunks, while ``byte_count`` covers the complete serialized object
including itself.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Final

PACK_SCHEMA: Final = "powercontext.context-pack.v1"
DELTA_SCHEMA: Final = "powercontext.context-pack-delta.v1"
DEFAULT_CHUNK_BYTES: Final = 512
_PACK_FIELDS: Final = frozenset({"schema", "items", "digest", "byte_count"})
_DELTA_FIELDS: Final = frozenset({"schema", "base_digest", "target_digest", "prefix_items", "appended"})


class ContextPackError(ValueError):
    """Report a protocol or chunk validation failure."""


@dataclass(frozen=True, slots=True)
class ContextPack:
    """One complete context encoded as ordered byte chunks plus protocol metadata."""

    items: tuple[bytes, ...]
    digest: str
    byte_count: int

    @classmethod
    def build(cls, items: tuple[bytes, ...]) -> ContextPack:
        """Build a complete pack from chunks that already hold the exact context bytes."""

        chunks = _require_chunks(items, field_name="items")
        _, digest, byte_count = _serialize_pack(chunks)
        return cls(items=chunks, digest=digest, byte_count=byte_count)

    @classmethod
    def from_content(cls, content: str | bytes, *, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> ContextPack:
        """Split ``content`` into deterministic, UTF-8 safe chunks of at most ``chunk_bytes``."""

        data = _content_bytes(content)
        return cls.build(_chunk_bytes(data, chunk_bytes))

    def to_bytes(self) -> bytes:
        """Serialize this pack using the canonical wire JSON encoding."""

        payload, digest, byte_count = _serialize_pack(self.items)
        if digest != self.digest or byte_count != self.byte_count:
            raise ContextPackError("pack metadata does not match its chunks")
        return payload

    @classmethod
    def from_bytes(cls, data: bytes | bytearray) -> ContextPack:
        """Decode and fully verify one serialized pack."""

        payload = _parse_json_object(data)
        if frozenset(payload) != _PACK_FIELDS:
            raise ContextPackError("pack must contain exactly schema, items, digest and byte_count")
        if payload["schema"] != PACK_SCHEMA:
            raise ContextPackError("unsupported pack schema")
        chunks = _decode_chunks(payload["items"], field_name="items")
        digest = _require_digest(payload["digest"], field_name="digest")
        byte_count = _require_byte_count(payload["byte_count"], field_name="byte_count")
        _, expected_digest, expected_count = _serialize_pack(chunks)
        if digest != expected_digest:
            raise ContextPackError("pack digest does not match its chunks")
        if byte_count != len(bytes(data)) or byte_count != expected_count:
            raise ContextPackError("pack byte_count does not match its serialized size")
        return cls(items=chunks, digest=digest, byte_count=byte_count)


@dataclass(frozen=True, slots=True)
class ContextPackDelta:
    """A target pack expressed as a reusable base prefix plus appended target chunks."""

    base_digest: str
    target_digest: str
    prefix_items: int
    appended: tuple[bytes, ...] = field(default_factory=tuple)

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack) -> ContextPackDelta:
        """Reuse the longest leading run of chunks shared by ``base`` and ``target``."""

        if not isinstance(base, ContextPack) or not isinstance(target, ContextPack):
            raise ContextPackError("delta endpoints must be ContextPack values")
        shared = 0
        limit = min(len(base.items), len(target.items))
        while shared < limit and base.items[shared] == target.items[shared]:
            shared += 1
        return cls(
            base_digest=base.digest,
            target_digest=target.digest,
            prefix_items=shared,
            appended=target.items[shared:],
        )

    @property
    def byte_count(self) -> int:
        """Return the exact serialized wire size of this delta."""

        return len(self.to_bytes())

    def to_bytes(self) -> bytes:
        """Serialize this delta using the canonical wire JSON encoding."""

        if not isinstance(self.appended, tuple):
            raise ContextPackError("appended must be a tuple of bytes")
        chunks = _require_chunks(self.appended, field_name="appended")
        payload = {
            "schema": DELTA_SCHEMA,
            "base_digest": _require_digest(self.base_digest, field_name="base_digest"),
            "target_digest": _require_digest(self.target_digest, field_name="target_digest"),
            "prefix_items": _require_prefix(self.prefix_items),
            "appended": [_encode_chunk(chunk) for chunk in chunks],
        }
        return _canonical_json(payload)

    @classmethod
    def from_bytes(cls, data: bytes | bytearray) -> ContextPackDelta:
        """Decode and fully verify one serialized delta."""

        payload = _parse_json_object(data)
        if frozenset(payload) != _DELTA_FIELDS:
            raise ContextPackError(
                "delta must contain exactly schema, base_digest, target_digest, prefix_items and appended"
            )
        if payload["schema"] != DELTA_SCHEMA:
            raise ContextPackError("unsupported delta schema")
        base_digest = _require_digest(payload["base_digest"], field_name="base_digest")
        target_digest = _require_digest(payload["target_digest"], field_name="target_digest")
        prefix_items = _require_prefix(payload["prefix_items"])
        appended = _decode_chunks(payload["appended"], field_name="appended")
        return cls(base_digest=base_digest, target_digest=target_digest, prefix_items=prefix_items, appended=appended)


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta) -> ContextPack:
    """Rebuild the target pack from a base pack and one delta, verifying every digest."""

    if not isinstance(base, ContextPack):
        raise ContextPackError("base must be a ContextPack")
    if not isinstance(delta, ContextPackDelta):
        raise ContextPackError("delta must be a ContextPackDelta")
    if base.digest != delta.base_digest:
        raise ContextPackError("delta base digest does not match the supplied base pack")
    prefix_items = _require_prefix(delta.prefix_items)
    if prefix_items > len(base.items):
        raise ContextPackError("prefix_items exceeds the base pack chunk count")
    rebuilt = ContextPack.build(base.items[:prefix_items] + _require_chunks(delta.appended, field_name="appended"))
    if rebuilt.digest != delta.target_digest:
        raise ContextPackError("rebuilt pack digest does not match the delta target digest")
    return rebuilt


def pack_against_base(base: ContextPack, content: str | bytes) -> tuple[ContextPack, int]:
    """Build a target pack that reuses every leading base chunk matching ``content``.

    Returns the target pack and the reused chunk count. The reused prefix is determined by
    comparing actual chunk bytes, so it is correct for any base chunking scheme.
    """

    if not isinstance(base, ContextPack):
        raise ContextPackError("base must be a ContextPack")
    data = _content_bytes(content)
    offset = 0
    reused = 0
    for chunk in base.items:
        if data[offset : offset + len(chunk)] != chunk:
            break
        offset += len(chunk)
        reused += 1
    rest = data[offset:]
    items = base.items[:reused] + _chunk_bytes(rest, DEFAULT_CHUNK_BYTES)
    return ContextPack.build(items), reused


def _content_bytes(content: str | bytes) -> bytes:
    if isinstance(content, str):
        return content.encode("utf-8")
    if isinstance(content, (bytes, bytearray)):
        return bytes(content)
    raise ContextPackError("content must be str or bytes")


def _chunk_bytes(data: bytes, chunk_bytes: int) -> tuple[bytes, ...]:
    if isinstance(chunk_bytes, bool) or not isinstance(chunk_bytes, int):
        raise ContextPackError("chunk_bytes must be an integer")
    if chunk_bytes <= 0:
        raise ContextPackError("chunk_bytes must be positive")
    chunks: list[bytes] = []
    start = 0
    stable_end = _stable_prefix_end(data, chunk_bytes)
    if stable_end:
        # Keep the invariant preamble in its own leading chunk: it survives every re-prepare,
        # so the next delta can reuse it even when a citation revision changes right after it.
        chunks.append(data[:stable_end])
        start = stable_end
    while start < len(data):
        end = min(start + chunk_bytes, len(data))
        end = _utf8_boundary(data, start, end)
        if end <= start:
            raise ContextPackError("chunk_bytes cannot hold one complete UTF-8 character")
        chunks.append(data[start:end])
        start = end
    return tuple(chunks)


def _stable_prefix_end(data: bytes, chunk_bytes: int) -> int:
    """Find a bounded leading chunk that precedes the first volatile citation revision."""

    marker = b'"revision":'
    position = data.find(marker)
    if position < 0:
        return 0
    boundary = position + len(marker)
    if boundary <= 0 or boundary > chunk_bytes or boundary > len(data):
        return 0
    return boundary


def _utf8_boundary(data: bytes, start: int, end: int) -> int:
    """Move ``end`` back onto a UTF-8 character boundary at or after ``start``."""

    while end > start and end < len(data) and (data[end] & 0xC0) == 0x80:
        end -= 1
    return end


def _require_chunks(items: Any, *, field_name: str) -> tuple[bytes, ...]:
    if isinstance(items, (str, bytes, bytearray)) or not isinstance(items, (tuple, list)):
        raise ContextPackError(f"{field_name} must be a tuple of byte chunks")
    chunks: list[bytes] = []
    for item in items:
        if not isinstance(item, (bytes, bytearray)):
            raise ContextPackError(f"{field_name} must contain only bytes")
        chunks.append(bytes(item))
    return tuple(chunks)


def _encode_chunk(chunk: bytes) -> str:
    return base64.b64encode(chunk).decode("ascii")


def _decode_chunks(value: Any, *, field_name: str) -> tuple[bytes, ...]:
    if not isinstance(value, list):
        raise ContextPackError(f"{field_name} must be a JSON array of Base64 strings")
    chunks: list[bytes] = []
    for encoded in value:
        if not isinstance(encoded, str):
            raise ContextPackError(f"{field_name} must be a JSON array of Base64 strings")
        try:
            chunks.append(base64.b64decode(encoded.encode("ascii"), validate=True))
        except (UnicodeError, binascii.Error, ValueError) as error:
            raise ContextPackError(f"{field_name} contains an invalid Base64 chunk") from error
    return tuple(chunks)


def _require_digest(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
        raise ContextPackError(f"{field_name} must be a sha256:<hex> digest")
    hexdigest = value.removeprefix("sha256:")
    if hexdigest != hexdigest.lower():
        raise ContextPackError(f"{field_name} must use lowercase hexadecimal")
    try:
        int(hexdigest, 16)
    except ValueError as error:
        raise ContextPackError(f"{field_name} must use lowercase hexadecimal") from error
    return value


def _require_prefix(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContextPackError("prefix_items must be an integer")
    if value < 0:
        raise ContextPackError("prefix_items must not be negative")
    return value


def _require_byte_count(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContextPackError(f"{field_name} must be a non-negative integer")
    return value


def _parse_json_object(data: bytes | bytearray) -> dict[str, Any]:
    if not isinstance(data, (bytes, bytearray)):
        raise ContextPackError("wire payload must be bytes")
    try:
        decoded = bytes(data).decode("utf-8")
    except UnicodeDecodeError as error:
        raise ContextPackError("wire payload must be valid UTF-8") from error
    try:
        payload = json.loads(decoded)
    except json.JSONDecodeError as error:
        raise ContextPackError("wire payload must be valid JSON") from error
    if not isinstance(payload, dict):
        raise ContextPackError("wire payload must be a JSON object")
    return payload


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _serialize_pack(items: tuple[bytes, ...]) -> tuple[bytes, str, int]:
    encoded_items = [_encode_chunk(chunk) for chunk in items]
    digest = "sha256:" + hashlib.sha256(_canonical_json({"schema": PACK_SCHEMA, "items": encoded_items})).hexdigest()
    payload: dict[str, Any] = {"schema": PACK_SCHEMA, "items": encoded_items, "digest": digest, "byte_count": 0}
    byte_count = 0
    for _ in range(16):
        payload["byte_count"] = byte_count
        size = len(_canonical_json(payload))
        if size == byte_count:
            break
        byte_count = size
    else:  # pragma: no cover - defensive: the byte count always converges within a few steps
        raise ContextPackError("pack byte_count did not converge")
    return _canonical_json(payload), digest, byte_count


__all__ = [
    "DEFAULT_CHUNK_BYTES",
    "DELTA_SCHEMA",
    "PACK_SCHEMA",
    "ContextPack",
    "ContextPackDelta",
    "ContextPackError",
    "apply_context_pack_delta",
    "pack_against_base",
]
