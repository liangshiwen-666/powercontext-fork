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

"""Relational storage for Scope evidence-dependency nodes and lifecycle events."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncConnection

from powercontext.artifacts import ArtifactRef, MemoryCitation
from powercontext.builtin.persistence.tables import (
    SOURCE_LIFECYCLE_EVENTS_TABLE,
    SOURCE_LIFECYCLE_NODES_TABLE,
)
from powercontext.sources import SourceRef

LifecycleNodeKind: TypeAlias = Literal["source", "memory", "topic"]
LifecyclePolicy: TypeAlias = Literal["all", "any"]

_NODES = SOURCE_LIFECYCLE_NODES_TABLE
_EVENTS = SOURCE_LIFECYCLE_EVENTS_TABLE


@dataclass(frozen=True, slots=True)
class StoredLifecycleNode:
    """One registered dependency node exactly as persisted.

    A Source node carries its caller-declared lifecycle ``revision`` plus the
    captured ``source`` reference. A derived node carries the ``all``/``any``
    ``policy`` and exactly one native binding: a Memory citation or a Topic
    Memory Artifact reference. ``ordinal`` preserves registration order, which is
    a topological order because parents must be registered first.
    """

    scope_id: str
    node_id: str
    ordinal: int
    kind: LifecycleNodeKind
    available: bool
    parents: tuple[str, ...] = ()
    policy: LifecyclePolicy | None = None
    revision: int | None = None
    source: SourceRef | None = None
    memory_citation: MemoryCitation | None = None
    artifact_ref: ArtifactRef | None = None

    @property
    def reference(self) -> SourceRef | MemoryCitation | ArtifactRef:
        """Return the original reference this node was registered with."""

        if self.kind == "source":
            assert self.source is not None
            return self.source
        if self.kind == "memory":
            assert self.memory_citation is not None
            return self.memory_citation
        assert self.artifact_ref is not None
        return self.artifact_ref


@dataclass(frozen=True, slots=True)
class StoredLifecycleEvent:
    """One accepted Source lifecycle event for one Source node."""

    revision: int
    sequence: int
    available: bool
    reason: str
    propagated: bool


class SourceLifecycleRepository:
    """Read and write lifecycle nodes and events for one Scope."""

    async def nodes(self, connection: AsyncConnection, scope_id: str) -> tuple[StoredLifecycleNode, ...]:
        rows = (
            await connection.execute(
                select(_NODES).where(_NODES.c.scope_id == scope_id).order_by(_NODES.c.ordinal)
            )
        ).mappings()
        return tuple(_node_from_row(row) for row in rows)

    async def node(
        self,
        connection: AsyncConnection,
        scope_id: str,
        node_id: str,
    ) -> StoredLifecycleNode | None:
        row = (
            (
                await connection.execute(
                    select(_NODES).where(_NODES.c.scope_id == scope_id, _NODES.c.node_id == node_id)
                )
            )
            .mappings()
            .one_or_none()
        )
        return None if row is None else _node_from_row(row)

    async def insert_node(self, connection: AsyncConnection, node: StoredLifecycleNode) -> None:
        await connection.execute(
            _NODES.insert().values(
                scope_id=node.scope_id,
                node_id=node.node_id,
                ordinal=node.ordinal,
                kind=node.kind,
                policy=node.policy,
                parents=json.dumps(list(node.parents), ensure_ascii=False, separators=(",", ":")).encode(),
                available=node.available,
                revision=node.revision,
                source_type=None if node.source is None else node.source.source_type,
                source_id=None if node.source is None else node.source.source_id,
                memory_artifact_id=(
                    None if node.memory_citation is None else node.memory_citation.memory_ref.artifact_id
                ),
                memory_revision=None if node.memory_citation is None else node.memory_citation.memory_ref.revision,
                entry_id=None if node.memory_citation is None else node.memory_citation.entry_id,
                entry_version_id=None if node.memory_citation is None else node.memory_citation.entry_version_id,
                topic_artifact_id=None if node.artifact_ref is None else node.artifact_ref.artifact_id,
                topic_revision=None if node.artifact_ref is None else node.artifact_ref.revision,
            )
        )

    async def set_availability(
        self,
        connection: AsyncConnection,
        scope_id: str,
        changes: Mapping[str, bool],
    ) -> None:
        for node_id, available in changes.items():
            await connection.execute(
                update(_NODES)
                .where(_NODES.c.scope_id == scope_id, _NODES.c.node_id == node_id)
                .values(available=available)
            )

    async def latest_event(
        self,
        connection: AsyncConnection,
        scope_id: str,
        node_id: str,
    ) -> StoredLifecycleEvent | None:
        row = (
            (
                await connection.execute(
                    select(_EVENTS)
                    .where(_EVENTS.c.scope_id == scope_id, _EVENTS.c.node_id == node_id)
                    .order_by(_EVENTS.c.sequence.desc())
                    .limit(1)
                )
            )
            .mappings()
            .one_or_none()
        )
        if row is None:
            return None
        return StoredLifecycleEvent(
            revision=int(row["revision"]),
            sequence=int(row["sequence"]),
            available=bool(row["available"]),
            reason=str(row["reason"]),
            propagated=bool(row["propagated"]),
        )

    async def append_event(
        self,
        connection: AsyncConnection,
        scope_id: str,
        node_id: str,
        *,
        revision: int,
        sequence: int,
        available: bool,
        reason: str,
        propagated: bool = False,
    ) -> None:
        await connection.execute(
            _EVENTS.insert().values(
                scope_id=scope_id,
                node_id=node_id,
                sequence=sequence,
                revision=revision,
                available=available,
                reason=reason,
                propagated=propagated,
            )
        )

    async def mark_event_propagated(
        self,
        connection: AsyncConnection,
        scope_id: str,
        node_id: str,
        sequence: int,
    ) -> None:
        await connection.execute(
            update(_EVENTS)
            .where(
                _EVENTS.c.scope_id == scope_id,
                _EVENTS.c.node_id == node_id,
                _EVENTS.c.sequence == sequence,
            )
            .values(propagated=True)
        )

    async def unavailable_topic_artifact_ids(
        self,
        connection: AsyncConnection,
        scope_id: str,
    ) -> frozenset[str]:
        rows = (
            await connection.execute(
                select(_NODES.c.topic_artifact_id).where(
                    _NODES.c.scope_id == scope_id,
                    _NODES.c.kind == "topic",
                    _NODES.c.available.is_(False),
                    _NODES.c.topic_artifact_id.is_not(None),
                )
            )
        ).scalars()
        return frozenset(str(value) for value in rows if value is not None)


def _node_from_row(row: Mapping[object, object]) -> StoredLifecycleNode:
    kind = cast(LifecycleNodeKind, str(row["kind"]))
    parents = tuple(str(parent) for parent in json.loads(_payload_text(row["parents"])))
    scope_id = str(row["scope_id"])
    node_id = str(row["node_id"])
    if kind == "source":
        return StoredLifecycleNode(
            scope_id=scope_id,
            node_id=node_id,
            ordinal=int(cast(int, row["ordinal"])),
            kind=kind,
            available=bool(row["available"]),
            revision=int(cast(int, row["revision"])),
            source=SourceRef(source_type=str(row["source_type"]), source_id=str(row["source_id"])),
        )
    policy = cast(LifecyclePolicy, str(row["policy"]))
    if kind == "memory":
        return StoredLifecycleNode(
            scope_id=scope_id,
            node_id=node_id,
            ordinal=int(cast(int, row["ordinal"])),
            kind=kind,
            available=bool(row["available"]),
            parents=parents,
            policy=policy,
            memory_citation=MemoryCitation(
                memory_ref=ArtifactRef(
                    family="memory",
                    artifact_id=str(row["memory_artifact_id"]),
                    revision=int(cast(int, row["memory_revision"])),
                ),
                entry_id=str(row["entry_id"]),
                entry_version_id=str(row["entry_version_id"]),
            ),
        )
    return StoredLifecycleNode(
        scope_id=scope_id,
        node_id=node_id,
        ordinal=int(cast(int, row["ordinal"])),
        kind=kind,
        available=bool(row["available"]),
        parents=parents,
        policy=policy,
        artifact_ref=ArtifactRef(
            family="topic-memory",
            artifact_id=str(row["topic_artifact_id"]),
            revision=int(cast(int, row["topic_revision"])),
        ),
    )


def _payload_text(value: object) -> str:
    if isinstance(value, bytes | bytearray | memoryview):
        return bytes(value).decode("utf-8")
    if isinstance(value, str):
        return value
    raise TypeError(f"unexpected lifecycle payload column type: {type(value)!r}")


__all__ = [
    "LifecycleNodeKind",
    "LifecyclePolicy",
    "SourceLifecycleRepository",
    "StoredLifecycleEvent",
    "StoredLifecycleNode",
]
