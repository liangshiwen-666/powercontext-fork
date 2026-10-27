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

"""Propagate Source lifecycle events through a Scope's evidence dependency DAG.

The caller registers the dependency graph explicitly: captured Sources are the
roots, and Memory entries and Topic Memory Artifacts are derived nodes whose
availability follows their parents under an ``all`` or ``any`` policy. A
withdrawal (or restore) of a Source is then propagated to every descendant and
reconciled with the native families, so Memory list/search/Context and Topic
Memory search/Context all agree on what may be generated.

The persisted authoritative state is the dependency graph plus the accepted
Source events. Derived availability is recomputed deterministically from those,
and the native Memory entry state is reconciled towards the computed state, so a
retry of an interrupted propagation completes it while a successful replay
stays inert.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, field_validator
from sqlalchemy.ext.asyncio import AsyncConnection

from powercontext.artifacts import ArtifactRef, MemoryCitation
from powercontext.builtin.artifacts.memory import MemoryEntryVersion, MemoryService
from powercontext.builtin.persistence.database import AsyncDatabase
from powercontext.builtin.persistence.source_lifecycle import (
    LifecycleNodeKind,
    LifecyclePolicy,
    SourceLifecycleRepository,
    StoredLifecycleNode,
)
from powercontext.builtin.persistence.tables import MAX_SOURCE_LIFECYCLE_NODE_ID_LENGTH
from powercontext.builtin.sources import validate_scope_id
from powercontext.errors import ArtifactNotFoundError
from powercontext.sources import SourceRef

if TYPE_CHECKING:
    from powercontext.builtin.runtime.application import BuiltinRuntime

_GENERATION = "generation"
_AUDIT = "audit"
_PURPOSES: frozenset[str] = frozenset({_GENERATION, _AUDIT})
_REASON = "source-lifecycle"


class SourceLifecycleError(ValueError):
    """Report a rejected lifecycle registration, event, or resolution."""

    def __init__(self, code: str, detail: object | None = None) -> None:
        self.code = code
        self.detail = detail
        messages = {
            "event": "apply requires a validated SourceLifecycleEvent",
            "node-id": (
                f"node id must be a non-blank string of at most "
                f"{MAX_SOURCE_LIFECYCLE_NODE_ID_LENGTH} characters"
            ),
            "source-ref": "source registration requires a captured SourceRef",
            "revision": "source lifecycle revision must be a positive integer",
            "policy": "dependency policy must be 'all' or 'any'",
            "parents": "derived nodes require at least one parent",
            "parent-duplicate": "parent references must not repeat",
            "parent-self": "a node cannot depend on itself",
            "parent-missing": "every parent must be registered before its child",
            "node-conflict": "node id is already registered with a different binding",
            "source-unreadable": "the registered Source reference is not readable in this Scope",
            "memory-unreadable": "the registered Memory citation is not readable in this Scope",
            "topic-unreadable": "the registered Topic Memory reference is not readable in this Scope",
            "memory-citation": "memory registration requires a MemoryCitation",
            "topic-reference": "topic registration requires a topic-memory ArtifactRef",
            "unknown-node": "the node is not registered in this Scope",
            "unknown-source": "an event may only target a registered Source node",
            "revision-mismatch": "the event revision does not match the Source lifecycle revision",
            "sequence-regression": "event sequence must strictly increase for each Source",
            "sequence-conflict": "the sequence is already used by an event with different content",
            "unknown-purpose": "purpose must be 'generation' or 'audit'",
            "unavailable-node": "the node is not available for generation",
            "lifecycle-unavailable": "Source lifecycle storage is not configured for this Runtime",
            "memory-binding": "a registered Memory binding no longer resolves in this Scope",
        }
        rendered = messages.get(code, code)
        if detail is not None:
            rendered = f"{rendered}: {detail}"
        super().__init__(rendered)


class SourceLifecycleEvent(BaseModel):
    """One validated Source availability transition at an exact lifecycle revision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str
    revision: StrictInt = Field(gt=0)
    sequence: StrictInt = Field(gt=0)
    available: StrictBool
    reason: str

    @field_validator("node_id", "reason")
    @classmethod
    def require_non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must be a non-empty, non-blank string")  # noqa: TRY003
        return value


LifecyclePurpose: TypeAlias = Literal["generation", "audit"]


class SourceLifecycleApplication:
    """Select the lifecycle service for one Scope."""

    def __init__(self, runtime: BuiltinRuntime, database: AsyncDatabase | None) -> None:
        self._runtime = runtime
        self._database = database

    def for_scope(self, scope_id: str, /) -> ScopedSourceLifecycle:
        return ScopedSourceLifecycle(self._runtime, self._database, scope_id)


class ScopedSourceLifecycle:
    """Register a Scope's dependency DAG, apply Source events, and resolve references."""

    def __init__(self, runtime: BuiltinRuntime, database: AsyncDatabase | None, scope_id: str) -> None:
        self._runtime = runtime
        self._database = database
        self.scope_id = validate_scope_id(scope_id)
        self._repository = SourceLifecycleRepository()

    # -- registration ----------------------------------------------------------------

    async def register_source(
        self,
        node_id: str,
        source_ref: SourceRef,
        /,
        *,
        revision: int,
    ) -> None:
        """Register one captured Source root at the caller-declared lifecycle revision."""

        checked_id = _require_node_id(node_id)
        if not isinstance(source_ref, SourceRef):
            raise SourceLifecycleError("source-ref")
        checked_revision = _require_positive_int(revision, "revision")
        await self._require_source_readable(source_ref)
        await self._register(
            StoredLifecycleNode(
                scope_id=self.scope_id,
                node_id=checked_id,
                ordinal=-1,
                kind="source",
                available=True,
                revision=checked_revision,
                source=source_ref,
            )
        )

    async def register_memory(
        self,
        node_id: str,
        citation: MemoryCitation,
        /,
        *,
        parents: Sequence[str],
        policy: LifecyclePolicy,
    ) -> None:
        """Register one Memory entry derived from already-registered parents."""

        checked_id = _require_node_id(node_id)
        if not isinstance(citation, MemoryCitation) or citation.memory_ref.family != "memory":
            raise SourceLifecycleError("memory-citation")
        checked_parents = _require_parents(checked_id, parents)
        checked_policy = _require_policy(policy)
        await self._require_memory_readable(citation)
        await self._register(
            StoredLifecycleNode(
                scope_id=self.scope_id,
                node_id=checked_id,
                ordinal=-1,
                kind="memory",
                available=True,
                parents=checked_parents,
                policy=checked_policy,
                memory_citation=citation,
            )
        )

    async def register_topic(
        self,
        node_id: str,
        artifact_ref: ArtifactRef,
        /,
        *,
        parents: Sequence[str],
        policy: LifecyclePolicy,
    ) -> None:
        """Register one Topic Memory revision derived from already-registered parents."""

        checked_id = _require_node_id(node_id)
        if not isinstance(artifact_ref, ArtifactRef) or artifact_ref.family != "topic-memory":
            raise SourceLifecycleError("topic-reference")
        checked_parents = _require_parents(checked_id, parents)
        checked_policy = _require_policy(policy)
        await self._require_topic_readable(artifact_ref)
        await self._register(
            StoredLifecycleNode(
                scope_id=self.scope_id,
                node_id=checked_id,
                ordinal=-1,
                kind="topic",
                available=True,
                parents=checked_parents,
                policy=checked_policy,
                artifact_ref=artifact_ref,
            )
        )

    # -- events ----------------------------------------------------------------------

    async def apply(self, event: SourceLifecycleEvent, /) -> list[str]:
        """Apply one Source event and return every available node id in sorted order."""

        if not isinstance(event, SourceLifecycleEvent):
            raise SourceLifecycleError("event")
        database = self._require_database()
        async with self._runtime._scope_operation(self.scope_id), self._runtime._locked(self.scope_id):
            async with database.transaction() as connection:
                nodes = await self._repository.nodes(connection, self.scope_id)
                by_id = {node.node_id: node for node in nodes}
                source = by_id.get(event.node_id)
                if source is None or source.kind != "source":
                    raise SourceLifecycleError("unknown-source")
                if source.revision != event.revision:
                    raise SourceLifecycleError("revision-mismatch")
                latest = await self._repository.latest_event(connection, self.scope_id, event.node_id)
                if latest is None or event.sequence > latest.sequence:
                    await self._repository.append_event(
                        connection,
                        self.scope_id,
                        event.node_id,
                        revision=event.revision,
                        sequence=event.sequence,
                        available=event.available,
                        reason=event.reason,
                    )
                    await self._repository.set_availability(
                        connection,
                        self.scope_id,
                        {event.node_id: event.available},
                    )
                    nodes = await self._repository.nodes(connection, self.scope_id)
                elif event.sequence == latest.sequence:
                    if (latest.revision, latest.available, latest.reason) != (
                        event.revision,
                        event.available,
                        event.reason,
                    ):
                        raise SourceLifecycleError("sequence-conflict")
                    if latest.propagated:
                        return sorted(node.node_id for node in nodes if node.available)
                else:
                    raise SourceLifecycleError("sequence-regression")
                states = await self._recompute(connection, nodes)
            await self._reconcile_memory_nodes(states, nodes)
            async with database.transaction() as connection:
                await self._repository.mark_event_propagated(
                    connection,
                    self.scope_id,
                    event.node_id,
                    event.sequence,
                )
        return sorted(node_id for node_id, available in states.items() if available)

    # -- resolution ------------------------------------------------------------------

    async def resolve(self, node_id: str, /, *, purpose: str) -> SourceRef | MemoryCitation | ArtifactRef:
        """Return the original registration reference for one node and purpose."""

        if purpose not in _PURPOSES:
            raise SourceLifecycleError("unknown-purpose")
        database = self._require_database()
        async with self._runtime._operation(), database.transaction() as connection:
            node = await self._repository.node(connection, self.scope_id, node_id)
        if node is None:
            raise SourceLifecycleError("unknown-node")
        if purpose == _GENERATION and not node.available:
            raise SourceLifecycleError("unavailable-node")
        return node.reference

    # -- native Topic Memory projection ---------------------------------------------

    async def unavailable_topic_artifact_ids(self) -> frozenset[str]:
        """Return registered Topic Memory artifact ids that may not be recalled."""

        if self._database is None:
            return frozenset()
        async with self._runtime._operation(), self._database.transaction() as connection:
            return await self._repository.unavailable_topic_artifact_ids(connection, self.scope_id)

    # -- internals -------------------------------------------------------------------

    async def _register(self, candidate: StoredLifecycleNode) -> None:
        database = self._require_database()
        async with self._runtime._scope_operation(self.scope_id), self._runtime._locked(self.scope_id):
            async with database.transaction() as connection:
                existing = await self._repository.node(connection, self.scope_id, candidate.node_id)
                if existing is not None:
                    _require_same_registration(existing, candidate)
                    node = existing
                else:
                    nodes = await self._repository.nodes(connection, self.scope_id)
                    by_id = {item.node_id: item for item in nodes}
                    parents = _require_registered_parents(candidate, by_id)
                    node = StoredLifecycleNode(
                        scope_id=candidate.scope_id,
                        node_id=candidate.node_id,
                        ordinal=0 if not nodes else nodes[-1].ordinal + 1,
                        kind=candidate.kind,
                        available=_derive_node_availability(
                            candidate.kind,
                            candidate.policy,
                            [by_id[parent].available for parent in parents],
                        ),
                        parents=parents,
                        policy=candidate.policy,
                        revision=candidate.revision,
                        source=candidate.source,
                        memory_citation=candidate.memory_citation,
                        artifact_ref=candidate.artifact_ref,
                    )
                    await self._repository.insert_node(connection, node)
            if node.kind == "memory":
                # A derived node is usable only from the moment its computed state is
                # reflected natively; an idempotent re-registration repairs an
                # interrupted first attempt instead of silently leaving it unapplied.
                await self._reconcile_memory_nodes({node.node_id: node.available}, (node,))

    async def _recompute(
        self,
        connection: AsyncConnection,
        nodes: Sequence[StoredLifecycleNode],
    ) -> dict[str, bool]:
        """Recompute derived availability in registration (topological) order."""

        states: dict[str, bool] = {}
        for node in nodes:
            if node.kind == "source":
                states[node.node_id] = node.available
                continue
            support = [states[parent] for parent in node.parents]
            states[node.node_id] = all(support) if node.policy == "all" else any(support)
        changes = {node.node_id: states[node.node_id] for node in nodes if states[node.node_id] != node.available}
        if changes:
            await self._repository.set_availability(connection, self.scope_id, changes)
        return states

    async def _reconcile_memory_nodes(
        self,
        states: Mapping[str, bool],
        nodes: Sequence[StoredLifecycleNode],
    ) -> None:
        """Drive native Memory entry states towards the computed availability."""

        groups: dict[str, list[StoredLifecycleNode]] = {}
        for node in nodes:
            if node.kind != "memory" or node.memory_citation is None:
                continue
            groups.setdefault(node.memory_citation.memory_ref.artifact_id, []).append(node)
        if not groups:
            return
        async with self._runtime._context(self.scope_id) as context:
            service = context.artifacts.memory
            for artifact_id, group in groups.items():
                await _reconcile_memory_artifact(service, artifact_id, group, states)

    async def _require_source_readable(self, source_ref: SourceRef) -> None:
        try:
            await self._runtime._records().get_source(
                self.scope_id,
                source_ref.source_type,
                source_ref.source_id,
            )
        except (LookupError, ValueError) as error:
            raise SourceLifecycleError("source-unreadable", source_ref) from error

    async def _require_memory_readable(self, citation: MemoryCitation) -> None:
        async with self._runtime._context(self.scope_id) as context:
            service = context.artifacts.memory
            try:
                await service.validate_citation(citation)
            except (LookupError, ValueError) as error:
                raise SourceLifecycleError("memory-unreadable", citation) from error

    async def _require_topic_readable(self, artifact_ref: ArtifactRef) -> None:
        try:
            await self._runtime._records().get_artifact_revision(
                self.scope_id,
                artifact_ref.family,
                artifact_ref.artifact_id,
                artifact_ref.revision,
            )
        except (LookupError, ValueError) as error:
            raise SourceLifecycleError("topic-unreadable", artifact_ref) from error

    def _require_database(self) -> AsyncDatabase:
        if self._database is None:
            raise SourceLifecycleError("lifecycle-unavailable")
        return self._database


async def _reconcile_memory_artifact(
    service: MemoryService,
    artifact_id: str,
    nodes: Sequence[StoredLifecycleNode],
    states: Mapping[str, bool],
) -> None:
    """Apply only the Memory entry state transitions that are still missing."""

    try:
        head = await service.head(artifact_id)
    except ArtifactNotFoundError as error:
        raise SourceLifecycleError("memory-binding", artifact_id) from error
    manifest = {item.entry_id: item.state for item in head.content.manifest.entries}
    try:
        versions = {version.entry_id: version for version in await service.entries(head)}
    except (LookupError, ValueError) as error:
        raise SourceLifecycleError("memory-binding", artifact_id) from error
    deactivate: list[MemoryEntryVersion] = []
    activate: list[MemoryEntryVersion] = []
    for node in sorted(nodes, key=lambda item: item.ordinal):
        citation = node.memory_citation
        if citation is None:
            raise SourceLifecycleError("memory-binding", node.node_id)
        current_state = manifest.get(citation.entry_id)
        version = versions.get(citation.entry_id)
        if current_state is None or version is None:
            raise SourceLifecycleError("memory-binding", citation.entry_id)
        desired = states[node.node_id]
        if desired and current_state == "inactive":
            activate.append(version)
        elif not desired and current_state == "active":
            deactivate.append(version)
    if deactivate:
        head = await service.forget(head, entries=deactivate, reason=_REASON)
    if activate:
        await service.reactivate(head, entries=activate, reason=_REASON)


def _require_same_registration(existing: StoredLifecycleNode, candidate: StoredLifecycleNode) -> None:
    """Accept an identical re-registration and reject a conflicting rebinding."""

    if (
        existing.kind != candidate.kind
        or existing.policy != candidate.policy
        or existing.parents != candidate.parents
        or existing.revision != candidate.revision
        or existing.source != candidate.source
        or existing.memory_citation != candidate.memory_citation
        or existing.artifact_ref != candidate.artifact_ref
    ):
        raise SourceLifecycleError("node-conflict", candidate.node_id)


def _require_registered_parents(
    candidate: StoredLifecycleNode,
    by_id: Mapping[str, StoredLifecycleNode],
) -> tuple[str, ...]:
    if candidate.kind == "source":
        return ()
    for parent in candidate.parents:
        if parent not in by_id:
            raise SourceLifecycleError("parent-missing", parent)
    return candidate.parents


def _derive_node_availability(
    kind: LifecycleNodeKind,
    policy: LifecyclePolicy | None,
    parents: Sequence[bool],
) -> bool:
    if kind == "source":
        return True
    support = list(parents)
    if policy == "all":
        return all(support)
    return any(support)


def _require_node_id(value: object) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > MAX_SOURCE_LIFECYCLE_NODE_ID_LENGTH:
        raise SourceLifecycleError("node-id")
    return value


def _require_positive_int(value: object, field: str) -> int:
    if type(value) is not int or value < 1:
        raise SourceLifecycleError(field)
    return value


def _require_policy(value: object) -> LifecyclePolicy:
    if value not in {"all", "any"}:
        raise SourceLifecycleError("policy")
    return value  # type: ignore[return-value]


def _require_parents(node_id: str, parents: object) -> tuple[str, ...]:
    if isinstance(parents, str) or not isinstance(parents, Sequence) or not parents:
        raise SourceLifecycleError("parents")
    checked: list[str] = []
    for parent in parents:
        if not isinstance(parent, str) or not parent.strip():
            raise SourceLifecycleError("parents")
        if parent == node_id:
            raise SourceLifecycleError("parent-self")
        if parent in checked:
            raise SourceLifecycleError("parent-duplicate")
        checked.append(parent)
    return tuple(checked)


__all__ = [
    "LifecyclePurpose",
    "ScopedSourceLifecycle",
    "SourceLifecycleApplication",
    "SourceLifecycleError",
    "SourceLifecycleEvent",
]
