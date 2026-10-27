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

"""Cross-call embedding request coalescing with a bounded LRU vector cache.

One wrapper instance coordinates every ``embed`` call on the same event loop. Texts are
deduplicated within one call, across calls inside the coalescing window, and against
underlying requests that are still running, so identical text is only ever embedded once
while that work is pending. Successful vectors are cached in a bounded LRU; failures are
never cached. Underlying requests are owned tasks that are shielded from waiter
cancellation, so one caller giving up cannot cancel or lose the work shared with others.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from powercontext.builtin.inference.models import EmbeddingResult, EmbeddingVector, InferenceUsage

if TYPE_CHECKING:
    from powercontext.builtin.artifacts.memory.models import EmbeddingProfile
    from powercontext.builtin.inference.protocols import EmbeddingModel


class _Batch:
    """One underlying request worth of deduplicated texts."""

    __slots__ = ("texts", "usage")

    def __init__(self, texts: tuple[str, ...]) -> None:
        self.texts = texts
        self.usage: InferenceUsage | None = None


class _TextFlight:
    """The pending or in-flight computation of one distinct text."""

    __slots__ = ("batch", "future")

    def __init__(self, future: asyncio.Future[EmbeddingVector]) -> None:
        self.future = future
        self.batch: _Batch | None = None


class CoalescingEmbeddingModel:
    """Coalesce concurrent embedding calls and reuse successful vectors.

    ``max_batch_size`` caps the number of distinct texts in one underlying request;
    ``coalesce_ms`` is the window during which new texts may join the next request;
    ``cache_size`` bounds the cross-call LRU cache and ``0`` disables it while keeping
    in-flight request sharing.
    """

    profile: EmbeddingProfile

    def __init__(
        self,
        model: EmbeddingModel,
        *,
        max_batch_size: int = 64,
        coalesce_ms: float = 5.0,
        cache_size: int = 1024,
    ) -> None:
        if isinstance(max_batch_size, bool) or not isinstance(max_batch_size, int) or max_batch_size < 1:
            raise ValueError("max_batch_size must be a positive integer")  # noqa: TRY003
        if isinstance(coalesce_ms, bool) or not isinstance(coalesce_ms, (int, float)) or coalesce_ms < 0:
            raise ValueError("coalesce_ms must be a non-negative number")  # noqa: TRY003
        if coalesce_ms != coalesce_ms or coalesce_ms == float("inf"):
            raise ValueError("coalesce_ms must be finite")  # noqa: TRY003
        if isinstance(cache_size, bool) or not isinstance(cache_size, int) or cache_size < 0:
            raise ValueError("cache_size must be a non-negative integer")  # noqa: TRY003
        self.profile = model.profile
        self._model = model
        self._max_batch_size = max_batch_size
        self._coalesce_seconds = float(coalesce_ms) / 1_000.0
        self._cache_size = cache_size
        self._cache: OrderedDict[str, EmbeddingVector] = OrderedDict()
        self._flights: dict[str, _TextFlight] = {}
        self._pending: dict[str, _TextFlight] = {}
        self._dispatch_task: asyncio.Task[None] | None = None
        self._batch_tasks: set[asyncio.Task[None]] = set()

    async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        """Return one vector per input position, reusing and sharing underlying work."""

        values = tuple(texts)
        if not values:
            return EmbeddingResult(vectors=(), usage=InferenceUsage(requests=0))

        distinct = tuple(dict.fromkeys(values))
        cached: dict[str, EmbeddingVector] = {}
        flights: dict[str, _TextFlight] = {}
        registered = False
        for text in distinct:
            vector = self._cache_lookup(text)
            if vector is not None:
                cached[text] = vector
                continue
            flight = self._flights.get(text)
            if flight is None:
                flight = self._register(text)
                registered = True
            flights[text] = flight
        if registered:
            self._ensure_dispatch_task()
        for flight in flights.values():
            await asyncio.shield(flight.future)

        vectors: list[EmbeddingVector] = []
        batches: list[_Batch] = []
        seen_batches: set[int] = set()
        for text in values:
            vector = cached.get(text)
            if vector is None:
                flight = flights[text]
                vector = flight.future.result()
                batch = flight.batch
                if batch is not None and id(batch) not in seen_batches:
                    seen_batches.add(id(batch))
                    batches.append(batch)
            vectors.append(vector)
        return EmbeddingResult(vectors=tuple(vectors), usage=_aggregate_usage(batches))

    def _register(self, text: str) -> _TextFlight:
        flight = _TextFlight(asyncio.get_running_loop().create_future())
        flight.future.add_done_callback(_consume_future_exception)
        self._flights[text] = flight
        self._pending[text] = flight
        return flight

    def _ensure_dispatch_task(self) -> None:
        if self._dispatch_task is not None and not self._dispatch_task.done():
            return
        task = asyncio.get_running_loop().create_task(self._dispatch_loop())
        self._dispatch_task = task
        task.add_done_callback(self._dispatch_done)

    def _dispatch_done(self, task: asyncio.Task[None]) -> None:
        if self._dispatch_task is task:
            self._dispatch_task = None
        if not task.cancelled():
            task.exception()

    async def _dispatch_loop(self) -> None:
        try:
            while self._pending:
                await self._coalesce_window()
                if not self._pending:
                    break
                self._dispatch_ready_batches()
        finally:
            if self._dispatch_task is asyncio.current_task():
                self._dispatch_task = None

    async def _coalesce_window(self) -> None:
        if self._coalesce_seconds <= 0.0:
            await asyncio.sleep(0)
            return
        await asyncio.sleep(self._coalesce_seconds)

    def _dispatch_ready_batches(self) -> None:
        while self._pending:
            texts = tuple(list(self._pending)[: self._max_batch_size])
            batch = _Batch(texts)
            for text in texts:
                self._pending.pop(text).batch = batch
            task = asyncio.get_running_loop().create_task(self._run_batch(batch))
            self._batch_tasks.add(task)
            task.add_done_callback(self._batch_tasks.discard)

    async def _run_batch(self, batch: _Batch) -> None:
        try:
            result = await self._model.embed(batch.texts)
            vectors = _validated_vectors(result.vectors, len(batch.texts))
        except asyncio.CancelledError:
            self._fail_batch(batch, asyncio.CancelledError())
            return
        except Exception as error:  # noqa: BLE001 - every failure belongs to this batch only
            self._fail_batch(batch, error)
            return
        batch.usage = result.usage
        for text, vector in zip(batch.texts, vectors, strict=True):
            self._complete_text(text, vector)

    def _complete_text(self, text: str, vector: EmbeddingVector) -> None:
        flight = self._flights.pop(text, None)
        if flight is None:
            return
        self._pending.pop(text, None)
        self._cache_store(text, vector)
        if not flight.future.done():
            flight.future.set_result(vector)

    def _fail_batch(self, batch: _Batch, error: BaseException) -> None:
        for text in batch.texts:
            flight = self._flights.pop(text, None)
            if flight is None:
                continue
            self._pending.pop(text, None)
            if not flight.future.done():
                flight.future.set_exception(error)

    def _cache_lookup(self, text: str) -> EmbeddingVector | None:
        if not self._cache:
            return None
        vector = self._cache.get(text)
        if vector is None:
            return None
        self._cache.move_to_end(text)
        return vector

    def _cache_store(self, text: str, vector: EmbeddingVector) -> None:
        if self._cache_size == 0:
            return
        self._cache[text] = vector
        self._cache.move_to_end(text)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)


def _validated_vectors(vectors: Sequence[Sequence[float]], expected: int) -> tuple[EmbeddingVector, ...]:
    pool = tuple(vectors)
    if len(pool) != expected:
        raise ValueError("embedding model returned the wrong number of vectors")  # noqa: TRY003
    normalized: list[EmbeddingVector] = []
    for vector in pool:
        try:
            normalized.append(tuple(float(component) for component in vector))
        except (TypeError, ValueError) as error:
            raise ValueError("embedding model returned an invalid vector") from error  # noqa: TRY003
    return tuple(normalized)


def _aggregate_usage(batches: Sequence[_Batch]) -> InferenceUsage:
    usages = [batch.usage for batch in batches if batch.usage is not None]
    return InferenceUsage(
        requests=sum(usage.requests for usage in usages),
        input_tokens=_sum_optional(usage.input_tokens for usage in usages),
        output_tokens=_sum_optional(usage.output_tokens for usage in usages),
    )


def _sum_optional(values: Iterable[int | None]) -> int | None:
    present = [value for value in values if value is not None]
    return sum(present) if present else None


def _consume_future_exception(future: asyncio.Future[EmbeddingVector]) -> None:
    if future.cancelled():
        return
    future.exception()


__all__ = ["CoalescingEmbeddingModel"]
