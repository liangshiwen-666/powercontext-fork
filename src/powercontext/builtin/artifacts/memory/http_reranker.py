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

"""HTTP cross-encoder reranking for coarse Memory search candidates."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import httpx

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import MemoryRerankDecision
from powercontext.builtin.inference import InferenceUsage

DEFAULT_POLICY_ID = "http-cross-encoder-v1"


class HTTPMemoryReranker:
    """Score a coarse candidate pool through bounded, schema-checked HTTP batches."""

    policy_id: str

    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = 32,
        policy_id: str = DEFAULT_POLICY_ID,
    ) -> None:
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise ValueError("rerank endpoint must be a non-empty string")  # noqa: TRY003
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("rerank batch_size must be a positive integer")  # noqa: TRY003
        if not isinstance(policy_id, str) or not policy_id.strip():
            raise ValueError("rerank policy_id must be a non-empty string")  # noqa: TRY003
        self.policy_id = policy_id
        self._endpoint = endpoint.strip()
        self._client = client
        self._batch_size = batch_size

    async def rerank(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        limit: int,
        /,
    ) -> MemoryRerankDecision:
        """Score every candidate once, then select the global top ``limit``.

        Batches keep the one-based rank of the complete input pool, so a high score in the
        last batch can still win. Every response is validated against exactly the ranks that
        its batch carried; malformed payloads fail with ``ValueError`` while non-2xx
        responses and transport failures keep their native ``httpx`` exception types.
        """

        pool = _validated_pool(candidates)
        _validate_limit(len(pool), limit)
        if not isinstance(query, str):
            raise ValueError("rerank query must be a string")  # noqa: TRY003

        scores: dict[int, float] = {}
        requests = 0
        for start in range(0, len(pool), self._batch_size):
            batch = pool[start : start + self._batch_size]
            ranks = tuple(range(start + 1, start + len(batch) + 1))
            payload = {
                "query": query,
                "candidates": [
                    {"rank": rank, "text": candidate.text} for rank, candidate in zip(ranks, batch, strict=True)
                ],
            }
            response = await self._client.post(self._endpoint, json=payload)
            requests += 1
            response.raise_for_status()
            scores.update(_batch_scores(response, ranks))

        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]
        return MemoryRerankDecision(
            selected_ranks=tuple(rank for rank, _score in ordered),
            usage=InferenceUsage(requests=requests),
        )


def _validated_pool(candidates: Sequence[MemoryHit]) -> tuple[MemoryHit, ...]:
    try:
        pool = tuple(candidates)
    except TypeError as error:
        raise ValueError("rerank candidates must be an iterable of MemoryHit") from error  # noqa: TRY003
    if not pool:
        raise ValueError("rerank requires at least one candidate")  # noqa: TRY003
    for candidate in pool:
        if not isinstance(candidate.text, str):
            raise ValueError("rerank candidates must expose text")  # noqa: TRY003
    return pool


def _validate_limit(candidate_count: int, limit: int) -> None:
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ValueError("rerank limit must be an integer")  # noqa: TRY003
    if limit < 1 or limit > candidate_count:
        raise ValueError("rerank limit must be between one and the candidate count")  # noqa: TRY003


def _batch_scores(response: httpx.Response, expected_ranks: tuple[int, ...]) -> dict[int, float]:
    try:
        payload: Any = response.json()
    except ValueError as error:
        raise ValueError("rerank response is not valid JSON") from error  # noqa: TRY003
    if not isinstance(payload, dict) or set(payload) != {"scores"}:
        raise ValueError("rerank response must contain exactly the scores field")  # noqa: TRY003
    entries = payload["scores"]
    if not isinstance(entries, list):
        raise ValueError("rerank scores must be a list")  # noqa: TRY003
    expected = set(expected_ranks)
    scores: dict[int, float] = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"rank", "score"}:
            raise ValueError("rerank score entries must contain exactly rank and score")  # noqa: TRY003
        rank = entry["rank"]
        value = entry["score"]
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise ValueError("rerank score rank must be an integer")  # noqa: TRY003
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("rerank score must be a number")  # noqa: TRY003
        try:
            numeric = float(value)
        except (OverflowError, ValueError) as error:
            raise ValueError("rerank score must be a finite number") from error  # noqa: TRY003
        if not math.isfinite(numeric):
            raise ValueError("rerank score must be finite")  # noqa: TRY003
        if rank not in expected:
            raise ValueError("rerank score rank is outside this batch")  # noqa: TRY003
        if rank in scores:
            raise ValueError("rerank score rank is duplicated")  # noqa: TRY003
        scores[rank] = numeric
    if len(scores) != len(expected_ranks):
        raise ValueError("rerank scores do not cover the batch ranks")  # noqa: TRY003
    return scores


__all__ = ["DEFAULT_POLICY_ID", "HTTPMemoryReranker"]
