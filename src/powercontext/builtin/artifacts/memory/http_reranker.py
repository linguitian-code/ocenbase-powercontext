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

import httpx

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import MemoryRerankDecision
from powercontext.builtin.inference import InferenceUsage

DEFAULT_HTTP_RERANK_POLICY_ID = "http-cross-encoder-v1"
"""Deployment-visible identity of the HTTP scoring policy."""


class HTTPMemoryReranker:
    """Score coarse Memory candidates through one batched HTTP rerank service.

    Candidate ranks are one-based positions in the complete candidate pool. Batching
    never renumbers them, so every score maps back to the original Memory entry.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = 32,
        policy_id: str = DEFAULT_HTTP_RERANK_POLICY_ID,
    ) -> None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")  # noqa: TRY003
        self._endpoint = endpoint
        self._client = client
        self._batch_size = batch_size
        self.policy_id = policy_id

    async def rerank(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        limit: int,
        /,
    ) -> MemoryRerankDecision:
        """Select the highest scoring original ranks across every batch."""

        _validate_bounds(len(candidates), limit)
        texts = tuple(candidate.text for candidate in candidates)
        scores: dict[int, float] = {}
        requests = 0

        for start in range(0, len(texts), self._batch_size):
            stop = min(start + self._batch_size, len(texts))
            batch_ranks = tuple(range(start + 1, stop + 1))
            response = await self._client.post(
                self._endpoint,
                json={
                    "query": query,
                    "candidates": [{"rank": rank, "text": texts[rank - 1]} for rank in batch_ranks],
                },
            )
            requests += 1
            response.raise_for_status()
            scores.update(_parse_batch_scores(response, batch_ranks))

        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return MemoryRerankDecision(
            selected_ranks=tuple(rank for rank, _ in ordered[:limit]),
            usage=InferenceUsage(requests=requests),
        )


def _parse_batch_scores(response: httpx.Response, expected_ranks: tuple[int, ...]) -> dict[int, float]:
    """Validate one response and return its rank to score mapping."""

    try:
        payload = response.json()
    except ValueError as error:
        raise ValueError("rerank response is not valid JSON") from error

    if not isinstance(payload, dict) or set(payload) != {"scores"}:
        raise ValueError("rerank response must contain exactly 'scores'")  # noqa: TRY003
    raw_scores = payload["scores"]
    if not isinstance(raw_scores, list):
        raise ValueError("rerank scores must be a list")  # noqa: TRY003

    expected = set(expected_ranks)
    observed: dict[int, float] = {}
    for item in raw_scores:
        if not isinstance(item, dict) or set(item) != {"rank", "score"}:
            raise ValueError("each rerank score must contain exactly 'rank' and 'score'")  # noqa: TRY003
        rank = item["rank"]
        score = item["score"]
        if not isinstance(rank, int) or isinstance(rank, bool):
            raise ValueError("rerank rank must be an integer")  # noqa: TRY003
        if rank not in expected:
            raise ValueError("rerank score rank is outside the requested batch")  # noqa: TRY003
        if rank in observed:
            raise ValueError("rerank score repeats a rank")  # noqa: TRY003
        if not isinstance(score, int | float) or isinstance(score, bool):
            raise ValueError("rerank score must be a number")  # noqa: TRY003
        if not math.isfinite(score):
            raise ValueError("rerank score must be finite")  # noqa: TRY003
        observed[rank] = float(score)

    if set(observed) != expected:
        raise ValueError("rerank response must cover every requested rank")  # noqa: TRY003
    return observed


def _validate_bounds(candidate_count: int, limit: int) -> None:
    if candidate_count < 1:
        raise ValueError("rerank requires at least one candidate")  # noqa: TRY003
    if limit < 1 or limit > candidate_count:
        raise ValueError("rerank limit must be between one and the candidate count")  # noqa: TRY003


__all__ = ["DEFAULT_HTTP_RERANK_POLICY_ID", "HTTPMemoryReranker"]
