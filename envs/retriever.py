"""Tiny in-memory retriever for dry-run. Seedable and fully reproducible.

Real runs use Search-R1's retriever server (Wikipedia + E5 dense / BM25) on HPC.
This mock does simple lexical-overlap scoring over a few-doc corpus so the
multi-turn search loop runs end-to-end on CPU with deterministic results.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class MockRetriever:
    corpus: list[str]
    seed: int = 0

    def __post_init__(self) -> None:
        self._toks = [self._tokenize(d) for d in self.corpus]

    @staticmethod
    def _tokenize(s: str) -> list[str]:
        return re.findall(r"\w+", s.lower())

    def retrieve(self, query: str, k: int = 2, step: int = 0) -> list[str]:
        q = set(self._tokenize(query))
        scored = []
        for i, dt in enumerate(self._toks):
            overlap = sum(1 for w in dt if w in q)
            scored.append((overlap, i))
        # deterministic order: score desc, then corpus index asc
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [self.corpus[i] for _, i in scored[:k]]

    def retrieve_batch(self, requests) -> list[list[str]]:
        return retrieve_batch_via_retrieve(self, requests)


def retrieve_batch_via_retrieve(retriever, requests) -> list[list[str]]:
    """Default `retrieve_batch`: serve a gathered round one query at a time.

    Mirrors `act_batch_via_act` -- the rollout driver targets ONE interface, and a
    retriever that cannot batch is not a special case. A plain loop over the SAME
    `retrieve`, so adding the surface is behaviour-preserving; only a backend that
    overrides it (the HTTP retriever, which can put every query of a round in one
    request) changes how the work is issued.
    """
    return [retriever.retrieve(query, k=k, step=step) for query, k, step in requests]


class NullRetriever:
    """A retriever that always returns NO documents. Same contract as MockRetriever /
    HTTPRetriever (retrieve + retrieve_batch), never touches a server.

    WHY IT EXISTS. The gap0 ladder (D3') needs the agentic protocol with the RETRIEVED CONTENT
    removed while everything else -- the template, the <search> action, the observation
    boundary, the multi-turn loop, the forced-answer seam -- stays exactly as the rollout
    coroutine does it. With docs == [] the coroutine injects an EMPTY <information> block
    (T.INFO_OPEN + "" + T.INFO_CLOSE), so the model sees the same structure and no evidence.
    A mocked corpus would put SOME text there; this puts none, which is the quantity wanted."""

    def retrieve(self, query: str, k: int = 2, step: int = 0) -> list[str]:  # noqa: ARG002
        return []

    def retrieve_batch(self, requests) -> list[list[str]]:
        return [[] for _ in requests]

