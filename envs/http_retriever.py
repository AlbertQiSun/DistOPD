"""Real retriever: HTTP client for the Search-R1 FastAPI server. HPC only.

Drop-in for MockRetriever -- same `retrieve(query, k, step) -> list[str]` interface,
so the rollout code path is unchanged. Speaks the Search-R1 /retrieve contract:

    POST {endpoint}  {"queries": [q], "topk": k, "return_scores": true}
    -> {"result": [[{"document": {"contents": "<title>\\n<text>"}, "score": ...}, ...]]}

We MUST send return_scores=true: the pinned Search-R1 retrieval_server.py unpacks
`results, scores = retriever.batch_search(..., return_score=request.return_scores)`
UNCONDITIONALLY, so batch_search returning a single value (return_scores=false) makes
that unpack raise "not enough values to unpack (expected 2, got 1)" -> HTTP 500. We
request scores and simply discard them (_doc_text ignores the "score" field), which keeps
the vendored server verbatim instead of forking its endpoint.

Uses stdlib urllib (no new dependency). The returned doc strings are injected into
the OBSERVATION span as-is; real Wikipedia passages are long and may carry special
characters, which is exactly what the adversarial-obs alignment test guards against.
"""
from __future__ import annotations

import http.client
import json
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass


class RetrieverHTTPError(RuntimeError):
    """Non-2xx from the retriever, WITH the server's stated reason attached.

    Same lesson as TeacherHTTPError: urllib raises before anyone reads the body, so the
    server's explanation is discarded on receipt and a failure reaches the operator as a
    bare "HTTP Error 500". Carries `.code` for status branching."""

    def __init__(self, code: int, detail: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.detail = detail


@dataclass
class HTTPRetriever:
    endpoint: str                 # e.g. http://<node>:8000/retrieve
    # 30s killed a real arm: the 2026-08-19 online run (150 steps x 256 trajectories,
    # ~31h) died mid-flight on ONE socket read timeout from a 256-query batch -- the
    # retriever shares its node's CPUs with up to three other jobs (OverSubscribe=FORCE:4),
    # so a transient stall past 30s is a matter of when, not if. The margin is bounded,
    # just no longer razor-thin.
    timeout: float = 120.0
    return_scores: bool = True    # MUST be True: server unconditionally unpacks (results, scores)
    # TRANSPORT retries. Retrieval is a pure index lookup, so re-sending the identical
    # request cannot change what is returned -- retrying alters only whether a transient
    # fault is fatal to a 31-hour run. Bounded: a DEAD server still fails loudly after
    # ~2+4+8+16s of backoff rather than hanging a run forever. HTTP errors are NOT
    # retried -- a server that answered with a status made a statement, not a hiccup.
    retries: int = 4

    def _post(self, queries: list[str], k: int) -> list[list[str]]:
        """THE transport -- one POST carrying however many queries it was given.

        Single and batched retrieval share this, so a batch is not a second client with
        its own parsing; `retrieve` is simply this called with one query.
        """
        payload = json.dumps(
            {"queries": queries, "topk": k, "return_scores": self.return_scores}
        ).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint, data=payload, headers={"Content-Type": "application/json"}
        )
        body = None
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                # The server SPOKE -- this is its answer, not a transport fault. The
                # pinned Search-R1 server's 500s are deterministic contract violations;
                # retrying one re-asks a question whose answer will not change.
                try:
                    detail = exc.read().decode("utf-8", "replace")[:2000]
                except Exception:                               # noqa: BLE001
                    detail = "<no response body>"
                raise RetrieverHTTPError(
                    exc.code, detail,
                    f"retriever {self.endpoint} -> HTTP {exc.code} for {len(queries)} "
                    f"quer{'y' if len(queries) == 1 else 'ies'} (topk={k}). Server said: {detail}"
                ) from exc
            except (OSError, http.client.HTTPException, json.JSONDecodeError) as exc:
                # OSError covers URLError, socket timeouts and connection resets;
                # HTTPException covers a reply cut off mid-body; JSONDecodeError is the
                # same cut observed after a 200 status line.
                if attempt == self.retries:
                    raise RetrieverHTTPError(
                        0, "",
                        f"retriever {self.endpoint} unreachable: {type(exc).__name__} "
                        f"({exc}) persisted through {self.retries + 1} attempts for "
                        f"{len(queries)} queries (topk={k}, timeout={self.timeout}s)."
                    ) from exc
                delay = 2.0 * (2 ** attempt)
                print(f"[retriever] transient {type(exc).__name__} on attempt "
                      f"{attempt + 1}/{self.retries + 1} ({len(queries)} queries) -- "
                      f"retrying in {delay:.0f}s", flush=True)
                time.sleep(delay)

        results = body["result"]
        # ALIGNMENT GUARD -- load-bearing once more than one query rides in a request.
        # Results are matched to queries POSITIONALLY, so a server that dropped, added or
        # reordered entries would hand trajectory A the documents retrieved for B. That is
        # silent: the docs are real passages, the rollout continues, and the corruption
        # only ever shows up as unexplained retrieval metrics. Same shape as the teacher's
        # BOS length guard -- refuse rather than mis-align.
        if len(results) != len(queries):
            raise RetrieverHTTPError(
                200, "",
                f"retriever returned {len(results)} result groups for {len(queries)} queries "
                f"-- results are paired to queries BY POSITION, so this would cross documents "
                f"between trajectories. Refusing."
            )
        return [[self._doc_text(d) for d in group][:k] for group in results]

    def retrieve(self, query: str, k: int = 2, step: int = 0) -> list[str]:  # noqa: ARG002
        return self._post([query], k)[0]

    def retrieve_batch(self, requests) -> list[list[str]]:
        """Serve a whole rollout round in as few requests as possible.

        The server's contract already takes a `queries` LIST, so batching is what it was
        built for; per-query serial POSTs would become the bottleneck the moment rollout is
        batched (256 trajectories x up to 4 turns).

        NO CROSS-TRAJECTORY DEDUPLICATION, deliberately. Two trajectories may legitimately
        issue the same query in the same round, and each must get its own reply: the
        duplicate-query guard (Guard 2) is a WITHIN-trajectory rule -- repeating a query
        yields no new evidence for the trajectory that already ran it -- and collapsing
        across trajectories would apply that rule where it does not hold, besides
        reintroducing the scatter step this positional design exists to avoid.

        `topk` is per-REQUEST in the server contract, not per-query, so requests are
        grouped by k. In practice every caller passes rollout.top_k and there is one group.
        """
        requests = list(requests)
        if not requests:
            return []
        by_k: dict[int, list[int]] = defaultdict(list)
        for i, (_query, k, _step) in enumerate(requests):
            by_k[k].append(i)

        out: list[list[str]] = [None] * len(requests)          # type: ignore[list-item]
        for k, idxs in by_k.items():
            groups = self._post([requests[i][0] for i in idxs], k)
            for i, docs in zip(idxs, groups):
                out[i] = docs
        assert all(d is not None for d in out), "retrieve_batch left a request unanswered"
        return out

    @staticmethod
    def _doc_text(d) -> str:
        """Extract passage text from a Search-R1 result item, tolerating the common
        shapes: {"document": {"contents": ...}} | {"contents": ...} | plain str."""
        if isinstance(d, str):
            return d
        doc = d.get("document", d) if isinstance(d, dict) else d
        if isinstance(doc, dict):
            return doc.get("contents") or doc.get("text") or json.dumps(doc)
        return str(doc)
