"""Batched retrieval must be the single path, and must stay positionally aligned.

Once a rollout round puts many queries in ONE request, results are matched to queries by
POSITION. A server that dropped, added or reordered entries would hand trajectory A the
documents retrieved for B -- silently, because the documents are real passages and the
rollout carries on. The corruption would surface only as inexplicable retrieval metrics,
long after the run. So alignment is asserted at the boundary, the same way the teacher
client refuses a prompt_logprobs array whose length does not match its input.

Runs a fake Search-R1 /retrieve over stdlib http.server -- no HPC, no index.
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.http_retriever import HTTPRetriever, RetrieverHTTPError  # noqa: E402


def _make_handler(responder):
    class H(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n).decode("utf-8")) if n else {}
            code, obj = responder(req)
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):  # noqa: ARG002
            pass
    return H


@contextlib.contextmanager
def serve(responder):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(responder))
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}/retrieve"
    finally:
        httpd.shutdown()
        httpd.server_close()


def _echo(req):
    """Well-behaved server: one result group per query, in the order received, each doc
    naming its query so cross-wiring is visible in the assertions."""
    qs, k = req["queries"], req["topk"]
    return 200, {"result": [[{"document": {"contents": f"doc{j} for {q}"}} for j in range(k)]
                            for q in qs]}


def test_batch_matches_one_at_a_time():
    calls = []

    def responder(req):
        calls.append(len(req["queries"]))
        return _echo(req)

    with serve(responder) as url:
        r = HTTPRetriever(endpoint=url)
        reqs = [("alpha", 2, 0), ("beta", 2, 1), ("gamma", 2, 2)]
        singles = [r.retrieve(q, k=k, step=s) for q, k, s in reqs]
        batched = r.retrieve_batch(reqs)
    assert batched == singles
    # and the batch really was ONE round-trip, not a loop wearing a batch's clothes
    assert calls == [1, 1, 1, 3], calls


def test_results_are_positional_not_content_keyed():
    """Two trajectories issuing the IDENTICAL query in one round must each get a reply,
    and dedup must not silently collapse them (Guard 2 is a within-trajectory rule)."""
    seen = []

    def responder(req):
        seen.append(list(req["queries"]))
        return _echo(req)

    with serve(responder) as url:
        r = HTTPRetriever(endpoint=url)
        out = r.retrieve_batch([("same", 2, 0), ("other", 2, 0), ("same", 2, 1)])
    assert len(out) == 3
    assert out[0] == out[2], "identical queries must yield identical docs"
    assert out[1] != out[0]
    assert seen == [["same", "other", "same"]], f"queries were collapsed/reordered: {seen}"


def test_misaligned_response_is_refused():
    """The guard that matters: a server returning the wrong number of groups must fail
    loud. Without it, every trajectory after the missing one silently receives the NEXT
    trajectory's documents."""
    def dropper(req):
        code, obj = _echo(req)
        obj["result"] = obj["result"][:-1]          # drop one group
        return code, obj

    with serve(dropper) as url:
        r = HTTPRetriever(endpoint=url)
        with pytest.raises(RetrieverHTTPError) as ei:
            r.retrieve_batch([("a", 2, 0), ("b", 2, 0), ("c", 2, 0)])
        assert "2 result groups for 3 queries" in str(ei.value)


def test_mixed_topk_is_grouped_and_still_aligned():
    with serve(_echo) as url:
        r = HTTPRetriever(endpoint=url)
        out = r.retrieve_batch([("a", 1, 0), ("b", 3, 0), ("c", 1, 0)])
    assert [len(d) for d in out] == [1, 3, 1]
    assert "for a" in out[0][0] and "for b" in out[1][0] and "for c" in out[2][0]


def test_http_error_surfaces_server_reason():
    with serve(lambda req: (500, {"error": "index not loaded"})) as url:
        r = HTTPRetriever(endpoint=url)
        with pytest.raises(RetrieverHTTPError) as ei:
            r.retrieve_batch([("a", 2, 0), ("b", 2, 0)])
        assert ei.value.code == 500
        assert "index not loaded" in str(ei.value)
        assert "2 queries" in str(ei.value)


def test_empty_batch_is_a_noop():
    with serve(_echo) as url:
        assert HTTPRetriever(endpoint=url).retrieve_batch([]) == []


# ---------------------------------------------------------------------------
# transport retries: a transient fault must not be fatal to a 31-hour run
# ---------------------------------------------------------------------------
class _FakeResp:
    def __init__(self, obj):
        self._body = json.dumps(obj).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _ok_body(queries):
    return {"result": [[{"document": {"contents": f"doc-for:{q}"}}] for q in queries]}


def test_transient_timeout_is_retried_not_fatal(monkeypatch):
    """THE incident: the 2026-08-19 online real run (150x256, ~31h) died on ONE socket
    read timeout from a 256-query batch -- no retry existed. A retrieval POST is a pure
    idempotent lookup, so the client must absorb a bounded number of transport faults."""
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("timed out")
        qs = json.loads(req.data.decode("utf-8"))["queries"]
        return _FakeResp(_ok_body(qs))

    slept = []
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", lambda s: slept.append(s))
    r = HTTPRetriever(endpoint="http://x/retrieve")
    out = r.retrieve_batch([("q0", 2, 0), ("q1", 2, 0)])
    assert out == [["doc-for:q0"], ["doc-for:q1"]]
    assert calls["n"] == 2, "expected exactly one retry after the transient fault"
    assert slept and slept[0] > 0, "backoff must actually wait"


def test_a_dead_retriever_still_fails_loudly_after_bounded_retries(monkeypatch):
    calls = {"n": 0}

    def always_timeout(req, timeout=None):
        calls["n"] += 1
        raise TimeoutError("timed out")

    monkeypatch.setattr("urllib.request.urlopen", always_timeout)
    monkeypatch.setattr("time.sleep", lambda s: None)
    r = HTTPRetriever(endpoint="http://x/retrieve", retries=3)
    with pytest.raises(RetrieverHTTPError, match="4 attempts"):
        r.retrieve("q", k=2)
    assert calls["n"] == 4, "retries must be BOUNDED -- a dead server is not retried forever"


def test_http_errors_are_answers_not_hiccups_and_are_never_retried(monkeypatch):
    """A status line is the server SPEAKING: the pinned Search-R1 500s are deterministic
    contract violations, and retrying one re-asks a question whose answer will not
    change (while hammering a struggling server)."""
    import io
    import urllib.error
    calls = {"n": 0}

    def http_500(req, timeout=None):
        calls["n"] += 1
        raise urllib.error.HTTPError("http://x/retrieve", 500, "boom", {},
                                     io.BytesIO(b"unpack error"))

    monkeypatch.setattr("urllib.request.urlopen", http_500)
    r = HTTPRetriever(endpoint="http://x/retrieve")
    with pytest.raises(RetrieverHTTPError) as ei:
        r.retrieve("q", k=2)
    assert ei.value.code == 500
    assert calls["n"] == 1, "an HTTP error must not be retried"
