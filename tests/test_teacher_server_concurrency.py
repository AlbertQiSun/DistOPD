"""Two clients, one engine: scoring must be serialized, or their outputs cross.

WHAT HAPPENED. The teacher is served by ThreadingHTTPServer, which handles every request in
its own thread, and every thread reached the same in-process vLLM engine with no lock.
LLM.generate() drains whatever outputs the engine's step() produces, so two concurrent calls
steal each other's results. This surfaced 2026-08-17 within the first minute of the first
moment TWO clients ever existed -- the real-scale precompute and the online arm started
together, and both died in the same second: one got 4 outputs for its 3 sequences, the other
2 for its 3. One client's surplus was the other's deficit.

The server-side pairing guard ("results are paired to requests by POSITION ... Refusing") is
what turned silent logprob-crossing into a loud 400, and it must stay. These tests are for
the fix ON TOP of it: a lock around the scoring call, so the guard never needs to fire for
this reason again.

THE TEST RUNS THE REAL SERVER -- make_handler + ThreadingHTTPServer, real sockets, two real
client threads. Only the scorer is fake, and the fake is what makes the race PROVABLE: it
records how many calls are inside it at once. Asserting on the lock attribute's existence
would be decoration; asserting max_in_flight == 1 under fire is the property.

vLLM is NOT imported: make_handler takes the scorer as an argument, which is what makes the
server's concurrency behaviour testable on a laptop.
"""
from __future__ import annotations

import json
import pathlib
import sys
import threading
import time
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from http.server import ThreadingHTTPServer  # noqa: E402

sys.path.insert(0, str(ROOT / "slurm"))
from teacher_server import make_handler  # noqa: E402


class _OverlapDetectingScorer:
    """Counts concurrent entries. sleep() widens the race window so an unserialized server
    fails RELIABLY rather than only under lucky timing."""

    def __init__(self, dwell: float = 0.05) -> None:
        self.dwell = dwell
        self.in_flight = 0
        self.max_in_flight = 0
        self.calls = 0
        self._mu = threading.Lock()

    def _enter(self):
        with self._mu:
            self.in_flight += 1
            self.calls += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)

    def _exit(self):
        with self._mu:
            self.in_flight -= 1

    def __call__(self, input_ids, score_positions=None):
        self._enter()
        try:
            time.sleep(self.dwell)
            return [None] + [0.0] * (len(input_ids) - 1), len(input_ids)
        finally:
            self._exit()

    def batch(self, parsed):
        self._enter()
        try:
            time.sleep(self.dwell)
            return [([None] + [0.0] * (len(ids) - 1), len(ids)) for ids, _sp in parsed]
        finally:
            self._exit()


@pytest.fixture()
def server():
    scorer = _OverlapDetectingScorer()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(scorer, {"fp": "test"}))
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", scorer
    httpd.shutdown()


def _post(url, payload):
    req = urllib.request.Request(f"{url}/score_logprobs",
                                 data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def test_concurrent_batch_requests_never_overlap_inside_the_scorer(server):
    """THE RACE, reproduced: several clients fire batch requests simultaneously. Without the
    lock the fake scorer observes >1 caller inside it at once -- which for the real engine
    means crossed outputs -- and with it, never."""
    url, scorer = server
    payload = {"batch": [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5]}]}
    errors, threads = [], []

    def fire():
        try:
            r = _post(url, payload)
            assert len(r["results"]) == 2
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    for _ in range(6):
        threads.append(threading.Thread(target=fire))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    assert scorer.calls == 6
    assert scorer.max_in_flight == 1, (
        f"{scorer.max_in_flight} scoring calls ran INSIDE the engine simultaneously. "
        "LLM.generate() drains the engine's outputs, so concurrent calls steal each other's "
        "results -- this is the 'engine returned 4 outputs for 3 scored sequences' failure.")


def test_single_and_batch_requests_share_the_same_lock(server):
    """A lock covering only one of the two entry points still crosses outputs when a batch
    client and a single-sequence client arrive together -- the two shapes reach the same
    engine."""
    url, scorer = server
    errors, threads = [], []

    def fire_single():
        try:
            _post(url, {"input_ids": [1, 2, 3, 4]})
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    def fire_batch():
        try:
            _post(url, {"batch": [{"input_ids": [5, 6, 7]}]})
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    for fn in (fire_single, fire_batch, fire_single, fire_batch):
        threads.append(threading.Thread(target=fn))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    assert scorer.max_in_flight == 1, "the two entry points do not share one lock"


def test_health_checks_are_not_blocked_by_a_slow_scoring_call(server):
    """The reason to keep ThreadingHTTPServer at all. wait_for_teacher polls /health; if the
    lock covered the whole handler, a long scoring call would make the server look DOWN to
    every waiting job."""
    url, scorer = server
    scorer.dwell = 1.0

    slow = threading.Thread(target=lambda: _post(url, {"input_ids": [1, 2, 3]}))
    slow.start()
    time.sleep(0.15)          # ensure the scoring call is inside the engine

    t0 = time.perf_counter()
    with urllib.request.urlopen(f"{url}/health", timeout=5) as resp:
        health = json.loads(resp.read().decode("utf-8"))
    elapsed = time.perf_counter() - t0
    slow.join()

    assert health["status"] == "ok"
    assert elapsed < 0.5, (
        f"/health took {elapsed:.2f}s while a scoring call held the engine -- the lock is "
        "over the whole handler, so a busy teacher reads as a dead one to wait_for_teacher")
