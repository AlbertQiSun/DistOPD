"""DiagnosticLogger must never truncate a previous run's rows. (local, zero-GPU)

The 2026-08-18 online OOM left 11 steps of diagnostics in runs/diag_online.jsonl; the
relaunch opened the same path with "w" and silently destroyed them -- the exact rows
needed to diagnose the crash. These tests pin the fix at the PRODUCER: a new logger on
an occupied path must move the old file aside (named by ITS mtime, so rotated files
order by run), and every previously written row must survive a relaunch byte-for-byte.
"""
from __future__ import annotations

import json
import os

from opd.diag_logger import DiagnosticLogger


class _Rec:
    def __init__(self, d):
        self._d = d

    def to_dict(self):
        return self._d


def _lines(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def test_relaunch_rotates_instead_of_truncating(tmp_path):
    p = tmp_path / "diag_online.jsonl"
    with DiagnosticLogger(str(p)) as lg:
        lg.log(_Rec({"step": 0, "run": "crashed"}))
        lg.log(_Rec({"step": 1, "run": "crashed"}))

    with DiagnosticLogger(str(p)) as lg:          # the relaunch
        lg.log(_Rec({"step": 0, "run": "relaunch"}))

    siblings = sorted(x for x in os.listdir(tmp_path) if x != "diag_online.jsonl")
    assert len(siblings) == 1, f"old run must survive as exactly one rotated file: {siblings}"
    old = _lines(tmp_path / siblings[0])
    assert [r["run"] for r in old] == ["crashed", "crashed"]   # nothing lost, order kept
    assert [r["run"] for r in _lines(p)] == ["relaunch"]       # canonical name = new run only


def test_rotated_name_carries_the_old_runs_mtime(tmp_path):
    p = tmp_path / "diag.jsonl"
    with DiagnosticLogger(str(p)) as lg:
        lg.log(_Rec({"step": 0}))
    os.utime(p, (1_000_000_000, 1_000_000_000))    # pin mtime -> deterministic stamp
    import time
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(1_000_000_000))

    with DiagnosticLogger(str(p)):
        pass
    assert (tmp_path / f"diag.{stamp}.jsonl").exists()


def test_second_relaunch_in_the_same_second_does_not_clobber_the_first(tmp_path):
    p = tmp_path / "diag.jsonl"
    for run in ("a", "b", "c"):
        with DiagnosticLogger(str(p)) as lg:
            lg.log(_Rec({"run": run}))
        # same pinned mtime every time -> same timestamp base -> collision path
        os.utime(p, (1_000_000_000, 1_000_000_000))
    with DiagnosticLogger(str(p)):
        pass
    rotated = [x for x in os.listdir(tmp_path) if x != "diag.jsonl"]
    assert len(rotated) == 3, f"all three runs must survive: {rotated}"
    survived = sorted(_lines(tmp_path / x)[0]["run"] for x in rotated)
    assert survived == ["a", "b", "c"]


def test_empty_or_absent_file_is_not_rotated(tmp_path):
    p = tmp_path / "diag.jsonl"
    with DiagnosticLogger(str(p)):                 # absent -> plain create
        pass
    assert os.listdir(tmp_path) == ["diag.jsonl"]
    with DiagnosticLogger(str(p)):                 # exists but EMPTY -> nothing to save
        pass
    assert os.listdir(tmp_path) == ["diag.jsonl"]
