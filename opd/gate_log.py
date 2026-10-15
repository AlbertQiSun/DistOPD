"""Append-only record of REAL-MACHINE gate runs -- makes "is this gate stale?" greppable.

WHY THIS EXISTS
---------------
git records when a gate's CODE changed, never when it RAN. So "when did this gate last
pass, and has anything since touched the path it validated?" was unanswerable from the
repo -- the only trace was whatever happened to be in slurm/logs, and in practice the
answer lived in one person's memory. That is how a gate goes stale, or turns out never to
have run at all, without anything failing: the determinism probe was the load-bearing
assumption under the whole online==offline argument and had NEVER been executed, which
only surfaced because someone thought to grep for it.

Each line records WHAT ran, its VERDICT, the COMMIT it ran at, and the CONFIGURATION it
ran under. The configuration matters as much as the verdict: a gate that passed under
settings production does not run certifies nothing about production (the probe used to
build its engine at gpu_memory_utilization 0.90 while the server ran 0.72).

    from opd.gate_log import append_gate_run
    append_gate_run("determinism_probe", "PASS", config="...", details={...})

    python -m opd.gate_log --gate resync_hot_update --verdict PASS \
        --config "..." --commit 6fbe2de --when 2026-08-12T10:27
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
from typing import Optional

DEFAULT_PATH = os.path.join("runs", "gate_log.jsonl")


def current_commit() -> Optional[str]:
    """HEAD at run time. Recorded per-entry because it is the only way to tell later
    whether a change since then touched the path this gate validated."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:                                        # noqa: BLE001
        return None


def append_gate_run(gate: str, verdict: str, *, config: str = "",
                    details: Optional[dict] = None, commit: Optional[str] = None,
                    when: Optional[str] = None, host: Optional[str] = None,
                    path: str = DEFAULT_PATH) -> dict:
    """Append one gate run. Never raises into the caller: a logging failure must not take
    down a gate that otherwise passed (and a missing line is visible, unlike a wrong one)."""
    rec = {
        "gate": gate,
        "verdict": verdict,
        "commit": commit or current_commit(),
        "when": when,                      # caller-supplied; None = "see the job log"
        # WHERE THE GATE RAN. Defaults to this machine, but must be passed when
        # backfilling a past run -- recording the laptop that typed the command
        # would make the provenance log itself a source of wrong provenance.
        "host": host or socket.gethostname(),
        "config": config,
        "details": details or {},
    }
    # REFUSE to write the real log from a test. This file is a PROVENANCE record: a
    # synthetic entry in it is worse than a missing one, because a later reader cannot tell
    # it from a real gate run. That already happened -- the probe's verdict tests exercised
    # the reporting path and wrote 8 fabricated "policy_batch_invariance_cost" records into
    # the tracked log. Tests must pass an explicit `path` (a tmp file); forgetting now fails
    # loudly instead of quietly corrupting the record.
    if os.environ.get("PYTEST_CURRENT_TEST") and path == DEFAULT_PATH:
        raise RuntimeError(
            "refusing to append to the real gate log from a test -- pass an explicit "
            "path=<tmp file>, or patch append_gate_run. A fabricated entry is "
            "indistinguishable from a real gate run to anyone reading this later."
        )
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[gate_log] {gate}: {verdict} (commit {rec['commit']}) -> {path}")
    except Exception as exc:                                 # noqa: BLE001
        print(f"[gate_log] WARNING: could not append to {path}: {exc!r}")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description="Append a real-machine gate run to the gate log.")
    ap.add_argument("--gate", required=True)
    ap.add_argument("--verdict", required=True, help="PASS | FAIL | <measured verdict>")
    ap.add_argument("--config", default="", help="engine/config summary the gate ran under")
    ap.add_argument("--commit", default=None, help="override; defaults to current HEAD")
    ap.add_argument("--when", default=None, help="ISO timestamp of the run")
    ap.add_argument("--host", default=None, help="node the gate RAN on (backfill)")
    ap.add_argument("--details", default=None, help="JSON object of extra fields")
    ap.add_argument("--path", default=DEFAULT_PATH)
    args = ap.parse_args()
    append_gate_run(args.gate, args.verdict, config=args.config,
                    details=json.loads(args.details) if args.details else None,
                    commit=args.commit, when=args.when, host=args.host,
                    path=args.path)


if __name__ == "__main__":
    main()
