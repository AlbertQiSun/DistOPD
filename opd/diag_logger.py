"""JSONL sink for DiagnosticRecords. One line per step; the plotting script is the
only reader. Kept separate from the runner so collection (what to measure) and
persistence (where it lands) evolve independently.

OFFLINE note: the offline source cycles a FROZEN precomputed set, so its context
quality does not drift with pi_theta -- it is the flat reference line the online
curve is read against. Nothing special is needed here; freezing lives in the data.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from opd.diagnostics import DiagnosticRecord


class DiagnosticLogger:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rotated = self._rotate_existing()
        if rotated is not None:
            print(f"[diag] {self.path} already held a previous run -> rotated to {rotated}")
        self._f = self.path.open("w", encoding="utf-8")

    def _rotate_existing(self) -> Path | None:
        """Move an existing non-empty log aside instead of truncating it.

        A plain open("w") here destroyed the evidence of a crashed run: the 2026-08-18
        online OOM left 11 steps of diagnostics on disk, and the relaunch silently
        overwrote them -- the exact forensics needed to diagnose the crash. Rotation is
        the the project conventions persistence rule applied to the writer: a rerun may claim the
        canonical filename, never a predecessor's rows. The suffix is the old file's
        mtime (when that run last wrote), so rotated names order by run, not by relaunch.
        """
        if not self.path.exists() or self.path.stat().st_size == 0:
            return None
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(self.path.stat().st_mtime))
        base = f"{self.path.stem}.{stamp}"
        cand = self.path.with_name(base + self.path.suffix)
        n = 1
        while cand.exists():                 # two relaunches inside one second
            cand = self.path.with_name(f"{base}.{n}" + self.path.suffix)
            n += 1
        self.path.rename(cand)
        return cand

    def log(self, record: DiagnosticRecord) -> None:
        self._f.write(json.dumps(record.to_dict()) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> "DiagnosticLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
