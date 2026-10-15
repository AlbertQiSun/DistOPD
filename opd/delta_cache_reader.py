"""Joins one trajectory's delta inputs from the two sharded caches next to the precompute:
    <cache>/ref_topk/part_<start>.npz      pi_ref top-K' at every scored position (step 1)
    <cache>/teacher_head/part_<start>.npz  the teacher head at the SOFT positions      (step 2)
into the DeltaCache opd.delta_loss consumes.  Parts are 1000 trajectories each and sorted by
(traj, pos); the offline arm walks trajectories in cache order, so at most one part of each
kind is resident at a time (the moe line's arrays are ~7 GB + ~20 GB on disk -- not RAM).
"""
from __future__ import annotations

import json
import os

import numpy as np

from .delta_loss import DeltaCache, build_delta_cache
from .loss import DEFAULT_TAU

REF_FIELDS = ("traj", "pos", "y", "top_ids", "top_lp", "lp_y")
HEAD_FIELDS = ("traj", "pos", "y", "kq", "q_ids", "q_lp", "extra_ids", "extra_lp", "floor_lp", "log_rest")


def _part_start(idx, shard):
    return (idx // shard) * shard


class _PartCursor:
    """The one part currently loaded, with a (traj -> row slice) index."""

    def __init__(self, directory, fields, shard):
        self.dir, self.fields, self.shard = directory, fields, shard
        self.start = None; self.arrays = None; self.bounds = None

    def rows(self, idx):
        s = _part_start(idx, self.shard)
        if s != self.start:
            p = os.path.join(self.dir, f"part_{s:06d}.npz")
            if not os.path.exists(p):
                raise FileNotFoundError(f"{p} missing -- the cache is incomplete at trajectory {idx}")
            z = np.load(p, allow_pickle=True)
            self.arrays = {f: z[f] for f in self.fields if f in z.files}
            tr = self.arrays["traj"]
            # parts are written in trajectory order; searchsorted gives each trajectory's slab
            self.bounds = (tr, np.searchsorted(tr, np.arange(s, s + self.shard), side="left"),
                           np.searchsorted(tr, np.arange(s, s + self.shard), side="right"))
            self.start = s
        tr, lo, hi = self.bounds
        a, b = int(lo[idx - s]), int(hi[idx - s])
        return {f: v[a:b] for f, v in self.arrays.items()}


class DeltaCacheReader:
    def __init__(self, precompute_dir, tau=DEFAULT_TAU, shard=1000, require_usable=True, support="union"):
        self.support = support                    # "union" | "teacher" (opd.delta_loss.ABLATIONS)
        self.ref = _PartCursor(os.path.join(precompute_dir, "ref_topk"), REF_FIELDS, shard)
        self.head = _PartCursor(os.path.join(precompute_dir, "teacher_head"), HEAD_FIELDS, shard)
        self.tau = tau
        for name, d in (("ref_topk", self.ref.dir), ("teacher_head", self.head.dir)):
            if not os.path.isdir(d):
                raise FileNotFoundError(f"{d} missing -- run scripts/{'ref_topk_dump' if name == 'ref_topk' else 'teacher_head_dump'}.py")
        if require_usable:
            for name in ("ref_topk", "teacher_head"):
                done = os.path.join(precompute_dir, f"{name}.npz")
                if not os.path.exists(done):
                    raise RuntimeError(f"{done} missing: the {name} dump is not COMPLETE (every part present) -- refusing to train on it")
                z = np.load(done, allow_pickle=True)
                if "usable" in z.files and not bool(z["usable"]):
                    raise RuntimeError(f"{done}: gates failed and not adjudicated -- refusing to train on it")

    def get(self, idx, teacher_logp):
        """DeltaCache for cache trajectory `idx`; teacher_logp = the record's align-to-t vector."""
        r = self.ref.rows(idx)
        if len(r["pos"]) == 0:
            raise RuntimeError(f"ref_topk has no rows for trajectory {idx}")
        h = self.head.rows(idx)
        head = {}
        for k in range(len(h["pos"])):
            kq = int(h["kq"][k])
            ex_ids = [int(t) for t in h["extra_ids"][k] if t >= 0]
            ex_lp = [None if not np.isfinite(v) else float(v) for v in h["extra_lp"][k][:len(ex_ids)]]
            head[int(h["pos"][k])] = dict(top_ids=[int(t) for t in h["q_ids"][k][:kq]], top_lp=[float(v) for v in h["q_lp"][k][:kq]],
                                          extra_ids=ex_ids, extra_lp=ex_lp, floor_lp=float(h["floor_lp"][k]), log_rest=float(h["log_rest"][k]))
        tl = np.asarray(teacher_logp, np.float64)[r["pos"]]
        return build_delta_cache(r, head, tl, tau=self.tau, support=self.support)
