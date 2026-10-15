"""M-SFT Stage 2 PROVENANCE gate: the export/verify pipeline must scope to what THIS run's
manifest marks kept -- NOT to whatever .pt files happen to sit in records/.

WHY THIS EXISTS
---------------
A prior run's contamination shipped 329 stale old-retriever trajectories into a shard because
both the exporter and the verifier globbed records/ ("what's on disk") instead of reading the
manifest ("what this run kept"). The stale files were structurally valid (mask/format/token
types all correct), so every structural gate PASSED -- the only reason it was caught was a
human noticing 3547 vs 3876. These tests make provenance a STRUCTURAL guarantee: a stale .pt at
an index the run marked dropped can never enter the shard, and a count mismatch fails loudly.
"""
from __future__ import annotations

import json
import os
import tempfile

import torch

from opd.trajectory import TerminationReason, TokenType, Trajectory
from sft.datagen import build_sft_record
from sft.export_llamafactory import export
from sft.pack import (count_manifest_kept, load_stage1_records_from_manifest,
                      pack_records_from_manifest)

P, A, O = TokenType.PROMPT, TokenType.ACTION, TokenType.OBSERVATION


def _record(seed: int) -> dict:
    """A small well-formed record with an OBSERVATION span. `seed` shifts the ids so a stale
    record is byte-distinguishable from a kept one."""
    types = [P, P, A, A, O, O, O, A, A]
    n = len(types)
    ids = torch.arange(seed, seed + n, dtype=torch.long)
    tt = torch.tensor([int(t) for t in types], dtype=torch.long)
    traj = Trajectory(ids, tt, terminated=TerminationReason.ANSWERED, num_search_calls=1,
                      meta={"question": f"q{seed}", "final_answer": "Paris",
                            "turns": [{"turn": 0, "query": "capital", "docs": ["Paris ..."]}]})
    return build_sft_record(traj)


def _write_record(records_dir: str, i: int, seed: int) -> str:
    path = os.path.join(records_dir, f"sft_{i:04d}.pt")
    torch.save(_record(seed), path)
    return path


def _build_shard(d: str, kept_idx: list[int], dropped_idx: list[int],
                 stale_idx: list[int]) -> str:
    """Materialize a Stage-1 out_dir with:
      - a real .pt at every kept index (manifest kept=True),
      - a manifest that also lists dropped indices (kept=False),
      - a STALE .pt planted at each `stale_idx` (an index the manifest marks dropped),
        simulating a prior run's leftover that datagen's drop path never deleted.
    Returns the out_dir.
    """
    records_dir = os.path.join(d, "records")
    os.makedirs(records_dir, exist_ok=True)
    manifest = []
    for i in kept_idx:
        _write_record(records_dir, i, seed=100 + i)
        manifest.append({"i": i, "kept": True, "path": f"{records_dir}/sft_{i:04d}.pt"})
    for i in dropped_idx:
        manifest.append({"i": i, "kept": False, "reason": "empty_query"})
    # plant stale leftovers at dropped indices (glob would pick these up; manifest must not).
    for i in stale_idx:
        assert i in dropped_idx, "a stale file only contaminates at a DROPPED index"
        _write_record(records_dir, i, seed=9000 + i)
    manifest.sort(key=lambda r: r["i"])
    with open(os.path.join(d, "manifest.jsonl"), "w", encoding="utf-8") as f:
        for r in manifest:
            f.write(json.dumps(r) + "\n")
    return d


def test_pack_from_manifest_ignores_stale_files_on_disk():
    with tempfile.TemporaryDirectory() as d:
        _build_shard(d, kept_idx=[0, 1, 3, 4], dropped_idx=[2, 5], stale_idx=[2, 5])
        # records/ has 6 files (4 kept + 2 stale); manifest marks only 4 kept.
        assert len(os.listdir(os.path.join(d, "records"))) == 6
        rows = pack_records_from_manifest(d)
        assert len(rows) == 4                                  # stale 2 & 5 did NOT enter
        # ids of the packed rows must be the KEPT seeds (100+i), never the stale (9000+i).
        first_ids = [r["input_ids"][0] for r in rows]
        assert first_ids == [100 + i for i in (0, 1, 3, 4)]
        assert count_manifest_kept(d) == 4


def test_missing_manifest_raises():
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "records"))
        _write_record(os.path.join(d, "records"), 0, seed=1)
        try:
            pack_records_from_manifest(d)
        except FileNotFoundError as e:
            assert "manifest.jsonl" in str(e)
        else:
            raise AssertionError("must refuse to pack when the manifest is absent (no provenance)")


def test_manifest_kept_but_file_missing_raises():
    with tempfile.TemporaryDirectory() as d:
        _build_shard(d, kept_idx=[0, 1], dropped_idx=[], stale_idx=[])
        os.remove(os.path.join(d, "records", "sft_0001.pt"))   # kept per manifest, gone from disk
        try:
            pack_records_from_manifest(d)
        except FileNotFoundError as e:
            assert "kept but" in str(e)
        else:
            raise AssertionError("must refuse a partial shard when a kept record is missing")


def test_verifier_and_exporter_agree_via_manifest():
    # the load-bearing double-blind-spot fix: the verifier's record loader is the SAME
    # manifest-scoped source as the exporter, so both ignore the stale files identically.
    with tempfile.TemporaryDirectory() as d:
        _build_shard(d, kept_idx=[0, 1, 3], dropped_idx=[2], stale_idx=[2])
        recs = load_stage1_records_from_manifest(d)
        assert len(recs) == 3
        assert [int(r["input_ids"][0]) for r in recs] == [100, 101, 103]


def test_export_end_to_end_scopes_to_manifest():
    # full exporter path with a stale file present: pretokenized.jsonl must have exactly the
    # kept count, and the Layer-3 count assertion must hold (no ValueError).
    with tempfile.TemporaryDirectory() as d:
        _build_shard(d, kept_idx=[0, 1, 3, 4], dropped_idx=[2], stale_idx=[2])
        out = os.path.join(d, "shard")
        summary = export(d, out, cutoff_len=64, base_model="fake/Qwen3-4B-Base")
        assert summary["records"] == 4
        with open(os.path.join(out, "pretokenized.jsonl"), encoding="utf-8") as f:
            assert sum(1 for line in f if line.strip()) == 4
