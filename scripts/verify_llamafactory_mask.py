"""M-SFT Stage 2 framework-seam GATE (run on HPC, where LlamaFactory is installed).

sft/pack.py proves LOCALLY that our pre-tokenized shard preserves the action-only mask
through a faithful round-trip. This script proves it against LlamaFactory's ACTUAL processed
data: it loads what LlamaFactory will TRAIN on (its tokenized/cached dataset) and, per
example, runs sft.pack.assert_mask_roundtrip vs the Stage-1 record -- catching a re-tokenize
(input_ids drift) or a role re-mask (OBSERVATION tokens supervised). This is the SFT analogue
of gate-1/gate-3: do not trust the pi_ref run until this passes.

The processed-dataset format is a LlamaFactory-version detail, so the loader is deliberately
tolerant -- it accepts either a HuggingFace arrow dir (datasets.load_from_disk) or a JSONL,
and reads `input_ids` + `labels` per row. Point --processed at whatever your LlamaFactory
build dumped (e.g. its `tokenized_path`, or a cached arrow under the run's cache dir). Rows
must be in the SAME ORDER as the exported shard (export writes records sorted by filename).

    python scripts/verify_llamafactory_mask.py <stage1_out_dir> \
        --processed <LlamaFactory tokenized dataset> --cutoff-len 4096

If --processed is omitted it self-checks the exported pretokenized shard (identity round-trip)
-- useful to confirm the export itself is clean before the LlamaFactory step.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sft.pack import check_mask_roundtrip, load_stage1_records_from_manifest


def _load_stage1_records(stage1_out_dir: str) -> list[dict]:
    """Stage-1 .pt records the run's manifest marks kept, in export order.

    Reads the MANIFEST (not a glob over records/) so a stale .pt from a prior run cannot enter
    the comparison. This is the SAME provenance guard the exporter uses: both sides agree on
    "what this run produced". Globbing here was the second half of the double blind spot -- a
    contaminated export (N+stale rows) matched a globbed record set (also N+stale), so the count
    check passed and the mask verify PASSED on mixed-provenance data.
    """
    return load_stage1_records_from_manifest(stage1_out_dir)


def _load_processed(processed: str) -> list[dict]:
    """Load what LlamaFactory will train on. Accepts an arrow dir or a JSONL; returns a list
    of {"input_ids": [...], "labels": [...]}."""
    if os.path.isdir(processed):
        from datasets import load_from_disk
        ds = load_from_disk(processed)
        if hasattr(ds, "keys"):                      # DatasetDict -> take the train split
            ds = ds[list(ds.keys())[0]]
        return [{"input_ids": ex["input_ids"], "labels": ex["labels"]} for ex in ds]
    rows = []
    with open(processed, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                rows.append({"input_ids": d["input_ids"], "labels": d["labels"]})
    return rows


def _load_self_shard(stage1_out_dir: str, shard_dir: str) -> list[dict]:
    path = os.path.join(shard_dir, "pretokenized.jsonl")
    return _load_processed(path)


def verify(stage1_out_dir: str, processed: str | None, cutoff_len: int,
           shard_dir: str | None) -> int:
    records = _load_stage1_records(stage1_out_dir)
    if processed is None:
        src = shard_dir or os.path.join(stage1_out_dir, "llamafactory_shard")
        proc = _load_self_shard(stage1_out_dir, src)
        where = f"self-check (exported shard {src})"
    else:
        proc = _load_processed(processed)
        where = f"LlamaFactory processed data ({processed})"

    if len(proc) != len(records):
        print(f"FAIL: {len(proc)} processed rows != {len(records)} Stage-1 records "
              "(order/count mismatch -- packing may have merged/reordered; align before trusting).")
        return 1

    n_fail = 0
    first_reasons = []
    for i, (rec, pr) in enumerate(zip(records, proc)):
        n = int(rec["input_ids"].shape[0])
        rt_ids = torch.tensor(pr["input_ids"][:n], dtype=torch.long)
        rt_lab = torch.tensor(pr["labels"][:n], dtype=torch.long)
        # a longer processed row means the record was PACKED with neighbors -- this per-record
        # verifier needs unpacked rows; if lengths differ, flag it (use neat-packing metadata
        # to split, or verify pre-packing).
        if len(pr["input_ids"]) != n:
            n_fail += 1
            if len(first_reasons) < 5:
                first_reasons.append(f"row {i}: processed len {len(pr['input_ids'])} != record len {n} "
                                     "(row is packed with neighbors; verify pre-packing or unpack first)")
            continue
        res = check_mask_roundtrip(rec["token_type"], rec["input_ids"], rt_ids, rt_lab)
        if not res.ok:
            n_fail += 1
            if len(first_reasons) < 5:
                first_reasons.append(f"row {i}: {res.reason}")

    print("=" * 70)
    print(f"SFT mask round-trip verify vs {where}")
    print(f"records={len(records)}  cutoff_len={cutoff_len}  failures={n_fail}")
    if n_fail:
        print("-" * 70)
        for r in first_reasons:
            print(f"  - {r}")
        print("VERDICT: FAIL -- LlamaFactory did not preserve the action-only mask. Do NOT "
              "train pi_ref on this. Feed pre-tokenized ids+labels; disable re-tokenization / "
              "role re-masking.")
    else:
        print("VERDICT: PASS -- every OBSERVATION token stayed masked; supervised set == ACTION.")
    print("=" * 70)
    return 1 if n_fail else 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Gate: assert the action-only mask survived "
                                             "LlamaFactory processing.")
    ap.add_argument("stage1_out_dir", help="the gen_sft_data.py --out-dir (has records/)")
    ap.add_argument("--processed", default=None,
                    help="LlamaFactory tokenized dataset (arrow dir or JSONL); omit to self-check "
                         "the exported shard")
    ap.add_argument("--shard-dir", default=None,
                    help="exported shard dir for the self-check (default <out_dir>/llamafactory_shard)")
    ap.add_argument("--cutoff-len", type=int, default=4096)
    args = ap.parse_args()
    sys.exit(verify(args.stage1_out_dir, args.processed, args.cutoff_len, args.shard_dir))


if __name__ == "__main__":
    main()
