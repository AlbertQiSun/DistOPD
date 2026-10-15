#!/usr/bin/env python3
"""Dump the OBSERVATION spans of saved SFT records so a human can EYEBALL the injected
retrieved-doc text -- the check the aggregate health report (check_sft_stats.py) is BLIND to.

WHY THIS EXISTS
---------------
check_sft_stats.py reports STRUCTURE (no_search / near_cap / hop dist). It cannot see whether
the CONTENT of an observation is clean. After the retriever migration we now POST
return_scores=true (the pinned Search-R1 endpoint unpacks (results, scores) unconditionally),
which changes the server's response SHAPE to {"document": {...}, "score": ...}. The client's
_doc_text is supposed to read only document.contents and drop the score, so the trajectory's
observation should be pure passage text. This tool PROVES that empirically: if a score float
or structural garbage ever leaked into the doc text, it would be tokenized straight into the
OBSERVATION span and silently poison the SFT data while every aggregate stat stays green.

WHAT IT SHOWS
-------------
For each of the first --n KEPT records (out_dir/records/sft_*.pt), it reconstructs each
contiguous OBSERVATION span (token_type == TokenType.OBSERVATION) from input_ids and decodes
it with the SAME tokenizer generation used (add_special_tokens=False round-trips), printing it
verbatim. A CLEAN observation is: OBS_PREFIX + <information> + "<title>\\n<passage>" lines +
</information> + OBS_SUFFIX -- and NO trailing score numbers mixed into the passage text.

    python scripts/dump_sft_observation.py runs/sft_shard1_smoke --n 3
    python scripts/dump_sft_observation.py runs/sft_shard1_smoke --n 3 \
        --base-model /path/to/scratch/.cache/models/Qwen3-4B-Base
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opd.trajectory import TokenType  # noqa: E402

_OBS = int(TokenType.OBSERVATION)
# OPD_MODELS_DIR is exported by slurm/sites/$OPD_SITE.sh; the literal is the H20 cluster's, kept as
# the fallback so a pre-migration invocation without a site still works.
_DEFAULT_MODEL = os.path.join(
    os.environ.get("OPD_MODELS_DIR", "/path/to/scratch/.cache/models"), "Qwen3-4B-Base")


def _obs_spans(token_type: torch.Tensor) -> list[tuple[int, int]]:
    """Contiguous [start, end) index ranges where token_type == OBSERVATION."""
    tt = token_type.tolist()
    spans, i, n = [], 0, len(tt)
    while i < n:
        if tt[i] == _OBS:
            j = i
            while j < n and tt[j] == _OBS:
                j += 1
            spans.append((i, j))
            i = j
        else:
            i += 1
    return spans


def main() -> None:
    ap = argparse.ArgumentParser(description="Eyeball OBSERVATION text of saved SFT records.")
    ap.add_argument("out_dir", help="SFT shard dir (contains records/sft_*.pt)")
    ap.add_argument("--base-model", default=_DEFAULT_MODEL,
                    help="tokenizer source (local model dir); must match generation tokenizer")
    ap.add_argument("--n", type=int, default=3, help="#records to dump")
    ap.add_argument("--max-chars", type=int, default=0,
                    help="truncate each observation to this many chars (0 = full)")
    args = ap.parse_args()

    records = sorted(glob.glob(os.path.join(args.out_dir, "records", "sft_*.pt")))
    if not records:
        raise SystemExit(f"no records under {args.out_dir}/records/ -- did the smoke keep any?")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base_model)

    shown = 0
    for path in records:
        if shown >= args.n:
            break
        rec = torch.load(path, map_location="cpu", weights_only=False)
        ids = rec["input_ids"]
        tt = rec["token_type"]
        meta = rec.get("meta", {})
        spans = _obs_spans(tt)
        print("=" * 88)
        print(f"[{os.path.basename(path)}]  Q: {meta.get('question')}")
        print(f"  num_search_calls={meta.get('num_search_calls')}  "
              f"terminated={meta.get('terminated')}  observation_spans={len(spans)}")
        if not spans:
            print("  (no OBSERVATION span -- no_search trajectory; nothing retrieved to inject)")
        for k, (s, e) in enumerate(spans):
            text = tok.decode(ids[s:e].tolist())
            if args.max_chars and len(text) > args.max_chars:
                text = text[:args.max_chars] + f"... [+{len(text) - args.max_chars} chars]"
            print(f"  --- observation #{k} (tokens {s}:{e}, {e - s} tok) ---")
            print(text)
        shown += 1

    print("=" * 88)
    print(f"dumped {shown} record(s). CLEAN = passage text inside <information>...</information>, "
          "NO score floats mixed into the doc text.")


if __name__ == "__main__":
    main()
