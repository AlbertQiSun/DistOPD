#!/usr/bin/env python3
"""Produce one CELL of the teacher x trajectory-batch cross-scoring matrix.

    # off-diagonal: score cache C's trajectories with teacher T's running server
    python scripts/cross_score_cache.py --cache precomputed_dense \\
        --teacher-endpoint http://<node>:8135/score_logprobs --teacher-name dense \\
        --cache-name dense --tokenizer $OPD_MODELS_DIR/pi_ref_miro \\
        --n 64 --out runs/cross/cell_miroT_denseC.pt

    # diagonal: the cache already holds this teacher's logps -- no server, no GPU
    python scripts/cross_score_cache.py --cache precomputed_miro --reuse-cached \\
        --teacher-name miro --cache-name miro \\
        --tokenizer $OPD_MODELS_DIR/pi_ref_miro --out runs/cross/cell_miroT_miroC.pt

WHAT THE MATRIX SEPARATES. Three teachers x three precompute caches:
  - down a COLUMN (same trajectories, different teachers): any metric difference is a
    property of the MODEL -- the trajectories are bit-identical.
  - along a ROW (same teacher, different trajectory batches): any difference is a property
    of the TRAJECTORIES (whose student produced them, what they talk about).
  - off-diagonal vs diagonal: the distribution-shift penalty itself -- is "uniformly lower
    logp on someone else's trajectories" every teacher's behaviour, or one teacher's?

This script only COLLECTS: per sampled trajectory, the teacher's logp at every ACTION token
(token_type==1), each token's role (think_body / query_body / answer_body / tags), and
whether each adjacent scored pair is CONTIGUOUS or separated by an observation gap. The
metrics live in analyze_cross_scores.py, so re-analysis never re-scores -- by design,
the raw series is the artifact, not the summary.

TOKENIZER IDENTITY IS A PRECONDITION, NOT AN ASSUMPTION. The cached input_ids were produced
by the STUDENT's tokenizer; sending them to a teacher is meaningful only if that teacher's
tokenizer maps the same ids to the same strings. Within a line this was verified by
scripts/check_tokenizer_identity.py; for CROSS cells run it once per (teacher, cache-owner)
pair before trusting the numbers. The served tokenizer_name from /health is recorded into
the cell file so the analyzer can display what actually scored it.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import urllib.request

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opd.trajectory import TokenType
from scripts.advantage_by_role import roles_for


def cache_paths(cache_dir: str) -> list[str]:
    man = os.path.join(cache_dir, "manifest.json")
    if os.path.exists(man):
        with open(man, encoding="utf-8") as f:
            names = json.load(f)["paths"]
        return [os.path.join(cache_dir, p) for p in names]
    return sorted(glob.glob(os.path.join(cache_dir, "traj_*.pt")))


def served_identity(endpoint: str) -> dict:
    """What /health says is running -- recorded verbatim into the cell."""
    base = endpoint.rsplit("/", 1)[0]
    with urllib.request.urlopen(f"{base}/health", timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8")).get("backend", {})


def collect_record(rec: dict, tok, logp_full: torch.Tensor) -> dict:
    """One trajectory's raw series, aligned to its ACTION positions.

    `gap_before[i]` marks whether scored position i is NOT adjacent to scored position i-1
    in the original sequence -- i.e. an observation (or prompt) was injected between them.
    The analyzer must never pool a Δlogp across such a gap with the within-run Δs: a jump
    across injected retrieval text measures the teacher's reaction to new state, not the
    smoothness of its distribution, and the two hypotheses those numbers feed are different.
    """
    ids = rec["input_ids"].long()
    tt = rec["token_type"].long()
    pos = torch.nonzero(tt == int(TokenType.ACTION), as_tuple=False).squeeze(-1)
    roles = roles_for(ids.tolist(), tok)
    lp = logp_full[pos]
    gap_before = [False] + [(int(pos[i]) - int(pos[i - 1])) != 1
                            for i in range(1, len(pos))]
    return {
        "positions": pos.tolist(),
        "logp": lp.float().tolist(),
        "roles": [roles[int(p)] for p in pos],
        "gap_before": gap_before,
        "terminated": str(rec.get("terminated")),
        "num_search_calls": int(rec.get("num_search_calls") or 0),
        "n_tokens": int(ids.shape[0]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", required=True, help="precompute dir (manifest.json + traj_*.pt)")
    ap.add_argument("--teacher-endpoint", default=None,
                    help=".../score_logprobs of a RUNNING teacher server")
    ap.add_argument("--reuse-cached", action="store_true",
                    help="diagonal cell: read the cache's own teacher_logp; no server")
    ap.add_argument("--teacher-name", required=True, help="row label (miro|moe|dense)")
    ap.add_argument("--cache-name", required=True, help="column label (miro|moe|dense)")
    ap.add_argument("--tokenizer", required=True,
                    help="tokenizer path for ROLE decoding (any checkpoint sharing the "
                         "student vocab; roles are text-space, see advantage_by_role)")
    ap.add_argument("--n", type=int, default=64, help="trajectories to sample")
    ap.add_argument("--stride", type=int, default=None,
                    help="take every stride-th record (default: spread n evenly)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if bool(args.teacher_endpoint) == bool(args.reuse_cached):
        raise SystemExit("exactly one of --teacher-endpoint / --reuse-cached")

    paths = cache_paths(args.cache)
    if not paths:
        raise SystemExit(f"FATAL: no trajectories under {args.cache}")
    stride = args.stride or max(1, len(paths) // args.n)
    picked = paths[::stride][:args.n]
    print(f"[cross] cell ({args.teacher_name} teacher x {args.cache_name} cache): "
          f"{len(picked)} of {len(paths)} trajectories, stride {stride}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    scorer = None
    backend = {"source": "cache teacher_logp (diagonal)"}
    if args.teacher_endpoint:
        backend = served_identity(args.teacher_endpoint)
        print(f"[cross] serving: {backend.get('tokenizer_name', '?')}")

    records = []
    for i, p in enumerate(picked):
        rec = torch.load(p, weights_only=False, map_location="cpu")
        if args.reuse_cached:
            tlp = rec.get("teacher_logp")
            if tlp is None:
                raise SystemExit(f"FATAL: {p} has no teacher_logp; this cache cannot be a "
                                 "diagonal cell -- score it via --teacher-endpoint")
            logp_full = tlp.float()
        else:
            if scorer is None:
                from teacher.http_scorer import TeacherHTTPScorer
                scorer = TeacherHTTPScorer(
                    args.teacher_endpoint, dtype="bfloat16",
                    tokenizer_name=str(backend.get("tokenizer_name", "")), verify=False)
            tt = rec["token_type"].long()
            pos = torch.nonzero(tt == int(TokenType.ACTION), as_tuple=False).squeeze(-1)
            logp_full = scorer.logprobs(rec["input_ids"].long(),
                                        score_positions=pos.tolist())
        records.append({"path": os.path.basename(p),
                        **collect_record(rec, tok, logp_full)})
        if (i + 1) % 8 == 0 or (i + 1) == len(picked):
            print(f"[cross]   {i + 1}/{len(picked)}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"teacher": args.teacher_name, "cache": args.cache_name,
                "cache_dir": os.path.abspath(args.cache), "backend": backend,
                "records": records}, args.out)
    print(f"[cross] wrote {args.out} ({len(records)} trajectories)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
