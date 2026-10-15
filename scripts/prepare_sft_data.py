#!/usr/bin/env python3
"""Stage the *question* sets (NQ / HotpotQA / MuSiQue) with gold answers -- TRAIN for SFT
generation, DEV for Stage-3 acceptance (scripts/stage3_eval.py).

RUN THIS ON THE LOGIN NODE (internet is available there; compute nodes are OFFLINE).
It pulls a split (`--split train|dev`) from HuggingFace `RUC-NLPIR/FlashRAG_datasets` -- the
cleanest source because its unified `golden_answers` field is EXACTLY what
dataio.dataset.extract_gold prefers, so conversion is near-zero (we only re-emit question +
golden_answers, nothing else).

WHY TRAIN *and* DEV FROM THE SAME REPO
--------------------------------------
SFT trajectories are generated from the TRAIN split; Stage-3 acceptance evaluates on the DEV
split. Pulling BOTH from FlashRAG makes DEV disjoint from TRAIN by construction AND keeps the
gold field shape identical, so the contamination guard in stage3_eval.py compares apples to
apples (normalized-question intersection). (The guard still runs and PROVES disjointness --
FlashRAG is a second-hand aggregator and its train/dev partition is not guaranteed official.)

DELIBERATE, LOCKED DESIGN DECISION -- SHARED CORPUS (do NOT "optimize" this away)
--------------------------------------------------------------------------------
All Wikipedia-based simple sets (NQ / HotpotQA / 2Wiki / MuSiQue) share ONE Wikipedia dump +
E5 index at retrieval time. This is an INTENTIONAL controlled-variable choice, not an
oversight: (1) it is the standard Search-R1/FlashRAG setup we benchmark against; (2) the study
is about OPD training DYNAMICS, not the retrieval system -- a shared corpus removes a confound
(otherwise "π_ref worse on HotpotQA than NQ" cannot separate task difficulty from corpus
quality). The accepted cost is mild dump-version drift (gold evidence may sit in the dump in a
slightly different form than at annotation time) -- which is exactly why Stage-3 Gate B
(retrieval hit@k) is read as a LOWER-BOUND signal, not a precise search-ability measurement.
NEVER recommend per-dataset corpora for these Wikipedia sets. BrowseComp-Plus / FRAMES (the
final long-horizon eval) need a DIFFERENT, still-TBD corpus/retriever -- that is SEPARATE
final-eval infrastructure, handled on its own, never folded into this Wikipedia+E5 setup.

WHY these three files (and no quality filtering): configs/opd_qwen3.yaml's sft.data.sources
declares a difficulty MIX by per-source `limit` (1hop NQ / 2hop HotpotQA / multihop MuSiQue).
This script only STAGES the raw question pools; the mix + any drops happen later in the loader
and the generator, never here. We do NOT filter by answerability -- that is the "zero quality
filtering" principle of the SFT stage.

Path discovery is robust: we DON'T hardcode the in-repo layout. We list the repo files and
pick, per dataset, the file whose path contains the dataset dir AND a train split, preferring
.jsonl. If a dataset can't be resolved we FAIL LOUD (better than silently staging the wrong split).

Usage (login node, conda OPD, internet on):
    # SFT train pools (-> configs/opd_qwen3.yaml sft.data.sources):
    python scripts/prepare_sft_data.py --out-dir /path/to/scratch/opd_sft_data
    # Stage-3 DEV eval sets (-> scripts/stage3_eval.py --eval-data):
    python scripts/prepare_sft_data.py --out-dir /path/to/scratch/opd_sft_data \
        --split dev --datasets hotpotqa,nq
  options:
    --split train|dev                  # train = SFT generation pool; dev = Stage-3 eval set
    --datasets nq,hotpotqa,musique     # which of the known sets to stage
    --repo RUC-NLPIR/FlashRAG_datasets # HF dataset repo id
    --limit 0                          # 0 = stage ALL rows (the loader applies the real per-source limit)

Output basenames: train -> <ds>_with_gold.jsonl (unchanged; keeps existing config refs),
dev -> <ds>_dev_with_gold.jsonl (never overwrites the train file). After it prints the
produced absolute paths: TRAIN -> paste into configs/opd_qwen3.yaml sft.data.sources;
DEV -> pass to scripts/stage3_eval.py --eval-data (and the matching TRAIN file to --sft-data
for the contamination guard).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataio.dataset import load  # noqa: E402  (validation after write)

# dataset dir name in the FlashRAG repo  ->  output basename expected by the config
KNOWN = {
    "nq":       "nq_with_gold.jsonl",
    "hotpotqa": "hotpotqa_with_gold.jsonl",
    "musique":  "musique_with_gold.jsonl",
    # the transfer benchmarks (Search-R1's splits: 2wiki dev, bamboogle test; musique dev above)
    "2wikimultihopqa": "2wiki_with_gold.jsonl",
    "bamboogle":       "bamboogle_with_gold.jsonl",
}


def _pick_split_file(files: list[str], dataset: str, split: str) -> str:
    """Choose the `split` file for `dataset` from the repo file list.

    Prefer a path that sits under a dir named exactly `dataset`, mentions the requested
    split (train/dev), and ends in .jsonl. Filtering to files that CONTAIN the split keyword
    is what keeps a `--split dev` request from ever resolving to the train file (and vice
    versa). Fail loud with candidates if we cannot resolve unambiguously."""
    ds = dataset.lower()
    sp = split.lower()
    cands = [f for f in files if f.lower().startswith(f"{ds}/") or f"/{ds}/" in f.lower()]
    if not cands:
        # some layouts are flat: <dataset>_train.jsonl etc.
        cands = [f for f in files if ds in f.lower()]
    with_split = [f for f in cands if sp in f.lower()]   # excludes the OTHER split
    pool = with_split or cands
    jsonl = [f for f in pool if f.lower().endswith(".jsonl")]
    pool = jsonl or pool
    if not pool:
        raise SystemExit(
            f"ERROR: could not find a {split!r} file for {dataset!r} in {len(files)} repo "
            f"files. Candidates seen: {cands[:20]}")
    # shortest path is usually the canonical top-level split (avoids subset variants)
    pool.sort(key=lambda p: (0 if sp in p.lower() else 1, len(p)))
    return pool[0]


def _out_name(dataset: str, split: str) -> str:
    """train -> <ds>_with_gold.jsonl (unchanged; keeps existing config refs);
    dev -> <ds>_dev_with_gold.jsonl (never overwrites the train file)."""
    base = KNOWN[dataset]
    return base if split == "train" else base.replace("_with_gold", f"_{split}_with_gold")


def _download_with_retry(repo: str, src: str, *, retries: int = 6,
                         base_delay: float = 5.0) -> str:
    """hf_hub_download with exponential backoff on TRANSIENT throttling.

    The HF Hub returns 429 ('Too Many Requests' / 'maximum queue size reached') under
    load; it surfaces as HfHubHTTPError, often re-wrapped as LocalEntryNotFoundError.
    Both are transient -- retry with backoff (5,10,20,40,60,60s). A genuine 404
    (missing file) is NOT retried -- re-raised immediately so a real error still fails loud."""
    import time
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    last = None
    for attempt in range(1, retries + 1):
        try:
            return hf_hub_download(repo, src, repo_type="dataset")
        except (EntryNotFoundError, RepositoryNotFoundError):
            raise  # genuine 404 (file/repo missing) -> fail loud so the caller can fall back
        except (HfHubHTTPError, LocalEntryNotFoundError) as e:
            last = e
            status = getattr(getattr(e, "response", None), "status_code", None)
            transient = isinstance(e, LocalEntryNotFoundError) or status in (429, 500, 502, 503, 504)
            if not transient:
                raise  # other HTTP/auth error -> fail loud, don't mask it
            if attempt == retries:
                break
            delay = min(base_delay * (2 ** (attempt - 1)), 60.0)
            print(f"  [retry {attempt}/{retries}] HF throttled "
                  f"({status or type(e).__name__}); sleeping {delay:.0f}s...")
            time.sleep(delay)
    raise SystemExit(
        f"failed to download {src} after {retries} retries (HF rate-limit persisted): {last}. "
        "Wait a few minutes and re-run -- resolved files land in the HF cache, so completed "
        "datasets are skipped on the next attempt.")


def _list_repo_files_with_retry(repo: str, *, retries: int = 6, base_delay: float = 5.0) -> list[str]:
    """list_repo_files with the SAME backoff. Only used as a FALLBACK -- the recursive
    tree API is the heaviest/most-throttled call on this large repo, so we avoid it unless
    the canonical path is genuinely absent."""
    import time
    from huggingface_hub import list_repo_files
    from huggingface_hub.errors import HfHubHTTPError

    last = None
    for attempt in range(1, retries + 1):
        try:
            return list_repo_files(repo, repo_type="dataset")
        except HfHubHTTPError as e:
            last = e
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status not in (429, 500, 502, 503, 504) or attempt == retries:
                if status not in (429, 500, 502, 503, 504):
                    raise
                break
            delay = min(base_delay * (2 ** (attempt - 1)), 60.0)
            print(f"  [list retry {attempt}/{retries}] HF throttled ({status}); sleeping {delay:.0f}s...")
            time.sleep(delay)
    raise SystemExit(f"failed to list repo files after {retries} retries: {last}. Wait and re-run.")


def _resolve_and_download(repo: str, dataset: str, split: str) -> str:
    """Resolve the split file, avoiding the throttle-prone recursive tree listing.

    FAST PATH: FlashRAG's layout is the stable `<dataset>/<split>.jsonl`, so try that
    canonical path DIRECTLY (one resolve call, no tree API). FALLBACK (only on a genuine
    404): discover via the retry-wrapped listing + picker, preserving robustness if the
    layout ever changes."""
    from huggingface_hub.errors import EntryNotFoundError

    canonical = f"{dataset}/{split}.jsonl"
    try:
        local = _download_with_retry(repo, canonical)
        print(f"[{dataset}/{split}] resolved (canonical path, no tree listing): {canonical}")
        return local
    except EntryNotFoundError:
        print(f"[{dataset}/{split}] canonical {canonical} absent; falling back to repo listing")
    files = _list_repo_files_with_retry(repo)
    src = _pick_split_file(files, dataset, split)
    print(f"[{dataset}/{split}] resolved (via listing): {src}")
    return _download_with_retry(repo, src)


def _stage_one(repo: str, dataset: str, out_dir: str, limit: int, split: str, as_dev: bool = False) -> str:
    local = _resolve_and_download(repo, dataset, split)
    # --as-dev: bamboogle ships only a `test` split; the eval reads <ds>_dev_with_gold.jsonl
    out_path = os.path.join(out_dir, _out_name(dataset, "dev" if as_dev else split))
    kept = 0
    with open(local, "r", encoding="utf-8") as fin, \
            open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            q = (row.get("question") or "").strip()
            gold = row.get("golden_answers")
            if gold is None and "answers" in row:      # tolerate the NQ-style field name
                gold = row["answers"]
            if not q or not gold:
                continue
            fout.write(json.dumps({"question": q, "golden_answers": gold},
                                  ensure_ascii=False) + "\n")
            kept += 1
            if limit and kept >= limit:
                break
    print(f"[{dataset}/{split}] wrote {kept} rows -> {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage SFT question sets with gold (login node).")
    ap.add_argument("--out-dir", required=True, help="scratch dir to write *_with_gold.jsonl into")
    ap.add_argument("--as-dev", action="store_true", help="write the dev-named file whatever the split (bamboogle has only test)")
    ap.add_argument("--split", default="train", choices=["train", "dev", "test"],
                    help="train = SFT generation pool; dev = Stage-3 eval set (disjoint by "
                         "construction; contamination guard still proves it)")
    ap.add_argument("--datasets", default="nq,hotpotqa,musique",
                    help="comma list from: " + ",".join(KNOWN))
    ap.add_argument("--repo", default="RUC-NLPIR/FlashRAG_datasets")
    ap.add_argument("--limit", type=int, default=0,
                    help="0 = ALL rows; the real per-source limit is applied later by the loader")
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    unknown = [d for d in datasets if d not in KNOWN]
    if unknown:
        raise SystemExit(f"unknown datasets {unknown}; known = {list(KNOWN)}")

    os.makedirs(args.out_dir, exist_ok=True)
    produced = {}
    for ds in datasets:
        produced[ds] = _stage_one(args.repo, ds, args.out_dir, args.limit, args.split, as_dev=args.as_dev)

    # --- validate each through the ACTUAL loader so we fail here, not mid-SFT ---
    print("\n=== validation via dataio.dataset.load (parses gold the way training will) ===")
    for ds, path in produced.items():
        rows = load(path, name=ds, limit=3)
        n_total = len(load(path, name=ds))
        assert rows, f"{ds}: loader parsed 0 usable rows from {path}"
        r0 = rows[0]
        print(f"[{ds}] {n_total} usable rows | sample gold={r0['gold_answers']!r} "
              f"| Q: {r0['question'][:70]}")

    if args.split == "train":
        print("\n=== paste these into configs/opd_qwen3.yaml sft.data.sources ===")
        for ds, path in produced.items():
            print(f"  {ds}: {path}")
    else:
        print(f"\n=== Stage-3 eval sets ({args.split}) -> scripts/stage3_eval.py ===")
        for ds, path in produced.items():
            train_file = os.path.join(args.out_dir, KNOWN[ds])
            print(f"  {ds}:  --eval-data {path}  --dataset-name {ds}  --sft-data {train_file}")


if __name__ == "__main__":
    main()
