"""M-SFT Stage 2: build the LlamaFactory `tokenized_path` arrow dataset from the exported
pre-tokenized shard. Emits a save_to_disk() arrow dir; does NOT run LlamaFactory.

WHY THIS EXISTS (the load-bearing Stage-2 seam)
-----------------------------------------------
LlamaFactory has NO dataset_info route that ingests raw input_ids/labels -- every dataset_info
converter tokenizes from TEXT, which DRIFTS ids off MiroThinker's exact emission and re-masks by
chat role (supervising the retrieved-doc OBSERVATION tokens). Those are exactly the two silent
failures sft/pack.py fights. The ONLY LlamaFactory contract that loads our exact input_ids/labels
VERBATIM is `tokenized_path`: get_dataset() (src/llamafactory/data/loader.py) checks
has_tokenized_data(tokenized_path) and, if present, calls datasets.load_from_disk() and RETURNS
EARLY -- bypassing all tokenization, role-masking, and packing. So we pre-build that arrow dataset
here and point the training yaml's tokenized_path at it.

CONTRACT (verified against LlamaFactory source)
-----------------------------------------------
  * columns: input_ids, attention_mask, labels  -- exactly what the supervised collator reads.
    Missing multimodal keys (images/videos/audios) are handled by the collator's
    `feature.pop(k, None) or []`, so a text-only row needs none of them.
  * shape: a DatasetDict with a "train" split -> get_dataset_module() maps train -> train_dataset.
  * ORDER: rows are kept in pretokenized.jsonl order (export writes records sorted by filename),
    which is the SAME order scripts/verify_llamafactory_mask.py expects when it gates the mask.

Run this in the LlamaFactory env (so the `datasets` version that save_to_disk() writes with is
the same one load_from_disk() reads with at train time).

    python -m sft.build_tokenized_dataset <shard>/pretokenized.jsonl --out <shard>/tokenized
"""
from __future__ import annotations

import argparse
import json


def build(pretokenized_jsonl: str, out_dir: str) -> tuple[int, str]:
    from datasets import Dataset, DatasetDict

    rows: list[dict] = []
    with open(pretokenized_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            # keep ONLY the training columns, in the framework's expected order; never carry
            # _token_type (that is gate-only provenance and would confuse the collator).
            rows.append({
                "input_ids": d["input_ids"],
                "attention_mask": d["attention_mask"],
                "labels": d["labels"],
            })
    if not rows:
        raise ValueError(f"no rows in {pretokenized_jsonl} (run sft.export_llamafactory first)")

    ds = Dataset.from_list(rows)
    DatasetDict({"train": ds}).save_to_disk(out_dir)
    return len(rows), out_dir


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build the LlamaFactory tokenized_path arrow dataset from a pre-tokenized shard.")
    ap.add_argument("pretokenized_jsonl", help="<shard>/pretokenized.jsonl from sft.export_llamafactory")
    ap.add_argument("--out", required=True, help="tokenized_path dir to save_to_disk (arrow)")
    args = ap.parse_args()
    n, out = build(args.pretokenized_jsonl, args.out)
    print(f"built tokenized_path arrow dataset: {n} rows -> {out}")
    print("  columns: input_ids, attention_mask, labels (DatasetDict['train']; loaded verbatim "
          "via load_from_disk -> NO re-tokenize, NO role re-mask)")
    print(f"  point the yaml's tokenized_path at: {out}")


if __name__ == "__main__":
    main()
