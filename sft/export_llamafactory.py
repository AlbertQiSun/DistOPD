"""M-SFT Stage 2: export Stage-1 records -> a LlamaFactory-ingestible PRE-TOKENIZED shard
(+ a dataset_info stub + a training-config stub). Emits files; does NOT run LlamaFactory.

WHY PRE-TOKENIZED (not sharegpt text)
-------------------------------------
Handing LlamaFactory chat TEXT invites the two silent failures sft/pack.py fights: a
re-tokenize drifts ids off MiroThinker's exact emission, and role-based masking supervises
the retrieved-doc (OBSERVATION) tokens inlined in the assistant turn. So we export the exact
input_ids + labels and require the framework to honor them verbatim.

LlamaFactory has NO dataset_info route that ingests raw input_ids/labels -- every dataset_info
converter tokenizes from TEXT. The ONE contract that loads our exact ids/labels verbatim is
`tokenized_path`: get_dataset() (data/loader.py) calls datasets.load_from_disk(tokenized_path)
and returns early, BYPASSING all tokenization/preprocessing/packing. So the emitted yaml uses
tokenized_path, and sft/build_tokenized_dataset.py pre-builds that arrow dataset from the
pretokenized.jsonl this exporter writes. After LlamaFactory loads it, run
scripts/verify_llamafactory_mask.py --processed <tokenized_path> to assert (against the actual
loaded data) that the action-only mask survived -- the framework-seam gate.

HARD GUARD -- cutoff_len >= rollout.max_len
-------------------------------------------
Any record longer than cutoff_len is truncated by the packer, clipping ACTION tokens and
corrupting the mask. Export REFUSES if any record exceeds cutoff_len (set cutoff_len >= the
Stage-1 rollout.max_len, 4096).

    python -m sft.export_llamafactory <stage1_out_dir> --out llamafactory_shard \
        --cutoff-len 4096 --base-model /path/to/Qwen3-4B-Base
"""
from __future__ import annotations

import argparse
import json
import os

from sft.pack import count_manifest_kept, pack_records_from_manifest, write_packed_jsonl


def export(stage1_out_dir: str, out_dir: str, cutoff_len: int, base_model: str,
           dataset_name: str = "sft_pi_ref", grad_accum: int = 8,
           deepspeed: str | None = None, max_steps: int | None = None,
           ckpt_name: str = "pi_ref_ckpt", save_steps: int = 500) -> dict:
    records_dir = os.path.join(stage1_out_dir, "records")
    if not os.path.isdir(records_dir):
        raise FileNotFoundError(f"no records/ under {stage1_out_dir} (run gen_sft_data.py first)")
    os.makedirs(out_dir, exist_ok=True)

    # PROVENANCE GUARD: pack ONLY what THIS run's manifest marks kept -- never glob records/. A
    # stale .pt from a prior/aborted run (datagen's drop path leaves it on disk at a now-dropped
    # index) is then structurally unable to enter the shard, instead of relying on a pre-export
    # `rm -rf records/`. This is the fix for the mixed-provenance export that revived the exact
    # old-retriever confound the wiki-18 migration exists to eliminate.
    rows = pack_records_from_manifest(stage1_out_dir)
    if not rows:
        raise ValueError(f"manifest under {stage1_out_dir} marks zero kept records")

    # LAYER-3 COUNT BACKSTOP: even if the pack logic ever regressed, assert the exported count
    # equals the manifest's INDEPENDENT kept count. The 3547-vs-3876 class of bug then fails
    # LOUDLY here instead of silently surviving into pi_ref training -- turning last time's
    # lucky eyeball-catch into a structural guarantee.
    manifest_kept = count_manifest_kept(stage1_out_dir)
    if len(rows) != manifest_kept:
        raise ValueError(
            f"export/manifest count mismatch: packed {len(rows)} but manifest marks "
            f"{manifest_kept} kept. Provenance guard tripped -- do NOT train on this shard.")

    # HARD GUARD: nothing may exceed cutoff_len (else silent truncation -> mask corruption).
    too_long = [(r["_id"], len(r["input_ids"])) for r in rows if len(r["input_ids"]) > cutoff_len]
    if too_long:
        raise ValueError(
            f"{len(too_long)} record(s) exceed cutoff_len {cutoff_len} (e.g. {too_long[:3]}). "
            "They would be TRUNCATED, clipping ACTION tokens. Raise --cutoff-len to >= the "
            "Stage-1 rollout.max_len, or regenerate with a smaller max_len.")

    # (a) the file LlamaFactory trains on: pre-tokenized, training fields ONLY.
    train_path = os.path.join(out_dir, "pretokenized.jsonl")
    with open(train_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"input_ids": r["input_ids"], "labels": r["labels"],
                                "attention_mask": r["attention_mask"]}, ensure_ascii=False) + "\n")

    # (b) full rows WITH _token_type, for scripts/verify_llamafactory_mask.py (gate provenance).
    verify_path = os.path.join(out_dir, "packed_with_token_type.jsonl")
    write_packed_jsonl(rows, verify_path)

    # (c) dataset_info.json -- FALLBACK/validation-satisfier ONLY. The real ingestion goes
    # through `tokenized_path` (see the yaml + sft/build_tokenized_dataset.py): when that arrow
    # dir exists, LlamaFactory's get_dataset() load_from_disk()s it and returns early, so THIS
    # entry is never consulted. It is kept only so `dataset:` name-validation passes. It maps
    # to input_ids/labels ON PURPOSE: if tokenized_path is ever missing, the alpaca converter
    # errors LOUDLY on the absent prompt/response columns instead of silently mis-tokenizing.
    dataset_info = {
        dataset_name: {
            "file_name": "pretokenized.jsonl",
            "formatting": "alpaca",
            "columns": {"input_ids": "input_ids", "labels": "labels",
                        "attention_mask": "attention_mask"},
        }
    }
    with open(os.path.join(out_dir, "dataset_info.json"), "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)

    # (d) training-config stub. The action-only mask survives ONLY if LlamaFactory loads our
    # exact input_ids/labels WITHOUT re-tokenizing or re-masking by chat role. The single
    # contract that guarantees that is `tokenized_path`: LlamaFactory data/loader.py::get_dataset
    # calls datasets.load_from_disk(tokenized_path) and returns early, BYPASSING all
    # tokenization/preprocessing (dataset_info converters + packing are all skipped). So we
    # PRE-BUILD the arrow dataset (sft/build_tokenized_dataset.py) at <out_dir>/tokenized and
    # point tokenized_path there; `dataset`/`dataset_dir` only satisfy arg-validation and are
    # ignored once the arrow loads. Because preprocessing is skipped, packing is a NO-OP here
    # (nothing to pack across) -- keep packing/neat_packing FALSE so the collator uses standard
    # 2D padding (no position_ids requirement). SFT hyperparameters: Lightning OPD SFT stage
    # (lr 8e-5, cosine). Batch sizes are memory-driven defaults (unpacked seqs up to cutoff_len).
    abs_out = os.path.abspath(out_dir)
    # DeepSpeed line (optional). ZeRO-2 shards the 64 GB full-FT optimizer/grad state across ranks
    # so it fits <48 GB cards (e.g. L20x8). It touches ONLY the optimizer/grad wrapping -- the data
    # path is untouched: get_dataset() still hits tokenized_path and returns early, so both mask
    # gates still hold. Empty when --deepspeed is unset (H20/A800 pure DDP: 64 GB fits one card).
    ds_line = f"deepspeed: {os.path.abspath(deepspeed)}\n" if deepspeed else ""
    # token-matched budget (scripts/sft_budget.py) for the continued-SFT baselines: LlamaFactory
    # stops at max_steps when it is > 0, whatever num_train_epochs says. 0 / None = epochs rule.
    steps_line = f"max_steps: {int(max_steps)}                   # token-matched budget, overrides num_train_epochs\n" if max_steps else ""
    yaml_stub = f"""### M-SFT Stage 2 pi_ref training (LlamaFactory) -- PRE-TOKENIZED via tokenized_path.
### The action-only mask is preserved ONLY because LlamaFactory loads our exact input_ids/labels
### verbatim: get_dataset() load_from_disk()s tokenized_path and returns early, skipping ALL
### tokenization/preprocessing. PRE-BUILD it first:
###   python -m sft.build_tokenized_dataset {abs_out}/pretokenized.jsonl --out {abs_out}/tokenized
model_name_or_path: {base_model}
stage: sft
do_train: true
finetuning_type: full            # pi_ref is a full SFT of Qwen3-4B-Base
dataset: {dataset_name}          # IGNORED once tokenized_path loads (arg-validation satisfier only)
dataset_dir: {abs_out}
tokenized_path: {abs_out}/tokenized   # <- THE data source; pre-built arrow, loaded verbatim
cutoff_len: {cutoff_len}         # >= Stage-1 rollout.max_len; export refuses longer records
packing: false                   # tokenized_path BYPASSES preprocessing, so packing is a no-op;
neat_packing: false              # keep both false -> standard 2D padding collator (no position_ids)
preprocessing_num_workers: 8
overwrite_cache: true
### Lightning OPD SFT stage:
learning_rate: 8.0e-5
lr_scheduler_type: cosine
num_train_epochs: 3.0            # or set max_steps -- calibrate to shard size
{steps_line}per_device_train_batch_size: 1   # unpacked seqs up to cutoff_len ({cutoff_len}); keep 1, scale grad_accum
gradient_accumulation_steps: {grad_accum}   # effective batch = 1 * grad_accum * NPROC (keep = 32)
gradient_checkpointing: true     # long-tail (~14k-token) trajectories -> checkpoint activations
{ds_line}bf16: true
warmup_ratio: 0.03
logging_steps: 10
save_steps: {int(save_steps)}                  # the SeqKD baseline sets this to its steps-per-epoch (addendum 2026-09-10)
report_to: none                  # compute nodes have NO internet; LlamaFactory auto-enables wandb
                                 # + mlflow, which then crash on api_key(no-tty)/DNS. none = disable
                                 # ALL reporters (loss still prints to stdout via logging_steps).
overwrite_output_dir: true       # START FRESH -- never silently auto-resume from a checkpoint a
                                 # PRIOR run left in output_dir. With this false (LlamaFactory's
                                 # default) it calls get_last_checkpoint(output_dir) and resumes; a
                                 # stale FINAL checkpoint then makes the run a 0-step no-op
                                 # (train_runtime ~0, loss 0, "skip the first N epochs") while BOTH
                                 # gates still pass -- pi_ref is then a prior run's weights of
                                 # UNVERIFIED provenance, not a fresh train on this shard. Resume is
                                 # opt-in via resume_from_checkpoint, never implicit.
output_dir: {abs_out}/{ckpt_name}
### AFTER LlamaFactory loads this dataset, GATE the mask BEFORE trusting pi_ref:
###   python scripts/verify_llamafactory_mask.py {stage1_out_dir} \\
###       --processed {abs_out}/tokenized --cutoff-len {cutoff_len}
"""
    with open(os.path.join(out_dir, "qwen3_4b_sft.yaml"), "w", encoding="utf-8") as f:
        f.write(yaml_stub)

    n_action = sum(sum(1 for x in r["labels"] if x != -100) for r in rows)
    n_tokens = sum(len(r["input_ids"]) for r in rows)
    summary = {"records": len(rows), "tokens": n_tokens, "action_tokens": n_action,
               "action_frac": (n_action / n_tokens) if n_tokens else 0.0,
               "cutoff_len": cutoff_len, "out_dir": os.path.abspath(out_dir)}
    with open(os.path.join(out_dir, "export_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Export Stage-1 SFT records to a LlamaFactory "
                                             "pre-tokenized shard (+ config stubs).")
    ap.add_argument("stage1_out_dir", help="the gen_sft_data.py --out-dir (has records/)")
    ap.add_argument("--out", required=True, help="output dir for the shard + stubs")
    ap.add_argument("--cutoff-len", type=int, default=4096,
                    help="LlamaFactory cutoff_len; MUST be >= Stage-1 rollout.max_len")
    ap.add_argument("--base-model", default="REPLACE_ME/Qwen3-4B-Base",
                    help="the SFT base (pi_ref = full SFT of Qwen3-4B-Base)")
    ap.add_argument("--dataset-name", default="sft_pi_ref")
    ap.add_argument("--grad-accum", type=int, default=8,
                    help="gradient_accumulation_steps; scale so 1*grad_accum*NPROC stays ~32 "
                         "(4 DDP -> 8; 8 ranks -> 4)")
    ap.add_argument("--max-steps", type=int, default=0, help="token-matched step budget; 0 = the epochs rule")
    ap.add_argument("--ckpt-name", default="pi_ref_ckpt", help="output_dir basename under --out")
    ap.add_argument("--save-steps", type=int, default=500, help="intermediate checkpoint interval (steps)")
    ap.add_argument("--deepspeed", default=None,
                    help="path to a DeepSpeed config (e.g. configs/ds_z2_config.json) to emit a "
                         "`deepspeed:` line; needed for full-FT on <64 GB cards (L20). Omit for "
                         "H20/A800 pure DDP.")
    args = ap.parse_args()
    s = export(args.stage1_out_dir, args.out, args.cutoff_len, args.base_model, args.dataset_name,
               args.grad_accum, args.deepspeed, args.max_steps or None, args.ckpt_name, args.save_steps)
    print(f"exported {s['records']} records, {s['tokens']} tokens "
          f"({s['action_frac']:.1%} action) -> {s['out_dir']}")
    print(f"  pretokenized.jsonl (feed to LlamaFactory) + dataset_info.json + qwen3_4b_sft.yaml")
    print(f"  AFTER LlamaFactory preprocesses: run scripts/verify_llamafactory_mask.py to gate "
          f"the mask round-trip.")


if __name__ == "__main__":
    main()
