# DistOPD

Code for **Rethinking Offline On-Policy Distillation: From Scalar Advantages to Distributional Supervision**.

> Paper link will be added when the arXiv version is public.

## Overview

On-policy distillation (OPD) trains a student on its own trajectories with token-level
supervision from a teacher. Offline OPD scores a frozen set of student rollouts once and
reuses the scores, which removes the live teacher from training. Existing offline OPD
compresses the teacher's supervision into a scalar per-token advantage on the sampled token.
This works for reasoning but degrades in multi-turn agentic search.

DistOPD keeps a compact distributional teacher signal instead. At every action position it
caches teacher probabilities over a small token set (the reference policy's top-K tokens
together with the teacher's top tokens) plus the remaining tail mass. The student is trained
with a forward KL toward an interpolated target between the frozen reference policy and the
teacher, controlled by `beta`. The teacher scores each rollout once, so training never queries
a live teacher.

The setting in this repository is multi-turn agentic search: a Qwen3-4B student issues
`<search>` calls against a wiki-18 retriever and answers multi-hop questions. Two teachers are
supported, MiroThinker-1.7-mini (`OPD_LINE=miro`) and Qwen3-32B (`OPD_LINE=dense`).

## Repository layout

| path | contents |
|---|---|
| `opd/` | losses (`loss.py` for scalar-advantage OPD, `delta_loss.py` for DistOPD), masking, trainer |
| `teacher/` | teacher scoring backends (vLLM over HTTP, HF), top-K head extraction |
| `rollout/` | multi-turn search rollouts at the token-id level, vLLM student policy |
| `envs/` | retriever client, tokenizer helpers |
| `sft/`, `dataio/` | reference-policy SFT data generation and export, dataset loading |
| `configs/` | `opd_qwen3.yaml` (all hyperparameters), GPU channel configs, chat template |
| `scripts/` | entry points: `train.py`, `stage3_eval.py`, cache builders, smoke tests |
| `slurm/` | job templates for every pipeline stage, site configuration |
| `tests/` | CPU unit tests (GPU and model-weight tests are marked `slow`) |

## Quick check on CPU

No model weights, GPU or retriever needed. This runs the full online and offline code path on a
mock policy and teacher.

```bash
pip install -r requirements.txt
pytest                              # slow tests are deselected by default
python scripts/dry_run_smoke.py     # online and offline parity on one trajectory
python scripts/dry_run_train.py     # a few trainer steps in both modes
```

## Full pipeline (SLURM)

Install the GPU stack from `requirements-hpc.txt` (install vLLM first and let it choose torch).
Then describe your cluster once:

```bash
cp slurm/sites/example.sh slurm/sites/mycluster.sh   # fill in paths, partition, gres
export OPD_SITE=mycluster
```

Every stage below is submitted with `bash slurm/submit.sh <n_gpus> <template>`, which fills in
the partition and gres from the site file. Each template documents its own environment
variables in its header.

1. **Retriever.** Download the wiki-18 corpus and index with `scripts/download_wiki18.sh`, then
   start the Search-R1 retriever with `slurm/retriever_wiki18.sbatch`.
2. **Data.** Build question and gold-answer JSONL files with `scripts/prepare_sft_data.py`.
3. **Reference policy.** Generate teacher trajectories with `slurm/gen_sft.sbatch`, then run full
   SFT of Qwen3-4B-Base with `slurm/sft_train.sbatch` (LlamaFactory). The result is `pi_ref`,
   the starting point of every OPD run.
4. **Teacher server.** Start `slurm/teacher_server.sbatch` for the chosen teacher.
5. **Rollout cache.** Sample `pi_ref` rollouts and score them with the teacher once:
   `OPD_PRECOMPUTE_DIR=precomputed_<line>_full slurm/precompute_teacher.sbatch`.
6. **DistOPD caches.** `slurm/ref_topk_dump.sbatch` stores the reference policy's top-K tokens
   and `slurm/teacher_head.sbatch` stores teacher probabilities on the token set.
7. **Training.**
   * DistOPD: `OPD_LINE=<line> OPD_BETA=0.5 slurm/train_delta.sbatch`
   * scalar-advantage offline OPD (Lightning OPD loss): `slurm/train_offline.sbatch`
   * online OPD: `slurm/train_online.sbatch`
8. **Evaluation.** `slurm/stage3_eval.sbatch` reports EM and F1 with the same retriever.

All training hyperparameters (learning rate, batch size, steps, turn cap, sequence budget,
advantage clip) live in `configs/opd_qwen3.yaml`. GPU-specific memory settings live in
`configs/channels/`.

## Consistency checks

Online and offline OPD share one code path, and only the rollout source differs. Several gates
guard this. `slurm/certify_channel.sbatch` runs the tokenizer identity, teacher determinism and
policy invariance probes on a new GPU type. `scripts/check_channel_parity.py` confirms that two
runs being compared used the same teacher backend and GPU channel.

## Citation

A BibTeX entry will be added with the arXiv version.

## License

This code is released under the [MIT License](LICENSE).
