#!/usr/bin/env bash
#SBATCH --job-name=opd-gen-sft
#
# PARTITION / ACCOUNT / GRES ARE DELIBERATELY NOT HERE. They are site vocabulary, and
# slurm/submit.sh supplies all three:
#
#     OPD_SITE=<site> bash slurm/submit.sh <n_gpus> <this file>
#
# They cannot stay as #SBATCH lines. A command line beats an #SBATCH line only when it SUPPLIES
# the option -- so a site that declares NO account (the A100 cluster: AllowAccounts=ALL) leaves
# the H20 cluster's `--account=<netid>` in force and the submission is rejected outright. Same trap for
# a gres TYPE: `gpu:h20:2` survives into a cluster that has no h20. Observed 2026-08-16.
# GPUS: 1                 # the GENERATOR only = MiroThinker via vLLM (SFT samples
#                                        # tokens, never scores logp -> no separate teacher GPU).
#                                        # PIN the type (H20 = newer driver); do NOT leave plain gpu:1.
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/gen-sft-%j.out
#
# M-SFT Stage 1: generate MiroThinker SFT data in the OPD env (same retriever / template /
# tag protocol / max_turns-max_len-top_k the OPD run uses). Zero quality filtering -- we take
# MiroThinker's teacher distribution as-is; only structurally-broken trajectories drop. The
# generator IS the OPD teacher (cfg.model.teacher = MiroThinker); generate_sft_dataset asserts
# that model identity before writing anything.

set -euo pipefail
source slurm/env.sh

python scripts/dry_run_smoke.py          # preflight (CPU/mock; proves wiring on this node)

# ---- resolve the retriever endpoint published by slurm/retriever_wiki18.sbatch ----
ENDPOINT=$(cat ${OPD_RUNSTATE_DIR}/opd_retriever_host.txt)
echo "[gen-sft] using retriever endpoint: $ENDPOINT"

# PER-LINE. Three teacher lines write three shards; the default keeps the historical
# single-line path, so nothing that used to work changes.
OUT=${OPD_SFT_OUT:-runs/sft_shard1}

# ---- STEP A: tiny SMOKE (32 questions) -- GREEN on the NEW wiki-18 retriever (2026-07-22):
# no_search=0, near_cap=0, and scripts/dump_sft_observation.py confirmed the injected
# observation text is clean (return_scores=true drops the score field -- only passage text is
# injected into the OBSERVATION span, no score floats leak in). Commented out so we don't reload
# MiroThinker to re-run a passed smoke; to re-gate, uncomment these two lines AND re-add an
# `exit 0` after them so the job stops for inspection. ----
# python scripts/gen_sft_data.py --config configs/opd_qwen3.yaml --out-dir ${OUT}_smoke \
#   --retriever-endpoint "$ENDPOINT" --limit 32
# python scripts/check_sft_stats.py ${OUT}_smoke

# ---- STEP B: full first shard. sft.data.sources sets the difficulty mix; with zero filtering
# KEPT ~= generated, so budget ~3000-3200 questions for ~3000 kept.
# LAYER-1 PROVENANCE: start from an EMPTY records/ so datagen's drop path (which leaves the old
# sft_{i}.pt on disk at a now-dropped index) can never ACCUMULATE stale trajectories across runs.
# The export is manifest-scoped and would ignore stragglers anyway, but this keeps the dir itself
# honest so a manual glob or a human `ls` never sees a leftover from a prior/aborted generation. ----
rm -rf ${OUT}/records
python scripts/gen_sft_data.py --config configs/opd_qwen3.yaml --out-dir $OUT \
  --retriever-endpoint "$ENDPOINT"
python scripts/check_sft_stats.py $OUT             # HEALTH REPORT (only no_search can FAIL);
                                                   # reads gen_config.max_len from stats.json

# ---- STEP C: export the pre-tokenized shard for LlamaFactory (Stage 2). ----
python -m sft.export_llamafactory $OUT --out $OUT/llamafactory_shard \
  --cutoff-len 16384 --base-model ${OPD_MODELS_DIR}/Qwen3-4B-Base
  # cutoff-len MUST match rollout.max_len (16384) or the export re-clips what generation kept.
python scripts/verify_llamafactory_mask.py $OUT --cutoff-len 16384   # self-check export is clean

echo "STEP B/C done. Read the health report + mask verify VERDICT above."
