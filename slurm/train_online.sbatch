#!/usr/bin/env bash
#SBATCH --job-name=opd-online
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
# GPUS: 2                 # student trainer + policy pi_theta (vLLM sampling). The
#                                        # teacher is NOT here -- it is the SEPARATE served process
#                                        # (teacher_server.sbatch, its own card); this job only CALLS
#                                        # it over HTTP. PIN the type (H20 = newer driver). a800 is NOT a
#                                        # verified alt: <node>'s A800 has a CUDA 12.2 driver this torch
#                                        # cannot initialise. Do NOT leave it as plain gpu:2 -- SLURM may
#                                        # land a 4090d whose driver is too old. Driver is per-NODE:
#                                        # scripts/gpu_preflight.py is the check, gres type is not.
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=36:00:00                  # 38,400 trajectories: rollout AND training in ONE
#                                        # job = 30.9h, plus margin. Ceiling is 48h.
#                                        # measured, not guessed: 1.348 s/trajectory of
#                                        # training (weighted over the real length
#                                        # distribution) + 1.55 s/trajectory of rollout, from
#                                        # scripts/bench_train_step.py on an H20.
#                                        #
#                                        # --time is the MEASURED NEED PLUS MARGIN, not the
#                                        # 48h ceiling. A request longer than the work needs
#                                        # cannot backfill: on this partition other users hold
#                                        # 3-5 day allocations, so the scheduler only starts a
#                                        # lower-priority job it can prove finishes before the
#                                        # next reservation. That is what kept the retriever
#                                        # pending at 48h with an idle node available.
#                                        # 31h IS a bad shape to queue. Splitting this into
#                                        # resumable chunks is the fix, and it is a SCHEDULING
#                                        # argument -- the run fits the walltime fine.
#SBATCH --output=slurm/logs/online-%j.out
#
# ONLINE OPD: trajectories sampled from the CURRENT pi_theta each step, teacher scored
# live. Same OPDTrainer + opd_train_step as offline -- only the rollout source differs.
# Diagnostics (confounds #1/#2/#3) stream to runs/diag_online.jsonl.

set -euo pipefail
source slurm/env.sh

python scripts/dry_run_smoke.py          # preflight

# ---- node-level GPU preflight: this node's DRIVER, not just its gpu type ----
# `--gres=gpu:h20:N` pins hardware uniformity, not a working driver: <node> has an
# A800 whose CUDA 12.2 driver this torch cannot initialise. Fail in seconds here
# rather than after the queue wait, model load and endpoint handshakes.
# The retry budget is a knob because the transient it waits out is not ours. SLURM hands out
# the lowest-index free GPU first, so "our device 0" is systematically the card most recently
# released on that node -- which is why six consecutive cudaErrorDevicesUnavailable failures
# were ALL on device 0 and never on device 1. The default 6 x 10 s was not enough on the
# churned partitions (measured: six failures, none recovering inside 50 s), while later jobs
# on the same nodes ran fine, so the card does free -- just later. Raise it when submitting
# into a pool someone else is cycling; a genuinely dead card then costs the longer wait
# instead of one minute, which is the trade.
python scripts/gpu_preflight.py --device-tries "${OPD_DEVICE_TRIES:-6}"                                 --device-delay "${OPD_DEVICE_DELAY:-10}"

# ---- resolve the endpoints published by retriever_wiki18.sbatch + teacher_server.sbatch ----
# BLOCK until both answer -- online rolls out through the retriever AND scores the teacher
# live on EVERY step, so both must be up before the first step, not eventually.
wait_for_retriever
opd_require_student_ckpt || exit 1   # the student is this job's input
wait_for_teacher
RETRIEVER=$(cat ${OPD_RUNSTATE_DIR}/opd_retriever_host.txt)
TEACHER=$(cat "$(opd_teacher_host_file)")
echo "[online] retriever: $RETRIEVER"
echo "[online] teacher  : $TEACHER   (live-scored every step)"

# Keep vLLM's V1 EngineCore IN-PROCESS so sync_from can reach the live model to hot-update its
# weights every step. With the default subprocess EngineCore the sampled model lives in another
# process and sync_from has no in-process handle (AttributeError / it would fail loud). Set ONLY
# here (not env.sh): the served teacher runs its own vLLM and does NOT want this.
export VLLM_ENABLE_V1_MULTIPROCESSING=0

# Allocator config, NOT a numerical parameter: expandable segments let the caching
# allocator grow/shrink mappings instead of stranding fragments in fixed-size blocks.
# The 2026-08-18 online OOM missed by 10 MiB (wanted 3.96 GiB, 3.95 free) on a ~74 GiB
# resident card -- exactly the margin fragmentation eats. Set IDENTICALLY in
# train_offline.sbatch: the training allocator is part of run provenance (stamped into
# every diag record's memory.cuda_alloc_conf), and the arms must not differ in it.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- online run: SAME SHAPE VOCABULARY AS train_offline.sbatch ----
# resync_every=1 => the policy vLLM pulls current training weights EVERY step, so
# policy.staleness must read 0 every step (the "online cannot be secretly stale" check).
#
# Defaults are the 64-question smoke; the REAL run (the arm the paper's figure needs)
# overrides at submit time and MUST use the same shape as the offline arm it will be
# compared against:
#
#   OPD_LIMIT=38400 OPD_STEPS=150 bash slurm/submit.sh 2 slurm/train_online.sbatch
#
# These were HARDCODED at the smoke shape before -- the offline arm already read
# OPD_LIMIT/OPD_STEPS, so launching "the real run" with the same submit line on both
# would have silently re-run a 40-step smoke on this side only.
LIMIT="${OPD_LIMIT:-64}"
STEPS="${OPD_STEPS:-40}"
EPOCHS=$(python -c "import yaml; c=yaml.safe_load(open('configs/opd_qwen3.yaml')); print(${STEPS}*c['optim']['global_batch']/${LIMIT})")
echo "[online] shape: limit=$LIMIT steps=$STEPS -> epochs=$EPOCHS"
python scripts/train.py --config configs/opd_qwen3.yaml --mode online --out-dir "${OPD_OUT_DIR:-runs}" \
  --retriever-endpoint "$RETRIEVER" --teacher-endpoint "$TEACHER" \
  --channel "$OPD_CHANNEL" \
  --precompute-dir "${OPD_PRECOMPUTE_DIR:-precomputed}" \
  --save-every "${OPD_SAVE_EVERY:-25}" \
  --limit "$LIMIT" --steps "$STEPS" --epochs "$EPOCHS"
# EPOCHS IS NOT DECORATION (same note as train_offline.sbatch): at the smoke defaults it
# derives 160 -- online resamples fresh per visit while offline replays its cache, a
# data-diversity difference ON TOP OF rollout source, which is what makes the smoke a
# smoke. At the real shape (38400 x 150) it derives 1.0: each question drawn once per arm,
# and train.py prints "each drawn once per arm" instead of the smoke warning.
echo "[online] DONE -- diagnostics streamed to runs/diag_online.jsonl (watch policy.staleness==0)."
