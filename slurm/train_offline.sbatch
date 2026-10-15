#!/usr/bin/env bash
#SBATCH --job-name=opd-offline
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
# GPUS: 1                 # ONE card: the student trainer. Offline training with a
#                                        # manifest present is READ-ONLY -- train.py skips the vLLM
#                                        # policy and the retriever entirely (it prints "[offline]
#                                        # manifest present -> read-only"), so the second card sat
#                                        # idle while doubling the queue wait. The teacher is the
#                                        # SEPARATE served process (teacher_server.sbatch) and is only
#                                        # CALLED over HTTP for the gate-1 /health fingerprint.
#                                        # PIN the type (H20). a800 is NOT a
#                                        # verified alternative: <node>'s A800 carries a CUDA 12.2
#                                        # driver this torch cannot initialise, same failure class as
#                                        # the consumer 4090d. Driver version is per-NODE, so another
#                                        # a800 node may work -- run scripts/gpu_preflight.py there
#                                        # before believing it.
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=18:00:00                  # 38,400 trajectories read from the cache and
#                                        # trained = 14.4h, plus margin. No rollout here.
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
#SBATCH --output=slurm/logs/offline-%j.out
#
# OFFLINE OPD: reads the precomputed teacher_logp cache (precompute_teacher.sbatch must
# have run first). gate 1 (assert_same_backend) fires at the disk boundary, so a cache
# scored by a different backend than this run's expected teacher is rejected, not used.
# Diagnostics stream to runs/diag_offline.jsonl (context-quality is the FROZEN baseline).

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
# offline still rolls out trajectories through the retriever, and scores the SAME served
# teacher online uses (gate 1 holds by construction). If precompute_teacher.sbatch already
# wrote precomputed/manifest.json, this job READS it; otherwise it self-precomputes here.
# The teacher is ALWAYS needed (make_teacher runs the gate-1 /health check and supplies the
# expected backend for the disk boundary), so block on it unconditionally. The retriever is
# needed ONLY when there is no manifest to read -- in that case train.py self-precomputes and
# rolls out. Waiting on the retriever regardless would block (and eventually fail) a run that
# never touches it, so gate that wait on the same condition the code uses.
opd_require_student_ckpt || exit 1   # the student is this job's input
wait_for_teacher
# The self-precompute fallback is now a HARD ERROR, not a silent path. It would build the
# vLLM policy (cuda:0) alongside the student (cuda:1) -- two cards -- and this job requests
# one. Rather than let it die on "invalid device ordinal" twenty minutes into a queue wait,
# refuse up front and name the job that produces the cache.
if [ ! -f precomputed/manifest.json ]; then
  echo "[offline] FATAL: precomputed/manifest.json is missing." >&2
  echo "[offline] This job is the READ-ONLY offline arm and requests a single card." >&2
  echo "[offline] Run \`sbatch slurm/precompute_teacher.sbatch\` first (it rolls out and" >&2
  echo "[offline] teacher-scores to disk on its own card), then resubmit this." >&2
  exit 1
fi
RETRIEVER=$(cat ${OPD_RUNSTATE_DIR}/opd_retriever_host.txt)
TEACHER=$(cat "$(opd_teacher_host_file)")
echo "[offline] retriever: $RETRIEVER"
echo "[offline] teacher  : $TEACHER"

# ---- REAL offline run (first signal: reads precomputed/ cache, trains off disk) ----
# gate 1 (assert_same_backend) fires when the manifest is read: a cache scored by a
# different backend fp than this run's expected teacher is REJECTED, not silently used.
#
# SHAPE: defaults are the 64-question smoke; the real run overrides at submit time and MUST
# use the same LIMIT the cache was precomputed with, or it trains on a subset of what was
# scored:
#
#   OPD_LIMIT=38400 OPD_STEPS=150 bash slurm/submit.sh 1 slurm/train_offline.sbatch --time=16:00:00
#
# Allocator config, NOT a numerical parameter -- see train_online.sbatch for the OOM
# incident that introduced it. Set IDENTICALLY in both arms: the training allocator is
# part of run provenance (stamped into every diag record's memory.cuda_alloc_conf),
# and the arms must not differ in it.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LIMIT="${OPD_LIMIT:-64}"
STEPS="${OPD_STEPS:-40}"
EPOCHS=$(python -c "import yaml; c=yaml.safe_load(open('configs/opd_qwen3.yaml')); print(${STEPS}*c['optim']['global_batch']/${LIMIT})")
# ~16 GiB per slot for a 4B fp32 student, two slots per arm. The final ckpt is written
# separately, so budget ~48 GiB per arm on scratch. 0 disables it.
SAVE_EVERY="${OPD_SAVE_EVERY:-25}"
echo "[offline] shape: limit=$LIMIT steps=$STEPS -> epochs=$EPOCHS"
python scripts/train.py --config configs/opd_qwen3.yaml --mode offline --out-dir "${OPD_OUT_DIR:-runs}" \
  --retriever-endpoint "$RETRIEVER" --teacher-endpoint "$TEACHER" \
  --channel "$OPD_CHANNEL" \
  --precompute-dir "${OPD_PRECOMPUTE_DIR:-precomputed}" \
  --save-every "$SAVE_EVERY" \
  --limit "$LIMIT" --steps "$STEPS" --epochs "$EPOCHS"
# EPOCHS IS NOT DECORATION. At the smoke defaults it is 160: 40 steps x global_batch 256 =
# 10,240 draws over 64 cached questions, so offline replays each cached trajectory 160 times
# while online resamples a fresh one each visit. That is a data-diversity difference ON TOP OF
# the rollout-source difference, which makes the smoke a smoke and not an experiment -- and
# train.py prints exactly that whenever epochs != 1. The real run (LIMIT 38400, STEPS 150)
# derives epochs = 1.0 and prints "each drawn once per arm" instead.
echo "[offline] DONE -- diagnostics streamed to runs/diag_offline.jsonl (the FROZEN baseline)."
